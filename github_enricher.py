"""GitHub enrichment for the resume extractor.

Adds four things to each resume's row:

    github_profile       GitHub profile URL(s) found in the resume
    repos_count          repositories analysed (forks skipped by default)
    github_tech_stack    technologies detected across those repositories
    project_categories   repository categories with counts, e.g. "MERN (3); Frontend (2)"

The heavy lifting is done by two files that are used UNCHANGED:

    resume.py     finds GitHub profiles in a resume, including links that are hidden
                  behind text such as "GitHub" (PDF annotations, DOCX relationships)
    analyzer.py   lists a profile's repositories (GitHub API), clones each one without
                  file contents, detects technologies, and computes the categories

This module only connects them to the rest of the extractor:

    extract_links(data)                 hidden hyperlink targets of a downloaded resume
    find_profiles(text, links)          GitHub usernames, most likely first
    analyze_profiles(profile_lists, ..) one GitHubInfo per resume, all repositories
                                        analysed together (a repository shared by
                                        several resumes is analysed once)
    rate_limit_status(token)            how many GitHub API requests are left (free call)
    summarize(infos)                    counts per outcome, to show after a run

Listing a profile's repositories costs one GitHub API request, and GitHub allows only 60 per
hour without a token (5,000 with one). A batch with more profiles than that does not fail
gracefully by itself: the first ~60 profiles succeed and every later one fails. Pass the
result of ``rate_limit_status`` to ``analyze_profiles`` (``max_profiles`` / ``rate``) so
profiles that cannot fit are skipped with a clear reason instead.

Needs network access to api.github.com and github.com, and the ``git`` command.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

import requests

import analyzer
import resume as resume_reader

logger = logging.getLogger(__name__)

__all__ = [
    "GITHUB_COLUMNS",
    "PLACEHOLDER_USERNAMES",
    "RESERVED_REQUESTS",
    "GitHubInfo",
    "RateLimit",
    "analyze_profiles",
    "extract_links",
    "find_profiles",
    "git_available",
    "profile_url",
    "rate_limit_status",
    "summarize",
]

# Resume templates often keep their placeholder text ("github.com/yourusername"); these are
# not real accounts and would only waste an API request and end up as "user not found".
PLACEHOLDER_USERNAMES = frozenset({
    "yourusername", "your-username", "username", "yourname", "your-name", "yourgithub",
    "firstlast", "first-last", "firstname-lastname", "johndoe", "john-doe", "janedoe", "jane-doe-example",
    "user", "example", "github",
})
# Requests kept in reserve when sizing a run: users with more than 100 repositories need extra
# pages, and other programs on the same network address may be using the same quota.
RESERVED_REQUESTS = 3

GITHUB_COLUMNS = [
    "github_profile", "repos_count", "github_tech_stack", "project_categories",
    "github_status", "github_notes",
]


def git_available() -> bool:
    """The analyzer clones repositories with the ``git`` command."""
    return shutil.which("git") is not None


def profile_url(username: str) -> str:
    return f"https://github.com/{username}"


# --------------------------------------------------------------------------- #
# Profile detection
# --------------------------------------------------------------------------- #

def extract_links(data: bytes, url: str = "") -> list[str]:
    """Hyperlink targets of a downloaded resume (PDF annotations, DOCX relationships, HTML anchors).

    Visible text is read elsewhere; this exists because a link whose visible text is just
    "GitHub" only exists as an annotation. Best effort: returns ``[]`` if the file cannot be parsed.
    """
    try:
        _text, links = resume_reader.extract_text_and_links(data, "", url)
        return list(links)
    except Exception as exc:  # noqa: BLE001 - a bad link table must never fail the resume
        logger.debug("could not extract hyperlinks: %s", exc)
        return []


def find_profiles(text: str, links: Iterable[str]) -> list[str]:
    """GitHub usernames found in the resume (at most a few), most likely first.

    Reserved github.com paths (``github.com/features``), words that follow "GitHub:" but are
    not usernames, and template placeholders (``yourusername``) are ignored; hyperlink targets
    outweigh visible URLs.
    """
    found = resume_reader.find_github_profiles(text or "", list(links))
    return [u for u in found if u.lower() not in PLACEHOLDER_USERNAMES]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RateLimit:
    """The GitHub API quota as reported by ``GET /rate_limit`` (which does not use up quota)."""

    limit: int
    remaining: int
    reset: int                    # epoch seconds
    authenticated: bool

    def reset_text(self) -> str:
        return time.strftime("%H:%M", time.localtime(self.reset))

    def budget(self, reserve: int = RESERVED_REQUESTS) -> int:
        """How many profiles can safely be listed right now (one request each)."""
        return max(self.remaining - reserve, 0)


def rate_limit_status(token: str | None = None, timeout: float = 10) -> RateLimit | None:
    """Current API quota, or ``None`` if it cannot be determined (offline, bad token, ...)."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        core = r.json()["resources"]["core"]
        return RateLimit(int(core["limit"]), int(core["remaining"]), int(core["reset"]), bool(token))
    except Exception as exc:  # noqa: BLE001 - the quota check is advisory
        logger.debug("could not read the GitHub rate limit: %s", exc)
        return None


def _limit_note(rate: RateLimit | None) -> str:
    if rate is None:
        return "Skipped: GitHub API limit reached. Add a GitHub token or re-run later"
    kind = "with a token" if rate.authenticated else "without a token (limit 60/hour)"
    return (
        f"Skipped: only {rate.remaining} of {rate.limit} GitHub API requests were left {kind}; resets at "
        f"{rate.reset_text()}. Add a GitHub token (5,000/hour) or re-run after the reset"
    )


@dataclass
class GitHubInfo:
    """GitHub results for one resume."""

    profiles: list[str]
    status: str = ""                       # "" | no_github_profile | done | failed | skipped_rate_limit
    repos_count: int | None = None         # None when the profile could not be read
    tech_stack: list[str] = field(default_factory=list)            # most common first
    categories: list[tuple[str, int]] = field(default_factory=list)  # (category, repository count)
    notes: list[str] = field(default_factory=list)

    def columns(self) -> dict[str, str | int]:
        """CSV cells. Lists are '; '-separated like the other technology columns."""
        return {
            "github_profile": "; ".join(profile_url(p) for p in self.profiles),
            "repos_count": "" if self.repos_count is None else self.repos_count,
            "github_tech_stack": "; ".join(self.tech_stack),
            "project_categories": analyzer.format_categories(self.categories),
            "github_status": self.status,
            "github_notes": "; ".join(self.notes),
        }


def analyze_profiles(
    profile_lists: list[list[str]],
    token: str | None = None,
    skip_forks: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    max_profiles: int | None = None,
    rate: RateLimit | None = None,
) -> list[GitHubInfo]:
    """Analyse the GitHub profiles of many resumes at once.

    ``profile_lists[i]`` holds the usernames found in resume ``i``; the result is aligned with
    it. Every repository is analysed once even when several resumes point at the same profile,
    results are cached on disk by ``analyzer`` (``tech-cache.json`` in the working directory),
    and nothing here raises: problems are reported in ``status`` and ``notes``.

    ``on_progress(done, total)`` counts repositories; ``total == 0`` while repositories are
    still being listed.

    ``max_profiles`` limits how many DISTINCT profiles are listed (first come, first served in
    resume order); resumes whose profiles do not fit get status ``skipped_rate_limit`` and a note
    built from ``rate`` instead of failing halfway. Size it with ``rate_limit_status(token).budget()``.
    """
    infos = [GitHubInfo(profiles=list(dict.fromkeys(pl))) for pl in profile_lists]

    allowed: set[str] | None = None
    if max_profiles is not None:
        allowed = set()
        for info in infos:
            for user in info.profiles:
                if len(allowed) < max(max_profiles, 0):
                    allowed.add(user.lower())

    people: list[dict] = []
    owners: list[GitHubInfo] = []
    for info in infos:
        if not info.profiles:
            info.status = "no_github_profile"
            info.notes.append("No GitHub profile found in resume")
            continue
        kept = info.profiles if allowed is None else [u for u in info.profiles if u.lower() in allowed]
        dropped = [u for u in info.profiles if u not in kept]
        if not kept:
            info.status = "skipped_rate_limit"
            info.notes.append(_limit_note(rate))
            continue
        person = analyzer.new_person("")
        person["profiles"] = list(kept)
        person["_dropped"] = dropped
        people.append(person)
        owners.append(info)
    if not people:
        return infos

    def progress(_people: list[dict], done: int, total: int) -> None:
        if on_progress:
            on_progress(done, total)

    try:
        analyzer.run_report(people, token or None, skip_forks, progress)
    except Exception as exc:  # noqa: BLE001 - e.g. git missing, disk full
        logger.exception("GitHub analysis failed")
        for info in owners:
            info.status = "failed"
            info.notes.append(f"GitHub analysis failed: {exc}")
        return infos

    for info, person in zip(owners, people):
        info.status = "done" if person["status"] == "Done" else "failed"
        if info.status == "done":
            info.repos_count = person["repositories"]
        info.tech_stack = list(person["technologies"])
        info.categories = [(name, count) for name, count in person["categories"]]
        info.notes = list(person["errors"])
        if person["failed_repos"]:
            info.notes.append(f"{person['failed_repos']} repos could not be read")
        if person.get("_dropped"):
            info.notes.append("Not analysed (GitHub API limit): " + ", ".join(person["_dropped"]))
        if len(info.profiles) > 1:
            info.notes.append("Several GitHub profiles found in resume: " + ", ".join(info.profiles))
    return infos


def summarize(infos: Iterable[GitHubInfo]) -> dict[str, int]:
    """How many resumes ended in each outcome (only non-zero keys), for a post-run message.

    Keys: done, no_github_profile, skipped_rate_limit, rate_limited (the quota ran out DURING the
    run), user_not_found, bad_token, failed_other.
    """
    counts: Counter[str] = Counter()
    for info in infos:
        if info.status in ("done", "no_github_profile", "skipped_rate_limit"):
            counts[info.status] += 1
        elif info.status == "failed":
            text = " ".join(info.notes).lower()
            if "rate limit" in text or "403" in text or "429" in text:
                counts["rate_limited"] += 1
            elif "token is invalid" in text:
                counts["bad_token"] += 1
            elif "not found" in text:
                counts["user_not_found"] += 1
            else:
                counts["failed_other"] += 1
    return dict(counts)