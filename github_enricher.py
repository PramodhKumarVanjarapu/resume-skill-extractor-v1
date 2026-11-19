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
    find_profiles(text, links, name)    GitHub usernames, the candidate's own first (see github_profile.py)
    find_profile_candidates(...)        the same with scores, reasons and what was rejected
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
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable

import requests

import analyzer
import github_profile
import resume as resume_reader
from github_profile import ProfileSearch, find_profile_candidates, guess_name

logger = logging.getLogger(__name__)

__all__ = [
    "GITHUB_COLUMNS",
    "MAX_FALLBACKS",
    "MAX_REPOS_PER_ACCOUNT",
    "RESERVED_REQUESTS",
    "ProfileSearch",
    "find_profile_candidates",
    "guess_name",
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

# Requests kept in reserve when sizing a run: users with more than 100 repositories need extra
# pages, and other programs on the same network address may be using the same quota.
RESERVED_REQUESTS = 3
# When the best candidate does not exist on GitHub, the next one is tried (at most this many times per resume).
MAX_FALLBACKS = 2
# Repositories analysed per account. The analyzer itself has no limit: an account such as `google` (thousands of
# repositories) would cost dozens of API requests and thousands of clones. A job applicant rarely has more than this
# many; for those who do, the most recently updated ones are analysed.
MAX_REPOS_PER_ACCOUNT = 100

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


def find_profiles(text: str, links: Iterable[str], name_hint: str = "") -> list[str]:
    """GitHub usernames found in the resume, the candidate's own account first.

    Delegates to ``github_profile.find_profile_candidates`` (use that for scores, reasons and what was rejected):
    profile URLs outrank repository URLs, hyperlinks outrank text, usernames that resemble ``name_hint`` win ties,
    and placeholders, products, organisations and invalid usernames are ignored.
    """
    return find_profile_candidates(text or "", list(links), name_hint).usernames


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


# --------------------------------------------------------------------------- #
# Guarded repository listing: no organisations, no unbounded accounts
# --------------------------------------------------------------------------- #

_REAL_LIST = analyzer.list_repositories      # the analyzer's own function, captured before anything replaces it
_RUN_LOCK = threading.Lock()                 # analyzer.list_repositories is swapped during a run: one run at a time


def _paged_listing(username: str, token: str | None, skip_forks: bool, max_repos: int | None, truncated: dict) -> list[dict]:
    """``analyzer.list_repositories`` with two additions (identical results and errors otherwise; see the parity tests):

    * the owner of the first repository tells whether the account is an Organization: that is refused after ONE
      request (GitHub already put ``owner.type`` in the page we fetched, so this costs nothing extra);
    * paging stops as soon as more than ``max_repos`` usable repositories are known; the most recently updated
      ``max_repos`` are kept and the account is recorded in ``truncated``.
    """
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    repos: list[dict] = []
    page = 1
    while True:
        r = analyzer.requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"type": "owner", "per_page": 100, "page": page, "sort": "updated"},
            headers=headers,
            timeout=30,
        )
        if r.status_code == 404:
            raise analyzer.GitHubError("GitHub user not found")
        if r.status_code == 401:
            raise analyzer.GitHubError("GitHub token is invalid or expired")
        if r.status_code == 429 or (r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0"):
            reset = r.headers.get("x-ratelimit-reset")
            when = datetime.fromtimestamp(int(reset)).strftime("%I:%M %p") if reset else "later"
            raise analyzer.GitHubError(f"GitHub rate limit reached. Add a GitHub token or try again after {when}")
        r.raise_for_status()

        batch = r.json()
        if page == 1 and batch and (batch[0].get("owner") or {}).get("type") == "Organization":
            raise analyzer.GitHubError("this is a GitHub organisation, not a person")
        repos.extend(batch)
        usable = [repo for repo in repos if not (skip_forks and repo.get("fork"))]
        if max_repos is not None and len(usable) > max_repos:
            truncated[username.lower()] = max_repos
            return usable[:max_repos]
        if len(batch) < 100:
            return usable
        page += 1


@contextmanager
def _guarded_listing(max_repos: int | None, truncated: dict):
    """Within the block ``analyzer.run_report`` lists repositories through the guard above.

    If something else replaced ``analyzer.list_repositories`` (a test double, a custom client) that function is
    still used, and only the checks that can be made afterwards are applied (organisation by ``owner.type``, cap).
    The original is always restored, and runs are serialised: a second batch waits instead of doubling the load.
    """
    with _RUN_LOCK:
        current = analyzer.list_repositories
        use_paging = current is _REAL_LIST

        def guarded(username: str, token: str | None, skip_forks: bool = True) -> list[dict]:
            if use_paging:
                return _paged_listing(username, token, skip_forks, max_repos, truncated)
            repos = current(username, token, skip_forks)
            if repos and (repos[0].get("owner") or {}).get("type") == "Organization":
                raise analyzer.GitHubError("this is a GitHub organisation, not a person")
            if max_repos is not None and len(repos) > max_repos:
                truncated[username.lower()] = max_repos
                return repos[:max_repos]
            return repos

        analyzer.list_repositories = guarded
        try:
            yield
        finally:
            analyzer.list_repositories = current


@dataclass
class GitHubInfo:
    """GitHub results for one resume."""

    profiles: list[str]                    # the account(s) analysed (the best candidate unless all_profiles=True)
    status: str = ""                       # "" | no_github_profile | done | failed | skipped_rate_limit
    repos_count: int | None = None         # None when the profile could not be read
    tech_stack: list[str] = field(default_factory=list)            # most common first
    categories: list[tuple[str, int]] = field(default_factory=list)  # (category, repository count)
    notes: list[str] = field(default_factory=list)
    others: list[str] = field(default_factory=list)                # other accounts found in the resume, not analysed

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


def _analyze_round(
    profile_lists: list[list[str]],
    token: str | None = None,
    skip_forks: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    max_profiles: int | None = None,
    rate: RateLimit | None = None,
    max_repos: int | None = MAX_REPOS_PER_ACCOUNT,
) -> list[GitHubInfo]:
    """One pass over the given profiles (every profile in every list is analysed; see ``analyze_profiles``).

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

    truncated: dict[str, int] = {}
    try:
        with _guarded_listing(max_repos, truncated):
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
        for user in person["profiles"]:
            if user.lower() in truncated:
                info.notes.append(
                    f"{user}: more than {truncated[user.lower()]} repositories, so only the "
                    f"{truncated[user.lower()]} most recently updated were analysed"
                )
        if person.get("_dropped"):
            info.notes.append("Not analysed (GitHub API limit): " + ", ".join(person["_dropped"]))
        if len(info.profiles) > 1:
            info.notes.append("Several GitHub profiles found in resume: " + ", ".join(info.profiles))
    return infos


def _unusable(info: GitHubInfo) -> bool:
    """The account does not exist or is an organisation: another candidate may be the person."""
    return any("not found" in note.lower() or "organisation" in note.lower() for note in info.notes)


def analyze_profiles(
    profile_lists: list[list[str]],
    token: str | None = None,
    skip_forks: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    max_profiles: int | None = None,
    rate: RateLimit | None = None,
    all_profiles: bool = False,
    fallback: bool = True,
    max_repos: int | None = MAX_REPOS_PER_ACCOUNT,
) -> list[GitHubInfo]:
    """Analyse the GitHub account of many resumes at once.

    ``profile_lists[i]`` holds the candidate usernames of resume ``i``, the candidate's own account FIRST (the order
    ``find_profiles`` returns). By default only that first account is analysed: merging the repositories of every
    account a resume mentions mixes other people's work into the tech stack. The remaining usernames are kept in
    ``info.others`` and named in the notes. ``all_profiles=True`` restores the merge of every account.

    ``max_repos`` caps the repositories analysed per account (``None`` = no cap): accounts with more get their most
    recently updated ones, and GitHub organisations (``google``, ``linkedin``, ...) are refused after one request.

    ``fallback``: when the chosen account does not exist on GitHub, or is an organisation, the next candidate is
    tried (up to ``MAX_FALLBACKS`` times) and the notes say so. The result is aligned with ``profile_lists``; every repository is
    analysed once even if several resumes share it; nothing here raises (see ``status`` and ``notes``).
    ``on_progress``, ``max_profiles`` and ``rate`` are as in ``_analyze_round``: the budget counts distinct accounts,
    so with the default it counts one per resume.
    """
    full = [list(dict.fromkeys(pl)) for pl in profile_lists]
    chosen = full if all_profiles else [pl[:1] for pl in full]
    infos = _analyze_round(chosen, token, skip_forks, on_progress, max_profiles, rate, max_repos)
    remaining = [
        [u for u in pl if u.lower() not in {p.lower() for p in info.profiles}] for pl, info in zip(full, infos)
    ]

    if fallback and not all_profiles:
        for _ in range(MAX_FALLBACKS):
            retry = [i for i, info in enumerate(infos) if info.status == "failed" and _unusable(info) and remaining[i]]
            if not retry:
                break
            redo = _analyze_round([[remaining[i][0]] for i in retry], token, skip_forks, on_progress, None, rate, max_repos)
            for i, new in zip(retry, redo):
                reason = "is an organisation, not a person" if any("organisation" in n.lower() for n in infos[i].notes) else "was not found on GitHub"
                new.notes.insert(0, f"{infos[i].profiles[0]} {reason}; analysed {new.profiles[0]} instead")
                remaining[i] = remaining[i][1:]
                infos[i] = new

    for info, rest in zip(infos, remaining):
        info.others = rest
        if rest and info.status != "no_github_profile":
            info.notes.append("Other GitHub accounts found in the resume (not analysed): " + ", ".join(rest))
    return infos


def summarize(infos: Iterable[GitHubInfo]) -> dict[str, int]:
    """How many resumes ended in each outcome (only non-zero keys), for a post-run message.

    Keys: done, no_github_profile, skipped_rate_limit, rate_limited (the quota ran out DURING the
    run), organisation (the account is a GitHub organisation, not a person), user_not_found, bad_token,
    failed_other.
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
            elif "organisation" in text:
                counts["organisation"] += 1
            elif "not found" in text:
                counts["user_not_found"] += 1
            else:
                counts["failed_other"] += 1
    return dict(counts)