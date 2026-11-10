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

Needs network access to api.github.com and github.com, and the ``git`` command.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Callable, Iterable

import analyzer
import resume as resume_reader

logger = logging.getLogger(__name__)

__all__ = [
    "GITHUB_COLUMNS",
    "GitHubInfo",
    "analyze_profiles",
    "extract_links",
    "find_profiles",
    "git_available",
    "profile_url",
]

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

    Reserved github.com paths (``github.com/features``) and words that follow
    "GitHub:" but are not usernames are ignored; hyperlink targets outweigh visible URLs.
    """
    return resume_reader.find_github_profiles(text or "", list(links))


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

@dataclass
class GitHubInfo:
    """GitHub results for one resume."""

    profiles: list[str]
    status: str = ""                       # "" | no_github_profile | done | failed
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
) -> list[GitHubInfo]:
    """Analyse the GitHub profiles of many resumes at once.

    ``profile_lists[i]`` holds the usernames found in resume ``i``; the result is aligned with
    it. Every repository is analysed once even when several resumes point at the same profile,
    results are cached on disk by ``analyzer`` (``tech-cache.json`` in the working directory),
    and nothing here raises: problems are reported in ``status`` and ``notes``.

    ``on_progress(done, total)`` counts repositories; ``total == 0`` while repositories are
    still being listed.
    """
    infos = [GitHubInfo(profiles=list(dict.fromkeys(pl))) for pl in profile_lists]

    people: list[dict] = []
    owners: list[GitHubInfo] = []
    for info in infos:
        if not info.profiles:
            info.status = "no_github_profile"
            info.notes.append("No GitHub profile found in resume")
            continue
        person = analyzer.new_person("")
        person["profiles"] = list(info.profiles)
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
        if len(info.profiles) > 1:
            info.notes.append("Several GitHub profiles found in resume: " + ", ".join(info.profiles))
    return infos
