"""Metadata-only ordering of repositories before any Git clone.

Metadata is a relevance hint. It never contributes verified technology evidence.

    pre_analysis_exclusion(repo)   repositories that are never projects (the profile README), skipped unscanned
    practice_signal(repo)          name/description/topics look like DSA practice, a tutorial or an assignment;
                                   such repositories are scanned last, and ``analyzer.assess_repository`` excludes
                                   them only when the scan finds no application implementation
    score_repository / select_for_candidate   relevance order and per-candidate scan budget
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Iterable, Mapping

from internal_projects import find_internal

LANGUAGE_HINTS = {
    "django": "python", "fastapi": "python", "flask": "python", "pytorch": "python",
    "react": "javascript", "next.js": "javascript", "express.js": "javascript",
    "spring boot": "java", "spring": "java", "asp.net": "c#", "rails": "ruby",
}
KEYWORD_HINTS = {
    "postgresql": ("postgres", "database", "backend"),
    "mysql": ("mysql", "database", "backend"),
    "mongodb": ("mongo", "database", "backend"),
    "redis": ("redis", "cache", "backend"),
    "docker": ("docker", "container", "deploy"),
}


@dataclass(frozen=True)
class Relevance:
    score: int
    priority: str
    signals: tuple[str, ...]


def _words(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _split_name(value: object) -> str:
    """'LeetCodeSolutions' / 'dsa_practice-2' -> 'leet code solutions' / 'dsa practice 2'."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return _words(text)


# Coding-platform names are distinctive enough to match inside a word ("leetcodesolutions").
PLATFORMS = re.compile(r"leetcode|hackerrank|codechef|codeforces|geeksforgeeks|interviewbit|codewars|codingninjas"
                       r"|codestudio|projecteuler|adventofcode|hackerearth|atcoder|spoj")
PRACTICE_WORDS = re.compile(
    r"\b(?:leet code|hacker rank|gfg|dsa|data structures?|algorithms?|competitive programming|cp|"
    r"problem solving|coding (?:practice|questions|problems|challenges?)|interview (?:prep|preparation|questions)|"
    r"\d+ ?days? of (?:code|coding|python|javascript|js|java|dsa|react|web dev)|daily coding|practice|practise|"
    r"exercises?|programs|programming questions)\b"
)
TUTORIAL_WORDS = re.compile(
    r"\b(?:tutorials?|courses?|coursework|bootcamp|udemy|coursera|freecodecamp|fcc|nptel|crash course|"
    r"learn|basics|beginners?|hello world|assignments?|homework|lab ?\d+|labs|class ?work|"
    r"sample|samples|playground|first repo|test repo|demo repo)\b"
)
# Descriptions are free text ("uses ML algorithms"): only unambiguous mentions count there.
PRACTICE_DESCRIPTION = re.compile(r"\b(?:dsa|data structures and algorithms|competitive programming|"
                                  r"coding problems|problem solutions)\b")


def pre_analysis_exclusion(repo: Mapping) -> str | None:
    """Repositories that are never candidate projects, decided without any request.

    The profile README repository (named exactly like its owner) only holds the profile page, and projects assigned
    by the training company (``dictionaries/internal_projects.txt``) are not the candidate's own work.
    Forks are handled by repository listing (``skip_forks``) and empty repositories by size.
    """
    owner = (repo.get("owner") or {}).get("login") or str(repo.get("full_name", "")).split("/", 1)[0]
    name = repo.get("name") or str(repo.get("full_name", "")).rsplit("/", 1)[-1]
    if owner and str(name).casefold() == str(owner).casefold():
        return "profile_readme"
    if find_internal(str(name)):
        return "internal_training_project"
    return None


def practice_signal(repo: Mapping) -> str | None:
    """``coding_practice`` / ``tutorial_or_assignment`` when the repository's own name, description or topics say so.

    A signal, not a verdict: a "dsa-visualizer" built with React is a project. See ``analyzer.assess_repository``.
    """
    raw_name = repo.get("name") or str(repo.get("full_name", "")).rsplit("/", 1)[-1]
    name = _split_name(raw_name)
    text = " ".join((name, " ".join(_words(t) for t in repo.get("topics") or [])))
    description = _words(repo.get("description"))
    if (PLATFORMS.search(text.replace(" ", "")) or PLATFORMS.search(description.replace(" ", ""))
            or PRACTICE_WORDS.search(text) or PRACTICE_DESCRIPTION.search(description)):
        return "coding_practice"
    if TUTORIAL_WORDS.search(text):
        return "tutorial_or_assignment"
    return None


def score_repository(repo: Mapping, required_groups: Iterable[Iterable[str]] = (),
                     language_hints: Mapping[str, str] = LANGUAGE_HINTS,
                     keyword_hints: Mapping[str, tuple[str, ...]] = KEYWORD_HINTS) -> Relevance:
    """Score name, description, topics, language and recency; no hard language exclusion."""
    required_groups = tuple(tuple(group) for group in required_groups)
    if repo.get("size") == 0:
        return Relevance(0, "DEFERRED", ("empty_repository",))
    name = _words(repo.get("name") or str(repo.get("full_name", "")).rsplit("/", 1)[-1])
    text = _words(" ".join((str(repo.get("description") or ""),
                            " ".join(str(t) for t in (repo.get("topics") or [])))))
    language = _words(repo.get("language"))
    score, signals = 0, []
    for group in required_groups:
        best, reason = 0, ""
        for technology in group:
            term = _words(technology)
            if not term:
                continue
            compact = term.replace(" ", "")
            if compact and compact in name.replace(" ", ""):
                points, signal = 8, f"name:{technology}"
            elif re.search(r"\b" + re.escape(term) + r"\b", text):
                points, signal = 6, f"description_or_topic:{technology}"
            elif language == term or language == _words(language_hints.get(str(technology).casefold(), "")) and language:
                points, signal = 4, f"language:{language}"
            elif any(hint in text for hint in keyword_hints.get(str(technology).casefold(), ())):
                points, signal = 2, f"related_keyword:{technology}"
            else:
                points, signal = 0, ""
            if points > best:
                best, reason = points, signal
        score += best  # an OR group contributes once
        if reason:
            signals.append(reason)
    if not required_groups:
        if name or text or language:
            score, signals = 2, ["metadata_available"]
    pushed = repo.get("pushed_at")
    if pushed:
        try:
            date = datetime.fromisoformat(str(pushed).replace("Z", "+00:00"))
            if date.tzinfo and 0 <= (datetime.now(timezone.utc) - date).days <= 180:
                score += 1
                signals.append("recent_push")
        except ValueError:
            pass
    practice = practice_signal(repo)
    if practice:
        # Scanned after real projects; never excluded on the name alone.
        score -= 5
        signals.append(practice)
    if repo.get("archived"):
        signals.append("archived")
    if not (name or text or language):
        signals.append("metadata_missing")
    priority = "HIGH_PRIORITY" if score >= 8 else "MEDIUM_PRIORITY" if score > 0 else "DEFERRED"
    return Relevance(score, priority, tuple(signals))


def select_for_candidate(repos: list[dict], requirements: Iterable[Iterable[str]],
                         budget: int | None) -> tuple[list[dict], list[dict]]:
    """Select up to budget uncached repos, stably ordered by metadata relevance."""
    groups = tuple(tuple(group) for group in requirements)
    ordered = sorted(repos, key=lambda repo: -score_repository(repo, groups).score)
    if budget is None:
        return ordered, []
    return ordered[:max(0, budget)], ordered[max(0, budget):]
