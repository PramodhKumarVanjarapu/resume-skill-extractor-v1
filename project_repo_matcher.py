"""Explainable matching between extracted resume projects and analysed GitHub repos."""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from urllib.parse import urlparse


def _name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(value).casefold()) if len(token) > 2}


def _repo_key(value: str) -> str:
    parsed = urlparse(str(value).strip().rstrip("/"))
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return ""
    parts = parsed.path.strip("/").split("/")
    return "/".join(parts[:2]).casefold().removesuffix(".git") if len(parts) >= 2 else ""


def match_projects(projects: list[dict], repositories: list[dict], high: float = .80,
                   medium: float = .55, ambiguity_gap: float = .08) -> list[dict]:
    """Return a record for every project, including unmatched and ambiguous ones."""
    if not 0 <= medium <= high <= 1:
        raise ValueError("confidence thresholds must satisfy 0 <= medium <= high <= 1")
    used: set[str] = set()
    output = []
    for project in projects:
        pname = str(project.get("project_name") or "")
        links = {_repo_key(link) for link in project.get("links", [])}
        links.discard("")
        choices = []
        for repo in repositories:
            full = str(repo.get("repository") or "")
            url = str(repo.get("url") or "")
            key = _repo_key(url)
            direct = bool(key and key in links)
            repo_name = full.rsplit("/", 1)[-1]
            name_score = SequenceMatcher(None, _name(pname), _name(repo_name)).ratio() if pname and repo_name else 0
            left, right = _tokens(project.get("description") or ""), _tokens(repo.get("description") or "")
            description = len(left & right) / len(left | right) if left and right else 0
            ptech = set(project.get("technologies") or [])
            rtech = set(repo.get("technologies") or [])
            overlap = len(ptech & rtech) / len(ptech | rtech) if ptech and rtech else 0
            # URL absent: renormalise 25/15/10 across available signals.
            signals = [(name_score, .25, bool(pname and repo_name)),
                       (description, .15, bool(left and right)), (overlap, .10, bool(ptech and rtech))]
            denominator = sum(weight for _, weight, available in signals if available)
            score = 1.0 if direct else sum(value * weight for value, weight, available in signals if available) / denominator if denominator else 0
            reasons = (["explicit_repository_url"] if direct else [])
            if name_score >= .7:
                reasons.append("project_name_similarity")
            if description >= .25:
                reasons.append("description_similarity")
            if overlap:
                reasons.append("technology_overlap")
            # A shared technology list by itself cannot establish identity.
            independent = direct or name_score >= .7 or (name_score >= .45 and description >= .25)
            if independent and (direct or score >= medium):
                choices.append((score, direct, full, url, reasons))
        choices.sort(key=lambda item: (-item[0], item[2]))
        best = choices[0] if choices else None
        ambiguous = bool(best and not best[1] and len(choices) > 1
                         and best[0] - choices[1][0] < ambiguity_gap)
        conflict = bool(best and best[2].casefold() in used and not best[1])
        confidence = ("high" if best and (best[1] or best[0] >= high and len(best[4]) >= 2)
                      else "medium" if best and best[0] >= medium else "no_match")
        if ambiguous or conflict:
            confidence = "low"
        if best and not ambiguous and not conflict:
            used.add(best[2].casefold())
        output.append({"project_name": pname, "repository_name": best[2] if best else "",
                       "repository_url": best[3] if best else "", "match_score": round(best[0], 3) if best else 0,
                       "confidence": confidence, "matching_reasons": best[4] if best else [],
                       "requires_review": ambiguous or conflict or confidence == "low",
                       "alternative_repositories": [c[2] for c in choices[1:3]] if ambiguous else []})
    return output
