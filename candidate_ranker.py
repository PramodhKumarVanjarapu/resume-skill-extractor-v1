"""Rank extracted resumes against recruiter technology groups without network calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Iterable, Mapping

from technology_matcher import TechnologyMatcher, normalize_surface

SOURCES = {"github": "github_tech_stack", "project": "tech_projects", "experience": "tech_experience"}
PRIORITY_ORDER = {"P1": 0, "P2": 1, "P3": 2, "Unclassified": 3, "Pending Analysis": 4}
RANKING_COLUMNS = [
    "technology_match_percentage", "evidence_score", "priority", "matched_requirements",
    "missing_requirements", "matched_alternative_groups", "github_matched_technologies",
    "project_matched_technologies", "experience_matched_technologies", "optional_matches",
    "matched_mandatory_groups", "total_mandatory_groups", "potential_priority",
    "github_evidence_quality", "ranking_reason", "ranking_warnings",
    "total_projects", "matching_projects_count", "matching_project_names", "project_threshold_met",
    "technology_threshold_met", "shortlisting_eligible",
    "screening_status", "unverified_requirements",
]


@dataclass(frozen=True)
class Requirement:
    technologies: tuple[str, ...]
    mandatory: bool = True

    @property
    def label(self) -> str:
        return " OR ".join(self.technologies)


def _cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(math.isnan(value)):  # pandas NaN
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _canonical(value: str, matcher: TechnologyMatcher) -> str | None:
    surface = matcher.dictionary.surfaces.get(normalize_surface(value.strip()))
    return surface.name if surface else None


def normalize_requirements(
    entries: Iterable[tuple[str, bool]], matcher: TechnologyMatcher
) -> tuple[Requirement, ...]:
    """Parse one technology or OR alternatives per entry; reject unknown names."""
    result = []
    seen = set()
    for raw, mandatory in entries:
        names = []
        for part in re.split(r"\s+OR\s+|[,;\n]", raw.strip(), flags=re.I):
            if not part.strip():
                continue
            name = _canonical(part, matcher)
            if name is None:
                raise ValueError(f"Technology not in dictionary: {part.strip()}")
            if name not in names:
                names.append(name)
        if names:
            key = tuple(names)
            if key not in seen:
                result.append(Requirement(key, mandatory))
                seen.add(key)
    return tuple(result)


def _source_set(value: object, matcher: TechnologyMatcher) -> set[str]:
    # Extracted output cells use semicolons. Compare whole entries, never substrings.
    return {name for part in _cell(value).split(";") if (name := _canonical(part, matcher))}


def _matching_projects(row: Mapping[str, object], mandatory: list[Requirement], matcher: TechnologyMatcher):
    """Count unique extracted resume projects with a mandatory technology."""
    raw = row.get("projects_json")
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        projects = data["projects"]
        if not isinstance(projects, list):
            raise ValueError("projects must be a list")
    except (TypeError, ValueError, KeyError):
        return None, []
    required = {tech for group in mandatory for tech in group.technologies}
    seen, matching = set(), []
    for project in projects:
        if not isinstance(project, dict) or not isinstance(project.get("technologies"), list):
            return None, []
        name = _cell(project.get("project_name")).strip()
        start, end = project.get("start"), project.get("end")
        identity = ("span", start, end) if isinstance(start, int) and isinstance(end, int) else (
            "content", name.casefold(), _cell(project.get("description")).strip().casefold()
        )
        if identity in seen:
            continue
        seen.add(identity)
        technologies = {_canonical(str(value), matcher) for value in project["technologies"]}
        if required & technologies:
            matching.append(name or "(untitled project)")
    return len(matching), matching


def score_candidate(
    row: Mapping[str, object], requirements: tuple[Requirement, ...], matcher: TechnologyMatcher,
    weights: Mapping[str, float] | None = None, threshold: float = 60, min_matching_projects: int = 2,
) -> dict[str, object]:
    """Score one output-schema row, keeping coverage, evidence, and availability distinct."""
    weights = weights or {"github": 50, "project": 30, "experience": 20}
    if any(float(weights.get(s, 0)) < 0 for s in SOURCES) or sum(float(weights.get(s, 0)) for s in SOURCES) <= 0:
        raise ValueError("Evidence weights must be nonnegative and sum to more than zero")
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")
    if min_matching_projects < 0:
        raise ValueError("Minimum matching projects cannot be negative")
    sources = {s: _source_set(row.get(col), matcher) for s, col in SOURCES.items()}
    mandatory = [r for r in requirements if r.mandatory]
    optional = [r for r in requirements if not r.mandatory]
    supported = lambda req: {s: [t for t in req.technologies if t in names] for s, names in sources.items()}
    matched, missing, alternatives, bonus = [], [], [], []
    source_groups = {s: 0 for s in SOURCES}
    source_techs = {s: [] for s in SOURCES}
    for req in mandatory + optional:
        hits = supported(req)
        names = list(dict.fromkeys(t for s in SOURCES for t in hits[s]))
        if req.mandatory:
            if names:
                matched.append(f"{', '.join(names)} ({req.label})" if len(req.technologies) > 1 else names[0])
                if len(req.technologies) > 1:
                    alternatives.append(f"{req.label}: {', '.join(names)}")
            else:
                missing.append(req.label)
            for source in SOURCES:
                if hits[source]:
                    source_groups[source] += 1
        elif names:
            bonus.append(f"{req.label}: {', '.join(names)}")
        for source in SOURCES:
            source_techs[source].extend(hits[source])
    total = len(mandatory)
    coverage = 100 * len(matched) / total if total else 0.0
    try:
        pending_repos = int(_cell(row.get("github_deferred_repos")) or 0)
    except ValueError:
        pending_repos = 0
    project_count, project_names = _matching_projects(row, mandatory, matcher)
    try:
        total_projects = int(_cell(row.get("num_projects")))
    except ValueError:
        total_projects = None
    tech_met = bool(total and coverage >= threshold)
    project_met = None if project_count is None else project_count >= min_matching_projects
    eligible = tech_met and project_met is True
    weight_total = sum(float(weights.get(s, 0)) for s in SOURCES)
    evidence = 100 * sum(
        (float(weights.get(s, 0)) / weight_total) * (source_groups[s] / total if total else 0)
        for s in SOURCES
    )
    active = {s for s, count in source_groups.items() if count}
    if not total:
        priority = "Unclassified"
        reason = "No mandatory requirements configured."
    elif {"github", "project", "experience"} <= active:
        priority = "P1"
        reason = "Mandatory requirements have GitHub, project, and experience evidence."
    elif {"github", "project"} <= active:
        priority = "P2"
        reason = "Mandatory requirements have GitHub and project evidence."
    elif "project" in active:
        priority = "P3"
        reason = "Mandatory requirements have project evidence."
    else:
        priority = "Unclassified"
        reason = "No matched mandatory requirement has project evidence."
    if not eligible:
        if not tech_met:
            reason += f" Technology match below {threshold:.2f}% ({coverage:.2f}%)."
        if project_met is None:
            reason += " Matching project threshold unverified: project-wise data unavailable."
        elif not project_met:
            reason += f" Matching projects below {min_matching_projects} ({project_count})."
        display_priority = "Below Threshold"
    else:
        display_priority = priority
    if pending_repos and not tech_met and project_met is True:
        display_priority = "Pending Analysis"
        reason += f" {pending_repos} GitHub repositories remain unanalyzed; technology coverage is provisional."
    gh_status = _cell(row.get("github_status"))
    warnings = []
    if gh_status != "done":
        warnings.append("GitHub analysis unavailable" + (f" ({gh_status})" if gh_status else ""))
    if pending_repos:
        warnings.append(f"{pending_repos} GitHub repositories not yet analyzed")
    if not _cell(row.get("tech_projects")):
        warnings.append("No project technology evidence")
    if project_count is None:
        warnings.append("Project threshold unverified")
    if not _cell(row.get("tech_experience")):
        warnings.append("No experience technology evidence")
    if _cell(row.get("status")) in {"error", "partial"}:
        warnings.append(f"Resume processing {_cell(row.get('status'))}")
    for column in ("github_notes", "warnings"):
        note = _cell(row.get(column)).strip()
        if note:
            warnings.append(note)
    return {
        "technology_match_percentage": round(coverage, 2), "evidence_score": round(evidence, 2),
        "priority": display_priority, "potential_priority": priority,
        "matched_requirements": "; ".join(matched), "missing_requirements": "; ".join(missing),
        "matched_alternative_groups": "; ".join(alternatives),
        **{f"{s}_matched_technologies": "; ".join(dict.fromkeys(source_techs[s])) for s in SOURCES},
        "optional_matches": "; ".join(bonus), "matched_mandatory_groups": len(matched),
        "total_mandatory_groups": total, "ranking_reason": reason,
        "ranking_warnings": "; ".join(warnings),
        "github_evidence_quality": "unspecified" if gh_status == "done" else "unavailable",
        "screening_status": "partial" if pending_repos else "complete",
        "unverified_requirements": "; ".join(missing) if pending_repos else "",
        "total_projects": total_projects, "matching_projects_count": project_count,
        "matching_project_names": "; ".join(project_names), "project_threshold_met": project_met,
        "technology_threshold_met": tech_met, "shortlisting_eligible": eligible,
    }


def rank_candidates(
    rows: Iterable[Mapping[str, object]], requirements: tuple[Requirement, ...], matcher: TechnologyMatcher,
    weights: Mapping[str, float] | None = None, threshold: float = 60, min_matching_projects: int = 2,
) -> list[dict[str, object]]:
    scored = [dict(row, **score_candidate(row, requirements, matcher, weights, threshold, min_matching_projects))
              for row in rows]
    return sorted(scored, key=lambda r: (
        r["priority"] in {"Below Threshold", "Pending Analysis"}, PRIORITY_ORDER.get(str(r["priority"]), 5),
        -float(r["technology_match_percentage"]), -float(r["evidence_score"]),
    ))
