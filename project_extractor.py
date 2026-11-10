"""Project-wise technology extraction from a resume's Projects section.

No LLM and no ML model: boundaries come from explainable structural rules,
technologies come from an injected ``TechnologyMatcher``.

Public API
----------
``ProjectExtractor(matcher).extract(projects_section_text) -> ProjectExtractionResult``

``ProjectExtractionResult`` is a ``dict`` shaped like the spec::

    {"projects": [{"project_name": ..., "description": ..., "technologies": [...],
                   <extra fields: evidence, offsets, review flags, ...>}, ...]}

``result.to_simple()`` returns exactly the three spec fields per project.

How boundaries are found
------------------------
Every non-blank line is first classified (pass 1) as one of: bullet, sentence,
labelled line ("Technologies: ..."), tech-only list, date, URL, noise
(sub-headings such as "Academic Projects"), or *title candidate*. Pass 2 walks
the lines and decides which candidates start a new project using additive,
named cues (numbering, blank line before, short title-case text, project noun,
followed by a description, "Name | Tech" shape ...). Each project records the
cues that started it (``boundary_reasons``).

When evidence is weak the extractor never silently merges projects: it either
splits and sets ``needs_review`` ("weak_boundary"), or keeps the line in the
previous project and flags that project ("possible_missed_boundary").

Technologies are matched PER PROJECT BLOCK, so a technology can only be
attributed to the project whose text contains it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Project", "ProjectExtractionResult", "ProjectExtractor", "TechnologyRef"]

# --------------------------------------------------------------------------- #
# Vocabulary (all extensible: they are plain module-level constants)
# --------------------------------------------------------------------------- #

# Lines like "Technologies: ..." / "Frontend: ..." -> technology information.
_TECH_LABELS = frozenset({
    "technologies", "technologies used", "technology", "technology used", "tech stack", "technical stack",
    "tech", "tools", "tools used", "tools and technologies", "technologies and tools", "stack",
    "environment", "technical environment", "languages", "languages used", "frameworks", "libraries",
    "built with", "skills used", "platform", "platforms", "key technologies", "software", "software used",
    "frontend", "front end", "backend", "back end", "database", "databases", "db", "deployment", "hosting",
    "testing", "version control", "cloud", "devops", "api", "apis", "machine learning", "ml",
})
_META_LABELS = frozenset({
    "role", "my role", "duration", "team size", "team", "link", "links", "github", "github link", "git",
    "url", "website", "live", "live demo", "demo", "date", "dates", "period", "timeline", "client", "domain",
    "type", "project type", "status", "course", "guide", "organization", "organisation", "company",
    "location", "source code", "repository", "repo",
})
# Labels whose lines only carry links; the label word ("GitHub:") is not a technology mention.
_LINK_LABELS = frozenset({"github", "github link", "git", "link", "links", "url", "website", "live", "live demo",
                          "demo", "source code", "repository", "repo"})
_DESC_LABELS = frozenset({
    "description", "project description", "overview", "summary", "about", "objective", "purpose",
    "responsibilities", "key responsibilities", "roles and responsibilities", "features", "key features",
    "highlights", "achievements", "outcome", "outcomes", "results", "impact", "contribution", "contributions",
    "note", "notes", "remarks",
})
_LABEL_KIND: dict[str, str] = {
    **{k: "tech" for k in _TECH_LABELS}, **{k: "meta" for k in _META_LABELS}, **{k: "desc" for k in _DESC_LABELS},
}

# Sub-headings that may survive inside the section text.
_SUBHEADINGS = frozenset({
    "projects", "project", "academic projects", "academic project", "personal projects", "personal project",
    "key projects", "major projects", "notable projects", "selected projects", "project experience",
    "technical projects", "project work", "side projects", "college projects", "relevant projects",
    "software projects", "mini projects", "projects undertaken",
})

_ACTION_VERBS = frozenset("""
developed built created designed implemented used using worked integrated deployed developing building creating
designing implementing led managed wrote written engineered architected automated analyzed analysed applied
achieved added authored configured collaborated conducted constructed containerized containerised contributed
coordinated enhanced established evaluated executed extracted facilitated improved incorporated launched
maintained migrated modeled modelled monitored optimized optimised orchestrated performed produced programmed
provided published reduced refactored researched resolved responsible scraped simulated solved streamlined
structured supported tested trained transformed tracked utilized utilised validated visualized visualised
leveraged leveraging employed ensured handled hosted fetched secured scheduled processed generated focused
participated delivered demonstrated drove explored helped introduced learned learnt mentored operated planned
prepared presented proposed realized realised rebuilt replaced revamped scaled shipped spearheaded stored
taught tuned updated uploaded wrote crafted compiled converted debugged documented enabled expanded fixed
gathered identified installed investigated leveraged mapped merged negotiated organized organised paired
predicted prototyped queried ranked recommended reported reviewed rewrote sorted standardized standardised
submitted tailored tackled thrived translated unified verified wired won
""".split())
# Adjectives that look like verbs: "Automated Attendance System" is a title.
_VERB_LIKE_ADJECTIVES = frozenset(
    "automated integrated distributed advanced embedded connected optimized optimised enhanced customized "
    "customised personalized personalised dedicated decentralized decentralised unified standardized "
    "standardised intelligent".split()
)
_PROJECT_NOUNS = frozenset("""
system application app portal platform website web site management tracker predictor prediction detection detector
recognition classifier chatbot bot dashboard clone engine tool tools analyzer analyser generator manager planner
calculator game simulator recommender recommendation monitoring automation pipeline framework library api service
services store shop marketplace booking reservation portfolio blog forum network assistant scheduler organizer
visualizer visualiser converter scanner extension plugin simulation analysis analytics
""".split())
_TITLE_STOPWORDS = frozenset("of and for the a an in on with to using by via at from & - vs".split())
_LEADING_ARTICLES = frozenset({"a", "an", "the", "this", "it", "our", "my", "we", "i"})
_IGNORED_UNRECOGNIZED = frozenset({"etc", "others", "other", "more", "and more", "tools", "framework",
                                   "frameworks", "libraries", "apis", "api", "technologies", "n/a", "none"})

# --------------------------------------------------------------------------- #
# Regexes
# --------------------------------------------------------------------------- #

_BULLET = re.compile(r"^[•●▪■◦·►▸➢✓✔»*\-–—]\s+")
_NUMBERING = re.compile(
    r"^(?:(?:project|proj\.?)\s*#?\s*(?P<a>\d{1,2})\s*[:.)\-–—]?|\(?(?P<b>\d{1,2})[.)]|#\s*(?P<c>\d{1,2})"
    r"|(?P<d>\d{1,2})\s*[-–—:])\s+",
    re.IGNORECASE,
)
_LABEL = re.compile(r"^(?P<label>[A-Za-z][A-Za-z &/\-]{1,30}?)\s*[:：]\s*(?P<rest>.*)$")
_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
_YEAR = r"(?:19|20)\d{2}"
_POINT = rf"(?:(?:{_MONTH}\s+)?{_YEAR}|\d{{1,2}}/{_YEAR})"
_END = rf"(?:{_POINT}|Present|Current|Ongoing|Now|Till\s+date)"
_DATE = rf"{_POINT}(?:\s*(?:-|–|—|to)\s*{_END})?"
_DATE_ONLY = re.compile(rf"^[\(\[]?\s*{_DATE}\s*[\)\]]?$", re.IGNORECASE)
_TRAILING_DATE = re.compile(
    rf"^(?P<name>.*?)(?:\s*[\(\[]\s*(?P<d1>{_DATE})\s*[\)\]]|\s+[|–—-]\s+(?P<d2>{_DATE})|\s{{2,}}(?P<d3>{_DATE})|\t+(?P<d4>{_DATE})"
    rf"|\s+(?P<d5>{_MONTH}\s+{_YEAR}(?:\s*(?:-|–|—|to)\s*{_END})?))\s*$",
    re.IGNORECASE,
)
_URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:github|gitlab|bitbucket)\.com/\S+", re.IGNORECASE)
_URL_ONLY = re.compile(r"^(?:https?://|www\.)\S+$|^(?:github|gitlab|bitbucket)\.com/\S+$", re.IGNORECASE)
_PIPE = re.compile(r"\s*\|\s*")
_DASH = re.compile(r"\s+[–—-]\s+")
_COLON = re.compile(r"^(?P<left>[^:]{2,80}?)\s*:\s+(?P<right>\S.*)$")
_TOKEN_SPLIT = re.compile(r"\s*(?:,|;|/|\||&|•|·|\band\b)\s*", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+#'_\-]*")
_CONTINUATION_WORDS = frozenset(
    "and or with using in on for to of the a an by as via from into like including such &".split()
)
_BODY_KINDS = frozenset({"bullet", "sentence", "label_tech", "label_meta", "label_desc", "tech_only", "date", "url"})


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TechnologyRef:
    """A technology attributed to one project, with its evidence."""

    name: str
    category: str
    matched_text: str            # first mention, as written
    start: int                   # offsets of the first mention in the SECTION text
    end: int
    evidence: str                # sentence of the first mention
    all_evidence: tuple[str, ...]
    occurrences: int
    found_in: tuple[str, ...]    # subset of ("title", "description", "technology_line")
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "category": self.category, "matched_text": self.matched_text,
            "start": self.start, "end": self.end, "evidence": self.evidence,
            "all_evidence": list(self.all_evidence), "occurrences": self.occurrences,
            "found_in": list(self.found_in), "reason": self.reason,
        }


@dataclass
class Project:
    name: str
    description: str
    technologies: list[str]
    technology_details: list[TechnologyRef]
    unrecognized_technologies: list[str]   # listed in a tech line but not in the dictionary
    dates: str
    links: list[str]
    title_style: str                       # numbered | pipe | dash | colon | bullet | plain | untitled
    boundary_reasons: list[str]            # cues that started this project
    needs_review: bool
    review_reasons: list[str]
    notes: list[str]                       # informational (no_description, no_technologies)
    start: int                             # block offsets in the section text
    end: int
    name_start: int
    name_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.name,
            "description": self.description,
            "technologies": list(self.technologies),
            "technology_details": [t.to_dict() for t in self.technology_details],
            "unrecognized_technologies": list(self.unrecognized_technologies),
            "dates": self.dates,
            "links": list(self.links),
            "title_style": self.title_style,
            "boundary_reasons": list(self.boundary_reasons),
            "needs_review": self.needs_review,
            "review_reasons": list(self.review_reasons),
            "notes": list(self.notes),
            "start": self.start, "end": self.end,
            "name_start": self.name_start, "name_end": self.name_end,
        }


class ProjectExtractionResult(dict):
    """``{"projects": [...]}`` plus metadata.

    * ``projects``     list of :class:`Project`
    * ``needs_review`` True if any project (or the whole section) is uncertain
    * ``warnings``     result-level codes, e.g. ``no_project_titles_found``
    * ``preamble``     text before the first detected title (not a project)
    """

    def __init__(self, projects: list[Project], warnings: list[str], preamble: str) -> None:
        super().__init__(projects=[p.to_dict() for p in projects])
        self.projects = projects
        self.warnings = warnings
        self.preamble = preamble
        self.needs_review = bool(warnings and any(w in _REVIEW_WARNINGS for w in warnings)) or any(
            p.needs_review for p in projects
        )

    def to_simple(self) -> dict[str, Any]:
        """Exactly the spec shape: project_name, description, technologies."""
        return {
            "projects": [
                {"project_name": p.name, "description": p.description, "technologies": list(p.technologies)}
                for p in self.projects
            ]
        }


_REVIEW_WARNINGS = frozenset({"no_project_titles_found"})


# --------------------------------------------------------------------------- #
# Internal line model
# --------------------------------------------------------------------------- #

@dataclass
class _Split:
    name: str
    kind: str                      # pipe | dash | colon
    desc: str = ""
    tech: str = ""
    dates: str = ""
    links: list[str] = field(default_factory=list)


@dataclass
class _Line:
    start: int
    end: int
    indent: int
    raw: str
    kind: str = "blank"
    text: str = ""
    label: str = ""
    numbering: int | None = None
    bullet: bool = False
    split: _Split | None = None
    dates: str = ""
    links: list[str] = field(default_factory=list)


@dataclass
class _Builder:
    title: _Line | None
    reasons: list[str]
    flags: list[str]
    style: str
    lines: list[_Line] = field(default_factory=list)   # body lines (after the title)
    missed: list[str] = field(default_factory=list)    # title-like lines kept in the body


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #

def _strip_decor(s: str) -> str:
    return s.strip().strip("*_#`").strip()


def _norm_label(s: str) -> str:
    return " ".join(s.lower().replace("&", " and ").replace("-", " ").replace("/", " ").split())


def _words(s: str) -> list[str]:
    return _WORD.findall(s)


def _cap_ratio(name: str) -> float:
    content = [w for w in _words(name) if w.lower() not in _TITLE_STOPWORDS]
    if not content:
        return 0.0
    good = sum(1 for w in content if w[0].isupper() or w[0].isdigit())
    return good / len(content)


def _title_exception(text: str) -> bool:
    """Title Case phrases that start with a verb-looking word are still titles:
    'Automated Attendance System', 'Building Management System'."""
    words = _words(text)
    if len(words) < 2 or len(words) > 6 or text.rstrip().endswith((".", "!", "?")):
        return False
    first = words[0].lower()
    if first in _VERB_LIKE_ADJECTIVES:
        return words[1][0].isupper()
    if first.endswith("ing") and first in _ACTION_VERBS:
        return _cap_ratio(text) >= 0.99 and any(w.lower() in _PROJECT_NOUNS for w in words)
    return False


def _is_sentence(text: str) -> bool:
    words = _words(text)
    if not words:
        return False
    stripped = text.rstrip()
    n = len(words)
    if _title_exception(text):
        return False
    first = words[0].lower()
    if first in _VERB_LIKE_ADJECTIVES or first in _ACTION_VERBS:
        return True
    cap = _cap_ratio(text)
    if stripped.endswith((".", "!", "?")):
        # a terminal period means a sentence, unless it is a short Title Case name with a project noun
        titleish = cap >= 0.8 and n <= 5 and any(w.lower() in _PROJECT_NOUNS for w in words)
        return not titleish
    if n > 14:
        return True
    if first in _LEADING_ARTICLES and (cap < 0.7 or n > 6):
        return True
    low = f" {stripped.lower()} "
    if n >= 6 and any(k in low for k in (" that ", " which ", " where ", " allows ", " enables ", " allowing ")):
        return True
    return False


def _strip_trailing_date(s: str) -> tuple[str, str]:
    m = _TRAILING_DATE.match(s)
    if not m:
        return s.strip(), ""
    date = next((m.group(g) for g in ("d1", "d2", "d3", "d4", "d5") if m.group(g)), "")
    return m.group("name").strip(), date.strip()


def _clean_name(s: str) -> str:
    s = _strip_decor(s)
    s = _URL.sub("", s)
    s = s.strip().strip("\"'“”‘’")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" \t:–—|,;-")


def _is_continuation(prev: str, cur: str) -> bool:
    """Does line ``cur`` continue the sentence that ``prev`` started (wrapped line)?"""
    prev = prev.rstrip()
    if not prev or not cur:
        return False
    if prev[-1] in ".!?:":
        return False
    if prev[-1] in ",;(&/-":
        return True
    tail = re.search(r"[A-Za-z']+$", prev)
    if tail and tail.group(0).lower() in _CONTINUATION_WORDS:
        return True
    return cur[0].islower()


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

class ProjectExtractor:
    """Splits a Projects section into projects and attributes technologies to each.

    Parameters
    ----------
    matcher:
        A ``TechnologyMatcher`` (or anything with ``find_all(text)`` returning
        objects with name/category/matched_text/start/end/evidence/reason).
        ``None`` creates a default ``TechnologyMatcher``.
    title_threshold / strong_threshold:
        Cue-score cut-offs: below ``title_threshold`` a line is body text;
        between the two the boundary is "weak" and gets flagged for review.
    max_title_words:
        Longest text accepted as a project name.
    description_join:
        Separator between description lines (default newline).
    max_project_lines:
        Projects with more body lines are flagged as a possible merge.
    """

    def __init__(
        self,
        matcher: Any = None,
        *,
        title_threshold: float = 0.5,
        strong_threshold: float = 0.75,
        max_title_words: int = 12,
        description_join: str = "\n",
        max_project_lines: int = 25,
    ) -> None:
        if matcher is None:
            from technology_matcher import TechnologyMatcher

            matcher = TechnologyMatcher()
        if not hasattr(matcher, "find_all"):
            raise TypeError("matcher must provide find_all(text)")
        self.matcher = matcher
        self.title_threshold = title_threshold
        self.strong_threshold = strong_threshold
        self.max_title_words = max_title_words
        self.description_join = description_join
        self.max_project_lines = max_project_lines

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    def extract(self, section_text: str) -> ProjectExtractionResult:
        """Extract projects from the text of a Projects section.

        Raises TypeError for non-string input. Empty/whitespace input returns
        an empty result (no projects, no warnings).
        """
        if not isinstance(section_text, str):
            raise TypeError(f"section text must be str, got {type(section_text).__name__}")
        if not section_text.strip():
            return ProjectExtractionResult([], [], "")

        lines = self._prepare_lines(section_text)
        self._classify(lines)
        builders, pre_lines, warnings = self._assemble(lines)

        preamble = ""
        if builders and pre_lines:
            preamble = section_text[pre_lines[0].start:pre_lines[-1].end]
            warnings.append("text_before_first_title")
        if not builders:
            if not pre_lines:
                return ProjectExtractionResult([], warnings, "")
            builders = [_Builder(None, ["no_title_detected"], ["no_project_titles_found"], "untitled", pre_lines)]
            warnings.append("no_project_titles_found")

        projects = [self._finish(b, section_text) for b in builders]
        return ProjectExtractionResult(projects, warnings, preamble)

    # ------------------------------------------------------------------ #
    # pass 0: lines
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare_lines(text: str) -> list[_Line]:
        lines: list[_Line] = []
        offset = 0
        for piece in text.splitlines(keepends=True):
            body = piece.splitlines()[0] if piece else ""
            stripped = body.strip()
            if not stripped:
                lines.append(_Line(offset, offset, 0, ""))
            else:
                lead = len(body) - len(body.lstrip())
                lines.append(
                    _Line(
                        start=offset + lead,
                        end=offset + len(body.rstrip()),
                        indent=len(body[:lead].expandtabs(4)),
                        raw=stripped,
                    )
                )
            offset += len(piece)
        return lines

    # ------------------------------------------------------------------ #
    # pass 1: classify every line
    # ------------------------------------------------------------------ #
    def _tech_coverage(self, text: str) -> float:
        matches = self.matcher.find_all(text)
        if not matches:
            return 0.0
        covered = sum(len(re.sub(r"\W", "", m.matched_text)) for m in matches)
        total = len(re.sub(r"\W", "", text))
        return covered / total if total else 0.0

    def _looks_tech(self, text: str) -> bool:
        m = _LABEL.match(text)
        if m and _LABEL_KIND.get(_norm_label(m.group("label"))) == "tech":
            return True
        return self._tech_coverage(text) >= 0.5

    def _title_like(self, text: str) -> bool:
        text = text.strip()
        if not text or len(_words(text)) > self.max_title_words:
            return False
        if _is_sentence(text):
            return False
        m = _LABEL.match(text)
        return not (m and _norm_label(m.group("label")) in _LABEL_KIND)

    def _role_of(self, part: str) -> str:
        part = part.strip()
        if _DATE_ONLY.match(part):
            return "date"
        if _URL_ONLY.match(part):
            return "url"
        if self._looks_tech(part):
            return "tech"
        return "desc"

    def _try_split(self, text: str) -> _Split | None:
        """'Name | Tech', 'Name - description', 'Name: description' -> pieces."""
        parts: list[str] | None = None
        kind = ""
        if "|" in text:
            parts, kind = [p for p in _PIPE.split(text) if p.strip()], "pipe"
        elif _DASH.search(text):
            first = _DASH.split(text, maxsplit=1)
            parts, kind = [first[0], first[1]], "dash"
        else:
            m = _COLON.match(text)
            if m and _norm_label(m.group("left")) not in _LABEL_KIND:
                parts, kind = [m.group("left"), m.group("right")], "colon"
        if not parts or len(parts) < 2:
            return None
        name, name_date = _strip_trailing_date(_clean_name(parts[0]))
        if not name or not self._title_like(name):
            return None
        sp = _Split(name=_clean_name(name), kind=kind, dates=name_date)
        for part in parts[1:]:
            role = self._role_of(part)
            if role == "date":
                sp.dates = (sp.dates + " " + part.strip()).strip()
            elif role == "url":
                sp.links.append(part.strip())
            elif role == "tech":
                m = _LABEL.match(part)
                sp.tech = (sp.tech + ", " if sp.tech else "") + (m.group("rest") if m else part).strip()
            else:
                text_part, part_date = _strip_trailing_date(part.strip())
                if part_date:
                    sp.dates = (sp.dates + " " + part_date).strip()
                sp.desc = (sp.desc + " " if sp.desc else "") + text_part
        return sp

    def _classify(self, lines: list[_Line]) -> None:
        group: str | None = None
        group_num: int | None = None          # last number of a numbered list inside the group
        for ln in lines:
            if not ln.raw:
                ln.kind, group, group_num = "blank", None, None
                continue
            if _norm_label(ln.raw.rstrip(":")) in _SUBHEADINGS:
                ln.kind, group = "noise", None
                continue

            bullet_m = _BULLET.match(ln.raw)
            body = ln.raw[bullet_m.end():].strip() if bullet_m else ln.raw

            label_m = _LABEL.match(body)
            if label_m:
                kind = _LABEL_KIND.get(_norm_label(label_m.group("label")))
                if kind:
                    rest = label_m.group("rest").strip()
                    ln.kind, ln.text, ln.bullet = f"label_{kind}", rest, bool(bullet_m)
                    ln.label = _norm_label(label_m.group("label"))
                    group, group_num = (kind if not rest else None), None
                    continue

            if group and bullet_m:
                ln.kind, ln.text, ln.bullet = f"label_{group}", body, True
                continue
            if group and not bullet_m:
                nm = _NUMBERING.match(body)
                if nm:
                    n = int(next(g for g in nm.group("a", "b", "c", "d") if g))
                    # "Features:" followed by 1. 2. 3. is a sub-list of that label, not new projects
                    if (group_num is None and n == 1) or (group_num is not None and n == group_num + 1):
                        ln.kind, ln.text, ln.bullet = f"label_{group}", body[nm.end():].strip(), True
                        group_num = n
                        continue
            if group == "tech" and not _is_sentence(body) and len(_words(body)) <= 8:
                ln.kind, ln.text = "label_tech", body
                continue
            group = None

            if bullet_m:
                ln.kind, ln.text, ln.bullet = "bullet", _strip_decor(body), True
                continue

            numbering = None
            num_m = _NUMBERING.match(body)
            if num_m:
                numbering = int(next(g for g in num_m.group("a", "b", "c", "d") if g))
                body = body[num_m.end():].strip()
            body = _strip_decor(body)
            ln.numbering = numbering

            if _DATE_ONLY.match(body):
                ln.kind, ln.text = "date", body
                continue
            if _URL_ONLY.match(body):
                ln.kind, ln.text = "url", body
                continue

            split = self._try_split(body)
            if split is not None:
                ln.kind, ln.split, ln.text = "cand", split, split.name
                ln.dates, ln.links = split.dates, list(split.links)
                continue

            if _is_sentence(body):
                ln.kind, ln.text = "sentence", body
                continue
            if self._tech_coverage(body) >= 0.6 and (re.search(r"[,;/|]", body) or self._tech_coverage(body) >= 0.85):
                ln.kind, ln.text = "tech_only", body
                continue

            name, date = _strip_trailing_date(_clean_name(body))
            if not name or len(_words(name)) > self.max_title_words:
                ln.kind, ln.text = "sentence", body
                continue
            ln.kind, ln.text, ln.dates = "cand", name, date
            ln.links = [m.group(0) for m in _URL.finditer(body)]

    # ------------------------------------------------------------------ #
    # pass 2: decide boundaries
    # ------------------------------------------------------------------ #
    @staticmethod
    def _next_content(lines: list[_Line], i: int) -> _Line | None:
        for ln in lines[i + 1:]:
            if ln.kind not in ("blank", "noise"):
                return ln
        return None

    def _score(
        self, ln: _Line, prev: _Line | None, nxt: _Line | None, blank_before: bool, numbered_ok: bool
    ) -> tuple[float, list[str]]:
        name = ln.split.name if ln.split else ln.text
        cues: list[str] = []
        score = 0.0
        n = len(_words(name))
        if numbered_ok:
            score += 0.45
            cues.append("numbered")
        if blank_before:
            score += 0.25
            cues.append("blank_line_before" if prev is not None else "first_line")
        if n <= 8:
            score += 0.2
            cues.append("short")
        elif n <= self.max_title_words:
            score += 0.05
        else:
            score -= 0.35
            cues.append("long")
        if not name.rstrip().endswith((".", "!", "?")):
            score += 0.1
        cap = _cap_ratio(name)
        if cap >= 0.6:
            score += 0.25
            cues.append("title_case")
        elif cap >= 0.3:
            score += 0.1
        if any(w.lower() in _PROJECT_NOUNS for w in _words(name)):
            score += 0.15
            cues.append("project_noun")
        if nxt is not None and nxt.kind in _BODY_KINDS:
            score += 0.2
            cues.append("followed_by_description")
        if ln.split is not None:
            if ln.split.tech:
                score += 0.35
                cues.append("name_pipe_technology")
            elif ln.split.desc:
                score += 0.2
                cues.append("name_with_inline_description")
            else:
                score += 0.1
        return score, cues

    def _bullet_is_title(self, ln: _Line, nxt: _Line | None) -> bool:
        """'• Project name' followed by deeper-indented bullets."""
        if ln.kind != "bullet" or nxt is None or nxt.kind not in _BODY_KINDS or nxt.indent < ln.indent + 2:
            return False
        text = ln.text
        return self._title_like(text) and _cap_ratio(text) >= 0.6 and self._tech_coverage(text) < 0.6

    def _assemble(self, lines: list[_Line]) -> tuple[list[_Builder], list[_Line], list[str]]:
        builders: list[_Builder] = []
        pre: list[_Line] = []
        warnings: list[str] = []
        cur: _Builder | None = None
        expected: int | None = None
        prev: _Line | None = None
        prev_in_body = False           # previous content line was added to a project body
        blank_before = True

        for i, ln in enumerate(lines):
            if ln.kind == "blank":
                blank_before = True
                continue
            if ln.kind == "noise":
                continue
            nxt = self._next_content(lines, i)
            bullet_title = self._bullet_is_title(ln, nxt) and (
                cur is None or cur.title is None or ln.indent <= cur.title.indent
            )
            is_cand = ln.kind == "cand" or bullet_title

            if not is_cand:
                (cur.lines if cur else pre).append(ln)
                prev, prev_in_body, blank_before = ln, True, False
                continue

            prev_is_body = prev_in_body
            prev_is_title = prev is not None and cur is not None and prev is cur.title
            flags: list[str] = []

            # --- indentation: deeper than the current title => part of that project
            if cur is not None and cur.title is not None and ln.numbering is None and not bullet_title \
                    and ln.indent >= cur.title.indent + 2:
                cur.lines.append(ln)
                prev, prev_in_body, blank_before = ln, True, False
                continue

            # --- numbering sequence
            numbered_ok = False
            if ln.numbering is not None and not bullet_title:
                if expected is None or expected <= ln.numbering <= expected + 2:
                    numbered_ok = True
                    if expected is not None and ln.numbering > expected:
                        flags.append("numbering_gap")
                elif cur is not None and prev_is_body and not blank_before:
                    cur.lines.append(ln)                      # restarted numbering: sub-list item
                    prev, prev_in_body, blank_before = ln, True, False
                    continue
                elif cur is not None and cur.title is not None and ln.indent > cur.title.indent + 1:
                    cur.lines.append(ln)
                    prev, prev_in_body, blank_before = ln, True, False
                    continue

            if bullet_title:
                ln.text = _clean_name(ln.text)
            score, cues = self._score(ln, prev, nxt, blank_before, numbered_ok)
            if bullet_title:
                cues.append("bullet_with_nested_items")
                score += 0.3

            after_body_no_blank = prev_is_body and not blank_before
            extra_evidence = any(
                c in cues for c in ("project_noun", "followed_by_description", "name_pipe_technology",
                                    "bullet_with_nested_items")
            )
            strong = score >= self.strong_threshold and not (after_body_no_blank and not extra_evidence)

            if score < self.title_threshold:
                (cur.lines if cur else pre).append(ln)
                prev, prev_in_body, blank_before = ln, True, False
                continue

            if not strong:
                if blank_before or prev_is_title or prev is None:
                    flags.append("weak_boundary")
                elif cur is not None:
                    # Looks like a title but sits inside a body without a blank line:
                    # keep it with the current project and flag it instead of splitting.
                    cur.lines.append(ln)
                    cur.flags.append("possible_missed_boundary")
                    cur.missed.append(ln.text)
                    prev, prev_in_body, blank_before = ln, True, False
                    continue

            if ln.numbering is not None and not bullet_title:
                expected = ln.numbering + 1
            style = (
                "numbered" if numbered_ok else
                "bullet" if bullet_title else
                ln.split.kind if ln.split else "plain"
            )
            cur = _Builder(ln, cues, flags, style)
            builders.append(cur)
            prev, prev_in_body, blank_before = ln, False, False

        return builders, pre, warnings

    # ------------------------------------------------------------------ #
    # finishing: description, technologies, flags
    # ------------------------------------------------------------------ #
    def _finish(self, b: _Builder, text: str) -> Project:
        title = b.title
        body = b.lines
        all_lines = ([title] if title else []) + body
        start = all_lines[0].start
        end = all_lines[-1].end
        block = text[start:end]

        # ---- description (structured lines excluded) + dates + links
        desc_parts: list[tuple[str, bool]] = []     # (text, is_bullet)
        dates, links = "", []
        tech_texts: list[str] = []
        tech_buckets: list[str] = []   # one entry per technology line, keyed by its label
        if title is not None:
            dates = title.dates
            links = list(title.links)
            if title.split:
                if title.split.desc:
                    desc_parts.append((title.split.desc, False))
                if title.split.tech:
                    tech_texts.append(title.split.tech)
                    tech_buckets.append("<title>")
        for ln in body:
            if ln.kind in ("label_tech", "tech_only"):
                tech_texts.append(ln.text)
                if ln.kind == "tech_only":
                    tech_buckets.append("<unlabelled>")
                elif ln.label:                 # bullets inside a label group carry no label
                    tech_buckets.append(ln.label)
            elif ln.kind == "label_meta":
                links.extend(m.group(0) for m in _URL.finditer(ln.text))
                if ln.text and re.search(r"(?:19|20)\d{2}", ln.text) and not dates:
                    dates = ln.text
            elif ln.kind == "date":
                dates = (dates + " " + ln.text).strip()
            elif ln.kind == "url":
                links.append(ln.text)
            elif ln.text:
                desc_parts.append((ln.text, ln.bullet))

        paragraphs: list[str] = []
        for text_part, is_bullet in desc_parts:
            if paragraphs and not is_bullet and _is_continuation(paragraphs[-1], text_part):
                paragraphs[-1] = paragraphs[-1] + " " + text_part
            else:
                paragraphs.append(text_part)
        description = self.description_join.join(paragraphs)

        # ---- technologies for THIS block only (link lines are masked, keeping offsets intact)
        masked = list(block)
        for ln in all_lines:
            if ln.kind == "url" or (ln.kind == "label_meta" and ln.label in _LINK_LABELS):
                for k in range(ln.start - start, ln.end - start):
                    masked[k] = " "
        matches = self.matcher.find_all("".join(masked))
        refs = self._refs(matches, title, body, start)

        # ---- unrecognised items listed in technology lines (judged in line context)
        unrecognized: list[str] = []
        for t in tech_texts:
            spans = [(m.start, m.end) for m in self.matcher.find_all(t)]
            pos = 0
            segments: list[tuple[int, str]] = []
            for sep in _TOKEN_SPLIT.finditer(t):
                segments.append((pos, t[pos:sep.start()]))
                pos = sep.end()
            segments.append((pos, t[pos:]))
            for seg_start, seg in segments:
                token = seg.strip(" .;:()[]\"'")
                low = token.lower()
                if (len(token) < 2 or len(token) > 40 or len(_words(token)) > 4 or low in _IGNORED_UNRECOGNIZED
                        or re.fullmatch(r"[\d.\s]+", token)):
                    continue
                seg_end = seg_start + len(seg)
                if any(a < seg_end and b > seg_start for a, b in spans):
                    continue                      # overlaps a recognised technology
                if token not in unrecognized:
                    unrecognized.append(token)

        # ---- review flags / notes
        review = list(dict.fromkeys(b.flags))
        notes: list[str] = []
        if not description and not tech_texts:
            notes.append("no_description")
        elif not description:
            notes.append("no_description_text")
        if not refs:
            notes.append("no_technologies")
        if any(tech_buckets.count(label) >= 2 for label in set(tech_buckets)):
            # the SAME label twice ("Technologies:" ... "Technologies:") suggests two merged projects;
            # different labels (Frontend / Backend / Database) are normal for one project.
            review.append("multiple_technology_lines")
        if len(body) > self.max_project_lines:
            review.append("unusually_long_project")
        if b.missed:
            notes.append("title_like_lines_kept_in_body: " + "; ".join(b.missed))

        if title is not None:
            name = title.text
            line_text = text[title.start:title.end]
            idx = line_text.find(name)
            name_start = title.start + (idx if idx >= 0 else 0)
            name_end = name_start + (len(name) if idx >= 0 else len(line_text))
        else:
            name, name_start, name_end = "", start, start

        return Project(
            name=name,
            description=description,
            technologies=[r.name for r in refs],
            technology_details=refs,
            unrecognized_technologies=unrecognized,
            dates=dates,
            links=list(dict.fromkeys(links)),
            title_style=b.style,
            boundary_reasons=list(b.reasons),
            needs_review=bool(review),
            review_reasons=list(dict.fromkeys(review)),
            notes=notes,
            start=start,
            end=end,
            name_start=name_start,
            name_end=name_end,
        )

    @staticmethod
    def _refs(matches: list[Any], title: _Line | None, body: list[_Line], block_start: int) -> list[TechnologyRef]:
        """Group matcher hits by technology (first-mention order), with provenance."""
        name_end_abs = -1
        if title is not None and title.split is not None:
            idx = title.raw.find(title.split.name)
            name_end_abs = title.start + (idx + len(title.split.name) if idx >= 0 else len(title.raw))

        def where(abs_pos: int) -> str:
            if title is not None and title.start <= abs_pos < title.end:
                if title.split is not None and abs_pos >= name_end_abs:
                    return "technology_line" if title.split.tech else "description"
                return "title"
            for ln in body:
                if ln.start <= abs_pos < ln.end:
                    return "technology_line" if ln.kind in ("label_tech", "tech_only") else "description"
            return "description"

        grouped: dict[str, list[Any]] = {}
        for m in matches:
            grouped.setdefault(m.name, []).append(m)
        refs: list[TechnologyRef] = []
        for name, ms in grouped.items():
            first = ms[0]
            evidences = tuple(dict.fromkeys(m.evidence for m in ms))
            found_in = tuple(dict.fromkeys(where(block_start + m.start) for m in ms))
            refs.append(
                TechnologyRef(
                    name=name,
                    category=first.category,
                    matched_text=first.matched_text,
                    start=block_start + first.start,
                    end=block_start + first.end,
                    evidence=first.evidence,
                    all_evidence=evidences,
                    occurrences=len(ms),
                    found_in=found_in,
                    reason=first.reason,
                )
            )
        return refs
