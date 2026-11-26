"""Rule-based resume section detection (no LLM, no ML model).

Public API
----------
``SectionDetector.detect(text) -> SectionResult``

``SectionResult`` is a ``dict`` (canonical key -> section text) that also
carries rich metadata (original heading, character offsets, warnings), so it
can be used exactly like the plain dict in the project spec::

    sections = SectionDetector().detect(resume_text)
    sections.get("projects", "")        # plain dict access
    sections.metadata[0].start          # offsets into the ORIGINAL text

Design notes
------------
* A line is a heading only if, after normalisation, the WHOLE heading text
  equals a known alias.  We never do substring/"contains" matching, which is
  what keeps "Skills Assessment Platform" or "Experience in Python and Java"
  from being mistaken for headings.
* Lines that start with a bullet marker are never headings.
* Aliases that commonly appear *inside* projects/jobs ("Technologies",
  "Tech Stack", "Tools") are deliberately NOT registered, because
  "Technologies: Python, Django" is a legitimate project line.
* Known-but-irrelevant headings (Hobbies, Languages, References, ...) are
  "boundary-only": they end the previous section so their content does not
  leak into e.g. Education, but they are excluded from the result dict.
  They only count when they stand alone on the line: "Languages: Python, Java"
  inside a skills block is content, not a heading.
* All offsets refer to the original, unmodified input string.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDARY_ONLY_ALIASES",
    "OTHER_KEY",
    "SECTION_ALIASES",
    "Section",
    "SectionDetector",
    "SectionResult",
    "normalize_heading",
]

# --------------------------------------------------------------------------- #
# Dictionaries (extensible via SectionDetector(extra_aliases=...))
# --------------------------------------------------------------------------- #

OTHER_KEY = "other"

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "skills": (
        "skills", "technical skills", "key skills", "core skills",
        "core competencies", "technical competencies", "technical proficiency",
        "technical expertise", "skills and tools", "skills and technologies",
        "skill set", "skillset", "areas of expertise", "it skills",
        "computer skills", "programming skills", "professional skills",
        "technical skills and tools", "technical skill set", "core technical skills",
        "technical skills summary", "key competencies",
    ),
    "projects": (
        "projects", "project", "academic projects", "academic project",
        "personal projects", "personal project", "key projects",
        "major projects", "notable projects", "selected projects",
        "project experience", "technical projects", "project work",
        "projects undertaken", "side projects", "college projects",
        "relevant projects", "software projects", "mini projects",
        "key engineering projects", "engineering projects", "featured projects",
        "technical project", "major project", "projects portfolio",
    ),
    "experience": (
        "experience", "professional experience", "work experience",
        "employment history", "work history", "employment",
        "employment experience", "career history", "relevant experience",
        "industry experience", "professional background",
        "work experience details", "other work experience", "other experience",
    ),
    "internships": (
        "internships", "internship", "internship experience",
        "industrial training", "internships and training", "externship", "externships",
    ),
    "education": (
        "education", "educational qualifications", "educational qualification",
        "education qualification", "educational background",
        "academic background", "academic qualifications",
        "academic qualification", "academic details", "academics",
        "education details", "education and training",
    ),
    "certifications": (
        "certifications", "certification", "certificates",
        "licenses and certifications", "certifications and licenses",
        "professional certifications", "courses and certifications",
        "training and certifications", "certifications and training",
        "licenses",
    ),
    "achievements": (
        "achievements", "achievement", "accomplishments", "awards",
        "awards and honors", "awards and honours", "honors and awards",
        "honours and awards", "achievements and awards",
        "awards and achievements", "key achievements", "notable achievements",
        "honors", "honours", "recognition", "scholarships and awards",
        "competitions", "hackathons", "hackathons and competitions",
    ),
    "summary": (
        "summary", "professional summary", "career summary",
        "executive summary", "profile", "professional profile",
        "career profile", "career objective", "objective", "about me",
        "personal summary", "summary of qualifications", "profile summary",
        "professional overview", "career overview", "summary profile",
    ),
}

# Recognised headings we do not extract, but which must terminate the
# preceding section.
BOUNDARY_ONLY_ALIASES: tuple[str, ...] = (
    "languages", "language skills", "hobbies", "interests",
    "hobbies and interests", "interests and hobbies", "areas of interest",
    "extracurricular activities", "extra curricular activities",
    "co curricular activities", "activities", "references", "declaration",
    "personal details", "personal information", "personal profile",
    "publications", "research", "volunteer experience", "volunteering",
    "volunteer work", "leadership", "positions of responsibility",
    "strengths", "soft skills", "links", "contact", "contact information",
    "contact details", "training", "workshops", "conferences", "patents",
    "research publication", "research publications", "publication",
    "additional information", "other information", "extracurricular",
    "coding proficiency", "coding profiles", "coding profile", "coding practice",
    "competitive programming", "problem solving", "dsa", "online profiles", "problem solving dsa",
)

# --------------------------------------------------------------------------- #
# Heading normalisation
# --------------------------------------------------------------------------- #

_LEADING_DECOR = re.compile(r"^[\s#*_=~|>]+")
# 1.  1)  (1)  1.2  1:  |  I.  IV)  |  A.  b)
_NUMBERING = re.compile(
    r"^(?:\(?\d{1,2}(?:\.\d{1,2})*[.):]?|\(?[ivxlc]{1,5}[.)]|\(?[a-z][.)])\s+",
    re.IGNORECASE,
)
_SPACED_LETTERS = re.compile(r"^(?:[^\W\d_] ){2,}[^\W\d_]$")  # "P R O J E C T S"
_NON_WORD = re.compile(r"[\W_]+")
# PDF symbol fonts extract bullets as private-use / C1 control characters (U+F0B7, U+0087 ...)
_BULLET = re.compile(r"^\s*(?:[•●▪■◦·–—*-]\s+|[‣⁃\x80-\x9f-�]\s*)\S")
# "Heading: optional inline content" -- head is at most 60 chars, no colon.
_COLON_SPLIT = re.compile(
    r"^(?P<lead>\s*)(?P<head>[^:]{1,60}?)\s*:(?P<gap>\s*)(?P<rest>\S.*)?$"
)
# Two-column layout: heading flush left, a wide gap, then content.
_GAP_SPLIT = re.compile(r"^(?P<head>[A-Za-z][A-Za-z &/]{1,40}?)\s{3,}(?P<rest>\S.*)$")
# Singular labels with inline content name one item ("Project: X"), not a section.
_SINGULAR_INLINE = frozenset({"project"})


def _ends_mid_sentence(line: str) -> bool:
    """Does ``line`` look cut off by a hard wrap (so the next line continues it)?

    True when it is non-blank and ends with a lowercase letter or a joining
    character, e.g. "...for a polished user" or "Python, Java,".
    """
    stripped = line.rstrip()
    return bool(stripped) and (stripped[-1].islower() or stripped[-1] in ",;&/-–—(+")


def normalize_heading(raw: str) -> str:
    """Reduce a heading line to a canonical lowercase comparison key.

    Removes numbering, decoration, colons, case and spacing differences:
    ``"## 2) Technical  Skills:"`` -> ``"technical skills"``.
    Only used for *matching*; offsets always refer to the original text.
    """
    s = unicodedata.normalize("NFKC", raw).replace("\u00a0", " ")
    s = _LEADING_DECOR.sub("", s)
    s = _NUMBERING.sub("", s, count=1).strip()
    if _SPACED_LETTERS.match(s):
        s = s.replace(" ", "")
    s = s.replace("&", " and ").replace("/", " and ").casefold()
    return _NON_WORD.sub(" ", s).strip()


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Section:
    """One detected section.

    Invariants (verified in tests), with ``src`` the original input:
        ``src[heading_start:heading_end] == heading``
        ``src[start:end] == text``  (content, outer whitespace trimmed)
    """

    key: str            # canonical key, e.g. "skills" ("other" if boundary-only)
    label: str          # normalised heading as written, e.g. "academic projects"
    heading: str        # heading exactly as it appears in the source
    heading_start: int
    heading_end: int
    start: int          # content start offset
    end: int            # content end offset
    text: str           # original content slice
    inline: bool        # True if content started on the heading line ("Skills: A, B")
    target: bool        # False for boundary-only sections


class SectionResult(dict):
    """``dict[str, str]`` of canonical key -> text, plus metadata.

    * ``metadata``  - every detected :class:`Section`, in document order
    * ``preamble``  - text before the first heading (name, contact info, ...)
    * ``warnings``  - machine-readable flags for downstream review
    """

    def __init__(
        self,
        texts: Mapping[str, str],
        metadata: list[Section],
        preamble: str,
        warnings: list[str],
    ) -> None:
        super().__init__(texts)
        self.metadata = metadata
        self.preamble = preamble
        self.warnings = warnings

    def get_sections(self, key: str) -> list[Section]:
        """All sections for a canonical key (e.g. both 'Academic' and
        'Personal Projects' for ``"projects"``), with individual offsets."""
        return [s for s in self.metadata if s.key == key and s.target]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view including metadata."""
        return {
            "sections": dict(self),
            "metadata": [asdict(s) for s in self.metadata],
            "preamble": self.preamble,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _Hit:
    key: str
    target: bool
    label: str
    line_start: int
    line_end: int        # offset of the next line's start
    heading_start: int
    heading_end: int
    inline_start: int | None


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #

class SectionDetector:
    """Detects resume sections using exact-alias heading matching.

    Parameters
    ----------
    extra_aliases:
        ``{canonical_key: [alias, ...]}`` merged into the defaults. A new key
        creates a new section type.
    allow_inline_content:
        Accept ``"SKILLS: Python, Java"`` (content on the heading line).
    max_heading_length:
        Standalone lines longer than this are never headings.

    The alias lookup table is built once in ``__init__``; ``detect`` is a
    single linear pass over the lines, so one instance can be reused for any
    number of resumes.
    """

    def __init__(
        self,
        extra_aliases: Mapping[str, Iterable[str]] | None = None,
        allow_inline_content: bool = True,
        max_heading_length: int = 60,
    ) -> None:
        self.allow_inline_content = allow_inline_content
        self.max_heading_length = max_heading_length
        self._lookup: dict[str, tuple[str, bool]] = {}

        for alias in BOUNDARY_ONLY_ALIASES:
            self._lookup[normalize_heading(alias)] = (OTHER_KEY, False)
        merged: dict[str, list[str]] = {k: list(v) for k, v in SECTION_ALIASES.items()}
        for key, aliases in (extra_aliases or {}).items():
            merged.setdefault(key, []).extend(aliases)
        for key, aliases in merged.items():
            for alias in aliases:
                self._lookup[normalize_heading(alias)] = (key, True)
        self._despaced = {k.replace(" ", ""): v for k, v in self._lookup.items()}

    # ------------------------------------------------------------------ #
    def detect(self, text: str) -> SectionResult:
        """Split ``text`` into sections.

        Raises ``TypeError`` for non-string input (fail loudly rather than
        silently returning an empty result for a caller bug). Empty or
        whitespace-only strings return an empty result.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"resume text must be str, got {type(text).__name__}"
            )

        hits, rejected = self._find_headings(text)
        warnings: list[str] = [f"wrapped_line_not_heading:{h.label}" for h in rejected]
        if not hits:
            if text.strip():
                warnings.append("no_sections_detected")
            return SectionResult({}, [], text.strip(), warnings)

        preamble = text[: hits[0].line_start].strip()
        metadata: list[Section] = []
        texts: dict[str, list[str]] = {}

        for i, hit in enumerate(hits):
            raw_start = hit.inline_start if hit.inline_start is not None else hit.line_end
            raw_end = hits[i + 1].line_start if i + 1 < len(hits) else len(text)
            start, end = self._trim(text, raw_start, raw_end)
            section = Section(
                key=hit.key,
                label=hit.label,
                heading=text[hit.heading_start : hit.heading_end],
                heading_start=hit.heading_start,
                heading_end=hit.heading_end,
                start=start,
                end=end,
                text=text[start:end],
                inline=hit.inline_start is not None,
                target=hit.target,
            )
            metadata.append(section)
            if not hit.target:
                continue
            if not section.text:
                warnings.append(f"empty_section:{hit.key}")
                continue
            texts.setdefault(hit.key, []).append(section.text)

        merged = {}
        for key, parts in texts.items():
            if len(parts) > 1:
                warnings.append(f"merged_multiple_sections:{key}")
            merged[key] = "\n\n".join(parts)

        logger.debug("detected %d headings, keys=%s", len(hits), list(merged))
        return SectionResult(merged, metadata, preamble, warnings)

    # ------------------------------------------------------------------ #
    def _find_headings(self, text: str) -> tuple[list[_Hit], list[_Hit]]:
        """Return ``(headings, rejected)``.

        A lowercase standalone heading word directly under a line that ends
        mid-sentence is a hard-wrapped continuation ("...for a polished user"
        / "experience"), not a heading. This only applies when the resume's
        other headings are capitalised; a resume that uses lowercase headings
        (a styling choice) keeps them all.
        """
        hits: list[_Hit] = []
        gap_hits: list[_Hit] = []
        suspects: set[int] = set()
        lowercase_style = False
        capitalised_style = False
        offset = 0
        prev = ""
        for piece in text.splitlines(keepends=True):
            body = piece.splitlines()[0] if piece else ""
            match = self._match_line(body)
            if match is None and body.strip() and not _BULLET.match(body):
                gap = self._match_gap(body)
                if gap is not None:
                    key, target, label, h_start, h_end, inline = gap
                    gap_hits.append(_Hit(key, target, label, offset, offset + len(piece),
                                         offset + h_start, offset + h_end, offset + inline))
            if match is not None:
                key, target, label, h_start, h_end, inline = match
                heading = body[h_start:h_end]
                lowercase = heading[:1].islower()
                if lowercase and inline is None and _ends_mid_sentence(prev):
                    suspects.add(len(hits))
                elif lowercase:
                    lowercase_style = True
                else:
                    capitalised_style = True
                hits.append(
                    _Hit(
                        key=key,
                        target=target,
                        label=label,
                        line_start=offset,
                        line_end=offset + len(piece),
                        heading_start=offset + h_start,
                        heading_end=offset + h_end,
                        inline_start=None if inline is None else offset + inline,
                    )
                )
            offset += len(piece)
            prev = body
        rejected: list[_Hit] = []
        if not (lowercase_style or not capitalised_style or not suspects):
            rejected = [h for i, h in enumerate(hits) if i in suspects]
            hits = [h for i, h in enumerate(hits) if i not in suspects]
        # A two-column "Heading    content" layout is a whole-document style:
        # one such line alone is far more likely ordinary content.
        if len(gap_hits) >= 2:
            hits = sorted(hits + gap_hits, key=lambda h: h.line_start)
        return hits, rejected

    def _match_line(
        self, body: str
    ) -> tuple[str, bool, str, int, int, int | None] | None:
        """Return (key, target, label, head_start, head_end, inline_start)
        relative to the line, or None if the line is not a heading."""
        if not body.strip() or _BULLET.match(body):
            return None

        m = _COLON_SPLIT.match(body)
        if m:
            label = normalize_heading(m.group("head"))
            entry = self._resolve(label)
            if entry is None:
                return None
            rest = m.group("rest")
            # Inline content ("Languages: Python, Java") is only trusted for
            # real target sections. For boundary-only headings it is far more
            # likely to be a line INSIDE another section (a skills block).
            if rest and (not self.allow_inline_content or not entry[1]):
                return None
            # "Project: Payments revamp" inside a job entry names ONE project.
            if rest and label in _SINGULAR_INLINE:
                return None
            return (
                entry[0], entry[1], label,
                m.end("lead"), m.start("gap"),          # heading incl. colon
                m.start("rest") if rest else None,
            )

        core = body.strip()
        if len(core) > self.max_heading_length:
            return None
        label = normalize_heading(core)
        entry = self._resolve(label)
        if entry is None and core.isupper():
            entry = self._resolve_loose_compound(label)
        if entry is None:
            return None
        return (
            entry[0], entry[1], label,
            len(body) - len(body.lstrip()), len(body.rstrip()), None,
        )

    def _resolve_loose_compound(self, label: str) -> tuple[str, bool] | None:
        """ALL-CAPS standalone line whose first part is a known target heading and
        whose other parts are one or two words: "TECHNICAL SKILLS AND INTERSETS"
        (typo), "PROJECTS AND RESEARCH". Capitals on a line of their own are a
        strong heading cue, so one unknown short tail is tolerated."""
        parts = label.split(" and ")
        if len(parts) < 2:
            return None
        first = self._lookup.get(parts[0])
        if first is None or not first[1] or any(len(p.split()) > 2 for p in parts[1:]):
            return None
        return first

    def _match_gap(self, body: str) -> tuple[str, bool, str, int, int, int] | None:
        """Two-column layout: "Summary        Final-year student ..." (heading
        flush left, 3+ spaces, content). Only target sections qualify."""
        if not self.allow_inline_content:
            return None
        m = _GAP_SPLIT.match(body)
        if not m:
            return None
        label = normalize_heading(m.group("head"))
        entry = self._resolve(label)
        if entry is None or not entry[1] or label in _SINGULAR_INLINE:
            return None
        return entry[0], entry[1], label, 0, m.end("head"), m.start("rest")

    def _resolve(self, label: str) -> tuple[str, bool] | None:
        """Exact alias, letter-spaced alias ("P R O F E S S I O N A L S U M M A R Y"
        -> "professionalsummary") or compound heading ("projects and experience")."""
        entry = self._lookup.get(label)
        if entry is None and label and " " not in label:
            entry = self._despaced.get(label)
            if entry is None:
                # PDF text layers sometimes glue two headings: "PROJECTSEDUCATION"
                # (the visible heading plus a hidden duplicate). Both halves must be
                # known headings; the first one is the one that is displayed.
                entry = next((self._despaced[label[:i]] for i in range(3, len(label) - 2)
                              if label[:i] in self._despaced and label[i:] in self._despaced), None)
        return entry if entry is not None else self._resolve_compound(label)

    def _resolve_compound(self, label: str) -> tuple[str, bool] | None:
        """``"certifications and achievements"`` -> certifications.

        Every part must be a known heading (so "Experience in Python and
        Java" never qualifies). Projects wins when present, because a mixed
        "Projects & Experience" section is where the projects are; otherwise
        the first part decides.
        """
        parts = label.split(" and ")
        if not 2 <= len(parts) <= 4:
            return None
        # cover the parts left-to-right; aliases may contain "and" themselves
        cover: list[list[tuple[str, bool]] | None] = [None] * (len(parts) + 1)
        cover[0] = []
        for i in range(len(parts)):
            if cover[i] is None:
                continue
            for j in range(i + 1, len(parts) + 1):
                entry = self._lookup.get(" and ".join(parts[i:j]))
                if entry is not None and cover[j] is None:
                    cover[j] = cover[i] + [entry]
        entries = cover[-1]
        if not entries or len(entries) < 2:
            return None
        return next((e for e in entries if e[0] == "projects"), entries[0])

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[int, int]:
        """Shrink [start, end) so it excludes outer whitespace."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end
