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
        "technical skills and tools", "technical skill set",
    ),
    "projects": (
        "projects", "project", "academic projects", "academic project",
        "personal projects", "personal project", "key projects",
        "major projects", "notable projects", "selected projects",
        "project experience", "technical projects", "project work",
        "projects undertaken", "side projects", "college projects",
        "relevant projects", "software projects", "mini projects",
    ),
    "experience": (
        "experience", "professional experience", "work experience",
        "employment history", "work history", "employment",
        "employment experience", "career history", "relevant experience",
        "industry experience", "professional background",
        "work experience details",
    ),
    "internships": (
        "internships", "internship", "internship experience",
        "industrial training", "internships and training",
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
    ),
    "summary": (
        "summary", "professional summary", "career summary",
        "executive summary", "profile", "professional profile",
        "career profile", "career objective", "objective", "about me",
        "personal summary", "summary of qualifications",
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
_BULLET = re.compile(r"^\s*[•●▪■◦·–—*-]\s+\S")
# "Heading: optional inline content" -- head is at most 60 chars, no colon.
_COLON_SPLIT = re.compile(
    r"^(?P<lead>\s*)(?P<head>[^:]{1,60}?)\s*:(?P<gap>\s*)(?P<rest>\S.*)?$"
)


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

        hits = self._find_headings(text)
        warnings: list[str] = []
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
    def _find_headings(self, text: str) -> list[_Hit]:
        hits: list[_Hit] = []
        offset = 0
        for piece in text.splitlines(keepends=True):
            body = piece.splitlines()[0] if piece else ""
            match = self._match_line(body)
            if match is not None:
                key, target, label, h_start, h_end, inline = match
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
        return hits

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
            entry = self._lookup.get(label)
            if entry is None:
                return None
            rest = m.group("rest")
            # Inline content ("Languages: Python, Java") is only trusted for
            # real target sections. For boundary-only headings it is far more
            # likely to be a line INSIDE another section (a skills block).
            if rest and (not self.allow_inline_content or not entry[1]):
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
        entry = self._lookup.get(label)
        if entry is None:
            return None
        return (
            entry[0], entry[1], label,
            len(body) - len(body.lstrip()), len(body.rstrip()), None,
        )

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[int, int]:
        """Shrink [start, end) so it excludes outer whitespace."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end
