"""Dictionary-driven technology matching with evidence (no LLM, no ML model).

Public API
----------
``TechnologyMatcher.find_all(text)``  every occurrence, ordered by position
``TechnologyMatcher.extract(text)``   one entry per technology (deduplicated)
``TechnologyMatcher.extract_names(text)``  just the canonical names

Each result is a :class:`TechMatch` carrying the canonical name, category, the
text exactly as written, character offsets, and the *evidence sentence*.

Pipeline (every stage is engine-independent except stage 1)
-----------------------------------------------------------
1. ENGINE      finds candidate spans (regex trie by default, FlashText optional)
2. LOOKUP      normalised matched text -> canonical technology (alias handling)
3. VALIDATE    drops "Moment.js"-style fragments, "C+", URLs and e-mail addresses
4. OVERLAPS    longest phrase wins ("Spring Boot" beats "Spring")
5. AMBIGUITY   short/common words (Go, R, C, Swift, Spring, ...) need exact case
               AND technology context, otherwise they are rejected
6. EVIDENCE    sentence containing the match (wrapped lines are re-joined)

Matching rules
--------------
* Case-insensitive for normal terms; case-SENSITIVE for ambiguous terms.
* A match must not touch a word character on either side, so "Java" never
  matches inside "JavaScript" and "R" never matches inside "React".
* Spaces and hyphens in multi-word names are interchangeable and may wrap
  across lines: "Scikit-learn" == "scikit learn" == "Scikit\\nlearn".
* All offsets refer to the original input string.
"""

from __future__ import annotations

import bisect
import itertools
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_ALIASES",
    "DEFAULT_MASTER",
    "DictionaryError",
    "EngineUnavailable",
    "FlashTextEngine",
    "RegexTrieEngine",
    "SurfaceInfo",
    "TechDictionary",
    "TechMatch",
    "TechnologyMatcher",
    "build_dictionary",
    "load_dictionary",
    "normalize_surface",
    "resolve_overlaps",
    "sentence_span_at",
    "split_sentences",
]

DEFAULT_DICT_DIR = Path(__file__).resolve().parent / "dictionaries"
DEFAULT_MASTER = DEFAULT_DICT_DIR / "technologies_master.json"
DEFAULT_ALIASES = DEFAULT_DICT_DIR / "technology_aliases.json"


class DictionaryError(ValueError):
    """The technology dictionaries are missing, malformed or inconsistent."""


class EngineUnavailable(RuntimeError):
    """A requested matching engine cannot be used (e.g. FlashText not installed)."""


# --------------------------------------------------------------------------- #
# Dictionary
# --------------------------------------------------------------------------- #

_SEPARATORS = re.compile(r"[\s\-_]+")


def normalize_surface(text: str) -> str:
    """Canonical comparison key: lowercase, runs of space/hyphen/underscore -> one space.

    ``"Scikit-Learn"`` -> ``"scikit learn"``; ``"Node.JS"`` -> ``"node.js"``.
    """
    return _SEPARATORS.sub(" ", text.strip().lower())


@dataclass(frozen=True)
class SurfaceInfo:
    """What a normalised surface form means."""

    name: str                 # canonical technology name
    category: str
    kind: str                 # "canonical" | "alias"
    exact: str | None = None  # set => ambiguous: matched text must equal this exactly


@dataclass(frozen=True)
class TechDictionary:
    categories: Mapping[str, str]        # canonical name -> category
    surfaces: Mapping[str, SurfaceInfo]  # normalised surface -> info
    master: Mapping[str, Any]
    aliases: Mapping[str, str]

    def extended(
        self,
        extra_technologies: Mapping[str, Iterable[str]] | None = None,
        extra_aliases: Mapping[str, str] | None = None,
        extra_ambiguous: Iterable[str] | None = None,
    ) -> "TechDictionary":
        """Return a NEW dictionary with additions (the cached one is never mutated).

        Extra technologies that already exist (in any category) are ignored, so
        re-adding a known name is harmless; conflicting aliases still raise.
        """
        master = {
            "categories": {c: list(n) for c, n in self.master["categories"].items()},
            "ambiguous_surfaces": list(self.master.get("ambiguous_surfaces", [])),
        }
        known = set(self.categories)
        for category, names in (extra_technologies or {}).items():
            fresh = [n for n in names if n not in known]
            known.update(fresh)
            if fresh:
                master["categories"].setdefault(category, []).extend(fresh)
        master["ambiguous_surfaces"].extend(extra_ambiguous or [])
        aliases = dict(self.aliases)
        aliases.update(extra_aliases or {})
        return build_dictionary(master, aliases)


def build_dictionary(master: Mapping[str, Any], aliases: Mapping[str, str]) -> TechDictionary:
    """Validate raw dictionary data and build lookup tables.

    Raises DictionaryError for: duplicate technologies, aliases pointing at
    unknown technologies, one surface meaning two technologies, and ambiguous
    entries that match no surface (usually a typo).
    """
    categories = master.get("categories") if isinstance(master, Mapping) else None
    if not isinstance(categories, Mapping) or not categories:
        raise DictionaryError("master dictionary needs a non-empty 'categories' object")
    if not isinstance(aliases, Mapping):
        raise DictionaryError("alias dictionary must be an object of alias -> canonical name")

    name_to_category: dict[str, str] = {}
    for category, names in categories.items():
        if not isinstance(names, (list, tuple)):
            raise DictionaryError(f"category {category!r} must be a list of names")
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise DictionaryError(f"invalid technology name {name!r} in {category!r}")
            if name in name_to_category:
                raise DictionaryError(
                    f"technology {name!r} listed in both {name_to_category[name]!r} and {category!r}"
                )
            name_to_category[name] = category

    surfaces: dict[str, SurfaceInfo] = {}

    def add(surface: str, name: str, kind: str) -> None:
        key = normalize_surface(surface)
        if not key:
            raise DictionaryError(f"empty surface form for {name!r}")
        existing = surfaces.get(key)
        if existing is not None:
            if existing.name != name:
                raise DictionaryError(
                    f"{surface!r} means both {existing.name!r} and {name!r}"
                )
            return  # same meaning; keep the first (canonical) entry
        surfaces[key] = SurfaceInfo(name, name_to_category[name], kind)

    for name in name_to_category:
        add(name, name, "canonical")
    for alias, target in aliases.items():
        if target not in name_to_category:
            raise DictionaryError(f"alias {alias!r} points to unknown technology {target!r}")
        add(alias, target, "alias")

    for exact in master.get("ambiguous_surfaces", []):
        key = normalize_surface(exact)
        info = surfaces.get(key)
        if info is None:
            raise DictionaryError(f"ambiguous surface {exact!r} matches no technology or alias")
        surfaces[key] = replace(info, exact=exact)

    return TechDictionary(name_to_category, surfaces, dict(master), dict(aliases))


@lru_cache(maxsize=16)
def _load_cached(master_path: str, aliases_path: str, master_mtime: int, aliases_mtime: int) -> TechDictionary:
    # mtimes are part of the cache key: editing a JSON file invalidates the
    # entry, while repeated construction of matchers never re-parses the files.
    try:
        with open(master_path, encoding="utf-8") as fh:
            master = json.load(fh)
        with open(aliases_path, encoding="utf-8") as fh:
            aliases = json.load(fh)
    except json.JSONDecodeError as exc:
        raise DictionaryError(f"invalid JSON in dictionary file: {exc}") from exc
    logger.debug("loaded technology dictionaries (%s, %s)", master_path, aliases_path)
    return build_dictionary(master, aliases)


def load_dictionary(
    master_path: str | Path | None = None, aliases_path: str | Path | None = None
) -> TechDictionary:
    """Load (with caching) the master and alias dictionaries."""
    master = Path(master_path or DEFAULT_MASTER).resolve()
    aliases = Path(aliases_path or DEFAULT_ALIASES).resolve()
    try:
        return _load_cached(
            str(master), str(aliases), master.stat().st_mtime_ns, aliases.stat().st_mtime_ns
        )
    except OSError as exc:
        raise DictionaryError(f"cannot read dictionary file: {exc}") from exc


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TechMatch:
    """One technology mention (or, from ``extract``, one technology)."""

    name: str                  # canonical, e.g. "Node.js"
    category: str              # e.g. "Backend"
    matched_text: str          # exactly as written, e.g. "NodeJS"
    start: int                 # offsets of matched_text in the input
    end: int
    evidence: str              # sentence containing the match (whitespace collapsed)
    evidence_start: int        # offsets of the evidence in the input (before collapsing)
    evidence_end: int
    match_type: str            # "canonical" | "alias"
    reason: str                # human-readable explanation of why it matched
    occurrences: int = 1       # set by extract(): total mentions in the text
    all_evidence: tuple[str, ...] = field(default_factory=tuple)  # set by extract()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["all_evidence"] = list(self.all_evidence)
        return d


# --------------------------------------------------------------------------- #
# Engines (stage 1): text -> candidate spans
# --------------------------------------------------------------------------- #

def _trie_regex(node: dict) -> str:
    """Compile a trie into a regex fragment (longest continuation preferred)."""
    kids = sorted(k for k in node if k != "")
    if not kids:
        return ""
    parts = []
    for ch in kids:
        char_re = r"[\s\-_]+" if ch == " " else re.escape(ch)
        parts.append(char_re + _trie_regex(node[ch]))
    body = parts[0] if len(parts) == 1 else "(?:" + "|".join(parts) + ")"
    return f"(?:{body})?" if "" in node else body


class RegexTrieEngine:
    """Default engine: all surfaces compiled into ONE regex via a character trie.

    * Greedy optional groups => longest phrase wins at each position.
    * ``(?<!\\w)`` / ``(?!\\w)`` => word-boundary matching, also for C++ / C#.
    * Runs in C inside ``re``; flexible whitespace/hyphen between words.
    """

    name = "regex"

    def __init__(self, surfaces: Iterable[str]) -> None:
        root: dict = {}
        for surface in surfaces:
            node = root
            for ch in surface:
                node = node.setdefault(ch, {})
            node[""] = True
        self.pattern = (
            re.compile(r"(?<!\w)(?:" + _trie_regex(root) + r")(?!\w)", re.IGNORECASE)
            if root
            else None
        )

    def find(self, text: str) -> list[tuple[int, int]]:
        if self.pattern is None:
            return []
        return [m.span() for m in self.pattern.finditer(text)]


class FlashTextEngine:
    """Optional engine backed by the FlashText library (``pip install flashtext``).

    Differences from the regex engine: only single spaces/hyphens between words
    are recognised (variants are pre-generated), and FlashText lower-cases the
    text, so for the rare input whose lower-cased length differs
    (e.g. 'İ') ``find`` returns ``None`` and the matcher falls back to regex.
    """

    name = "flashtext"

    def __init__(self, surfaces: Iterable[str]) -> None:
        try:
            from flashtext import KeywordProcessor
        except ImportError as exc:
            raise EngineUnavailable("flashtext is not installed (pip install flashtext)") from exc
        self._kp = KeywordProcessor(case_sensitive=False)
        for surface in surfaces:
            words = surface.split(" ")
            for seps in itertools.product(" -", repeat=len(words) - 1):
                variant = words[0] + "".join(s + w for s, w in zip(seps, words[1:]))
                self._kp.add_keyword(variant, surface)

    def find(self, text: str) -> list[tuple[int, int]] | None:
        if len(text.lower()) != len(text):
            return None
        return [(s, e) for _name, s, e in self._kp.extract_keywords(text, span_info=True)]


# --------------------------------------------------------------------------- #
# Overlap resolution (stage 4)
# --------------------------------------------------------------------------- #

def resolve_overlaps(items: Iterable[Any]) -> list[Any]:
    """Keep a non-overlapping subset, preferring the longest span (leftmost on ties).

    ``items`` need ``.start`` / ``.end``. Result is sorted by position.
    Engines already return non-overlapping spans; this is the safety net that
    also makes engines such as spaCy's PhraseMatcher (which reports overlaps)
    pluggable.

    Runs in O(n log n): spans are sorted, grouped into clusters of mutually
    connected overlaps (clusters never interact), and the longest-first rule
    is applied inside each (small) cluster.
    """
    ordered = sorted(items, key=lambda x: (x.start, x.end))
    result: list[Any] = []
    cluster: list[Any] = []
    cluster_end = -1

    def flush() -> None:
        if len(cluster) <= 1:
            result.extend(cluster)
            return
        kept: list[Any] = []
        for cand in sorted(cluster, key=lambda x: (-(x.end - x.start), x.start)):
            if all(cand.end <= k.start or cand.start >= k.end for k in kept):
                kept.append(cand)
        result.extend(sorted(kept, key=lambda x: x.start))

    for item in ordered:
        if cluster and item.start >= cluster_end:
            flush()
            cluster = []
            cluster_end = -1
        cluster.append(item)
        cluster_end = max(cluster_end, item.end)
    flush()
    return result


# --------------------------------------------------------------------------- #
# Sentence / evidence location (stage 6)
# --------------------------------------------------------------------------- #

_BULLET_PREFIX = re.compile(r"^\s*(?:[•●▪■◦·►▸➢✓✔*\-–—]|\d{1,2}[.)]|\(\d{1,2}\))\s+")
_SENT_END = re.compile(r"(?P<end>[.!?]+[\"')\]]*)\s+(?=[A-Z0-9\"'(\[])")
_DOTTED_ABBREV = re.compile(r"^(?:[a-z]\.)+[a-z]$")
_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "etc", "vs", "approx", "mr", "mrs", "ms", "dr", "inc", "ltd",
     "jr", "sr", "st", "no", "co", "corp", "fig", "eg", "ie", "cf"}
)
_CONTINUATION_WORDS = frozenset(
    {"and", "or", "with", "using", "in", "on", "for", "to", "of", "the", "a", "an",
     "by", "as", "via", "from", "into", "like", "including", "such", "&"}
)
_MAX_WRAP_LINES = 12


def _is_abbreviation(token: str, punct: str) -> bool:
    if not punct.startswith(".") or len(punct.rstrip("\"')]")) != 1:
        return False
    stripped = token.lstrip("(\"'[")
    lowered = stripped.lower()
    if lowered in _ABBREVIATIONS or _DOTTED_ABBREV.match(lowered):
        return True
    return len(stripped) == 1 and stripped.isalpha() and stripped.isupper()  # initials: "R. Sharma"


def _continues(prev: str, cur: str) -> bool:
    """Does physical line ``cur`` continue the sentence that ``prev`` started?"""
    last = prev[-1]
    if last in ".!?:":
        return False
    if last in ",;(&/-":
        return True
    tail = re.search(r"[A-Za-z']+$", prev)
    if tail and tail.group(0).lower() in _CONTINUATION_WORDS:
        return True
    return cur[0].islower()


def _line_at(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, len(text) if end == -1 else end


def _segment_bounds(text: str, pos: int, wrap_lines: bool) -> tuple[int, int]:
    """Bounds of the logical paragraph/bullet around ``pos`` (outer whitespace trimmed)."""
    first = last = _line_at(text, pos)
    if wrap_lines:
        for _ in range(_MAX_WRAP_LINES):                       # extend upwards
            cur = text[first[0]:first[1]]
            if first[0] == 0 or _BULLET_PREFIX.match(cur):
                break
            prev_end = first[0] - 1
            prev_start = text.rfind("\n", 0, prev_end) + 1
            prev, cur = text[prev_start:prev_end].strip(), cur.strip()
            if not prev or not cur or not _continues(prev, cur):
                break
            first = (prev_start, prev_end)
        for _ in range(_MAX_WRAP_LINES):                       # extend downwards
            if last[1] >= len(text):
                break
            nxt_start = last[1] + 1
            nxt_end = text.find("\n", nxt_start)
            nxt_end = len(text) if nxt_end == -1 else nxt_end
            nxt = text[nxt_start:nxt_end]
            if not nxt.strip() or _BULLET_PREFIX.match(nxt):
                break
            if not _continues(text[last[0]:last[1]].strip(), nxt.strip()):
                break
            last = (nxt_start, nxt_end)
    seg_start = first[0] + (len(text[first[0]:first[1]]) - len(text[first[0]:first[1]].lstrip()))
    seg_end = last[0] + len(text[last[0]:last[1]].rstrip())
    return seg_start, max(seg_start, seg_end)


def split_sentences(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Sentence spans inside ``text[start:end]`` (offsets into ``text``).

    Splits after . ! ? followed by whitespace and a capital/digit/quote, except
    after abbreviations ("e.g."), dotted abbreviations and single-letter
    initials. A leading bullet/number marker is excluded from the first span.
    """
    end = len(text) if end is None else end
    seg = text[start:end]
    marker = _BULLET_PREFIX.match(seg)
    last = marker.end() if marker else 0
    spans: list[tuple[int, int]] = []
    for m in _SENT_END.finditer(seg, last):
        head = seg[last:m.start("end")].rsplit(None, 1)
        token = head[-1] if head else ""
        if _is_abbreviation(token, m.group("end")):
            continue
        spans.append((start + last, start + m.end("end")))
        last = m.end()
    if last < len(seg):
        spans.append((start + last, start + len(seg)))
    return spans


class _SentenceLocator:
    """Per-call helper: sentence span containing a position, with segment caching."""

    def __init__(self, text: str, wrap_lines: bool) -> None:
        self.text = text
        self.wrap_lines = wrap_lines
        self._cache: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def span_at(self, pos: int) -> tuple[int, int]:
        seg = _segment_bounds(self.text, pos, self.wrap_lines)
        spans = self._cache.get(seg)
        if spans is None:
            spans = self._cache[seg] = split_sentences(self.text, *seg)
        for s, e in spans:
            if s <= pos < e:
                return s, e
        return seg


def sentence_span_at(text: str, pos: int, wrap_lines: bool = True) -> tuple[int, int]:
    """Offsets of the sentence containing ``pos`` (convenience for other modules)."""
    return _SentenceLocator(text, wrap_lines).span_at(pos)


# --------------------------------------------------------------------------- #
# Ambiguity handling (stage 5)
# --------------------------------------------------------------------------- #

_TRIGGERS = frozenset(
    "using use used uses with in via on and or of for from by built develop developed "
    "developing written write wrote code coded coding implemented programmed programming "
    "language languages skills skill stack tools tool technologies framework frameworks "
    "proficient experienced experience knowledge expertise familiar familiarity".split()
)
_FOLLOW_WORDS = frozenset(
    "language languages programming lang framework developer developers backend library "
    "skills skill expertise proficiency scripting script scripts tool tools engineer engineers "
    "programmer programmers certified".split()
)
_SEASON_AFTER = re.compile(
    r"^\s*(?:'?\d{2,4}\b|semester|term|quarter|break|season|intake|batch|session|cohort|internship)", re.I
)
_INITIAL_AFTER = re.compile(r"^\.\s+[A-Z][a-z]+")
_PREV_WORD = re.compile(r"([A-Za-z']+)\W*$")
_NEXT_WORD = re.compile(r"^\W*([A-Za-z']+)")


def _judge_ambiguous(
    raw: str, name: str, before: str, after: str, supporting: set[str]
) -> tuple[bool, str, bool]:
    """Decide whether an ambiguous term is really the technology.

    Returns ``(accepted, reason, hard_reject)``. A *hard* reject (season
    usage, "R&D", a person's initial) can never be rescued by a neighbouring
    technology; a soft reject ("no technology context") can.

    Order: hard rejects -> co-occurring unambiguous technology -> initials
    check -> trigger words around the term -> soft reject.
    """
    if name == "Spring" and _SEASON_AFTER.match(after):
        return False, "season/semester usage", True
    if len(raw) == 1 and after[:1] == "&":
        return False, "looks like an abbreviation (e.g. R&D)", True
    others = sorted(supporting - {name})
    if others:
        return True, f"ambiguous term accepted: co-occurs with {others[0]}", False
    if len(raw) == 1 and _INITIAL_AFTER.match(after):
        return False, "looks like a person's initial", True
    prev = _PREV_WORD.search(before)
    if prev and prev.group(1).lower() in _TRIGGERS:
        return True, f"ambiguous term accepted: preceded by '{prev.group(1)}'", False
    nxt = _NEXT_WORD.match(after)
    if nxt and nxt.group(1).lower() in _FOLLOW_WORDS:
        return True, f"ambiguous term accepted: followed by '{nxt.group(1)}'", False
    return False, "no technology context", False


# "SAP and Oracle", "Go / Rust": only list connectors between two terms.
_LIST_CONNECTOR = re.compile(r"^\s*(?:[,;/&|]|\band\b|\bor\b|,\s*and\b|,\s*or\b)\s*$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Candidate validation (stage 3)
# --------------------------------------------------------------------------- #

# URLs with a scheme or www., e-mail addresses, and bare domains such as
# "github.com/jane/app". ".net" is deliberately NOT treated as a TLD so that
# "ASP.NET" is never mistaken for a domain; ".js" is not a TLD either.
_URL_OR_EMAIL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|(?<![\w.@-])(?:[\w-]+\.)+(?:com|org|io|dev|app|co|in|me|ai|edu)\b(?:/\S*)?",
    re.IGNORECASE,
)
_URL_HINT = re.compile(r"://|@|www\.|\.(?:com|org|io|dev|app|co|in|me|ai|edu)\b", re.IGNORECASE)


def _protected_spans(text: str) -> tuple[list[int], list[int]]:
    if _URL_HINT.search(text) is None:
        return [], []
    spans = [m.span() for m in _URL_OR_EMAIL.finditer(text)]
    return [s for s, _ in spans], [e for _, e in spans]


def _span_is_plausible(text: str, start: int, end: int) -> bool:
    # "Moment.js", "main.java": the match is the tail of a dotted identifier.
    if start >= 2 and text[start - 1] == "." and (text[start - 2].isalnum() or text[start - 2] == "_"):
        return False
    # "C+" / "C#x": a one-letter name directly followed by + or # is not C.
    if end - start == 1 and end < len(text) and text[end] in "+#":
        return False
    return True


@dataclass(frozen=True, slots=True)
class _Cand:
    start: int
    end: int
    info: SurfaceInfo


# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #

class TechnologyMatcher:
    """Finds technologies in text and records the evidence for each.

    Build one instance and reuse it: the dictionaries are parsed once (cached
    per file/mtime) and the engine is compiled once per instance.

    Parameters
    ----------
    master_path, aliases_path:
        Dictionary files (default: ``app/dictionaries``).
    engine:
        ``"regex"`` (default, also ``"auto"``) or ``"flashtext"``.
    extra_technologies / extra_aliases / extra_ambiguous:
        Runtime additions, e.g. ``extra_technologies={"Tools": ["Docker Compose"]}``.
    ignore_urls:
        Skip matches inside URLs and e-mail addresses ("github.com/jane").
    wrap_lines:
        Re-join hard-wrapped lines when locating the evidence sentence.
    max_evidence_chars:
        Longer evidence is cropped around the match (with "..." markers).
    """

    def __init__(
        self,
        master_path: str | Path | None = None,
        aliases_path: str | Path | None = None,
        *,
        engine: str = "regex",
        extra_technologies: Mapping[str, Iterable[str]] | None = None,
        extra_aliases: Mapping[str, str] | None = None,
        extra_ambiguous: Iterable[str] | None = None,
        ignore_urls: bool = True,
        wrap_lines: bool = True,
        max_evidence_chars: int = 400,
    ) -> None:
        dictionary = load_dictionary(master_path, aliases_path)
        if extra_technologies or extra_aliases or extra_ambiguous:
            dictionary = dictionary.extended(extra_technologies, extra_aliases, extra_ambiguous)
        self.dictionary = dictionary
        self.ignore_urls = ignore_urls
        self.wrap_lines = wrap_lines
        self.max_evidence_chars = max(40, max_evidence_chars)
        self.engine_name = "regex" if engine == "auto" else engine
        self._engine = self._make_engine(self.engine_name)
        self._fallback: RegexTrieEngine | None = None

    # ------------------------------------------------------------------ #
    def _make_engine(self, name: str):
        surfaces = list(self.dictionary.surfaces)
        if name == "regex":
            return RegexTrieEngine(surfaces)
        if name == "flashtext":
            return FlashTextEngine(surfaces)
        raise ValueError(f"unknown engine {name!r}; use 'regex' or 'flashtext'")

    @property
    def technologies(self) -> dict[str, str]:
        """Canonical technology name -> category."""
        return dict(self.dictionary.categories)

    def category_of(self, name: str) -> str | None:
        return self.dictionary.categories.get(name)

    # ------------------------------------------------------------------ #
    def find_all(self, text: str) -> list[TechMatch]:
        """Every technology mention in ``text``, ordered by position.

        Raises TypeError for non-string input; empty text returns ``[]``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return []

        candidates = resolve_overlaps(self._candidates(text))
        if not candidates:
            return []

        locator = _SentenceLocator(text, self.wrap_lines)
        sentences = [locator.span_at(c.start) for c in candidates]

        support: dict[tuple[int, int], set[str]] = {}
        for c, sent in zip(candidates, sentences):
            if c.info.exact is None:
                support.setdefault(sent, set()).add(c.info.name)

        # Stage 5: ambiguous terms. Pass 1 judges each on its own context;
        # pass 2 lets a soft-rejected term ride on an adjacent accepted one.
        verdict: dict[int, tuple[bool, str]] = {}
        soft: list[int] = []
        for i, (c, sent) in enumerate(zip(candidates, sentences)):
            if c.info.exact is None or text[c.start:c.end] != c.info.exact:
                continue  # unambiguous, or wrong case (ambiguous terms are case-sensitive)
            ok, why, hard = _judge_ambiguous(
                text[c.start:c.end], c.info.name, text[sent[0]:c.start], text[c.end:sent[1]],
                support.get(sent, set()),
            )
            verdict[i] = (ok, why)
            if not ok and not hard:
                soft.append(i)
        changed = True
        while changed and soft:
            changed = False
            for i in list(soft):
                for j in (i - 1, i + 1):
                    if not (0 <= j < len(candidates)) or sentences[j] != sentences[i]:
                        continue
                    if j not in verdict or not verdict[j][0]:
                        continue
                    lo, hi = sorted((i, j))
                    if _LIST_CONNECTOR.match(text[candidates[lo].end:candidates[hi].start]):
                        verdict[i] = (True, f"ambiguous term accepted: adjacent to {candidates[j].info.name}")
                        soft.remove(i)
                        changed = True
                        break

        results: list[TechMatch] = []
        for i, (c, sent) in enumerate(zip(candidates, sentences)):
            raw = text[c.start:c.end]
            is_canonical = normalize_surface(raw) == normalize_surface(c.info.name)
            if c.info.exact is None:
                reason = "canonical name" if is_canonical else f"alias '{normalize_surface(raw)}'"
            else:
                if raw != c.info.exact:
                    continue
                accepted, reason = verdict[i]
                if not accepted:
                    logger.debug("rejected ambiguous %r at %d: %s", raw, c.start, reason)
                    continue
            ev_start, ev_end, evidence = self._evidence(text, sent, c.start, c.end)
            results.append(
                TechMatch(
                    name=c.info.name,
                    category=c.info.category,
                    matched_text=raw,
                    start=c.start,
                    end=c.end,
                    evidence=evidence,
                    evidence_start=ev_start,
                    evidence_end=ev_end,
                    match_type="canonical" if is_canonical else "alias",
                    reason=reason,
                    all_evidence=(evidence,),
                )
            )
        return results

    def extract(self, text: str, unique: bool = True) -> list[TechMatch]:
        """Technologies found in ``text``.

        With ``unique=True`` (default) each technology appears once, in order of
        first mention: the first occurrence supplies offsets/evidence,
        ``occurrences`` counts all mentions and ``all_evidence`` lists the
        distinct evidence sentences. With ``unique=False`` this is ``find_all``.
        """
        matches = self.find_all(text)
        if not unique:
            return matches
        counts = Counter(m.name for m in matches)
        evidences: dict[str, list[str]] = {}
        first: dict[str, TechMatch] = {}
        for m in matches:
            first.setdefault(m.name, m)
            bucket = evidences.setdefault(m.name, [])
            if m.evidence not in bucket:
                bucket.append(m.evidence)
        return [
            replace(m, occurrences=counts[name], all_evidence=tuple(evidences[name]))
            for name, m in first.items()
        ]

    def extract_names(self, text: str) -> list[str]:
        """Unique canonical names in order of first mention."""
        return [m.name for m in self.extract(text)]

    # ------------------------------------------------------------------ #
    def _candidates(self, text: str) -> list[_Cand]:
        spans = self._engine.find(text)
        if spans is None:  # engine could not guarantee offsets for this text
            if self._fallback is None:
                self._fallback = RegexTrieEngine(self.dictionary.surfaces)
            spans = self._fallback.find(text)

        starts, ends = _protected_spans(text) if self.ignore_urls else ([], [])
        surfaces = self.dictionary.surfaces
        out: list[_Cand] = []
        for s, e in spans:
            info = surfaces.get(normalize_surface(text[s:e]))
            if info is None:
                logger.debug("engine span %r is not in the dictionary", text[s:e])
                continue
            if not _span_is_plausible(text, s, e):
                continue
            if starts:
                i = bisect.bisect_right(starts, s) - 1
                # Inside a URL/e-mail => skip, unless the "domain" IS the technology
                # name ("Socket.io", ".net") rather than part of a longer address.
                if i >= 0 and s < ends[i] and not (s == starts[i] and e == ends[i]):
                    continue
            out.append(_Cand(s, e, info))
        return out

    def _evidence(self, text: str, sent: tuple[int, int], s: int, e: int) -> tuple[int, int, str]:
        ev_start, ev_end = sent
        cap = self.max_evidence_chars
        front = back = False
        if ev_end - ev_start > cap:
            half = cap // 2
            new_start = max(ev_start, s - half)
            new_end = min(ev_end, e + half)
            if new_start > ev_start:
                gap = next((i for i in range(new_start, s) if text[i].isspace()), None)
                new_start = gap + 1 if gap is not None else new_start
                front = True
            if new_end < ev_end:
                gap = next((i for i in range(new_end, e, -1) if text[i - 1].isspace()), None)
                new_end = gap if gap is not None else new_end
                back = True
            ev_start, ev_end = new_start, new_end
        body = " ".join(text[ev_start:ev_end].split())
        return ev_start, ev_end, ("..." if front else "") + body + ("..." if back else "")