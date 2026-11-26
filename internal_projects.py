"""Projects assigned by the training company, which are not the candidate's own work.

The names live in ``dictionaries/internal_projects.txt`` (one per line; ``#`` starts a comment). Text in brackets
is an alternative name: ``Nxt Trendz ( ECommerce Clone - Amazon, Flipkart)`` matches "Nxt Trendz" and
"ECommerce Clone".

Matching is exact on normalised names, never on a word inside a longer title, so "AI Chatbot with RAG" is not the
internal "Chatbot". Normalisation ignores case, punctuation, camelCase / hyphens (repository names), plural ``s``,
filler words (app, application, page, website, project, ...) and a trailing description after a separator::

    "Nxt Trendz - E-commerce App | React"   -> matches "Nxt Trendz ( ECommerce Clone - Amazon, Flipkart)"
    "jobby-app", "JobbyApp"                 -> matches "Jobby App"

    find_internal(name)   the list entry a project / repository name matches, or None
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

__all__ = ["DEFAULT_PATH", "find_internal", "load_internal_projects"]

DEFAULT_PATH = Path(__file__).resolve().parent / "dictionaries" / "internal_projects.txt"
FILLER = {"app", "application", "page", "website", "site", "web", "webpage", "webapp", "project", "the", "my"}
SEPARATOR = re.compile(r"\s*[|:–—]\s*|\s+-\s*|\s*-\s+")
BRACKETS = re.compile(r"[(\[]([^)\]]*)[)\]]")


def _tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # NxtTrendz -> Nxt Trendz
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words]


def _keys(name: str) -> set[str]:
    """Normalised forms of one name: with and without filler words, spaced and compact."""
    variants = [BRACKETS.sub(" ", name)] + BRACKETS.findall(name)
    for variant in list(variants):
        head = SEPARATOR.split(variant.strip(), maxsplit=1)[0]
        if head and head != variant.strip():
            variants.append(head)
    keys: set[str] = set()
    for variant in variants:
        tokens = _tokens(variant)
        if not tokens:
            continue
        core = [t for t in tokens if t not in FILLER] or tokens
        for form in (tokens, core):
            keys.add(" ".join(form))
            keys.add("".join(form))
    return keys


@lru_cache(maxsize=4)
def load_internal_projects(path: str | Path = DEFAULT_PATH) -> dict[str, str]:
    """Normalised key -> list entry. A missing file means no internal projects."""
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    index: dict[str, str] = {}
    for line in lines:
        entry = line.split("#", 1)[0].strip()
        if entry:
            for key in _keys(entry):
                index.setdefault(key, entry)
    return index


def find_internal(name: str | None, path: str | Path = DEFAULT_PATH) -> str | None:
    """The internal project ``name`` refers to, or ``None`` for the candidate's own project."""
    if not name or not name.strip():
        return None
    index = load_internal_projects(path)
    for key in sorted(_keys(name), key=len, reverse=True):
        if key in index:
            return index[key]
    return None
