"""Filter engine for the profile table (pure pandas, no Streamlit).

A *filter* looks at one column; several filters are combined with ALL (AND) or ANY (OR).

Conditions
----------
contains / not_contains   keywords typed by the user; the match type decides how a keyword is found:
                            item       a whole entry of a ';'-separated list ("java" matches "Java" but NOT
                                       "JavaScript" and NOT "Java EE"; "react" does not match "React Native")
                            word       a whole word anywhere in the text
                            substring  the text anywhere ("java" also matches "JavaScript")
                          how several keywords combine:
                            contains      OR = has ANY of them, AND = has ALL of them
                            not_contains  AND = has NONE of them, OR = lacks AT LEAST ONE of them
in_list / not_in_list     the whole cell equals one of a list of values (typed, pasted, or loaded from
                          Excel / CSV / TXT); optional phone mode compares the last 10 digits
is_empty / not_empty      blank (or a placeholder such as NA, N/A, none, -) / not blank
num_ge / num_le / num_eq / num_between
                          numeric comparison for count columns (repos_count, num_projects)

Design rules (each one exists because the original filter got it wrong):

* Nothing is guessed from data types: every cell is handled as text, so ``08810931580`` keeps its zero,
  ``3`` is not ``3.0`` and ``NA`` is not silently turned into "missing".
* A filter that cannot be applied (no keywords, unknown column) is REPORTED, never silently dropped.
* ``not_contains`` negates each keyword first and then combines, which is what "exclude X and Y" means.
* Values are kept as a list from the moment they are read: a value such as "Doe, John" is never re-split.
* Lists read from Excel are read as text (no ``9876543210.0``) and the header row is optional.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import pandas as pd

__all__ = [
    "FilterResult", "FilterSpec", "ListLoad", "LOGICS", "MATCH_TYPES",
    "alias_resolver", "apply_filters", "cell_text", "column_mask", "describe", "norm",
    "parse_keywords", "parse_number", "parse_values", "phone_key", "problem", "read_table", "read_value_list", "split_items",
]

LOGICS = (
    "contains", "not_contains", "in_list", "not_in_list", "is_empty", "not_empty",
    "num_ge", "num_le", "num_eq", "num_between",
)
TEXT_LOGICS = frozenset({"contains", "not_contains"})
LIST_LOGICS = frozenset({"in_list", "not_in_list"})
NUMBER_LOGICS = frozenset({"num_ge", "num_le", "num_eq", "num_between"})
EMPTY_LOGICS = frozenset({"is_empty", "not_empty"})
MATCH_TYPES = ("item", "word", "substring")

# Cell values that mean "nothing here" for is_empty / not_empty.
NULL_TOKENS = frozenset({"", "na", "n/a", "nan", "none", "null", "-", "--"})

_ITEM_SPLIT = re.compile(r"[;|\n]+")
_COUNT_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")          # "MERN (3)" -> "MERN"
_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
_FLOAT_TEXT = re.compile(r"(\d+)\.0+")                  # "9876543210.0" -> "9876543210"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def cell_text(value: object) -> str:
    """Any cell as text: missing -> "", integral floats -> "3" (not "3.0"), everything else str()."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def norm(text: str) -> str:
    """Case-folded, Unicode-normalised, whitespace-collapsed text used for every comparison."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_items(cell: str) -> list[str]:
    """Entries of a list cell, normalised. Splits on ``;``, ``|`` and newlines; a cell with none of those but
    with ", " is treated as a comma list (other tools write lists that way). A trailing count such as the
    "(3)" in "MERN (3)" is dropped so categories can be matched by name."""
    parts = _ITEM_SPLIT.split(cell)
    if len(parts) == 1 and ", " in cell:
        parts = cell.split(",")
    return [norm(_COUNT_SUFFIX.sub("", p)) for p in parts if p.strip()]


def is_blank(text: str) -> bool:
    return norm(text) in NULL_TOKENS


def parse_number(cell: str) -> float | None:
    """The number in a cell, or None ("3" -> 3.0, "3.0" -> 3.0, "" / "n/a" / "3 projects" -> None)."""
    t = cell.strip()
    return float(t) if _NUMBER.fullmatch(t) else None


def phone_key(text: str) -> str:
    """Comparison key for phone numbers: the last 10 digits ("+91 98765 43210" == "09876543210" == "9876543210").
    Text with fewer than 7 digits is not a phone number and is compared as normal text."""
    digits = re.sub(r"\D", "", text)
    return "p:" + digits[-10:] if len(digits) >= 7 else "t:" + norm(text)


def parse_keywords(text: str) -> list[str]:
    """Keywords typed into a box: separated by commas, semicolons or new lines; trimmed; no empties or
    duplicates (case-insensitive, first spelling kept)."""
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\n]+", text or ""):
        part = part.strip()
        if part and norm(part) not in seen:
            seen.add(norm(part))
            out.append(part)
    return out


def parse_values(text: str) -> list[str]:
    """Values for in_list / not_in_list typed or pasted into a box.

    One value per line when the text has several lines (so a value such as "Doe, John" is kept whole);
    otherwise separated by commas or semicolons. Trimmed, no empties, no duplicates (case-insensitive).
    """
    raw = text or ""
    parts = raw.strip().splitlines() if "\n" in raw.strip() else re.split(r"[,;]+", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        part = part.strip()
        if part and norm(part) not in seen:
            seen.add(norm(part))
            out.append(part)
    return out


# --------------------------------------------------------------------------- #
# Filter specification
# --------------------------------------------------------------------------- #

@dataclass
class FilterSpec:
    column: str
    logic: str = "contains"
    values: list[str] = field(default_factory=list)   # keywords / list values (never re-split)
    match: str = "item"                               # item | word | substring (contains / not_contains)
    combine: str = "OR"                               # AND | OR across several keywords
    low: float | None = None                          # numeric conditions
    high: float | None = None
    phone_mode: bool = False                          # in_list / not_in_list: compare phone numbers


def problem(spec: FilterSpec, columns: Sequence[str]) -> str | None:
    """Why this filter cannot be applied, or None if it is complete."""
    if spec.column not in columns:
        return f"column '{spec.column}' is not in the table"
    if spec.logic not in LOGICS:
        return f"unknown condition '{spec.logic}'"
    if spec.logic in TEXT_LOGICS | LIST_LOGICS and not [v for v in spec.values if str(v).strip()]:
        return "no keywords / values entered"
    if spec.logic in TEXT_LOGICS and spec.match not in MATCH_TYPES:
        return f"unknown match type '{spec.match}'"
    if spec.logic in {"num_ge", "num_eq"} and spec.low is None:
        return "no number entered"
    if spec.logic == "num_le" and spec.high is None and spec.low is None:
        return "no number entered"
    if spec.logic == "num_between":
        if spec.low is None or spec.high is None:
            return "both numbers are needed"
        if spec.low > spec.high:
            return "the minimum is larger than the maximum"
    return None


def _fmt(n: float | None) -> str:
    return "" if n is None else (str(int(n)) if float(n).is_integer() else str(n))


_MATCH_WORDS = {"item": "whole item", "word": "whole word", "substring": "text anywhere"}


def describe(spec: FilterSpec) -> str:
    """One-line plain-language description, used in messages."""
    vals = ", ".join(spec.values[:6]) + (f" (+{len(spec.values) - 6} more)" if len(spec.values) > 6 else "")
    c = spec.column
    if spec.logic == "contains":
        word = "all of" if spec.combine == "AND" else "any of"
        return f"{c} contains {word}: {vals} [{_MATCH_WORDS.get(spec.match, spec.match)}]"
    if spec.logic == "not_contains":
        word = "none of" if spec.combine == "AND" else "not all of"
        return f"{c} contains {word}: {vals} [{_MATCH_WORDS.get(spec.match, spec.match)}]"
    if spec.logic == "in_list":
        return f"{c} is one of {len(spec.values)} value(s): {vals}"
    if spec.logic == "not_in_list":
        return f"{c} is none of {len(spec.values)} value(s): {vals}"
    if spec.logic == "is_empty":
        return f"{c} is empty"
    if spec.logic == "not_empty":
        return f"{c} is not empty"
    if spec.logic == "num_ge":
        return f"{c} >= {_fmt(spec.low)}"
    if spec.logic == "num_le":
        return f"{c} <= {_fmt(spec.high if spec.high is not None else spec.low)}"
    if spec.logic == "num_eq":
        return f"{c} = {_fmt(spec.low)}"
    return f"{c} between {_fmt(spec.low)} and {_fmt(spec.high)}"


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def _keyword_matcher(match: str, values: Iterable[str]) -> Callable[[str], list[bool]]:
    """text -> one bool per keyword."""
    kws = [norm(v) for v in values if str(v).strip()]

    if match == "item":
        def by_item(text: str) -> list[bool]:
            items = set(split_items(text))
            return [kw in items for kw in kws]
        return by_item

    if match == "word":
        pats = [re.compile(r"(?<![\w+#])" + re.escape(kw) + r"(?![\w+#])") for kw in kws]

        def by_word(text: str) -> list[bool]:
            normal = norm(text)
            return [bool(p.search(normal)) for p in pats]
        return by_word

    def by_substring(text: str) -> list[bool]:
        normal = norm(text)
        return [kw in normal for kw in kws]
    return by_substring


def column_mask(series: pd.Series, spec: FilterSpec) -> pd.Series:
    """Boolean mask: which rows of ``series`` pass ``spec`` (the spec must be complete: see ``problem``)."""
    texts = series.map(cell_text)
    logic = spec.logic

    if logic == "is_empty":
        return texts.map(is_blank).astype(bool)
    if logic == "not_empty":
        return (~texts.map(is_blank)).astype(bool)

    if logic in TEXT_LOGICS:
        matcher = _keyword_matcher(spec.match, spec.values)
        negate = logic == "not_contains"
        need_all = spec.combine == "AND"

        def one(t: str) -> bool:
            hits = matcher(t)
            if negate:                      # negate EACH keyword, then combine ("none of" / "not all of")
                hits = [not h for h in hits]
            return all(hits) if need_all else any(hits)

        return texts.map(one).astype(bool)

    if logic in LIST_LOGICS:
        key = phone_key if spec.phone_mode else norm
        wanted = {key(v) for v in spec.values if str(v).strip()}
        hit = texts.map(lambda t: key(t) in wanted and not is_blank(t)).astype(bool)
        return ~hit if logic == "not_in_list" else hit

    nums = texts.map(parse_number)           # float Series: a missing number becomes NaN, not None
    present = nums.notna()
    low, high = spec.low, spec.high

    def cmp(n: float | None) -> bool:
        if n is None or n != n:              # no number (None or NaN) never matches
            return False
        if logic == "num_ge":
            return n >= low
        if logic == "num_le":
            return n <= (high if high is not None else low)
        if logic == "num_eq":
            return n == low
        return low <= n <= high

    mask = nums.map(cmp).astype(bool)
    mask.attrs["rows_without_number"] = int((~present).sum())
    return mask


# --------------------------------------------------------------------------- #
# Applying several filters
# --------------------------------------------------------------------------- #

@dataclass
class FilterResult:
    frame: pd.DataFrame                                   # the matching rows (original index kept)
    mask: pd.Series
    total: int
    active: int                                           # filters actually applied
    per_filter: list[tuple[str, int]]                     # (description, rows matching that filter alone)
    ignored: list[str]                                    # filters NOT applied, with the reason
    notes: list[str] = field(default_factory=list)

    @property
    def matched(self) -> int:
        return int(self.mask.sum())


def apply_filters(df: pd.DataFrame, specs: Sequence[FilterSpec], combine: str = "AND") -> FilterResult:
    """Apply ``specs`` to ``df``. ``combine`` is "AND" (every filter must pass) or "OR" (any filter may pass).

    Filters that are incomplete are listed in ``ignored`` with the reason instead of being dropped silently.
    With no usable filter every row is returned.
    """
    columns = list(df.columns)
    masks: list[pd.Series] = []
    per_filter: list[tuple[str, int]] = []
    ignored: list[str] = []
    notes: list[str] = []

    for n, spec in enumerate(specs, start=1):
        why = problem(spec, columns)
        if why:
            ignored.append(f"Filter {n} ({spec.column or 'no column'}): {why}")
            continue
        column = df[spec.column]
        if isinstance(column, pd.DataFrame):               # duplicate column names: use the first
            column = column.iloc[:, 0]
        mask = column_mask(column, spec)
        masks.append(mask)
        per_filter.append((describe(spec), int(mask.sum())))
        missing = mask.attrs.get("rows_without_number")
        if missing:
            notes.append(f"Filter {n}: {missing} row(s) have no number in '{spec.column}' and can never match.")

    if not masks:
        mask = pd.Series(True, index=df.index)
    elif combine == "OR":
        mask = masks[0]
        for m in masks[1:]:
            mask = mask | m
    else:
        mask = masks[0]
        for m in masks[1:]:
            mask = mask & m
    return FilterResult(df[mask], mask, len(df), len(masks), per_filter, ignored, notes)


# --------------------------------------------------------------------------- #
# Reading tables and value lists (CSV / Excel / text)
# --------------------------------------------------------------------------- #

def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _is_index_column(name: str, column: pd.Series) -> bool:
    """The empty-header 0,1,2,... column that pandas / table download icons add."""
    if not (str(name).strip() == "" or str(name).startswith("Unnamed")):
        return False
    return column.tolist() == [str(i) for i in range(len(column))]


def read_table(data: bytes, filename: str) -> tuple[pd.DataFrame, list[str]]:
    """Read an uploaded CSV / XLSX as an all-text table. Returns (table, notes).

    Nothing is type-guessed: ``08810931580`` keeps its zero, counts stay ``3``, ``NA`` stays ``NA``.
    A leading row-number column (as written by pandas ``to_csv`` with its index) is dropped and mentioned.
    Raises ValueError with a readable message when the file cannot be read.
    """
    notes: list[str] = []
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(io.BytesIO(data), dtype=str, keep_default_na=False)
        else:
            df = pd.read_csv(io.StringIO(_decode(data)), dtype=str, keep_default_na=False)
    except ImportError as exc:
        raise ValueError("Reading .xlsx files needs openpyxl:  pip install openpyxl") from exc
    except Exception as exc:  # noqa: BLE001 - surface any parser problem as a message
        raise ValueError(f"Could not read '{filename}': {exc}") from exc
    df = df.astype(str)
    drop = [c for c in df.columns if _is_index_column(c, df[c])]
    if drop:
        df = df.drop(columns=drop)
        notes.append("Ignored the row-number column at the start of the file.")
    return df, notes


@dataclass
class ListLoad:
    values: list[str]
    columns: list[str]
    column: str | None
    error: str | None = None


def _clean_value(v: object) -> str:
    text = cell_text(v).strip()
    m = _FLOAT_TEXT.fullmatch(text)
    return m.group(1) if m else text


def read_value_list(data: bytes, filename: str, column: str | None = None, has_header: bool = True) -> ListLoad:
    """Values for in_list / not_in_list from an Excel, CSV or text file.

    ``column`` picks a column by name (first column when None / unknown). ``has_header=False`` means the first
    row is already a value (otherwise it would be silently lost). Everything is read as text, blanks and
    duplicates are removed, and ``9876543210.0`` is turned back into ``9876543210``.
    """
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(io.BytesIO(data), dtype=str, header=0 if has_header else None, keep_default_na=False)
        elif name.endswith(".txt"):
            lines = [ln for ln in _decode(data).splitlines()]
            df = pd.DataFrame({"Column 1": lines[1:] if has_header and lines else lines})
            if has_header and lines:
                df.columns = [lines[0].strip() or "Column 1"]
        else:
            df = pd.read_csv(io.StringIO(_decode(data)), dtype=str, header=0 if has_header else None, keep_default_na=False)
    except ImportError:
        return ListLoad([], [], None, "Reading .xlsx files needs openpyxl:  pip install openpyxl")
    except Exception as exc:  # noqa: BLE001
        return ListLoad([], [], None, f"Could not read '{filename}': {exc}")

    if not has_header:
        df.columns = [f"Column {i + 1}" for i in range(len(df.columns))]
    columns = [str(c) for c in df.columns]
    if not columns:
        return ListLoad([], [], None, "The file has no columns")
    chosen = column if column in columns else columns[0]
    series = df.iloc[:, columns.index(chosen)]
    values: list[str] = []
    seen: set[str] = set()
    for v in series:
        text = _clean_value(v)
        if text and norm(text) not in seen:
            seen.add(norm(text))
            values.append(text)
    return ListLoad(values, columns, chosen, None if values else "No values found in that column")


# --------------------------------------------------------------------------- #
# Technology aliases ("nodejs" -> "Node.js") for technology columns
# --------------------------------------------------------------------------- #

def alias_resolver(matcher) -> Callable[[str], str]:
    """Return ``resolve(keyword)`` that maps a technology alias to its canonical name using a TechnologyMatcher
    ("nodejs" / "Node JS" -> "Node.js", "sklearn" -> "Scikit-learn"). Keywords the dictionary does not recognise as
    a technology on their own are returned unchanged."""
    def resolve(keyword: str) -> str:
        text = keyword.strip()
        if not text:
            return keyword
        try:
            found = matcher.find_all(text)
        except Exception:  # noqa: BLE001 - aliases are a convenience, never a failure
            return keyword
        if len(found) == 1 and found[0].start == 0 and found[0].end == len(text):
            return found[0].name
        return keyword
    return resolve
