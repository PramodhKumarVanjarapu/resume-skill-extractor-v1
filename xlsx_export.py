"""Formatted Excel (.xlsx) export of the output table, for Excel and Google Sheets.

Why this exists: a CSV file cannot carry formatting. When it is opened in Google Sheets or Excel,
columns have default widths (long technology lists look cut off), there is no styled header, and
spreadsheet programs "helpfully" convert phone numbers to numbers, which drops leading zeros
(``08810931580`` -> ``8810931580``). An .xlsx file stores all of that, and Google Sheets imports it
with the formatting intact (File > Import > Upload).

What the sheet contains, for the columns passed in:

* a styled, frozen header row with filter buttons (and the first column frozen, normally ``phone``)
* a sensible width per known column, wrapped text, top-aligned cells
* ``phone`` stored as TEXT (leading zeros and "+" survive; no ="..." trick needed)
* ``num_projects`` / ``repos_count`` stored as real numbers
* a single resume / GitHub URL in a cell becomes a clickable link
* every other value is written as a plain string, never as a formula: resume text that happens to
  start with ``=`` cannot execute

Needs the ``xlsxwriter`` package (imported lazily, so nothing else depends on it).
"""

from __future__ import annotations

import io
import re
from typing import Iterable, Sequence

import pandas as pd

__all__ = ["COLUMN_WIDTHS", "dataframe_to_xlsx"]

COLUMN_WIDTHS: dict[str, int] = {
    "phone": 16, "resume_link": 38, "num_projects": 13, "project_names": 42,
    "tech_skills": 45, "tech_projects": 45, "tech_experience": 40, "tech_internships": 32,
    "tech_certifications": 32, "tech_achievements": 32,
    "github_profile": 34, "repos_count": 13, "github_tech_stack": 50, "project_categories": 34,
    # diagnostic columns
    "status": 12, "error_stage": 14, "error": 40, "projects_needs_review": 14, "projects_summary": 60,
    "unrecognized_technologies": 36, "github_status": 20, "github_notes": 50, "warnings": 40,
    "raw_text": 60,
}
DEFAULT_WIDTH = 30
NUMERIC_COLUMNS = frozenset({"num_projects", "repos_count"})
URL_COLUMNS = frozenset({"resume_link", "github_profile"})
TEXT_COLUMNS = frozenset({"phone"})

MAX_CELL_CHARS = 32767        # Excel's limit per cell
MAX_URL_CHARS = 2000          # Excel's limit is 2079
MAX_URLS = 60000              # Excel allows 65,530 hyperlinks per sheet
_SINGLE_URL = re.compile(r"https?://\S+")


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def dataframe_to_xlsx(
    df: pd.DataFrame, columns: Sequence[str] | None = None, sheet_name: str = "output"
) -> bytes:
    """Return the bytes of a formatted .xlsx file holding ``df[columns]`` (all columns by default)."""
    import xlsxwriter  # lazy: only needed when this export is used

    cols: list[str] = list(columns) if columns is not None else list(df.columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not in the table: {missing}")

    buffer = io.BytesIO()
    workbook = xlsxwriter.Workbook(
        buffer,
        {
            "in_memory": True,
            "strings_to_numbers": False,   # nothing is converted behind our back
            "strings_to_formulas": False,  # "=1+1" stays text
            "strings_to_urls": False,
        },
    )
    sheet = workbook.add_worksheet(sheet_name[:31])

    header_fmt = workbook.add_format({
        "bold": True, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "border": 1,
        "valign": "vcenter", "text_wrap": True,
    })
    body_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    text_fmt = workbook.add_format({"num_format": "@", "valign": "top"})          # phone: always text
    number_fmt = workbook.add_format({"valign": "top", "align": "center"})
    link_fmt = workbook.add_format({"font_color": "#0563C1", "underline": 1, "valign": "top", "text_wrap": True})

    sheet.set_row(0, 24)
    for c, name in enumerate(cols):
        sheet.write_string(0, c, name, header_fmt)
        sheet.set_column(c, c, COLUMN_WIDTHS.get(name, DEFAULT_WIDTH))

    urls_written = 0
    for r, record in enumerate(df[cols].itertuples(index=False, name=None), start=1):
        for c, (name, raw) in enumerate(zip(cols, record)):
            text = _cell_text(raw)
            if text == "":
                sheet.write_blank(r, c, None, body_fmt)
            elif name in NUMERIC_COLUMNS and re.fullmatch(r"\d+", text):
                sheet.write_number(r, c, int(text), number_fmt)
            elif (name in URL_COLUMNS and _SINGLE_URL.fullmatch(text)
                  and len(text) <= MAX_URL_CHARS and urls_written < MAX_URLS):
                sheet.write_url(r, c, text, link_fmt, string=text)
                urls_written += 1
            else:
                sheet.write_string(r, c, text[:MAX_CELL_CHARS], text_fmt if name in TEXT_COLUMNS else body_fmt)

    last_row = max(len(df), 1)
    if cols:
        sheet.freeze_panes(1, 1)                                 # header row + first column stay visible
        sheet.autofilter(0, 0, last_row, len(cols) - 1)
        if "phone" in cols:                                      # no "number stored as text" warnings
            p = cols.index("phone")
            sheet.ignore_errors({"number_stored_as_text": xlsxwriter.utility.xl_range(1, p, last_row, p)})
    workbook.close()
    return buffer.getvalue()
