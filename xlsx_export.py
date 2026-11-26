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
    "technology_match_percentage": 22, "evidence_score": 18, "priority": 18,
    "matched_requirements": 48, "missing_requirements": 48, "matched_alternative_groups": 48,
    "github_matched_technologies": 44, "project_matched_technologies": 44,
    "experience_matched_technologies": 44, "optional_matches": 40, "ranking_reason": 60,
    "ranking_warnings": 52,
    "total_projects": 16, "matching_projects_count": 24, "matching_project_names": 48,
    "project_threshold_met": 22, "technology_threshold_met": 24, "shortlisting_eligible": 22,
    "screening_status": 20, "unverified_requirements": 48,
    "github_evidence_details": 60, "verified_project_repositories": 60,
    "project_repo_match_confidence": 36, "experience_evidence_details": 60,
    "internship_evidence_details": 60, "recent_repositories_count": 22,
    "active_repositories_count": 22, "older_repositories_count": 22,
    "inactive_repositories_count": 24, "unknown_repositories_count": 24,
    "most_recent_github_activity": 26,
    "repository_activity_details": 60, "evidence_review_flags": 50,
    "github_cached_repos": 22, "github_selected_repos": 22, "github_deferred_repos": 22,
    "github_empty_repos": 20, "github_analysis_failures": 24,
    "github_deferred_repository_names": 54,
}
DEFAULT_WIDTH = 30
NUMERIC_COLUMNS = frozenset({"num_projects", "repos_count", "technology_match_percentage",
                             "evidence_score", "matched_mandatory_groups", "total_mandatory_groups",
                             "total_projects", "matching_projects_count"})
NUMERIC_COLUMNS = NUMERIC_COLUMNS | {"recent_repositories_count", "active_repositories_count",
                                     "older_repositories_count", "inactive_repositories_count",
                                     "unknown_repositories_count", "github_cached_repos", "github_selected_repos",
                                     "github_deferred_repos", "github_empty_repos", "github_analysis_failures"}
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
    priority_formats = {
        "P1": workbook.add_format({"bg_color": "#D9EAD3", "valign": "top"}),
        "P2": workbook.add_format({"bg_color": "#FFF2CC", "valign": "top"}),
        "P3": workbook.add_format({"bg_color": "#FCE5CD", "valign": "top"}),
        "Below Threshold": workbook.add_format({"bg_color": "#E7E6E6", "valign": "top"}),
        "Pending Analysis": workbook.add_format({"bg_color": "#D9EAF7", "valign": "top"}),
    }

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
            elif name in NUMERIC_COLUMNS and re.fullmatch(r"\d+(?:\.\d+)?", text):
                sheet.write_number(r, c, float(text), number_fmt)
            elif name == "priority" and text in priority_formats:
                sheet.write_string(r, c, text, priority_formats[text])
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
