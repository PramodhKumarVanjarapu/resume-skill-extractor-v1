"""Streamlit harness for testing section_detector + technology_matcher + project_extractor on real resumes.

    streamlit run app.py

Input : phone number + resume link (pasted list or CSV upload)
Output: output.csv - one row per resume with exactly these columns:
        phone, resume_link, num_projects, project_names, tech_skills,
        tech_projects, tech_experience, tech_internships, tech_certifications,
        tech_achievements, github_profile, repos_count, github_tech_stack,
        project_categories.
        (tech_<section> = unique technologies found in that section; the four
        github_* / project_* columns come from the resume's GitHub profile and
        its repositories - see github_enricher.py; the optional sidebar box
        "Add diagnostic columns" appends status, error, review flags,
        per-project technologies, GitHub status/notes and warnings.)
        Results can be downloaded as CSV or formatted XLSX.

Technology matching runs on the whole extracted text; each match is then
attributed to a section by its character offset. Project extraction runs on
the detected Projects section only and matches technologies per project.

All the logic lives in plain functions above ``main()`` so it can be tested
(and reused) without Streamlit. Supported files: PDF, DOCX, TXT.
Not supported: scanned/image PDFs (no OCR), legacy .doc, HTML pages.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import time
import zipfile
import threading
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import candidate_ranker
import experience_extractor
import internal_projects
import project_repo_matcher

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

# Every module this file imports. A missing one gets a clear startup error.
# are missing and where they belong instead.
REQUIRED_MODULES = (
    "section_detector", "technology_matcher", "project_extractor", "github_profile", "github_enricher", "analyzer",
    "resume", "xlsx_export", "pdf_parse", "candidate_ranker", "experience_extractor", "project_repo_matcher",
)
_here = Path(__file__).resolve().parent
_missing = [f"{name}.py" for name in REQUIRED_MODULES
            if not (_here / "app" / f"{name}.py").exists() and not (_here / f"{name}.py").exists()]
if _missing:
    raise ImportError(
        "app.py cannot start: these files are missing: " + ", ".join(_missing)
        + f". Put them in the folder '{_here / 'app'}' (next to the other modules) and run again. "
          "The latest resume_extractor.zip contains all of them."
    )
import github_enricher  # noqa: E402
import pdf_parse  # noqa: E402
import xlsx_export  # noqa: E402
import section_detector  # noqa: E402
import project_extractor  # noqa: E402
import technology_matcher  # noqa: E402
from project_extractor import ProjectExtractionResult, ProjectExtractor  # noqa: E402
from section_detector import SectionDetector, SectionResult  # noqa: E402
from technology_matcher import (  # noqa: E402
    DictionaryError, EngineUnavailable, TechMatch, TechnologyMatcher,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MAX_BYTES = 15 * 1024 * 1024          # refuse downloads larger than this
MAX_PDF_PAGES = 20                    # resumes are short; cap pathological files
# PDFs are parsed in separate processes (parsing is CPU work; threads cannot run it in parallel).
PARSE_PROCESSES = max(1, min(8, (os.cpu_count() or 2) - 1))
REQUEST_TIMEOUT = (10, 30)            # (connect, read) seconds
USER_AGENT = "Mozilla/5.0 (compatible; ResumeSectionTester/1.0)"

SECTION_KEYS = [
    "summary", "skills", "projects", "experience",
    "internships", "education", "certifications", "achievements",
]
BASE_COLUMNS = [
    "phone", "resume_link", "status", "error_stage", "error", "file_type",
    "text_chars", "num_sections", "sections_found",
    "num_technologies", "technologies", "num_projects", "project_names", "projects_needs_review", "warnings",
]
# Technologies per section; "preamble" = text before the first heading,
# "other" = headings we recognise but do not export (Hobbies, Languages, ...).
TECH_SECTION_KEYS = SECTION_KEYS + ["preamble", "other"]
TECH_COLUMNS = [f"tech_{k}" for k in TECH_SECTION_KEYS]
# internal_projects: resume projects assigned by the training company, removed from every project/technology column
PROJECT_COLUMNS = ["projects_summary", "unrecognized_technologies", "inferred_technologies", "internal_projects"]
EVIDENCE_COLUMNS = [
    "github_evidence_details", "verified_project_repositories", "project_repo_match_confidence",
    "experience_evidence_details", "internship_evidence_details", "recent_repositories_count",
    "active_repositories_count", "older_repositories_count", "inactive_repositories_count",
    "unknown_repositories_count", "most_recent_github_activity", "repository_activity_details",
    "evidence_review_flags",
]
# github_profile, repos_count, github_tech_stack, project_categories, github_status, github_notes
GITHUB_COLUMNS = github_enricher.GITHUB_COLUMNS
# how sure we are that github_profile is the candidate's own account, and every candidate with its evidence
PROFILE_SEARCH_COLUMNS = ["github_confidence", "github_candidates"]
TAIL_COLUMNS = [
    "preamble", "sections_metadata", "technology_matches", "projects_json",
    "download_s", "parse_s", "detect_ms", "tech_ms", "project_ms",
]
ALL_COLUMNS = (BASE_COLUMNS + SECTION_KEYS + TECH_COLUMNS + PROJECT_COLUMNS + GITHUB_COLUMNS
               + PROFILE_SEARCH_COLUMNS + TAIL_COLUMNS + EVIDENCE_COLUMNS)

# The columns of the downloadable output.csv, in this order.
OUTPUT_COLUMNS = [
    "phone", "resume_link", "num_projects", "project_names",
    "tech_skills", "tech_projects", "tech_experience", "tech_internships",
    "tech_certifications", "tech_achievements",
    "github_profile", "repos_count", "github_tech_stack", "project_categories",
]
# Optional extras (sidebar checkbox) that explain failures and uncertainty.
DIAGNOSTIC_COLUMNS = [
    "status", "error_stage", "error", "projects_needs_review", "projects_summary", "internal_projects",
    "unrecognized_technologies", "inferred_technologies", "github_confidence", "github_candidates", "github_status",
    "github_notes", "warnings",
]


class ResumeError(Exception):
    """A per-resume failure; ``stage`` says where it happened."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


# --------------------------------------------------------------------------- #
# Input parsing
# --------------------------------------------------------------------------- #

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)


def _digit_count(cell: str) -> int:
    return sum(ch.isdigit() for ch in cell)


def _looks_like_phone(cell: str) -> bool:
    """Mostly digits and at least 7 of them ("98765 43210", "+91-98765-43210", even "0l8897228905")."""
    compact = cell.replace(" ", "")
    return bool(compact) and _digit_count(compact) >= 7 and _digit_count(compact) / len(compact) >= 0.6


def parse_pasted(text: str) -> list[tuple[str, str]]:
    """Parse lines like ``+91 98765 43210, https://...`` into (phone, link).

    The URL is located by pattern, so separators (comma/tab/pipe/semicolon) don't matter. The
    remaining cells are searched for the phone number, so extra columns (a name, "NA", ...) are
    ignored instead of being glued onto the phone. If no cell looks like a phone number the first
    remaining cell is used. A line with no URL keeps an empty link, so the row still appears in the
    output (as a failure) instead of vanishing.
    """
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _URL_RE.search(line)
        link = m.group(0).rstrip(",;)\"'") if m else ""
        rest = (line[: m.start()] + "\t" + line[m.end():]) if m else line
        cells = [c.strip() for c in re.split(r"[\t,;|]+", rest) if c.strip()]
        phone = next((c for c in cells if _looks_like_phone(c)), cells[0] if cells else "")
        entries.append((phone, link))
    return entries


def guess_column(columns: Iterable[str], keywords: Iterable[str], exclude: str | None = None) -> int:
    cols = list(columns)
    for kw in keywords:
        for i, c in enumerate(cols):
            if c != exclude and kw in c.lower():
                return i
    return 0


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

_DRIVE_ID = re.compile(r"/file/d/([\w-]+)")
_DOCS_ID = re.compile(r"docs\.google\.com/document/d/([\w-]+)")


def normalize_url(url: str) -> str:
    """Turn common share links into direct-download links.

    Google Drive file links, Google Docs links and Dropbox ``?dl=0`` links are
    rewritten; everything else is returned unchanged (with a scheme added to
    bare ``www.`` links). Raises ResumeError for empty or non-http(s) links.
    """
    url = (url or "").strip()
    if not url:
        raise ResumeError("input", "Missing resume link")
    if url.lower().startswith("www."):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ResumeError("input", f"Unsupported or malformed link: {url[:80]}")

    host = parsed.netloc.lower()
    if host.endswith("drive.google.com"):
        m = _DRIVE_ID.search(parsed.path)
        file_id = m.group(1) if m else parse_qs(parsed.query).get("id", [None])[0]
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    if host.endswith("docs.google.com"):
        m = _DOCS_ID.search(url)
        if m:
            return f"https://docs.google.com/document/d/{m.group(1)}/export?format=pdf"
    if host.endswith("dropbox.com"):
        q = parse_qs(parsed.query)
        q["dl"] = ["1"]
        flat = "&".join(f"{k}={v[0]}" for k, v in q.items())
        return parsed._replace(query=flat).geturl()
    return url


def download(url: str) -> bytes:
    """Fetch ``url`` with timeouts and a hard size cap."""
    try:
        with requests.get(
            url, stream=True, timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": USER_AGENT}, allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                raise ResumeError("download", f"File too large ({int(declared):,} bytes)")
            buf = bytearray()
            for chunk in resp.iter_content(65536):
                buf += chunk
                if len(buf) > MAX_BYTES:
                    raise ResumeError("download", f"File exceeds {MAX_BYTES:,} bytes")
            return bytes(buf)
    except ResumeError:
        raise
    except requests.exceptions.Timeout:
        raise ResumeError("download", "Request timed out") from None
    except requests.exceptions.HTTPError as e:
        raise ResumeError("download", f"HTTP {e.response.status_code}") from None
    except requests.exceptions.RequestException as e:
        raise ResumeError("download", f"{type(e).__name__}: {str(e)[:150]}") from None


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #

def detect_kind(data: bytes) -> str:
    """Identify the file from its bytes (URLs rarely have reliable extensions)."""
    if not data:
        return "empty"
    head = data[:2048]
    if b"%PDF-" in head:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                return "docx" if "word/document.xml" in z.namelist() else "zip"
        except zipfile.BadZipFile:
            return "unknown"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "doc"
    if head.lstrip().startswith(b"{\\rtf"):
        return "rtf"
    if b"<html" in head.lower() or head.lstrip().lower().startswith(b"<!doctype html"):
        return "html"
    if b"\x00" not in head:
        return "txt"
    return "unknown"


_parse_pool: ProcessPoolExecutor | None = None
_parse_pool_lock = threading.Lock()
_parse_pool_broken = False


def _get_parse_pool() -> ProcessPoolExecutor | None:
    """One pool for the whole Streamlit server (module globals survive reruns); None = parse in-thread."""
    global _parse_pool
    if _parse_pool_broken:
        return None
    with _parse_pool_lock:
        if _parse_pool is None:
            _parse_pool = ProcessPoolExecutor(max_workers=PARSE_PROCESSES)
        return _parse_pool


def _fast_pdf(data: bytes) -> tuple[str, list[str]] | None:
    """Text + hyperlinks in one PyMuPDF pass, in a worker process when possible.

    None when PyMuPDF is not installed (the caller uses pdfplumber / pypdf instead).
    Raises ResumeError("parse") when the PDF cannot be read.
    """
    global _parse_pool_broken
    if not pdf_parse.available():
        return None
    pool = _get_parse_pool()
    if pool is not None:
        try:
            return pool.submit(pdf_parse.pdf_text_and_links, data, MAX_PDF_PAGES).result()
        except ValueError as e:
            raise ResumeError("parse", f"PDF could not be read: {e}") from None
        except (BrokenProcessPool, OSError, RuntimeError):
            _parse_pool_broken = True          # worker processes cannot start here: parse in-thread from now on
    try:
        return pdf_parse.pdf_text_and_links(data, MAX_PDF_PAGES)
    except ValueError as e:
        raise ResumeError("parse", f"PDF could not be read: {e}") from None


def _pdf_text(data: bytes) -> str:
    text = ""
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages[:MAX_PDF_PAGES])
    except Exception:  # fall through to pypdf
        text = ""
    if text.strip():
        return text
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        return "\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES])
    except Exception as e:
        raise ResumeError("parse", f"PDF could not be read: {type(e).__name__}: {str(e)[:120]}") from None


def _docx_text(data: bytes) -> str:
    """Body text in document order, including table cells (common in resumes)."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:
        raise ResumeError("parse", f"DOCX could not be read: {type(e).__name__}: {str(e)[:120]}") from None

    lines: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            lines.append(Paragraph(child, doc).text)
        elif tag == "tbl":
            for row in Table(child, doc).rows:
                seen: set[int] = set()
                for cell in row.cells:
                    if id(cell._tc) in seen:        # merged cells repeat
                        continue
                    seen.add(id(cell._tc))
                    lines.extend(p.text for p in cell.paragraphs)
    return "\n".join(lines)


def _plain_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


_UNSUPPORTED = {
    "html": "Link returned a web page, not a file (is the link public / a direct link?)",
    "doc": "Legacy .doc is not supported (save as .docx or PDF)",
    "rtf": "RTF is not supported",
    "zip": "ZIP archive is not a resume file",
    "empty": "Downloaded file is empty",
    "unknown": "Unrecognised file type",
}


def extract_text(data: bytes, kind: str) -> str:
    if kind in _UNSUPPORTED:
        raise ResumeError("format", _UNSUPPORTED[kind])
    text = {"pdf": _pdf_text, "docx": _docx_text, "txt": _plain_text}[kind](data)
    text = text.replace("\x00", "")
    if not text.strip():
        raise ResumeError(
            "empty_text",
            "No extractable text (scanned/image-only file? OCR is not supported)",
        )
    return text


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class Processed:
    """Outcome for one resume: CSV row, extracted text, detector + matcher results."""

    # Plain class (not @dataclass) so this entry script has no dependence on
    # how Streamlit registers the script module.
    def __init__(
        self,
        row: dict,
        text: str = "",
        result: SectionResult | None = None,
        match_rows: list[dict] | None = None,
        project_result: ProjectExtractionResult | None = None,
        profiles: list[str] | None = None,
        profile_search: github_enricher.ProfileSearch | None = None,
    ) -> None:
        self.row = row
        self.text = text
        self.result = result
        self.match_rows = match_rows or []
        self.project_result = project_result
        self.profiles = profiles or []                       # candidate usernames, the candidate's own account first
        self.profile_search = profile_search                 # scores and reasons behind that order
        self.github: github_enricher.GitHubInfo | None = None   # filled by run_github()


def section_for_offset(result: SectionResult, pos: int) -> str:
    """Which section does character offset ``pos`` of the resume text fall in?"""
    for s in result.metadata:
        if s.start <= pos < s.end:
            return s.key if s.target else "other"
    if result.metadata and pos < result.metadata[0].heading_start:
        return "preamble"
    return "other"


def match_technologies(
    text: str, result: SectionResult, matcher: TechnologyMatcher
) -> tuple[list[TechMatch], list[dict]]:
    """All technology mentions in ``text`` plus flat, JSON-able rows (with section).

    Includes skills implied by a written technology (``match_type="inferred"``,
    e.g. "Deep Learning" from "PyTorch") so skill columns and filters see them.
    """
    return technology_rows(matcher.find_all(text, include_inferred=True), result)


def technology_rows(matches: list[TechMatch], result: SectionResult) -> tuple[list[TechMatch], list[dict]]:
    """``(matches, JSON-able rows with their section)`` for already found matches."""
    rows = []
    for m in matches:
        d = m.to_dict()
        d.pop("occurrences", None)
        d.pop("all_evidence", None)
        d["section"] = section_for_offset(result, m.start)
        rows.append(d)
    return matches, rows


def _unique(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))


def parse_extra_technologies(text: str) -> dict[str, list[str]]:
    """Parse the sidebar box: ``Category: Name1, Name2`` or just ``Name`` (-> "Other")."""
    extra: dict[str, list[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, sep, names = line.partition(":")
        if not sep:
            category, names = "Other", line
        for name in names.split(","):
            if name.strip():
                extra.setdefault(category.strip() or "Other", []).append(name.strip())
    return extra


def fill_technology_columns(row: dict, matches: list[TechMatch], match_rows: list[dict]) -> None:
    """technologies, num_technologies, inferred_technologies, tech_<section> and technology_matches."""
    names = _unique(m.name for m in matches)
    row["technologies"] = "; ".join(names)
    row["num_technologies"] = len(names)
    sources: dict[str, list[str]] = {}
    for m in matches:
        if m.match_type == "inferred" and m.inferred_from not in sources.setdefault(m.name, []):
            sources[m.name].append(m.inferred_from)
    row["inferred_technologies"] = "; ".join(
        f"{name} (from {', '.join(froms)})" for name, froms in sources.items()
    )
    for key in TECH_SECTION_KEYS:
        row[f"tech_{key}"] = "; ".join(
            _unique(r["name"] for r in match_rows if r["section"] == key)
        )
    row["technology_matches"] = json.dumps(match_rows, ensure_ascii=False)


def drop_internal_projects(
    project_result: ProjectExtractionResult, text: str, section_text: str
) -> tuple[ProjectExtractionResult, list[str], list[tuple[int, int]]]:
    """Remove projects assigned by the training company (``internal_projects.find_internal``).

    Returns the remaining projects, the removed project names and the removed blocks' offsets in ``text``
    (so technologies written only inside them can be dropped too).
    """
    kept, removed, spans = [], [], []
    for project in project_result.projects:
        if internal_projects.find_internal(project.name):
            removed.append(project.name)
            block = section_text[project.start:project.end]
            at = text.find(block) if block.strip() else -1
            if at >= 0:
                spans.append((at, at + len(block)))
        else:
            kept.append(project)
    if not removed:
        return project_result, [], []
    return ProjectExtractionResult(kept, project_result.warnings, project_result.preamble), removed, spans


def summarize_projects(pr: ProjectExtractionResult) -> str:
    """One readable cell: 'Name: Tech1, Tech2 [REVIEW: reason] || Name2: ...'."""
    parts = []
    for p in pr.projects:
        label = p.name or "(untitled)"
        techs = ", ".join(p.technologies) or "-"
        flag = f" [REVIEW: {', '.join(p.review_reasons)}]" if p.needs_review else ""
        parts.append(f"{label}: {techs}{flag}")
    return " || ".join(parts)


def process_resume(
    phone: str,
    link: str,
    detector: SectionDetector,
    matcher: TechnologyMatcher | None = None,
    extractor: ProjectExtractor | None = None,
    blocked_accounts: tuple[str, ...] = (),
) -> Processed:
    """download -> parse -> detect sections -> match technologies -> extract projects for ONE resume.

    Never raises. If sections were detected but a later stage failed, the row
    keeps what it has and gets status ``partial`` with error_stage ``match`` or
    ``projects``.
    """
    row: dict = {c: "" for c in ALL_COLUMNS}
    row.update(phone=phone, resume_link=link, status="error", text_chars=0, num_sections=0)
    text, result, match_rows, project_result = "", None, [], None
    profiles: list[str] = []
    profile_search: github_enricher.ProfileSearch | None = None
    early_warnings: list[str] = []
    try:
        url = normalize_url(link)

        t0 = time.perf_counter()
        data = download(url)
        row["download_s"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        kind = detect_kind(data)
        row["file_type"] = kind
        fast = _fast_pdf(data) if kind == "pdf" else None
        if fast is not None:
            text = fast[0].replace("\x00", "")
            if not text.strip():
                raise ResumeError(
                    "empty_text",
                    "No extractable text (scanned/image-only file? OCR is not supported)",
                )
        else:
            text = extract_text(data, kind)
        row["parse_s"] = round(time.perf_counter() - t0, 3)
        row["text_chars"] = len(text)

        # GitHub profile: every GitHub account mentioned (visible URLs plus links hidden behind text such as
        # "GitHub": PDF annotations / DOCX relationships) is scored as evidence and ranked; github_profile is the
        # best one. A failure here must not fail the resume.
        try:
            # PDFs read by PyMuPDF already have their hyperlinks: no second parse.
            links = fast[1] if fast is not None else github_enricher.extract_links(data, url)
            profile_search = github_enricher.find_profile_candidates(
                text, links, github_enricher.guess_name(text, link), extra_blocked=blocked_accounts
            )
            profiles = profile_search.usernames
            row["github_confidence"] = profile_search.confidence
            row["github_candidates"] = profile_search.describe()
        except Exception:  # noqa: BLE001
            profiles = []
            early_warnings.append("github:profile_detection_failed")
        row["github_profile"] = github_enricher.profile_url(profiles[0]) if profiles else ""

        t0 = time.perf_counter()
        result = detector.detect(text)
        row["detect_ms"] = round((time.perf_counter() - t0) * 1000, 3)

        for key in SECTION_KEYS:
            row[key] = result.get(key, "")
        row["num_sections"] = len(result)
        row["sections_found"] = ", ".join(result)
        row["warnings"] = "; ".join([*result.warnings, *early_warnings])
        row["preamble"] = result.preamble
        row["sections_metadata"] = json.dumps(
            [
                {k: v for k, v in s.__dict__.items() if k != "text"}
                for s in result.metadata
            ],
            ensure_ascii=False,
        )
        row["status"] = "ok"

        if matcher is not None:
            try:
                t0 = time.perf_counter()
                matches, match_rows = match_technologies(text, result, matcher)
                row["tech_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                fill_technology_columns(row, matches, match_rows)
            except Exception as e:
                row["status"], row["error_stage"] = "partial", "match"
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
                match_rows = []

        if extractor is not None:
            try:
                t0 = time.perf_counter()
                project_result = extractor.extract(result.get("projects", ""))
                # Training-company projects are not the candidate's own work: drop them and the technologies
                # written only inside them (inference is redone so implied skills stay consistent).
                project_result, internal, spans = drop_internal_projects(
                    project_result, text, result.get("projects", ""))
                row["internal_projects"] = "; ".join(internal)
                if spans and matcher is not None and match_rows:
                    outside = [m for m in matches if m.match_type != "inferred"
                               and not any(a <= m.start < b for a, b in spans)]
                    matches, match_rows = technology_rows(sorted(
                        outside + matcher.infer(outside), key=lambda m: (m.start, m.match_type == "inferred")), result)
                    fill_technology_columns(row, matches, match_rows)
                row["project_ms"] = round((time.perf_counter() - t0) * 1000, 3)
                row["num_projects"] = len(project_result.projects)
                row["project_names"] = "; ".join(p.name or "(untitled)" for p in project_result.projects)
                row["projects_needs_review"] = project_result.needs_review
                row["projects_summary"] = summarize_projects(project_result)
                row["unrecognized_technologies"] = "; ".join(
                    dict.fromkeys(u for p in project_result.projects for u in p.unrecognized_technologies)
                )
                row["projects_json"] = json.dumps(project_result, ensure_ascii=False)
                extra = [f"projects:{w}" for w in project_result.warnings]
                row["warnings"] = "; ".join(filter(None, [row["warnings"], *extra]))
            except Exception as e:
                row["status"], row["error_stage"] = "partial", "projects"
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
                project_result = None
        if matcher is not None:
            for section_key, column in (("experience", "experience_evidence_details"),
                                        ("internships", "internship_evidence_details")):
                try:
                    detail = experience_extractor.extract_experiences(result.get(section_key, ""), matcher, section_key)
                    row[column] = json.dumps(detail, ensure_ascii=False)
                    if detail["warnings"]:
                        row["evidence_review_flags"] = "; ".join(filter(None, [row["evidence_review_flags"],
                                                      *detail["warnings"]]))
                except Exception as exc:
                    row["evidence_review_flags"] = "; ".join(filter(None, [row["evidence_review_flags"],
                                                  f"{section_key} attribution unavailable: {exc}"]))
    except ResumeError as e:
        row["error_stage"], row["error"] = e.stage, str(e)
    except Exception as e:  # defensive: one bad file must not kill the batch
        row["error_stage"], row["error"] = "unexpected", f"{type(e).__name__}: {str(e)[:150]}"
    return Processed(
        row=row, text=text, result=result, match_rows=match_rows, project_result=project_result, profiles=profiles,
        profile_search=profile_search,
    )


def run_batch(
    entries: list[tuple[str, str]],
    detector: SectionDetector,
    workers: int = 16,
    progress_cb: Callable[[int, int], None] | None = None,
    matcher: TechnologyMatcher | None = None,
    extractor: ProjectExtractor | None = None,
    blocked_accounts: tuple[str, ...] = (),
) -> list[Processed]:
    """Process resumes concurrently (downloads are I/O bound); order preserved."""
    results: list[Processed | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(process_resume, phone, link, detector, matcher, extractor, blocked_accounts): i
            for i, (phone, link) in enumerate(entries)
        }
        for done, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if progress_cb:
                progress_cb(done, len(entries))
    return results  # type: ignore[return-value]


def run_github(
    items: list[Processed],
    token: github_enricher.GitHubAuthorization = None,
    skip_forks: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
    max_profiles: int | None = None,
    rate: github_enricher.RateLimit | None = None,
    all_profiles: bool = False,
    max_repos: int | None = github_enricher.MAX_REPOS_PER_ACCOUNT,
    required_groups: tuple[tuple[str, ...], ...] = (),
    initial_repo_budget: int | None = 15,
    exhaustive: bool = False,
    screening_threshold: float = 60,
    min_matching_projects: int = 2,
    *,
    scan_method: github_enricher.RepositoryScanMethod = "clone",
) -> list[Processed]:
    """Fill repos_count, github_tech_stack, project_categories, github_status and github_notes.

    By default only the best candidate account of each resume is analysed (the others are named in github_notes);
    ``all_profiles=True`` merges the repositories of every account found. At most ``max_repos`` repositories are
    analysed per account (the most recently updated), and GitHub organisations are refused after one request.

    Only resumes that were read (status ok / partial) take part. The GitHub profile column is
    filled earlier, so it is present even when this step is skipped. ``max_profiles`` / ``rate``
    come from ``github_enricher.rate_limit_status``: profiles that do not fit in the remaining API
    quota are skipped with a clear note instead of failing halfway. Never raises.
    """
    targets = [p for p in items if p.row["status"] in ("ok", "partial")]
    needs_more_evidence = None
    if required_groups and not exhaustive:
        try:
            rank_matcher = TechnologyMatcher()
        except (DictionaryError, EngineUnavailable):
            rank_matcher = None
    else:
        rank_matcher = None
    if rank_matcher is not None:
        rank_requirements = tuple(candidate_ranker.Requirement(tuple(group)) for group in required_groups)
        rows_by_profile: dict[str, list[dict]] = {}
        for item in targets:
            for profile in item.profiles:
                rows_by_profile.setdefault(profile.casefold(), []).append(item.row)

        def needs_more_evidence(person: dict, analysed_repos: list[dict]) -> bool:
            stack = "; ".join(dict.fromkeys(
                technology for repo in analysed_repos if not repo.get("error")
                for technology in repo.get("technologies", [])
            ))
            linked_rows = [row for profile in person["profiles"]
                           for row in rows_by_profile.get(profile.casefold(), [])]
            for row in linked_rows:
                scored = candidate_ranker.score_candidate(
                    {**row, "github_tech_stack": stack, "github_status": "done"},
                    rank_requirements, rank_matcher, threshold=screening_threshold,
                    min_matching_projects=min_matching_projects,
                )
                if scored["project_threshold_met"] is not True:
                    continue  # Repository scans cannot add resume projects.
                has_experience = bool(scored["experience_matched_technologies"])
                best_priority = "P1" if has_experience else "P2"
                if not scored["technology_threshold_met"] or scored["potential_priority"] != best_priority:
                    return True
            return False

    infos = github_enricher.analyze_profiles(
        [p.profiles for p in targets], token, skip_forks, progress_cb, max_profiles=max_profiles, rate=rate,
        all_profiles=all_profiles, max_repos=max_repos, required_groups=required_groups,
        initial_repo_budget=initial_repo_budget, exhaustive=exhaustive,
        needs_more_evidence=needs_more_evidence, scan_method=scan_method,
    )
    for p, info in zip(targets, infos):
        p.github = info
        p.row.update(info.columns())
        activity = github_enricher.activity_summary(info.repositories)
        p.row["recent_repositories_count"] = activity["recent_repositories_count"]
        p.row["active_repositories_count"] = activity["active_repositories_count"]
        p.row["older_repositories_count"] = activity["older_repositories_count"]
        p.row["inactive_repositories_count"] = activity["inactive_repositories_count"]
        p.row["unknown_repositories_count"] = activity["unknown_repositories_count"]
        p.row["most_recent_github_activity"] = activity["most_recent_github_activity"]
        p.row["repository_activity_details"] = json.dumps(activity["repositories"], ensure_ascii=False)
        evidence = []
        for repo in info.repositories:
            if repo.get("excluded"):
                continue  # practice, tutorial, duplicate or code-less repositories are not evidence
            recorded = repo.get("technology_evidence")
            if recorded is None:
                recorded = [{"technology": tech, "type": "legacy_detection", "path": "",
                             "strength": "unspecified"} for tech in repo.get("technologies", [])]
            for item in recorded:
                evidence.append({**item, "repository": repo.get("repository", ""), "repository_url": repo.get("url", ""),
                                 "cache_refreshed_at": repo.get("cache_refreshed_at")})
        p.row["github_evidence_details"] = json.dumps(evidence, ensure_ascii=False)
        if p.project_result is not None:
            matches = project_repo_matcher.match_projects(p.project_result["projects"], info.repositories)
            p.row["verified_project_repositories"] = json.dumps(matches, ensure_ascii=False)
            p.row["project_repo_match_confidence"] = "; ".join(
                f"{m['project_name']}: {m['confidence']}" for m in matches)
            if any(m["requires_review"] for m in matches):
                p.row["evidence_review_flags"] = "; ".join(filter(None, [p.row["evidence_review_flags"],
                                              "Ambiguous project-repository match"]))
    return items


def profile_confidence_message(items: list[Processed]) -> tuple[str, str] | None:
    """How sure the profile detection is across the batch, as (level, text); None if nothing was found.

    Low confidence means the chosen account rests on weak evidence (for example a single repository link) or
    on a tie between accounts; see the github_candidates diagnostic column for the reasons."""
    counts = {"high": 0, "medium": 0, "low": 0}
    for p in items:
        level = p.row.get("github_confidence")
        if level in counts:
            counts[level] += 1
    found = sum(counts.values())
    if not found:
        return None
    text = (f"GitHub profile found in {found} resume(s): {counts['high']} high, {counts['medium']} medium, "
            f"{counts['low']} low confidence.")
    if counts["low"] or counts["medium"]:
        text += " Tick 'Add diagnostic columns' to see github_candidates (every account found, with the evidence)."
    return ("warning" if counts["low"] else "caption", text)


def github_limits_message(items: list[Processed], cap: int | None) -> tuple[str, str] | None:
    """Tell the user how many accounts hit the per-account repository cap (so the stack is from the newest repos only)."""
    capped = [p for p in items if p.github is not None and any("most recently updated were listed" in n for n in p.github.notes)]
    if not capped or cap is None:
        return None
    return ("caption", f"GitHub: {len(capped)} account(s) have more than {cap} repositories, so only the {cap} most recently "
                       "updated were listed for prioritization (raise 'Maximum repositories listed per account' "
                       "in the sidebar to change this).")


def parse_blocked_accounts(text: str) -> tuple[str, ...]:
    """GitHub account names to ignore, typed one per line or separated by commas ("github.com/name" is accepted too)."""
    names = []
    for part in re.split(r"[,\n;]+", text or ""):
        name = part.strip().rstrip("/").rsplit("/", 1)[-1].lstrip("@")
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return tuple(names)


def github_summary_messages(
    summary: dict[str, int], rate: github_enricher.RateLimit | None = None
) -> list[tuple[str, str]]:
    """Plain-language outcome of the GitHub step as (level, text); level is a Streamlit message type.

    Exists so a half-empty github_* column is never a mystery: every blank is explained here
    (and per row in the github_status / github_notes diagnostic columns).
    """
    out: list[tuple[str, str]] = []
    if summary.get("done"):
        out.append(("caption", f"GitHub: {summary['done']} resume(s) analysed."))
    if summary.get("skipped_rate_limit"):
        when = f" The limit resets at {rate.reset_text()}." if rate else ""
        if rate and not rate.authenticated:
            fix = "Add a GitHub token (5,000 requests/hour instead of 60), or re-run these rows after the reset."
        else:
            fix = "Re-run these rows after the limit resets."
        out.append(("warning", f"GitHub: {summary['skipped_rate_limit']} resume(s) were SKIPPED because the GitHub API "
                               f"request limit was too low for the whole batch.{when} {fix}"))
    if summary.get("rate_limited"):
        out.append(("warning", f"The GitHub API request limit interrupted profile listing for "
                               f"{summary['rate_limited']} resume(s)."))
    if summary.get("repository_rate_limited"):
        out.append(("warning", f"The GitHub API request limit interrupted repository scans for "
                               f"{summary['repository_rate_limited']} resume(s). Results may be incomplete."))
    if summary.get("bad_token"):
        out.append(("error", f"GitHub: {summary['bad_token']} resume(s) failed because the GitHub token is invalid or expired."))
    if summary.get("organisation"):
        out.append(("info", f"GitHub: {summary['organisation']} resume(s) pointed only at GitHub organisations (a company or "
                            f"project account, not a person) and no other account was found, so nothing was analysed for "
                            f"them. They were probably linked from a project, not the candidate's own profile."))
    if summary.get("user_not_found"):
        out.append(("info", f"GitHub: {summary['user_not_found']} profile(s) do not exist on GitHub (mistyped or placeholder "
                            f"usernames in the resume)."))
    if summary.get("failed_other"):
        out.append(("warning", f"GitHub: {summary['failed_other']} resume(s) failed for other reasons; tick 'Add diagnostic "
                               f"columns' to see github_notes."))
    if summary.get("no_github_profile"):
        out.append(("caption", f"GitHub: {summary['no_github_profile']} resume(s) contain no GitHub profile."))
    return out


def to_dataframe(items: list[Processed], include_raw_text: bool = False) -> pd.DataFrame:
    df = pd.DataFrame([p.row for p in items], columns=ALL_COLUMNS)
    if include_raw_text:
        df["raw_text"] = [p.text for p in items]
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """RFC 4180-compatible UTF-8 CSV, with a BOM for spreadsheet importers.

    CSV intentionally has no widths, wrapping, styles, or cell types. Values
    containing commas, quotes, or line breaks are quoted so Google Sheets and
    Excel import them as one cell. Use the XLSX download when visual formatting
    or text-typed phone numbers is required.
    """
    return df.to_csv(
        index=False,
        na_rep="",
        # Quote every field. This is valid RFC 4180 and makes the comma
        # delimiter unambiguous to spreadsheet importers with locale-specific
        # number/list separators.
        quoting=csv.QUOTE_ALL,
        lineterminator="\r\n",
    ).encode("utf-8-sig")


def output_columns(df: pd.DataFrame, include_diagnostics: bool = False) -> list[str]:
    """Columns of the user-facing output.csv: the ten requested ones, then optional extras."""
    cols = list(OUTPUT_COLUMNS)
    if include_diagnostics:
        cols += DIAGNOSTIC_COLUMNS
    if "raw_text" in df.columns:          # only present when the sidebar option asked for it
        cols.append("raw_text")
    cols += [name for name in candidate_ranker.RANKING_COLUMNS if name in df.columns]
    cols += [name for name in EVIDENCE_COLUMNS if name in df.columns]
    cols += [name for name in github_enricher.GITHUB_COLUMNS if name not in cols and name in df.columns]
    return cols


_EXCEL_PHONE = re.compile(r"^\+?\d[\d ]*$")


def excel_safe_phone(value: object) -> str:
    """``08810931580`` -> ``="08810931580"``: Excel then shows the number as typed instead of dropping the
    leading zero or switching long numbers to scientific notation. Anything that is not plain digits is untouched."""
    text = str(value)
    return f'="{text}"' if _EXCEL_PHONE.match(text) else text


def table_csv_bytes(frame: pd.DataFrame, columns: list[str], excel_phone: bool = False) -> bytes:
    """UTF-8 (with BOM) CSV of ``frame[columns]``; ``excel_phone`` writes phones as ="..." (see ``excel_safe_phone``)."""
    out = frame[columns].copy()
    if excel_phone and "phone" in out.columns:
        out["phone"] = out["phone"].map(excel_safe_phone)
    return to_csv_bytes(out)


def output_csv_bytes(df: pd.DataFrame, include_diagnostics: bool = False, excel_phone: bool = False) -> bytes:
    """The downloadable output.csv: exactly the requested columns, in order, no row-number column.

    ``excel_phone=True`` writes phone numbers as ="..." so Excel keeps leading zeros (other tools,
    e.g. pandas, then see those extra characters).
    """
    return table_csv_bytes(df, output_columns(df, include_diagnostics), excel_phone)


def output_xlsx_bytes(df: pd.DataFrame, include_diagnostics: bool = False) -> bytes:
    """output.xlsx: the same columns as output.csv, with formatting (styled frozen header, column widths,
    wrapped text, phone numbers stored as text). Needs ``xlsxwriter``; raises ImportError without it."""
    return xlsx_export.dataframe_to_xlsx(df, output_columns(df, include_diagnostics))


def render_candidate_ranking(st, df: pd.DataFrame, include_diag: bool, excel_phone: bool) -> pd.DataFrame | None:
    """Collect role requirements and rank cached extraction rows on demand."""
    st.divider()
    st.subheader("Candidate Ranking & Prioritization")
    st.caption("Enter one technology or alternatives separated by OR. Only project, professional experience, and "
               "analysed GitHub technology evidence contributes to ranking.")
    entries = st.session_state.setdefault("ranking_requirements", [])
    with st.form("ranking_add_requirement", clear_on_submit=True):
        a, b = st.columns([5, 1])
        raw = a.text_input("Technology or alternative group", placeholder="Django OR FastAPI OR Flask")
        mandatory = b.selectbox("Type", ["Mandatory", "Optional"])
        add = st.form_submit_button("Add requirement")
    if add and raw.strip():
        try:
            candidate_ranker.normalize_requirements([(raw, mandatory == "Mandatory")], TechnologyMatcher())
        except ValueError as exc:
            st.error(str(exc))
        else:
            entries.append((raw.strip(), mandatory == "Mandatory"))
    for index, (raw, required) in enumerate(list(entries)):
        c1, c2 = st.columns([8, 1])
        c1.write(f"{'Mandatory' if required else 'Optional'}: {raw}")
        if c2.button("Remove", key=f"rank_remove_{index}"):
            entries.pop(index)
            st.rerun()
    threshold = st.slider("Minimum Technology Match (%)", 0, 100, 60)
    min_projects = st.slider("Minimum Matching Projects", 0, 20, 2)
    st.caption("Only distinct resume projects with a mandatory technology count. Project eligibility is unverified "
               "when individual project extraction is unavailable.")
    w1, w2, w3 = st.columns(3)
    weights = {
        "github": w1.number_input("GitHub evidence weight", min_value=0, value=50),
        "project": w2.number_input("Project evidence weight", min_value=0, value=30),
        "experience": w3.number_input("Experience evidence weight", min_value=0, value=20),
    }
    if st.button("Rank Candidates", type="primary"):
        if not entries:
            st.info("Add a requirement before ranking.")
        elif sum(weights.values()) == 0:
            st.error("At least one evidence weight must be positive.")
        else:
            st.session_state["ranking_config"] = (list(entries), threshold, dict(weights), min_projects)
    config = st.session_state.get("ranking_config")
    if config is None:
        return None
    try:
        matcher = TechnologyMatcher()
        requirements = candidate_ranker.normalize_requirements(config[0], matcher)
        ranked = pd.DataFrame(candidate_ranker.rank_candidates(
            df.to_dict("records"), requirements, matcher, config[2], config[1], config[3] if len(config) > 3 else 2,
        ))
    except (ValueError, DictionaryError, EngineUnavailable) as exc:
        st.error(f"Could not rank candidates: {exc}")
        return None
    counts = ranked["priority"].value_counts()
    metrics = st.columns(6)
    for column, (label, count) in zip(metrics, [
        ("Total", len(ranked)), ("Meeting thresholds", int(ranked["shortlisting_eligible"].sum())),
        *[(p, int(counts.get(p, 0))) for p in ("P1", "P2", "P3", "Unclassified")],
    ]):
        column.metric(label, count)
    show_all = st.checkbox("Show all candidates, including below threshold")
    selected = st.multiselect("Priority", ["P1", "P2", "P3", "Unclassified", "Pending Analysis", "Below Threshold"],
                              default=["P1", "P2", "P3", "Unclassified", "Pending Analysis", "Below Threshold"])
    p1_only = st.checkbox("P1 candidates only")
    min_match = st.slider("Filter minimum technology match (%)", 0, 100, 0)
    min_evidence = st.slider("Filter minimum evidence score", 0, 100, 0)
    technology = st.text_input("Filter by matched technology")
    filtered = ranked[ranked["priority"].isin(["P1"] if p1_only else selected)].copy()
    if not show_all:
        filtered = filtered[~filtered["priority"].isin(["Below Threshold", "Pending Analysis"])]
    filtered = filtered[(filtered["technology_match_percentage"] >= min_match)
                        & (filtered["evidence_score"] >= min_evidence)]
    if technology.strip():
        canonical = candidate_ranker._canonical(technology, matcher)
        if canonical is None:
            st.warning("Technology is not in the dictionary.")
            filtered = filtered.iloc[0:0]
        else:
            filtered = filtered[filtered.apply(
                lambda row: canonical in set(candidate_ranker._source_set(
                    ";".join(str(row[c]) for c in ("github_matched_technologies", "project_matched_technologies",
                                                   "experience_matched_technologies")), matcher)), axis=1,
            )]
    columns = ["phone", "resume_link", *candidate_ranker.RANKING_COLUMNS]
    st.dataframe(filtered[columns], hide_index=True, use_container_width=True)
    for index, row in filtered.iterrows():
        with st.expander(f"Evidence: {row.get('phone') or row.get('resume_link') or index}"):
            st.write({name: row[name] for name in candidate_ranker.RANKING_COLUMNS})
            for title, column in (("GitHub technology provenance", "github_evidence_details"),
                                  ("Project verification", "verified_project_repositories"),
                                  ("Experience attribution", "experience_evidence_details"),
                                  ("Internship attribution", "internship_evidence_details"),
                                  ("Repository activity", "repository_activity_details")):
                st.markdown(f"**{title}**")
                try:
                    detail = json.loads(row.get(column) or "null")
                except (TypeError, ValueError):
                    detail = None
                st.write(detail if detail else "Unavailable")
            st.caption(f"GitHub evidence quality: {row['github_evidence_quality']}; repository authorship is not inferred.")
    st.download_button("Download filtered ranking CSV", output_csv_bytes(filtered, include_diag, excel_phone),
                       file_name="candidate_ranking.csv", mime="text/csv")
    try:
        st.download_button("Download filtered ranking XLSX", output_xlsx_bytes(filtered, include_diag),
                           file_name="candidate_ranking.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        pass
    return ranked


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def _github_tokens(secrets: Mapping[str, object], manual_token: str) -> dict[str, str]:
    """Return tokens in priority order, with their configuration sources for logging."""
    environment_token = os.getenv("GITHUB_TOKEN")
    try:
        configured = secrets.get("GITHUB_TOKENS", ())
        secret_token = secrets.get("GITHUB_TOKEN")
    except FileNotFoundError:
        configured, secret_token = (), None
    configured = configured if isinstance(configured, (list, tuple)) else ()
    values = (
        ("the sidebar GitHub token", manual_token),
        *((f"GITHUB_TOKENS entry #{index}", value) for index, value in enumerate(configured, 1)),
        ("GITHUB_TOKEN in Streamlit secrets", secret_token),
        ("the GITHUB_TOKEN environment variable", environment_token),
    )
    tokens: dict[str, str] = {}
    for source, value in values:
        if isinstance(value, str) and value.strip():
            tokens.setdefault(value.strip(), source)
    return tokens


def main() -> None:
    import streamlit as st

    try:                                   # optional: GITHUB_TOKEN from a .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    st.set_page_config(page_title="Resume Extractor Tester", layout="wide")
    st.title("Resume Shortlisting")
    st.caption(
        "Phone + resume link -> download -> extract text -> section_detector -> technology_matcher -> output.csv. "
        "Supports PDF, DOCX and TXT (no OCR)."
    )

    with st.sidebar:
        st.header("Options")
        workers = st.slider("Parallel downloads", 1, 50, 16)
        include_diag = st.checkbox(
            "Add diagnostic columns to the CSV", value=False,
            help="status, error, review flags, per-project technologies (projects_summary), "
                 "unrecognised technologies and warnings",
        )
        excel_phone = st.checkbox(
            "Excel-safe phone numbers", value=False,
            help="CSV only. Writes phones as =\"0881...\" so Excel keeps leading zeros and does not show long numbers "
                 "as 9.19E+11. Not needed for the .xlsx download (phones are stored as text there). Leave off for "
                 "Google Sheets and for other tools.",
        )
        include_raw = st.checkbox("Include raw extracted text in CSV", value=False)
        run_tech = st.checkbox("Run technology matcher", value=True)
        run_projects = st.checkbox("Extract projects (per-project technologies)", value=True)
        engine = st.selectbox("Matching engine", ["regex", "flashtext"], help="flashtext needs `pip install flashtext`")
        extra_text = st.text_area(
            "Extra technologies (this run only)",
            height=110,
            placeholder="Frontend: Redux, Svelte\nBackend: Hibernate\nGraphQL",
            help="One per line: 'Category: Name1, Name2' or just 'Name' (category Other). "
                 "To keep them permanently, add them to app/dictionaries/technologies_master.json.",
        )
        st.subheader("GitHub")
        run_gh = st.checkbox(
            "Analyse GitHub repositories", value=True,
            help="Adds repos_count, github_tech_stack and project_categories. The profile column is filled either way.",
        )
        scan_method = st.selectbox(
            "Repository scan method", ("clone", "api"),
            format_func={"clone": "Git clone", "api": "GitHub API (token)"}.__getitem__,
            help="Git clone requires Git installed. GitHub API uses a token to fetch only the file tree and "
                 "selected files. API scans use API quota for repository contents as well as listing.",
        )
        github_parallel_requests = st.slider(
            "Parallel GitHub API requests", min_value=1, max_value=github_enricher.MAX_PARALLEL_GITHUB_REQUESTS,
            value=github_enricher.DEFAULT_PARALLEL_GITHUB_REQUESTS, disabled=not run_gh,
            help="Maximum simultaneous GitHub API requests per batch. Applies to repository listing in both modes "
                 "and file downloads in API mode.",
        )
        gh_token = st.text_input(
            "GitHub token", type="password",
            help="You can enter a token here or configure GITHUB_TOKENS in Streamlit secrets. A token is required for "
                 "API scans and optional for Git clone.",
        )
        tokens = _github_tokens(st.secrets, gh_token) if run_gh else {}
        if run_gh and not tokens:
            if scan_method == "api":
                st.warning("Enter or configure a GitHub token to scan repositories through the API.")
            else:
                st.warning("Without a token, GitHub allows 60 API requests per hour. A batch with more than "
                           "about 55 GitHub profiles may be analysed in part.")
        skip_forks = st.checkbox("Skip forked repositories", value=True, help="Forks are mostly other people's code.")
        max_repos = int(st.number_input(
            "Maximum repositories listed per account", min_value=10, max_value=500,
            value=github_enricher.MAX_REPOS_PER_ACCOUNT, step=10,
            help="GitHub organisations and very large accounts are capped during listing. This limit is separate "
                 "from the number selected for analysis.",
        ))
        initial_repo_budget = int(st.number_input(
            "Initial repositories to analyze per candidate", min_value=1, max_value=500, value=15,
            help="Cached repositories are reused. Other repositories are prioritized from metadata before scanning.",
        ))
        exhaustive_gh = st.checkbox("Analyze all listed repositories", value=False,
                                    help="Disables the initial scan budget when completeness is required.")
        relevance_text = st.text_area(
            "GitHub relevance requirements (optional)", height=75,
            placeholder="Python\nDjango OR FastAPI OR Flask\nPostgreSQL",
            help="One mandatory technology or OR group per line. These hints determine the order of repository scans.",
        )
        blocked_text = st.text_area(
            "GitHub accounts to ignore (one per line)", height=70, placeholder="some-company\nsome-course-org",
            help="Names that should never be taken as a candidate's profile. Well-known organisations "
                 "(google, linkedin, microsoft, ...) are already ignored.",
        )
        all_profiles = st.checkbox(
            "Analyse every GitHub account found (not only the best match)", value=False,
            help="Off (recommended): only the account that best looks like the candidate's own is analysed; the others "
                 "are listed in github_notes. On: the repositories of every account mentioned are merged, which mixes "
                 "in other people's and organisations' work.",
        )
        st.caption(f"section_detector: `{section_detector.__file__}`")
        st.caption(f"technology_matcher: `{technology_matcher.__file__}`")
        st.caption(f"project_extractor: `{project_extractor.__file__}`")
        st.caption("Detector, matcher and extractor are rebuilt on every run, so edits to the modules "
                   "or the dictionary JSON files are picked up.")

    tab_paste, tab_csv = st.tabs(["Paste list", "Upload CSV"])
    paste_entries: list[tuple[str, str]] = []
    csv_entries: list[tuple[str, str]] = []

    with tab_paste:
        pasted = st.text_area(
            "One per line: phone, link",
            height=160,
            placeholder="+91 98765 43210, https://example.com/resume1.pdf\n9876501234 https://drive.google.com/file/d/XXXX/view",
            key="pasted",
        )
        if pasted.strip():
            paste_entries = parse_pasted(pasted)

    with tab_csv:
        uploaded = st.file_uploader("CSV with a phone column and a resume-link column", type=["csv"])
        if uploaded is not None:
            try:
                src = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                src = None
            if src is not None and len(src.columns) >= 2:
                cols = list(src.columns)
                c1, c2 = st.columns(2)
                p_idx = guess_column(cols, ("phone", "mobile", "cell", "whatsapp", "contact", "number"))
                phone_col = c1.selectbox("Phone column", cols, index=p_idx)
                l_idx = guess_column(cols, ("resume", "cv", "link", "url", "drive"), exclude=phone_col)
                link_col = c2.selectbox("Resume link column", cols, index=l_idx)
                csv_entries = [
                    (str(a).strip(), str(b).strip())
                    for a, b in zip(src[phone_col], src[link_col])
                ]
                st.caption(f"{len(csv_entries)} rows loaded")
            elif src is not None:
                st.error("CSV needs at least two columns (phone, resume link).")

    st.divider()
    # Streamlit cannot tell which tab is open, so use the first tab with input.
    source, entries = next(
        ((name, e) for name, e in (
            ("Upload CSV", csv_entries), ("Paste list", paste_entries),
        ) if e),
        ("", []),
    )
    st.write(f"**{len(entries)}** resume(s) queued" + (f" (from the *{source}* tab)" if source else ""))

    if st.button("Run", type="primary", disabled=not entries):
        detector = SectionDetector()
        matcher, extractor, shared = None, None, None
        if run_tech or run_projects:
            try:
                shared = TechnologyMatcher(engine=engine, extra_technologies=parse_extra_technologies(extra_text))
            except (EngineUnavailable, DictionaryError) as e:
                st.error(f"Technology matcher could not be created: {e}")
                return
            matcher = shared if run_tech else None
            extractor = ProjectExtractor(shared) if run_projects else None
        bar = st.progress(0.0, text="Starting...")
        t0 = time.perf_counter()
        items = run_batch(
            entries, detector, workers,
            progress_cb=lambda d, n: bar.progress(d / n, text=f"{d}/{n} processed"),
            matcher=matcher,
            extractor=extractor,
            blocked_accounts=parse_blocked_accounts(blocked_text),
        )
        bar.empty()

        st.session_state["github_summary"] = None            # never show a previous run's outcome
        if run_gh:
            retry_at = st.session_state.get("github_retry_at", 0.0)
            if scan_method == "api" and not tokens:
                st.error("GitHub analysis skipped. Enter or configure a token for API scanning.")
            elif scan_method == "clone" and not github_enricher.git_available():
                st.error("GitHub analysis skipped. Git clone requires the `git` command. GitHub profiles are still listed.")
            elif retry_at > time.time():
                when = time.strftime("%H:%M:%S", time.localtime(retry_at))
                st.warning(f"GitHub analysis paused until {when} because the API rate limit was reached.")
            elif any(p.profiles for p in items):
                credentials = github_enricher.GitHubCredentials(
                    tuple(tokens), token_sources=tokens, max_parallel_requests=github_parallel_requests,
                )
                gbar = st.progress(0.0, text="Finding GitHub repositories...")

                def gh_progress(done: int, total: int) -> None:
                    if total:
                        gbar.progress(done / total, text=f"{done} of {total} repositories analysed")
                    else:
                        gbar.progress(0.0, text="Finding GitHub repositories...")

                try:
                    requirement_entries = [(line.strip(), True) for line in relevance_text.splitlines() if line.strip()]
                    if not requirement_entries:
                        requirement_entries = list(st.session_state.get("ranking_requirements", []))
                    repo_requirements = candidate_ranker.normalize_requirements(
                        requirement_entries, shared or TechnologyMatcher())
                    required_groups = tuple(req.technologies for req in repo_requirements if req.mandatory)
                except (ValueError, DictionaryError, EngineUnavailable) as exc:
                    st.warning(f"GitHub relevance requirements ignored: {exc}")
                    required_groups = ()
                rate = github_enricher.rate_limit_status(credentials)
                budget = None
                if rate is not None:
                    needed = len({u.lower() for p in items for u in (p.profiles if all_profiles else p.profiles[:1])})
                    budget = rate.budget()
                    if needed > budget:
                        st.warning(
                            f"The GitHub API has {rate.remaining} of {rate.limit} requests left"
                            f"{'' if rate.authenticated else ' (no token)'}. Listing the repositories for {needed} "
                            f"distinct profiles needs at least one request each. Only the first {budget} profiles will "
                            f"be listed now. The quota resets at {rate.reset_text()}."
                        )
                screening_config = st.session_state.get("ranking_config")
                run_github(items, credentials, skip_forks, gh_progress, max_profiles=budget, rate=rate, all_profiles=all_profiles,
                           max_repos=max_repos, required_groups=required_groups,
                           initial_repo_budget=initial_repo_budget, exhaustive=exhaustive_gh, scan_method=scan_method,
                           screening_threshold=screening_config[1] if screening_config else 60,
                           min_matching_projects=screening_config[3] if screening_config and len(screening_config) > 3 else 2)
                st.session_state["github_retry_at"] = max(
                    credentials.retry_at, rate.reset if rate is not None and rate.remaining == 0 else 0,
                )
                st.session_state["github_max_repos"] = max_repos
                gbar.empty()
                if credentials.retry_at > time.time():
                    when = time.strftime("%H:%M:%S", time.localtime(credentials.retry_at))
                    st.caption(f"GitHub API requests can resume after {when}.")
                st.session_state["github_summary"] = (
                    github_enricher.summarize(p.github for p in items if p.github is not None), rate,
                )
            else:
                st.info("No GitHub profile was found in any resume, so there was nothing to analyse.")
        st.session_state["items"] = items
        st.session_state.pop("ranking_config", None)
        st.session_state["include_raw"] = include_raw
        st.session_state["elapsed"] = time.perf_counter() - t0

    items = st.session_state.get("items")
    if not items:
        return

    df = to_dataframe(items, st.session_state.get("include_raw", False))
    ok = int((df["status"] == "ok").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Processed", len(df))
    m2.metric("Succeeded", ok)
    m3.metric("Failed / partial", len(df) - ok)
    m4.metric("Wall time (s)", f"{st.session_state.get('elapsed', 0):.1f}")

    for level, text in github_summary_messages(*(st.session_state.get("github_summary") or ({}, None))):
        getattr(st, level)(text)
    confidence = profile_confidence_message(items)
    if confidence:
        getattr(st, confidence[0])(confidence[1])
    limits = github_limits_message(items, st.session_state.get("github_max_repos"))
    if limits:
        getattr(st, limits[0])(limits[1])
    github_infos = [p.github for p in items if p.github is not None]
    if github_infos:
        st.caption("GitHub repository analysis: "
                   f"{sum(info.cached_repos for info in github_infos)} cached, "
                   f"{sum(info.selected_repos for info in github_infos)} selected per candidate, "
                   f"{sum(info.deferred_repos for info in github_infos)} deferred, "
                   f"{sum(info.empty_repos for info in github_infos)} empty, "
                   f"{sum(info.analysis_failures for info in github_infos)} failed. "
                   "Shared repositories are scanned once. Deferred repositories are awaiting analysis.")

    ranked = render_candidate_ranking(st, df, include_diag, excel_phone)
    if ranked is not None:
        df = ranked

    st.download_button(
        "Download .csv (data only)", data=output_csv_bytes(df, include_diag, excel_phone), file_name="output.csv",
        mime="text/csv",
    )
    try:
        xlsx_data: bytes | None = output_xlsx_bytes(df, include_diag)
    except ImportError:
        xlsx_data = None
        st.info("For the formatted Excel / Google Sheets download, install xlsxwriter:  pip install xlsxwriter")
    if xlsx_data is not None:
        st.download_button(
            "Download .xlsx (recommended for Google Sheets)", data=xlsx_data, file_name="output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary",
        )
    st.caption("CSV is data-only: it cannot retain column widths, wrapping, or header styles. For Google Sheets, "
               "use the XLSX download and File > Import > Upload. If importing CSV, untick 'Convert text to numbers, "
               "dates and formulas' to preserve phone numbers with leading zeros.")
    csv_cols = output_columns(df, include_diag)
    # Text only: mixed ""/number columns would otherwise break the table, and the table never shows more than the CSV.
    st.dataframe(df[csv_cols].astype(str), hide_index=True)

    failed = df[df["status"] != "ok"]
    if len(failed):
        st.subheader("Failures")
        st.dataframe(failed[["phone", "resume_link", "status", "error_stage", "error"]])

if __name__ == "__main__":
    main()
