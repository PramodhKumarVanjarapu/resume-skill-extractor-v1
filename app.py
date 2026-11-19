"""Streamlit harness for testing section_detector + technology_matcher + project_extractor on real resumes.

    streamlit run app.py

Input : phone number + resume link (single, pasted list, or CSV upload)
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
        Below the table, a live "Filter profiles" panel narrows the profiles
        (technologies, counts, phone lists from Excel ...) and downloads exactly
        the filtered rows; a second panel filters a CSV / Excel file saved earlier.
        The on-screen inspector always shows the full detail: extracted text,
        detected sections, technology matches and projects with evidence.

Technology matching runs on the whole extracted text; each match is then
attributed to a section by its character offset. Project extraction runs on
the detected Projects section only and matches technologies per project.

All the logic lives in plain functions above ``main()`` so it can be tested
(and reused) without Streamlit. Supported files: PDF, DOCX, TXT.
Not supported: scanned/image PDFs (no OCR), legacy .doc, HTML pages.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import zipfile
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

# Every module this file imports. A missing one used to surface as "No module named 'filter_panel'"; say which files
# are missing and where they belong instead.
REQUIRED_MODULES = (
    "section_detector", "technology_matcher", "project_extractor", "github_profile", "github_enricher", "analyzer",
    "resume", "profile_filter", "filter_panel", "xlsx_export", "pdf_parse",
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
import filter_panel  # noqa: E402
import github_enricher  # noqa: E402
import pdf_parse  # noqa: E402
import profile_filter  # noqa: E402
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
PROJECT_COLUMNS = ["projects_summary", "unrecognized_technologies"]
# github_profile, repos_count, github_tech_stack, project_categories, github_status, github_notes
GITHUB_COLUMNS = github_enricher.GITHUB_COLUMNS
# how sure we are that github_profile is the candidate's own account, and every candidate with its evidence
PROFILE_SEARCH_COLUMNS = ["github_confidence", "github_candidates"]
TAIL_COLUMNS = [
    "preamble", "sections_metadata", "technology_matches", "projects_json",
    "download_s", "parse_s", "detect_ms", "tech_ms", "project_ms",
]
ALL_COLUMNS = (BASE_COLUMNS + SECTION_KEYS + TECH_COLUMNS + PROJECT_COLUMNS + GITHUB_COLUMNS
               + PROFILE_SEARCH_COLUMNS + TAIL_COLUMNS)

# The columns of the downloadable output.csv, in this order.
OUTPUT_COLUMNS = [
    "phone", "resume_link", "num_projects", "project_names",
    "tech_skills", "tech_projects", "tech_experience", "tech_internships",
    "tech_certifications", "tech_achievements",
    "github_profile", "repos_count", "github_tech_stack", "project_categories",
]
# Optional extras (sidebar checkbox) that explain failures and uncertainty.
DIAGNOSTIC_COLUMNS = [
    "status", "error_stage", "error", "projects_needs_review", "projects_summary",
    "unrecognized_technologies", "github_confidence", "github_candidates", "github_status", "github_notes", "warnings",
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
    """All technology mentions in ``text`` plus flat, JSON-able rows (with section)."""
    matches = matcher.find_all(text)
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
                names = _unique(m.name for m in matches)
                row["technologies"] = "; ".join(names)
                row["num_technologies"] = len(names)
                for key in TECH_SECTION_KEYS:
                    row[f"tech_{key}"] = "; ".join(
                        _unique(r["name"] for r in match_rows if r["section"] == key)
                    )
                row["technology_matches"] = json.dumps(match_rows, ensure_ascii=False)
            except Exception as e:
                row["status"], row["error_stage"] = "partial", "match"
                row["error"] = f"{type(e).__name__}: {str(e)[:150]}"
                match_rows = []

        if extractor is not None:
            try:
                t0 = time.perf_counter()
                project_result = extractor.extract(result.get("projects", ""))
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
    token: str | None = None,
    skip_forks: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
    max_profiles: int | None = None,
    rate: github_enricher.RateLimit | None = None,
    all_profiles: bool = False,
    max_repos: int | None = github_enricher.MAX_REPOS_PER_ACCOUNT,
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
    infos = github_enricher.analyze_profiles(
        [p.profiles for p in targets], token, skip_forks, progress_cb, max_profiles=max_profiles, rate=rate,
        all_profiles=all_profiles, max_repos=max_repos,
    )
    for p, info in zip(targets, infos):
        p.github = info
        p.row.update(info.columns())
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
    capped = [p for p in items if p.github is not None and any("most recently updated were analysed" in n for n in p.github.notes)]
    if not capped or cap is None:
        return None
    return ("caption", f"GitHub: {len(capped)} account(s) have more than {cap} repositories, so only the {cap} most recently "
                       f"updated were analysed (raise 'Max repositories analysed per account' in the sidebar to change this).")


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
        out.append(("warning", f"GitHub: {summary['rate_limited']} resume(s) FAILED because the API request limit ran out "
                               f"during the run. Add a GitHub token, or re-run these rows later."))
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
    """Every column of ``df`` as UTF-8 with BOM (so Excel opens non-ASCII text correctly)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def output_columns(df: pd.DataFrame, include_diagnostics: bool = False) -> list[str]:
    """Columns of the user-facing output.csv: the ten requested ones, then optional extras."""
    cols = list(OUTPUT_COLUMNS)
    if include_diagnostics:
        cols += DIAGNOSTIC_COLUMNS
    if "raw_text" in df.columns:          # only present when the sidebar option asked for it
        cols.append("raw_text")
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


def filter_alias_resolver(st):
    """Resolver that lets the technology columns of the filter understand aliases (nodejs = Node.js); built once per
    session. None if the dictionary cannot be loaded (the filter then simply matches the typed words)."""
    if "flt_alias_resolver" not in st.session_state:
        try:
            st.session_state["flt_alias_resolver"] = profile_filter.alias_resolver(TechnologyMatcher())
        except (DictionaryError, EngineUnavailable):
            st.session_state["flt_alias_resolver"] = None
    return st.session_state["flt_alias_resolver"]


def render_profile_filter(st, df: pd.DataFrame, csv_cols: list[str], excel_phone: bool) -> None:
    """Live filter panel for the profiles of this run: it filters exactly the columns of output.csv."""
    st.divider()
    st.subheader("Filter profiles")
    st.caption("Narrow down the profiles above: results update as you type, and the downloads contain exactly the "
               "profiles and columns you see here.")
    table = df[csv_cols].apply(lambda column: column.map(profile_filter.cell_text))      # all text: nothing is type-guessed
    filter_panel.render(
        st, table, scope="results", alias_resolver=filter_alias_resolver(st),
        csv_builder=lambda frame, cols: table_csv_bytes(frame, cols, excel_phone),
    )


def render_file_filter(st, excel_phone: bool) -> None:
    """Filter a CSV / Excel file saved earlier (for example last week's output.csv) without running anything."""
    with st.expander("Filter a CSV or Excel file you saved earlier"):
        upload = st.file_uploader("output.csv or output.xlsx", type=["csv", "xlsx"], key="flt_file_upload")
        if upload is None:
            st.caption("Upload a file to filter it with the same panel. Nothing is type-guessed, so phone numbers keep "
                       "their leading zeros.")
            return
        try:
            table, notes = profile_filter.read_table(upload.getvalue(), upload.name)
        except ValueError as exc:
            st.error(str(exc))
            return
        for note in notes:
            st.caption(note)
        st.caption(f"{len(table)} rows, {len(table.columns)} columns")
        filter_panel.render(
            st, table, scope="file", alias_resolver=filter_alias_resolver(st),
            csv_builder=lambda frame, cols: table_csv_bytes(frame, cols, excel_phone),
            file_stem=f"{Path(upload.name).stem}_filtered",
        )


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def main() -> None:
    import streamlit as st

    try:                                   # optional: GITHUB_TOKEN from a .env file
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    st.set_page_config(page_title="Resume Extractor Tester", layout="wide")
    st.title("Resume Extractor - test harness")
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
            help="Adds repos_count, github_tech_stack and project_categories. Needs network access and the "
                 "`git` command; can take minutes for people with many repositories. The profile column is "
                 "filled either way.",
        )
        gh_token = st.text_input(
            "GitHub token (optional)", value=os.getenv("GITHUB_TOKEN", ""), type="password",
            help="Read from GITHUB_TOKEN if set. Without a token GitHub allows 60 API requests per hour.",
        )
        if run_gh and not gh_token.strip():
            st.warning("No GitHub token: GitHub allows only 60 API requests per hour, so a batch with more than "
                       "about 55 GitHub profiles can only be analysed in part.")
        skip_forks = st.checkbox("Skip forked repositories", value=True, help="Forks are mostly other people's code.")
        max_repos = int(st.number_input(
            "Max repositories analysed per account", min_value=10, max_value=500, value=github_enricher.MAX_REPOS_PER_ACCOUNT,
            step=10, help="Protects against huge accounts: each repository is cloned and scanned. Accounts with more get "
                          "their most recently updated repositories, and GitHub organisations (google, linkedin, ...) are "
                          "skipped after one request.",
        ))
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

    tab_single, tab_paste, tab_csv = st.tabs(["Single resume", "Paste list", "Upload CSV"])
    single_entries: list[tuple[str, str]] = []
    paste_entries: list[tuple[str, str]] = []
    csv_entries: list[tuple[str, str]] = []

    with tab_single:
        phone = st.text_input("Phone number", key="single_phone")
        link = st.text_input("Resume link", key="single_link")
        if link.strip():
            single_entries = [(phone.strip(), link.strip())]

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
            ("Upload CSV", csv_entries), ("Paste list", paste_entries), ("Single resume", single_entries),
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
            if not github_enricher.git_available():
                st.error("GitHub analysis skipped: the `git` command was not found. Install git, or untick "
                         "'Analyse GitHub repositories'. GitHub profiles are still listed.")
            elif any(p.profiles for p in items):
                gbar = st.progress(0.0, text="Finding GitHub repositories...")

                def gh_progress(done: int, total: int) -> None:
                    if total:
                        gbar.progress(done / total, text=f"{done} of {total} repositories analysed")
                    else:
                        gbar.progress(0.0, text="Finding GitHub repositories...")

                token = gh_token.strip() or None
                rate = github_enricher.rate_limit_status(token)       # free call; None if it cannot be read
                budget = None
                if rate is not None:
                    needed = len({u.lower() for p in items for u in (p.profiles if all_profiles else p.profiles[:1])})
                    budget = rate.budget()
                    if needed > budget:
                        st.warning(
                            f"GitHub API quota: {rate.remaining} of {rate.limit} requests left"
                            f"{'' if rate.authenticated else ' (no token)'}, but {needed} distinct profiles need one each. "
                            f"Only the first {budget} will be analysed now; the rest are marked as skipped (quota resets "
                            f"at {rate.reset_text()}). Add a GitHub token to analyse them all."
                        )
                run_github(items, token, skip_forks, gh_progress, max_profiles=budget, rate=rate, all_profiles=all_profiles,
                           max_repos=max_repos)
                st.session_state["github_max_repos"] = max_repos
                gbar.empty()
                st.session_state["github_summary"] = (
                    github_enricher.summarize(p.github for p in items if p.github is not None), rate,
                )
            else:
                st.info("No GitHub profile was found in any resume, so there was nothing to analyse.")
        st.session_state["items"] = items
        st.session_state["include_raw"] = include_raw
        st.session_state["elapsed"] = time.perf_counter() - t0

    items = st.session_state.get("items")
    if not items:
        render_file_filter(st, excel_phone)
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

    st.download_button(
        "Download output.csv", data=output_csv_bytes(df, include_diag, excel_phone), file_name="output.csv",
        mime="text/csv", type="primary",
    )
    try:
        xlsx_data: bytes | None = output_xlsx_bytes(df, include_diag)
    except ImportError:
        xlsx_data = None
        st.info("For the formatted Excel / Google Sheets download, install xlsxwriter:  pip install xlsxwriter")
    if xlsx_data is not None:
        st.download_button(
            "Download output.xlsx (formatted - best for Google Sheets / Excel)", data=xlsx_data, file_name="output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    st.caption("Google Sheets: File > Import > Upload the **.xlsx** to keep column widths, the styled header and phone "
               "numbers exactly as they are. A .csv cannot carry formatting; if you import the CSV, untick "
               "'Convert text to numbers, dates and formulas' so Sheets does not drop leading zeros from phones.")
    csv_cols = output_columns(df, include_diag)
    st.caption("output.csv / output.xlsx contain: " + ", ".join(csv_cols))
    st.caption("Save the file with the **Download output.csv** button. The table below shows exactly the same columns "
               "(no row numbers). Rows that failed show blank cells; their reasons are listed under Failures.")
    # Text only: mixed ""/number columns would otherwise break the table, and the table never shows more than the CSV.
    st.dataframe(df[csv_cols].astype(str), hide_index=True)

    failed = df[df["status"] != "ok"]
    if len(failed):
        st.subheader("Failures")
        st.dataframe(failed[["phone", "resume_link", "status", "error_stage", "error"]])

    render_profile_filter(st, df, csv_cols, excel_phone)
    render_file_filter(st, excel_phone)

    st.subheader("Inspector")
    shown = [p for p in items if p.result is not None][:100]
    if len(items) > len(shown):
        st.caption("Showing up to 100 successfully parsed resumes.")
    for i, p in enumerate(shown):
        label = f"{p.row['phone'] or '(no phone)'} - {p.row['sections_found'] or 'NO SECTIONS DETECTED'}"
        with st.expander(label):
            if p.row["warnings"]:
                st.warning(p.row["warnings"])
            left, right = st.columns(2)
            with left:
                st.markdown("**Extracted text**")
                st.text_area("text", p.text, height=420, key=f"raw_{i}", label_visibility="collapsed")
            with right:
                st.markdown("**Detected sections**")
                if p.result.preamble:
                    st.markdown("`preamble`")
                    st.code(p.result.preamble, language=None)
                for s in p.result.metadata:
                    tag = "" if s.target else " (boundary only, not exported)"
                    st.markdown(
                        f"**{s.key}**{tag} - heading `{s.heading}` "
                        f"@ {s.heading_start}-{s.heading_end}, content {s.start}-{s.end}"
                    )
                    st.code(s.text or "(empty)", language=None)
            if p.row["github_profile"]:
                st.markdown("**GitHub**")
                if p.profile_search is not None and p.profile_search.candidates:
                    st.write(f"Profile confidence: {p.profile_search.confidence}  |  evidence: " + p.profile_search.describe())
                if p.github is not None and p.github.status:
                    st.write(
                        f"{p.row['github_profile']}  |  repositories: {p.row['repos_count'] or '-'}  |  "
                        f"status: {p.github.status}"
                    )
                    st.write("Project categories: " + (p.row["project_categories"] or "-"))
                    st.write("Tech stack: " + (p.row["github_tech_stack"] or "-"))
                    if p.row["github_notes"]:
                        st.warning(p.row["github_notes"])
                else:
                    st.write(p.row["github_profile"])
            if p.project_result is not None and p.project_result.projects:
                pr = p.project_result
                st.markdown(f"**Projects found ({len(pr.projects)})**" + ("  - needs review" if pr.needs_review else ""))
                st.dataframe(pd.DataFrame([
                    {
                        "project": q.name or "(untitled)",
                        "technologies": ", ".join(q.technologies),
                        "needs_review": q.needs_review,
                        "review_reasons": ", ".join(q.review_reasons),
                        "unrecognized_technologies": ", ".join(q.unrecognized_technologies),
                        "title_style": q.title_style,
                        "boundary_reasons": ", ".join(q.boundary_reasons),
                        "notes": "; ".join(q.notes),
                        "description": q.description,
                    }
                    for q in pr.projects
                ]))
                for q in pr.projects:
                    if q.technology_details:
                        with st.expander(f"Evidence: {q.name or '(untitled)'}"):
                            st.dataframe(pd.DataFrame([
                                {"technology": t.name, "category": t.category, "found_in": ", ".join(t.found_in),
                                 "matched_text": t.matched_text, "occurrences": t.occurrences, "evidence": t.evidence}
                                for t in q.technology_details
                            ]))
            if p.match_rows:
                st.markdown(f"**Technologies found ({len(p.match_rows)} mentions)**")
                cols = ["name", "category", "section", "matched_text", "start", "end",
                        "match_type", "reason", "evidence"]
                st.dataframe(pd.DataFrame(p.match_rows)[cols])
            elif p.row["technologies"] == "" and p.row["tech_ms"] != "":
                st.caption("No technologies found in this resume.")


if __name__ == "__main__":
    main()