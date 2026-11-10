"""Read a resume from a link: download it, extract text and hyperlinks,
find GitHub profiles and the projects listed in the resume.

Supported links:
  - direct PDF / DOCX / HTML / text links
  - Google Drive file links (the file must be shared as "Anyone with the link")
  - Google Docs links (exported as PDF so hidden hyperlinks survive)
  - Dropbox share links
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import requests

MAX_RESUME_BYTES = 15_000_000
RESUME_TIMEOUT = 30
MAX_PROFILES = 3          # keep at most this many GitHub profiles per resume
MAX_PROJECTS = 25

USER_AGENT = "Mozilla/5.0 (compatible; resume-reader/1.0)"

# =====================================================================
# Download
# =====================================================================


class ResumeError(Exception):
    pass


def normalize_url(url: str) -> str:
    """Rewrite share links so they return the file itself."""
    u = url.strip()
    parsed = urlparse(u)
    host = parsed.netloc.lower()

    if host.endswith("drive.google.com"):
        m = re.search(r"/file/d/([\w-]+)", parsed.path) or re.search(r"/d/([\w-]+)", parsed.path)
        file_id = m.group(1) if m else parse_qs(parsed.query).get("id", [None])[0]
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    if host.endswith("docs.google.com"):
        m = re.search(r"/document/d/([\w-]+)", parsed.path)
        if m:
            # PDF export keeps hyperlink targets; plain-text export drops them.
            return f"https://docs.google.com/document/d/{m.group(1)}/export?format=pdf"

    if host.endswith("dropbox.com"):
        query = re.sub(r"(^|&)dl=\d", "", parsed.query).strip("&")
        query = (query + "&dl=1").lstrip("&")
        return parsed._replace(query=query).geturl()

    return u


def download(url: str) -> tuple[bytes, str]:
    """Returns (content, content_type). Raises ResumeError."""
    target = normalize_url(url)
    try:
        with requests.get(
            target,
            headers={"User-Agent": USER_AGENT},
            timeout=RESUME_TIMEOUT,
            stream=True,
            allow_redirects=True,
        ) as r:
            if r.status_code in (401, 403):
                raise ResumeError("Link is private. Share it as 'Anyone with the link'")
            if r.status_code == 404:
                raise ResumeError("Link not found (404)")
            r.raise_for_status()

            if "accounts.google.com" in r.url or "ServiceLogin" in r.url:
                raise ResumeError("Link is private. Share it as 'Anyone with the link'")

            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > MAX_RESUME_BYTES:
                    raise ResumeError("File is too large (over 15 MB)")
                chunks.append(chunk)
            return b"".join(chunks), r.headers.get("content-type", "").lower()
    except ResumeError:
        raise
    except requests.RequestException as e:
        raise ResumeError(f"Could not download resume: {e}") from e


# =====================================================================
# Text + link extraction
# =====================================================================


def pdf_extract(data: bytes) -> tuple[str, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ResumeError("Install pypdf to read PDF resumes:  pip install pypdf") from e

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                raise ResumeError("PDF is password protected")
        texts, links = [], []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                continue
            # Hyperlinks whose visible text is just "GitHub" only exist as annotations.
            try:
                for annot in page.get("/Annots") or []:
                    obj = annot.get_object()
                    action = obj.get("/A")
                    if action is not None:
                        uri = action.get_object().get("/URI")
                        if uri:
                            links.append(str(uri))
            except Exception:  # noqa: BLE001
                continue
        return "\n".join(texts), links
    except ResumeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ResumeError(f"Could not read PDF: {e}") from e


def docx_extract(data: bytes) -> tuple[str, list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
            try:
                rels = z.read("word/_rels/document.xml.rels").decode("utf-8", "replace")
            except KeyError:
                rels = ""
    except (zipfile.BadZipFile, KeyError) as e:
        raise ResumeError("Could not read Word file") from e

    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:(?:br|tab)\s*/>", " ", xml)
    text = unescape(re.sub(r"<[^>]+>", "", xml))
    links = [
        unescape(m.group(1))
        for m in re.finditer(r'Target="([^"]+)"[^>]*TargetMode="External"', rels)
    ]
    links += [
        unescape(m.group(1))
        for m in re.finditer(r'TargetMode="External"[^>]*Target="([^"]+)"', rels)
    ]
    return text, links


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_extract(data: bytes) -> tuple[str, list[str]]:
    parser = _HTMLText()
    parser.feed(data.decode("utf-8", "replace"))
    return "".join(parser.parts), parser.links


def extract_text_and_links(data: bytes, content_type: str, url: str = "") -> tuple[str, list[str]]:
    if data[:5] == b"%PDF-":
        return pdf_extract(data)
    if data[:2] == b"PK":
        return docx_extract(data)
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ResumeError("Old .doc files are not supported. Save as PDF or .docx")
    head = data[:2000].lstrip().lower()
    if "html" in content_type or head.startswith((b"<!doctype html", b"<html")) or b"<body" in data[:5000].lower():
        return html_extract(data)
    return data.decode("utf-8", "replace"), []


# =====================================================================
# GitHub profile detection
# =====================================================================

# github.com/<this> paths that are not user profiles
RESERVED = {
    "about", "apps", "assets", "blog", "codespaces", "collections", "contact", "copilot",
    "customer-stories", "enterprise", "events", "explore", "features", "gist", "issues",
    "join", "login", "marketplace", "new", "news", "notifications", "organizations", "orgs",
    "pricing", "pulls", "readme", "resources", "search", "security", "settings", "site",
    "solutions", "sponsors", "team", "topics", "trending", "users", "enterprises", "premium-support",
}
# words that follow "GitHub:" in a resume but are not usernames
FALLBACK_STOPWORDS = {
    "link", "profile", "url", "portfolio", "account", "page", "repo", "repository", "repositories",
    "actions", "pages", "copilot", "projects", "project", "and", "com", "io", "the", "at",
}

USER_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
GITHUB_URL = re.compile(rf"github\.com/\s*({USER_NAME})(?![A-Za-z0-9-])", re.IGNORECASE)
GITHUB_REPO_URL = re.compile(rf"github\.com/({USER_NAME})/([\w.-]+)", re.IGNORECASE)
GITHUB_PAGES = re.compile(rf"\b({USER_NAME})\.github\.io\b", re.IGNORECASE)
GITHUB_LABEL = re.compile(rf"\bgit\s?hub\b\s*[:\-–—|/]\s*@?({USER_NAME})(?![A-Za-z0-9-])", re.IGNORECASE)
GITHUB_HANDLE = re.compile(rf"(?<![\w.])@({USER_NAME})(?![\w.@-])")


def _valid(user: str) -> bool:
    low = user.lower()
    return low not in RESERVED and not low.endswith("-")


def find_github_profiles(text: str, links: list[str]) -> list[str]:
    """Usernames found in the resume, most likely first."""
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}

    def add(user: str, weight: int) -> None:
        if _valid(user):
            counts[user.lower()] += weight
            display.setdefault(user.lower(), user)

    # Hyperlink targets are the most reliable, then visible URLs.
    for link in links:
        for m in GITHUB_URL.finditer(link):
            add(m.group(1), 3)
    # PDFs sometimes break URLs across lines or put a space after the slash.
    flat = re.sub(r"(github\.com/)\s+", r"\1", text, flags=re.IGNORECASE)
    for m in GITHUB_URL.finditer(flat):
        add(m.group(1), 2)

    if not counts:
        # Weaker signals, only used when there is no github.com link at all.
        for m in GITHUB_PAGES.finditer(text):
            if m.group(1).lower() not in FALLBACK_STOPWORDS:
                add(m.group(1), 1)
        for m in GITHUB_LABEL.finditer(text):
            if m.group(1).lower() not in FALLBACK_STOPWORDS:
                add(m.group(1), 1)

    ranked = [display[u] for u, _ in counts.most_common()]
    return ranked[:MAX_PROFILES]


def find_repo_links(text: str, links: list[str]) -> list[str]:
    seen, out = set(), []
    for source in [*links, re.sub(r"(github\.com/)\s+", r"\1", text, flags=re.IGNORECASE)]:
        for m in GITHUB_REPO_URL.finditer(source):
            if not _valid(m.group(1)):
                continue
            repo = re.sub(r"\.git$", "", m.group(2).rstrip(".,;)"))
            key = f"{m.group(1)}/{repo}".lower()
            if key not in seen:
                seen.add(key)
                out.append(f"https://github.com/{m.group(1)}/{repo}")
    return out


# =====================================================================
# Projects section
# =====================================================================

PROJECT_HEADING = re.compile(
    r"^(?:(?:personal|academic|key|major|notable|selected|technical|side|college|live|open[- ]source)\s+)?"
    r"projects?(?:\s*(?:&|and|/)\s*[a-z ]{3,30})?(?:\s+(?:undertaken|worked on|highlights))?$",
    re.IGNORECASE,
)
OTHER_HEADINGS = {
    "experience", "work experience", "professional experience", "employment", "work history",
    "internships", "internship", "education", "academic background", "qualifications",
    "skills", "technical skills", "key skills", "core competencies", "tech stack", "technologies",
    "certifications", "certificates", "courses", "coursework", "achievements", "awards",
    "accomplishments", "honors", "summary", "professional summary", "objective",
    "career objective", "profile", "about me", "publications", "research", "interests",
    "hobbies", "languages", "extracurricular", "extracurricular activities", "activities",
    "positions of responsibility", "responsibilities", "volunteering", "references",
    "declaration", "contact", "personal details", "training", "leadership",
}
BULLET = re.compile(r"^\s*(?:[•●▪■◦○➢►▶➤✓✔*\-–—·]|\d+[.)])\s*")


def _heading_key(line: str) -> str:
    return re.sub(r"[^a-z&/ ]", "", line.lower()).strip()


def _is_other_heading(line: str) -> bool:
    key = _heading_key(line)
    return key in OTHER_HEADINGS and len(line) <= 40


def _is_projects_heading(line: str) -> bool:
    s = line.strip().rstrip(":").strip()
    return 0 < len(s) <= 45 and bool(PROJECT_HEADING.match(_heading_key(s)))


def _clean_title(line: str) -> str:
    s = BULLET.sub("", line).strip()
    s = re.split(r"\s+[|–—]\s+|\s+-\s+|\s*\(|\s*\|\s*", s)[0].strip(" :-–—|")
    s = re.sub(r"\s*(?:link|github|live|demo|source code)\s*$", "", s, flags=re.IGNORECASE).strip(" :-–—|")
    return s[:90]


def extract_projects(text: str) -> list[dict]:
    """Best-effort: returns [{"name": ..., "description": ...}] from the Projects section."""
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    start = None
    for i, line in enumerate(lines):
        if _is_projects_heading(line):
            start = i + 1
            break
    if start is None:
        return []

    section: list[str] = []
    for line in lines[start:]:
        if _is_other_heading(line):
            break
        if _is_projects_heading(line):
            continue  # a second "Projects" heading
        section.append(line)

    projects: list[dict] = []
    current: dict | None = None
    prev_complete = True  # previous line ended a bullet / entry

    for line in section:
        if not line:
            prev_complete = True
            continue
        is_bullet = bool(BULLET.match(line))
        is_title = (
            not is_bullet
            and len(line) <= 110
            and not line.endswith(".")
            and (prev_complete or current is None)
        )
        if is_title:
            name = _clean_title(line)
            if len(name) >= 3:
                current = {"name": name, "description": ""}
                projects.append(current)
                prev_complete = False
                continue
        if current is not None:
            body = BULLET.sub("", line)
            current["description"] = (current["description"] + " " + body).strip()[:300]
        prev_complete = line.endswith((".", "!", ":")) or is_bullet

    # drop duplicates and empty junk
    seen, out = set(), []
    for p in projects:
        key = p["name"].lower()
        if key in seen or not re.search(r"[A-Za-z]{3}", p["name"]):
            continue
        seen.add(key)
        out.append(p)
    return out[:MAX_PROJECTS]


# =====================================================================
# Entry point
# =====================================================================


def read_resume(url: str) -> dict:
    """Returns {"profiles": [...], "projects": [...], "repo_links": [...], "error": str | None}."""
    result = {"profiles": [], "projects": [], "repo_links": [], "error": None}
    try:
        data, content_type = download(url)
        text, links = extract_text_and_links(data, content_type, url)
    except ResumeError as e:
        result["error"] = str(e)
        return result

    if len(text.strip()) < 30 and not links:
        result["error"] = "No text found in the resume (scanned image PDF?)"
        return result

    result["profiles"] = find_github_profiles(text, links)
    result["projects"] = extract_projects(text)
    result["repo_links"] = find_repo_links(text, links)
    return result
