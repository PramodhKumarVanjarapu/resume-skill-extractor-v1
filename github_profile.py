"""Find the candidate's OWN GitHub profile in a resume (no network, no ML).

The question is not "which GitHub usernames appear?" but "which one is this person's account?".
A resume mentions several: the candidate's profile, links to their projects, tutorials they cloned,
libraries they used, a team project owned by an organisation, placeholder text left over from a template.
This module collects every mention as *evidence*, scores each username, and ranks them.

Evidence, strongest first (weights are the constants ``W_*`` below, so every decision is explainable):

* a profile URL ``github.com/<user>`` that is a real hyperlink            (not just visible text)
* a profile URL typed as text
* a bare "GitHub: <user>" label, ``<user>.github.io``, ``@<user>`` next to the word GitHub
* repository URLs ``github.com/<user>/<repo>``: weak on their own, but several repositories under the same
  owner point at the candidate's account (capped, so a library link can never beat a profile link)
* bonuses: a "GitHub" label right before the link, being in the header block, and a username that resembles the
  candidate's name (taken from the resume's first line or the file name)

What is never a candidate (kept in ``ProfileSearch.rejected`` with the reason, so nothing disappears silently):

* template placeholders (``your-id``, ``yourusername``, ``firstlast``, ...)
* reserved github.com paths (``features``, ``orgs``, ``sponsors``, ...)
* well-known organisations / products that resumes link to (``mysql``, ``tensorflow``, ``facebook``, ...)
* text that is not a valid GitHub username. ``github.com/john_doe`` is NOT ``john``: usernames cannot contain
  ``_`` or ``.``, and cutting the text at that point would silently name a different person.

Also handled: usernames the PDF text extractor broke across a line, a space after the slash, ``gist.github.com``,
``www.``, upper / lower case, trailing punctuation, and zero-width characters.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote, urlparse

__all__ = [
    "Candidate", "ProfileSearch", "find_profile_candidates", "guess_name", "is_valid_username",
    "KNOWN_NON_PERSONAL", "RESERVED_PATHS",
]

MAX_CANDIDATES = 5

# ---- evidence weights -------------------------------------------------------
W_LINK_PROFILE = 7.0       # hyperlink target github.com/<user> (always beats the same account typed as text + header)
W_TEXT_PROFILE = 5.0       # visible text github.com/<user>
W_LINK_REPO = 2.0          # each hyperlink github.com/<user>/<repo>  (capped)
W_TEXT_REPO = 1.5          # each visible github.com/<user>/<repo>    (capped)
CAP_REPO = 6.0
W_GIST = 2.0
W_LABEL_ONLY = 3.5         # "GitHub: <user>" without a URL
W_HANDLE = 2.0             # "@user" within a few words of the word GitHub
W_PAGES = 1.5              # <user>.github.io
W_LABEL = 2.0              # the word GitHub / Git Hub right before the link
W_HEADER = 1.5             # found in the header block (contact line)
W_NAME_MAX = 3.0           # username resembles the candidate's name
P_ORGISH = -2.0            # looks like a team / lab / company account

# a candidate this strong, clearly ahead of the next one, is "high" confidence
HIGH_SCORE, HIGH_MARGIN, MEDIUM_SCORE = 6.0, 2.5, 3.0

# ---- what is not a person -----------------------------------------------------
RESERVED_PATHS = frozenset("""
about account apps assets blog codespaces collections contact copilot customer-stories dashboard discussions
education enterprise enterprises events explore features gist github issues join login logout marketplace mobile
new news notifications organizations orgs partners password_reset premium-support pricing projects pulls readme
resources search security settings signup site sitemap solutions sponsors stars support team topics trending users
watching
""".split())

_PLACEHOLDERS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"(your|my)[-_]?(github|git)?[-_]?(user)?[-_]?(name|id|username|handle|profile|account)",
        r"(first|firstname)[-_]?(last|lastname|surname)",
        r"(github|git)[-_]?(user)?[-_]?(name|id|username|handle|profile|account)?",
        r"user(name)?\d*", r"example\d*", r"john[-_]?doe\d*", r"janedoe\d*", r"name[-_]?surname", r"xxx+",
    )
]

KNOWN_NON_PERSONAL = frozenset("""
google microsoft facebook meta amazon aws apple ibm oracle apache mozilla gitlab nodejs npm expressjs reactjs
facebookresearch vuejs angular sveltejs vercel nextjs tailwindlabs twbs django pallets tiangolo psf python pypa
pytorch tensorflow keras-team scikit-learn numpy pandas-dev matplotlib jupyter ipython huggingface openai anthropics
langchain-ai langchain kubernetes docker hashicorp ansible jenkinsci mysql postgres postgresql mongodb redis elastic
spring-projects springframework eclipse jetbrains flutter dart-lang android kotlinlang golang rust-lang bitnami
freecodecamp ohmyzsh nvidia intel samsung adobe netflix airbnb uber twitter spotify shopify stripe paypal
linkedin instagram youtube whatsapp tiktok snapchat pinterest reddit discord slackhq telegram signalapp atlassian
dropbox yahoo baidu alibaba tencent bytedance huawei cisco dell hp hpe vmware redhat canonical sap salesforce
googlecloudplatform google-research googlesamples firebase aws-samples awslabs azure azure-samples microsoftdocs
github githubuniverse gnome kde archlinux ubuntu debian freebsd llvm denoland oven-sh cncf openstack hyperledger
ethereum bitcoin solana-labs onnx apple-oss-distributions
zomato swiggy flipkart flipkart-incubator paytm phonepe razorpay infosys tcs wipro hcl hcltech accenture capgemini
cognizant deloitte techmahindra tech-mahindra oyo byjus freshworks zoho cred meesho nxtwave ccbp
codecademy coursera udacity edx ossu thealgorithms trekhleb kamranahmedse public-apis donnemartin sindresorhus
jwasham awesome-selfhosted
""".split())

_ORGISH = ("team", "labs", "official", "inc", "corp", "solutions", "technologies", "softwares", "studios", "community",
           "foundation", "university", "college", "academy", "society", "ventures", "systems")

_USERNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}")
_PROFILE_TABS = frozenset({"repositories", "followers", "following", "stars", "projects", "packages", "achievements",
                           "sponsoring", "overview"})
_NOT_A_USERNAME_WORDS = frozenset({
    "link", "profile", "url", "portfolio", "account", "page", "repo", "repository", "repositories", "actions", "pages",
    "projects", "project", "and", "com", "io", "the", "at", "id", "username", "handle", "here", "name",
    # other profiles / contact items that follow "GitHub /" or "GitHub |" in a contact line
    "linkedin", "twitter", "leetcode", "hackerrank", "kaggle", "medium", "email", "gmail", "behance", "dribbble",
    "instagram", "youtube", "codechef", "codeforces", "geeksforgeeks", "stackoverflow", "website", "web", "site",
    "phone", "mobile", "contact", "resume", "cv",
    # words that follow "GitHub -" in a description line ("GitHub - Developed REST APIs", "GitHub: version control")
    "developed", "used", "using", "version", "control", "hosting", "collaboration", "source", "open", "managed",
    "maintained", "code", "copilot", "desktop", "cli", "flow", "workflow", "workflows", "repos",
})

_URL = re.compile(
    r"(?<![A-Za-z0-9.@-])(?:https?://)?(?:www\.)?(?P<host>gist\.github\.com|github\.com)/"
    r"(?P<user>[A-Za-z0-9][A-Za-z0-9_.-]*)(?P<rest>(?:/[^\s<>\"')\],;]*)?)",
    re.IGNORECASE,
)
_PAGES = re.compile(r"(?<![A-Za-z0-9.-])(?P<user>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\.github\.io\b", re.IGNORECASE)
_LABELLED = re.compile(r"\bgit\s?hub\b\s*[:\-\u2013\u2014|/]\s*@?(?P<user>[A-Za-z0-9][A-Za-z0-9_.-]*)", re.IGNORECASE)
_HANDLE = re.compile(r"(?<![\w.@])@(?P<user>[A-Za-z0-9][A-Za-z0-9-]*)(?![\w@-])")
_LABEL_BEFORE = re.compile(r"git\s?hub[^A-Za-z0-9]{0,12}$", re.IGNORECASE)
_SECTION_HEADING = re.compile(
    r"^\s*(?:professional\s+)?(?:summary|objective|education|skills|technical skills|projects?|experience|work experience|"
    r"internships?|certifications?|achievements?|profile)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    username: str
    score: float
    reasons: list[str]
    first_position: int = 0


@dataclass
class ProfileSearch:
    candidates: list[Candidate] = field(default_factory=list)        # best first
    rejected: list[tuple[str, str]] = field(default_factory=list)    # (text as found, why it was ignored)
    name_hint: str = ""

    @property
    def primary(self) -> str | None:
        return self.candidates[0].username if self.candidates else None

    @property
    def usernames(self) -> list[str]:
        return [c.username for c in self.candidates]

    @property
    def confidence(self) -> str:
        """"high" / "medium" / "low", or "" when nothing was found."""
        if not self.candidates:
            return ""
        top = self.candidates[0].score
        runner_up = self.candidates[1].score if len(self.candidates) > 1 else None
        if top >= HIGH_SCORE and (runner_up is None or top - runner_up >= HIGH_MARGIN):
            return "high"
        return "medium" if top >= MEDIUM_SCORE else "low"

    def describe(self) -> str:
        """One line for the diagnostics column: every candidate with score and reasons, then what was ignored."""
        parts = [f"{c.username} [{c.score:.1f}: {', '.join(c.reasons)}]" for c in self.candidates]
        parts += [f"{raw} (ignored: {why})" for raw, why in self.rejected]
        return " | ".join(parts)


@dataclass
class _Evidence:
    kind: str                 # profile | repo | gist | label | pages | handle
    source: str               # link | text
    position: int
    labelled: bool = False
    in_header: bool = False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def is_valid_username(name: str) -> bool:
    """GitHub rule: letters, digits and single hyphens, no leading / trailing hyphen, at most 39 characters."""
    return bool(_USERNAME.fullmatch(name))


def _placeholder(name: str) -> bool:
    return any(p.fullmatch(name) for p in _PLACEHOLDERS)


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", t)
    t = re.sub(r"(github\.com/)\s+", r"\1", t, flags=re.IGNORECASE)                  # "github.com/ user"
    # a username that the PDF extractor split after a hyphen: "github.com/nitheesh-\nkumar"
    t = re.sub(r"(github\.com/[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-)[ \t]*\r?\n[ \t]*(?=[A-Za-z0-9])", r"\1", t, flags=re.IGNORECASE)
    return t


def _header_end(text: str) -> int:
    match = _SECTION_HEADING.search(text)
    return min(match.start() if match else 700, 1500)


def _kind_of(host: str, rest: str) -> str:
    if host.lower().startswith("gist."):
        return "gist"
    path = re.split(r"[?#]", rest, maxsplit=1)[0].strip("/")
    first = path.split("/", 1)[0].lower() if path else ""
    return "profile" if not first or first in _PROFILE_TABS else "repo"


_NAME_STOP = frozenset("""
resume cv curriculum vitae profile mr mrs ms dr page final updated new copy latest docx pdf engineer developer student
fresher software full stack
""".split())


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z]{3,}", unicodedata.normalize("NFKD", name or "").lower()) if t not in _NAME_STOP]


def guess_name(text: str, link: str = "") -> str:
    """The candidate's name: the first line of the resume that looks like a name, else the words of the file name
    (``1793_GOPI_KRISHNA_THEEGALA_b7285353-....pdf`` -> "GOPI KRISHNA THEEGALA"). "" when neither works."""
    for line in (text or "").splitlines()[:8]:
        words = line.strip().split()
        if (2 <= len(words) <= 4 and 5 <= len(line.strip()) <= 40 and "@" not in line
                and all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", w) for w in words)
                and not any(w.lower().strip(".") in _NAME_STOP for w in words)
                and not _SECTION_HEADING.fullmatch(line.strip())):
            return line.strip()
    if link:
        stem = unquote(urlparse(link).path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
        words = [w for w in re.split(r"[_\-\s.]+", stem) if re.fullmatch(r"[A-Za-z]{2,}", w) and w.lower() not in _NAME_STOP]
        if 1 <= len(words) <= 4:
            return " ".join(words)
    return ""


def _name_bonus(username: str, tokens: list[str]) -> float:
    letters = re.sub(r"[^a-z]", "", username.lower())
    if not tokens or len(letters) < 3:
        return 0.0
    joined = "".join(tokens)
    hit = sum(len(t) for t in tokens if t in letters) / len(joined)
    inside = len(letters) / len(joined) if len(letters) >= 5 and letters in joined else 0.0
    return round(W_NAME_MAX * min(max(hit, inside), 1.0), 2)


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #

def find_profile_candidates(
    text: str,
    links: Iterable[str] = (),
    name_hint: str = "",
    extra_blocked: Iterable[str] = (),
) -> ProfileSearch:
    """Rank the GitHub accounts mentioned in a resume.

    ``links`` are hyperlink targets (PDF annotations, DOCX relationships, HTML anchors), ``name_hint`` the
    candidate's name (see ``guess_name``), ``extra_blocked`` more organisation / product names to ignore.
    Returns a ``ProfileSearch``: ranked ``candidates`` (best first, at most ``MAX_CANDIDATES``) and everything
    that was ``rejected`` with its reason.
    """
    blocked = KNOWN_NON_PERSONAL | {b.lower() for b in extra_blocked}
    flat = _normalise(text)
    header_end = _header_end(flat)
    evidence: dict[str, list[_Evidence]] = {}
    shown: dict[str, str] = {}
    rejected: list[tuple[str, str]] = []

    def reject(raw: str, why: str) -> None:
        if (raw, why) not in rejected:
            rejected.append((raw, why))

    def accept(token: str, ev: _Evidence) -> None:
        low = token.lower()
        shown.setdefault(low, token)
        evidence.setdefault(low, []).append(ev)

    def check(token: str) -> str | None:
        """Why ``token`` cannot be a personal profile, or None."""
        low = token.lower()
        if low in RESERVED_PATHS:
            return "reserved github.com path"
        if not is_valid_username(token):
            return "not a valid GitHub username (letters, digits and single hyphens only)"
        if _placeholder(token):
            return "template placeholder"
        if low in blocked:
            return "well-known organisation or product, not a person"
        return None

    def scan_urls(source_text: str, source: str, base: int, labelled_check: bool) -> None:
        for m in _URL.finditer(source_text):
            raw = m.group("user")
            token = raw.rstrip(".")
            why = check(token)
            if why:
                reject(token if token else raw, why)
                continue
            before = source_text[max(0, m.start() - 25):m.start()]
            accept(token, _Evidence(
                kind=_kind_of(m.group("host"), m.group("rest") or ""), source=source, position=base + m.start(),
                labelled=labelled_check and bool(_LABEL_BEFORE.search(before)),
                in_header=source == "text" and m.start() < header_end,
            ))

    for link in links:
        if isinstance(link, str):
            scan_urls(_normalise(link), "link", -1_000_000, False)
    scan_urls(flat, "text", 0, True)

    for m in _PAGES.finditer(flat):
        token = m.group("user")
        if token.lower() in _NOT_A_USERNAME_WORDS:
            continue
        why = check(token)
        if why:
            reject(token, why)
        else:
            accept(token, _Evidence("pages", "text", m.start(), in_header=m.start() < header_end))
    for m in _LABELLED.finditer(flat):
        raw = m.group("user")
        token = raw.rstrip(".")
        if "." in token or token.lower() in _NOT_A_USERNAME_WORDS or token.lower().startswith(("github", "www", "http")):
            continue                                   # "GitHub: github.com/x" is read by the URL pattern, not here
        why = check(token)
        if not why and m.start() >= header_end and not re.search(r"[\d-]", token):
            why = "text after the word GitHub, outside the header, that does not look like a username"
        if why:
            reject(token, why)
        else:
            accept(token, _Evidence("label", "text", m.start(), labelled=True, in_header=m.start() < header_end))
    for m in _HANDLE.finditer(flat):
        window = flat[max(0, m.start() - 40):m.start()]
        token = m.group("user")
        if not re.search(r"git\s?hub", window, re.IGNORECASE) or token.lower() in _NOT_A_USERNAME_WORDS:
            continue
        why = check(token)
        if not why and m.start() >= header_end and not re.search(r"[\d-]", token):
            why = "@name outside the header that does not look like a username"
        if why:
            reject(token, why)
        else:
            accept(token, _Evidence("handle", "text", m.start(), labelled=True, in_header=m.start() < header_end))

    tokens = _name_tokens(name_hint)
    candidates: list[Candidate] = []
    for low, items in evidence.items():
        name = shown[low]
        score, reasons = 0.0, []
        if any(e.kind == "profile" and e.source == "link" for e in items):
            score += W_LINK_PROFILE
            reasons.append("profile hyperlink")
        elif any(e.kind == "profile" for e in items):
            score += W_TEXT_PROFILE
            reasons.append("profile URL in the text")
        if any(e.kind == "label" for e in items):
            score += W_LABEL_ONLY
            reasons.append("'GitHub: name' label")
        if any(e.kind == "handle" for e in items):
            score += W_HANDLE
            reasons.append("@name next to GitHub")
        if any(e.kind == "gist" for e in items):
            score += W_GIST
            reasons.append("gist link")
        if any(e.kind == "pages" for e in items):
            score += W_PAGES
            reasons.append("github.io page")
        repo_links = sum(1 for e in items if e.kind == "repo" and e.source == "link")
        repo_text = sum(1 for e in items if e.kind == "repo" and e.source == "text")
        if repo_links or repo_text:
            score += min(W_LINK_REPO * repo_links + W_TEXT_REPO * repo_text, CAP_REPO)
            reasons.append(f"{repo_links + repo_text} repository link(s)")
        if any(e.labelled for e in items):
            score += W_LABEL
            reasons.append("labelled GitHub")
        if any(e.in_header for e in items):
            score += W_HEADER
            reasons.append("in the header")
        bonus = _name_bonus(name, tokens)
        if bonus >= 0.5:
            score += bonus
            reasons.append("resembles the candidate's name")
        if any(word in low for word in _ORGISH):
            score += P_ORGISH
            reasons.append("looks like a team / company account")
        candidates.append(Candidate(name, round(score, 2), reasons, min(e.position for e in items)))

    candidates.sort(key=lambda c: (-c.score, c.first_position))
    return ProfileSearch(candidates[:MAX_CANDIDATES], rejected, name_hint)
