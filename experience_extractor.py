"""Conservative job-level technology attribution within detected resume sections."""

from __future__ import annotations

import re
from technology_matcher import TechnologyMatcher

_ROLE = re.compile(r"\b(developer|engineer|intern|analyst|architect|consultant|programmer|scientist|manager)\b", re.I)
_DATE = re.compile(r"\b(?:(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?(20\d{2})\b", re.I)
_END = re.compile(r"\b(present|current|ongoing)\b", re.I)
_LABEL = re.compile(r"^\s*(tech(?:nologies)?(?: stack)?|tools|environment)\s*:", re.I)


def _dates(line: str) -> tuple[str | None, str | None]:
    found = list(_DATE.finditer(line))
    def value(match):
        month = match.group(1)
        number = next((i for i, label in enumerate(
            ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)
            if month and month.lower().startswith(label)), None)
        return f"{match.group(2)}-{number:02d}" if number else match.group(2)
    return (value(found[0]) if found else None,
            None if _END.search(line) else value(found[1]) if len(found) > 1 else None)


def extract_experiences(section_text: str, matcher: TechnologyMatcher, section: str = "experience") -> dict:
    """Split role headings into entries; flag text that cannot be attributed safely."""
    lines = section_text.splitlines()
    starts = [i for i, line in enumerate(lines) if _ROLE.search(line) and not line.lstrip().startswith(("-", "•"))
              and len(line.split()) <= 12 and not line.rstrip().endswith(".")]
    if not starts:
        return {"experiences": [], "section": section, "attribution_status": "unavailable" if section_text.strip() else "empty",
                "warnings": ["No reliable job boundaries"] if section_text.strip() else []}
    output = []
    warnings = []
    for index, start in enumerate(starts):
        block = lines[start:starts[index + 1] if index + 1 < len(starts) else len(lines)]
        role = block[0].strip()
        company = None
        for line in block[1:4]:
            clean = line.strip()
            if clean and not _DATE.search(clean) and not clean.startswith(("-", "•")) and not _LABEL.search(clean):
                if len(clean.split()) <= 7 and not clean.endswith("."):
                    company = clean
                break
        date_line = next((line for line in block[:5] if _DATE.search(line)), "")
        start_date, end_date = _dates(date_line)
        if not company or not start_date:
            warnings.append(f"Entry {index + 1}: company or date unavailable")
        body = "\n".join(block)
        evidence = []
        for match in matcher.extract(body):
            line = next((line.strip() for line in block if match.matched_text in line), match.evidence)
            strength = ("medium" if _LABEL.search(line) else "strong" if line.startswith(("-", "•"))
                        or re.search(r"\b(used|built|developed|implemented|integrated|deployed)\b", line, re.I)
                        else "weak")
            evidence.append({"technology": match.name, "sentence": match.evidence, "strength": strength})
        output.append({"company": company, "role": role, "start_date": start_date, "end_date": end_date,
                       "technologies": [item["technology"] for item in evidence], "evidence": evidence,
                       "attribution_confidence": "medium" if company and start_date else "low"})
    return {"experiences": output, "section": section, "attribution_status": "available", "warnings": warnings}
