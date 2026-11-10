# Resume Extractor (no LLM) — Phase 1

Rule-based extraction of sections, technologies and per-project technology usage.

## Status
| Module | State |
|---|---|
| `app/section_detector.py` | implemented, tested |
| `app/technology_matcher.py` | not started |
| `app/project_extractor.py` | not started |

## Architecture
```
resume text ──► SectionDetector ──► {"projects": "...", ...} + offsets
                                         │ projects text
                  TechnologyMatcher ◄────┤ (injected into)
                                         ▼
                                  ProjectExtractor ──► per-project technologies + evidence
```
Dependencies point one way: `project_extractor` takes a matcher instance; neither
`section_detector` nor `technology_matcher` imports anything from the others.

## Running the tests
```
python -m unittest tests.test_section_detector -v   # unit tests
python tests/test_section_detector.py               # report: input / expected / actual / status, P/R/F1, timing
```
Modules in `app/` import each other as top-level modules (`from section_detector import ...`);
tests add `app/` to `sys.path`.

## section_detector notes
* Heading = whole normalised line equals a known alias (never substring matching).
* Bullet lines are never headings. "Technologies", "Tech Stack", "Tools" are intentionally not
  aliases, since `Technologies: Python, Django` is a normal project line.
* Boundary-only headings (Hobbies, Languages, References, ...) end the previous section but are
  not returned. Inline form (`Languages: Python, Java`) is not treated as a heading.
* Extend with `SectionDetector(extra_aliases={"skills": ["tech arsenal"]})`.
