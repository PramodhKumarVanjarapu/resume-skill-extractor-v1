# Repository guide for AI agents

## Purpose

This is a no-LLM resume-processing tool. It extracts text from candidate
resumes, detects resume sections, identifies technologies from local
dictionaries, splits the Projects section into individual projects, optionally
enriches candidates from GitHub, and exports/filter results in a Streamlit UI.

The code intentionally favours deterministic, explainable rules over model
inference. Preserve evidence, offsets, warnings, and review flags when making
changes to extraction logic.

## Layout

The current code is **flat at the repository root**; there is no `app/`
directory.

| Path | Role |
| --- | --- |
| `app.py` | Streamlit entry point and end-to-end resume batch pipeline. |
| `section_detector.py` | Rule-based section headings and offset-aware results. |
| `technology_matcher.py` | Dictionary-driven matching, ambiguity checks, and evidence spans. |
| `project_extractor.py` | Project boundary detection and per-project technology attribution. |
| `resume.py`, `pdf_parse.py` | Downloading and PDF/DOCX/text extraction helpers. |
| `github_profile.py`, `github_enricher.py`, `analyzer.py` | GitHub-profile discovery and repository technology analysis. |
| `profile_filter.py`, `filter_panel.py` | DataFrame filtering and Streamlit filter controls. |
| `xlsx_export.py` | Formatted XLSX export. |
| `dictionaries/` | Technology master list and alias mapping; edit these rather than hard-coding ordinary technology additions. |
| `test_section_detector.py` | Table-driven `unittest` coverage for section detection. |

## Run and verify

Create/use the local virtual environment and install dependencies:

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The core extractor modules use only the Python standard library. The UI and
exports additionally use the packages listed in `requirements.txt`.

Run the current test file directly after correcting its stale import path if
needed:

```powershell
python test_section_detector.py
```

Important: `README.md`, comments, and `test_section_detector.py` still refer
to an old `app/` directory and `tests/` package. The repository now stores the
modules and test at its root. `test_section_detector.py` currently inserts the
parent's nonexistent `app/` directory into `sys.path`, so it may fail to import
`section_detector` unless the repository root is already on `PYTHONPATH` or
that path setup is fixed. Keep documentation and tests aligned with the flat
layout whenever touching this area.

## Engineering conventions

- Use Python type hints and keep public APIs documented through module
  docstrings and `__all__` where present.
- Do not introduce an LLM/ML dependency for core matching or extraction.
- `SectionResult`, `ProjectExtractionResult`, and matching data classes expose
  richer metadata than their simple dictionary forms. Maintain backwards-
  compatible dictionary behaviour.
- Character offsets must continue to refer to the original, unmodified resume
  text.
- Prefer changing `dictionaries/technologies_master.json` and
  `dictionaries/technology_aliases.json` for vocabulary/alias changes. Verify
  ambiguous short terms (for example `C`, `R`, `Go`) remain context-sensitive.
- `ProjectExtractor` receives a `TechnologyMatcher` instance; avoid reversing
  this dependency or importing higher-level UI modules into core modules.
- Preserve error-stage reporting and nonfatal per-resume failures in batch
  processing; one bad file should not abort a candidate batch.

## Data, security, and generated files

- Resume inputs and exports can contain phone numbers and other personal data.
  Do not commit raw resumes, exports, caches, tokens, or debug samples with
  personal information.
- GitHub enrichment needs network access and may use `GITHUB_TOKEN`; keep it
  in environment/Streamlit secrets, never in source or commits.
- `tech-cache.json` is a large runtime cache and is ignored. Do not modify or
  commit it as part of routine code changes.
- PDF/DOCX parsing accepts untrusted files; retain size/page limits and error
  handling when modifying parsers.

## Before handing off a change

1. Run the narrowest relevant test or a small, synthetic local example.
2. For extraction changes, check both extracted values and metadata/evidence
   offsets.
3. For UI or export changes, launch `streamlit run app.py` when dependencies
   are available and validate the affected flow.
4. State any behaviour intentionally changed and any unrun verification.
