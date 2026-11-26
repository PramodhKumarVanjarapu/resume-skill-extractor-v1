# Resume Extractor (no LLM) — Phase 1

## GitHub repository scheduling

Under **GitHub → Repository scan method**, choose **Git clone** or **GitHub API
(token)**. Git clone is the default and needs Git installed. API mode fetches the default branch
file tree and the selected manifests and source files through GitHub's API.
API mode reads the file tree with one REST request and the selected files in
batches of up to 40 per GraphQL request, so a repository usually costs 2 requests.
If GitHub refuses GraphQL (for example its separate quota is exhausted), the run
falls back to one REST request per file. Clone mode runs 8 blobless clones at once.

Use **Parallel GitHub API requests** in the sidebar to set the per-batch limit.
The default is 20, and the range is 1–32. This applies to repository listing in
both modes and to downloading files in API mode.

Configure one or more credentials in `.streamlit/secrets.toml` or in the deployed
app's Streamlit Secrets settings.

```toml
GITHUB_TOKENS = ["replace-with-primary-token", "replace-with-backup-token"]
```

A token entered in the sidebar takes priority, followed by `GITHUB_TOKENS`, a
single `GITHUB_TOKEN` in Streamlit secrets, and the `GITHUB_TOKEN` environment
variable. The app switches credentials when GitHub returns HTTP 401. When GitHub
rate-limits requests, the app pauses GitHub analysis until the retry window ends.
Tokens from the same GitHub account share a quota. [Streamlit secrets documentation](https://docs.streamlit.io/develop/concepts/connections/secrets-management) [GitHub rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)

Both modes use the same technology detectors and check the same cache before scanning.
Uncached, nonempty repositories are ordered from repository name, description,
topics, primary language and recent push metadata. These are scheduling hints,
not verified technologies. The initial scan budget defaults to 15 repositories
per candidate; the sidebar can change it or enable exhaustive analysis. Shared
repositories are scanned once. Deferred repositories remain unanalysed and are
reported separately from failed analysis. A caller of `analyzer.run_report` may
provide `needs_more_evidence(person, analysed_repositories)` to request further
deferred repositories after the initial pass, without copying screening rules
into the GitHub extractor.

### What a repository scan detects

Technologies come from the implementation, never from the README or the
description: dependency manifests (`package.json`, `requirements.txt`,
`pyproject.toml`, `pom.xml`, `*.csproj`, Docker/compose image references),
imports in Python and JavaScript/TypeScript/Vue/Svelte source, config files that
exist only when a tool is set up (`angular.json`, `next.config.js`, `manage.py`,
...), server code (`app.listen`, `http.createServer`, Express routers) and file
extensions. `node_modules`, `vendor`, build output, lock files, minified and
generated files are ignored. Names are normalised through
`dictionaries/technology_aliases.json` (`NodeJS` -> `Node.js`).

Each repository result records `technology_usage` per technology: `direct`
(imported, configured or implemented), `declared` (dependency only) or
`inferred` (GitHub's language label only). Peer dependencies and the entries of
`pip freeze` / `conda env export` dumps are `indirect_technologies` and are left
out of `technologies` unless the source imports them. Evidence is `strong` when
two independent sources agree (for example a dependency and an import).

Project categories need implementation evidence (`implementation` in the result):
**Frontend** requires React, Angular, Vue.js, Svelte, Next.js or Nuxt.js to be
imported, configured or backed by component files; a backend stack requires its
framework in source code (or a dependency plus server files); **Full Stack** /
MERN / MEAN require both. Languages alone never produce a category, and
**Static Website** requires HTML files and no framework.

Repositories that are not projects are kept in the results with `excluded` and
`exclusion_reason`, and left out of the tech stack, categories and evidence:
the profile README repository (never scanned), empty repositories, repositories
without source code, single-file scripts without libraries, DSA/coding-practice
collections, tutorials/assignments, and near-identical copies of another of the
candidate's repositories (`duplicate_of:<repo>`, compared by source blob hashes).
Practice- or tutorial-like names only move a repository to the end of the scan
order; they exclude it only when the scan finds no application implementation
and no libraries, so a "dsa-visualizer" built with React still counts. Stars,
commits and contributors are never used. Private, deleted (HTTP 404) or blocked
repositories are reported as failures and are not cached.

The cache (`tech-cache.json`) is saved even when a run is interrupted, and saving
drops entries from older detection versions and superseded pushes.

The sidebar's maximum *listed* repositories is an earlier API paging cap. Raise
it if relevant repositories may be outside the newest listed pages. A deferred
repository can still contain a technology absent from analyzed repositories.

Logs use INFO by default, with console and rotating `logs/github_extractor.log`
output. Set `GITHUB_EXTRACTOR_LOG_LEVEL=DEBUG` before launching Streamlit for
repository scoring and cache details. Logs use hashed candidate identifiers and
omit token values, phone numbers and resume content. Authentication and rate-limit
warnings identify the token by its configuration source, such as
`GITHUB_TOKENS entry #2`. Batch summaries distinguish
cache reuse, metadata deferral, empty repositories, failures, and clone avoidance.

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
