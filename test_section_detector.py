"""Tests for section_detector.py.

Run as unit tests:   python -m unittest tests.test_section_detector -v
                     (or ``pytest tests/test_section_detector.py``)
Run as a report:     python tests/test_section_detector.py
                     -> per-case Input / Expected / Actual / PASS-FAIL,
                        section-level precision/recall/F1 and timing.

Expected values below are written BY HAND from the intended behaviour; they
are never generated from the detector's own output.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from section_detector import SectionDetector, SectionResult  # noqa: E402

DETECTOR = SectionDetector()

SPEC_RESUME = """
TECHNICAL SKILLS
Python, Java, SQL, React

PROJECTS
E-commerce Application
Developed using React, Node.js and MongoDB.

PROFESSIONAL EXPERIENCE
Software Developer - ABC Technologies
Worked with Java and Spring Boot.

EDUCATION
B.Tech Computer Science
"""

# --------------------------------------------------------------------------- #
# Table-driven cases: (id, description, input, expected dict)
# --------------------------------------------------------------------------- #
CASES: list[dict] = [
    {
        "id": "spec_example",
        "desc": "Example from the specification",
        "input": SPEC_RESUME,
        "expected": {
            "skills": "Python, Java, SQL, React",
            "projects": "E-commerce Application\nDeveloped using React, Node.js and MongoDB.",
            "experience": "Software Developer - ABC Technologies\nWorked with Java and Spring Boot.",
            "education": "B.Tech Computer Science",
        },
    },
    {
        "id": "lowercase_headings",
        "desc": "All-lowercase headings",
        "input": "skills\npython, java\n\nprojects\nblog app\nbuilt with django\n\neducation\nb.sc physics",
        "expected": {
            "skills": "python, java",
            "projects": "blog app\nbuilt with django",
            "education": "b.sc physics",
        },
    },
    {
        "id": "title_case_headings",
        "desc": "Title-case headings incl. Professional Summary / Work Experience",
        "input": "Technical Skills\nPython\n\nProfessional Summary\nBackend engineer.\n\nWork Experience\nDev at X",
        "expected": {"skills": "Python", "summary": "Backend engineer.", "experience": "Dev at X"},
    },
    {
        "id": "numbered_headings",
        "desc": "1.  2)  III.  (4)  numbering styles",
        "input": "1. SKILLS\nPython\n2) Projects\nApp one\nIII. EDUCATION\nB.Tech\n(4) Certifications\nAWS CP",
        "expected": {
            "skills": "Python", "projects": "App one",
            "education": "B.Tech", "certifications": "AWS CP",
        },
    },
    {
        "id": "colon_headings",
        "desc": "Headings with trailing colon (and space before colon)",
        "input": "SKILLS:\nPython, Go\nPROJECTS :\nApp\nExperience:\nIntern at Y",
        "expected": {"skills": "Python, Go", "projects": "App", "experience": "Intern at Y"},
    },
    {
        "id": "inline_content",
        "desc": "Content on the same line as the heading",
        "input": "SKILLS: Python, Java, SQL\nEDUCATION: B.Tech CSE\nPROJECTS\nApp",
        "expected": {"skills": "Python, Java, SQL", "education": "B.Tech CSE", "projects": "App"},
    },
    {
        "id": "academic_personal_projects",
        "desc": "Academic + Personal Projects both map to 'projects'",
        "input": "ACADEMIC PROJECTS\nProject A\nPERSONAL PROJECTS\nProject B",
        "expected": {"projects": "Project A\n\nProject B"},
    },
    {
        "id": "experience_variants",
        "desc": "Work Experience + Employment History",
        "input": "WORK EXPERIENCE\nDev at A\n\nEMPLOYMENT HISTORY\nQA at B",
        "expected": {"experience": "Dev at A\n\nQA at B"},
    },
    {
        "id": "internships_separate",
        "desc": "Internships kept as its own key",
        "input": "INTERNSHIPS\nIntern at Z\nEDUCATION\nMBA",
        "expected": {"internships": "Intern at Z", "education": "MBA"},
    },
    {
        "id": "certs_achievements_summary",
        "desc": "Certifications / Achievements / Summary",
        "input": "CERTIFICATIONS\nAWS Cloud Practitioner\nACHIEVEMENTS\nWon hackathon\nSUMMARY\nFresher.",
        "expected": {
            "certifications": "AWS Cloud Practitioner",
            "achievements": "Won hackathon", "summary": "Fresher.",
        },
    },
    {
        "id": "missing_sections",
        "desc": "Only one section present",
        "input": "EDUCATION\nB.Tech CSE",
        "expected": {"education": "B.Tech CSE"},
    },
    {
        "id": "description_not_heading",
        "desc": "Project names/sentences that merely START with a heading word",
        "input": (
            "PROJECTS\nSkills Assessment Platform\n"
            "Experience in Python and Java was used to build it.\n"
            "Education Portal\nBuilt a portal.\n\nEDUCATION\nB.Tech"
        ),
        "expected": {
            "projects": (
                "Skills Assessment Platform\n"
                "Experience in Python and Java was used to build it.\n"
                "Education Portal\nBuilt a portal."
            ),
            "education": "B.Tech",
        },
    },
    {
        "id": "technologies_line_not_heading",
        "desc": "'Technologies:' / 'Tools & Technologies:' inside a project",
        "input": (
            "PROJECTS\nChat App\nTechnologies: Python, Django, PostgreSQL\n"
            "Tools & Technologies: Git, Docker\nEDUCATION\nMCA"
        ),
        "expected": {
            "projects": (
                "Chat App\nTechnologies: Python, Django, PostgreSQL\n"
                "Tools & Technologies: Git, Docker"
            ),
            "education": "MCA",
        },
    },
    {
        "id": "bullet_lines_not_headings",
        "desc": "Bulleted lines equal to a heading word are content",
        "input": "EXPERIENCE\nDeveloper at X\n- Skills: Python\n• Projects\nEDUCATION\nB.E.",
        "expected": {
            "experience": "Developer at X\n- Skills: Python\n• Projects",
            "education": "B.E.",
        },
    },
    {
        "id": "markdown_headings",
        "desc": "## Projects, **Education**, # Skills",
        "input": "## Projects\nApp\n**Education**\nB.Tech\n# Skills\nPython",
        "expected": {"projects": "App", "education": "B.Tech", "skills": "Python"},
    },
    {
        "id": "decorated_headings",
        "desc": "=== SKILLS ===, --- EXPERIENCE ---, ____ EDUCATION ____",
        "input": "=== SKILLS ===\nPython\n--- EXPERIENCE ---\nDev\n________ EDUCATION ________\nB.Tech",
        "expected": {"skills": "Python", "experience": "Dev", "education": "B.Tech"},
    },
    {
        "id": "spaced_letters",
        "desc": "'P R O J E C T S' letter-spaced headings",
        "input": "P R O J E C T S\nApp\nE D U C A T I O N\nB.Tech",
        "expected": {"projects": "App", "education": "B.Tech"},
    },
    {
        "id": "crlf_line_endings",
        "desc": "Windows line endings",
        "input": "SKILLS\r\nPython\r\n\r\nPROJECTS\r\nApp\r\n",
        "expected": {"skills": "Python", "projects": "App"},
    },
    {
        "id": "boundary_only_hobbies",
        "desc": "Unlisted headings (Hobbies/Languages) must not leak into Education",
        "input": "EDUCATION\nB.Tech\nHOBBIES\nChess, reading\nLANGUAGES\nEnglish",
        "expected": {"education": "B.Tech"},
    },
    {
        "id": "empty_input",
        "desc": "Empty string",
        "input": "",
        "expected": {},
    },
    {
        "id": "whitespace_only",
        "desc": "Whitespace-only string",
        "input": "   \n\n  \t\n",
        "expected": {},
    },
    {
        "id": "no_headings",
        "desc": "Resume text with no recognisable headings",
        "input": "John Doe\njohn@x.com\nBackend developer with Java experience",
        "expected": {},
    },
    {
        "id": "indented_headings",
        "desc": "Leading/trailing spaces and tabs around headings",
        "input": "   SKILLS   \n  Python, Java\n\n\tPROJECTS\n\tApp",
        "expected": {"skills": "Python, Java", "projects": "App"},
    },
    {
        "id": "duplicate_heading_merged",
        "desc": "Same section appearing twice is merged",
        "input": "SKILLS\nPython\nEDUCATION\nB.Tech\nSKILLS\nJava",
        "expected": {"skills": "Python\n\nJava", "education": "B.Tech"},
    },
    {
        "id": "ampersand_and_slash",
        "desc": "'Skills & Tools' and 'Awards / Honors'",
        "input": "SKILLS & TOOLS\nGit\nAWARDS / HONORS\nTopper",
        "expected": {"skills": "Git", "achievements": "Topper"},
    },
    {
        "id": "nbsp_preserved",
        "desc": "NBSP in heading is tolerated; NBSP in content is preserved",
        "input": "PROJECTS\u00a0\nApp\u00a0one\nEDUCATION\nB.Tech",
        "expected": {"projects": "App\u00a0one", "education": "B.Tech"},
    },
    {
        "id": "different_order",
        "desc": "Sections in non-standard order",
        "input": "EDUCATION\nB.Tech\nPROJECTS\nApp\nSKILLS\nPython",
        "expected": {"education": "B.Tech", "projects": "App", "skills": "Python"},
    },
    {
        "id": "inline_projects",
        "desc": "Projects: with inline content",
        "input": "Projects: Chat App, Blog Engine\nSkills: Go",
        "expected": {"projects": "Chat App, Blog Engine", "skills": "Go"},
    },
    {
        "id": "career_objective_profile",
        "desc": "Career Objective + Profile -> summary",
        "input": "CAREER OBJECTIVE\nSeeking a role.\n\nPROFILE\nHello.",
        "expected": {"summary": "Seeking a role.\n\nHello."},
    },
    {
        "id": "skills_block_with_languages_line",
        "desc": "'Languages: Python, Java' inside Skills is content, not a boundary",
        "input": (
            "TECHNICAL SKILLS\nLanguages: Python, Java\nFrameworks: React\n"
            "Interests: Chess\nEDUCATION\nB.Tech"
        ),
        "expected": {
            "skills": "Languages: Python, Java\nFrameworks: React\nInterests: Chess",
            "education": "B.Tech",
        },
    },
    {
        "id": "standalone_languages_still_boundary",
        "desc": "'Languages:' alone on its line still ends the previous section",
        "input": "EDUCATION\nB.Tech\nLanguages:\nEnglish, Hindi",
        "expected": {"education": "B.Tech"},
    },
    {
        "id": "sentence_starting_with_projects",
        "desc": "Long sentence beginning with a heading word is content",
        "input": "SKILLS\nPython\nProjects that I have worked on are listed below\nJava",
        "expected": {"skills": "Python\nProjects that I have worked on are listed below\nJava"},
    },
]


def _run(case: dict) -> dict:
    return dict(DETECTOR.detect(case["input"]))


class TestSectionDetectorCases(unittest.TestCase):
    def test_table_driven_cases(self):
        self.assertGreaterEqual(len(CASES), 20)
        for case in CASES:
            with self.subTest(case=case["id"]):
                self.assertEqual(_run(case), case["expected"])

    def test_offsets_slice_back_to_original_text(self):
        for case in CASES:
            with self.subTest(case=case["id"]):
                src = case["input"]
                result = DETECTOR.detect(src)
                prev_end = 0
                for s in result.metadata:
                    self.assertEqual(src[s.heading_start : s.heading_end], s.heading)
                    self.assertEqual(src[s.start : s.end], s.text)
                    self.assertLessEqual(s.heading_end, s.start if s.text else s.heading_end)
                    self.assertGreaterEqual(s.heading_start, prev_end)
                    prev_end = s.end


class TestSectionMetadata(unittest.TestCase):
    def test_spec_example_metadata(self):
        r = DETECTOR.detect(SPEC_RESUME)
        self.assertEqual([s.key for s in r.metadata], ["skills", "projects", "experience", "education"])
        first = r.metadata[0]
        self.assertEqual(first.heading, "TECHNICAL SKILLS")
        self.assertEqual(first.label, "technical skills")
        self.assertEqual(SPEC_RESUME[first.heading_start : first.heading_end], "TECHNICAL SKILLS")
        self.assertFalse(first.inline)

    def test_original_heading_preserved_with_decoration(self):
        r = DETECTOR.detect("## 2) Technical  Skills:\nPython")
        self.assertEqual(r.metadata[0].heading, "## 2) Technical  Skills:")
        self.assertEqual(r.metadata[0].label, "technical skills")

    def test_subtype_visible_for_merged_projects(self):
        r = DETECTOR.detect("ACADEMIC PROJECTS\nA\nPERSONAL PROJECTS\nB")
        labels = [s.label for s in r.get_sections("projects")]
        self.assertEqual(labels, ["academic projects", "personal projects"])
        self.assertIn("merged_multiple_sections:projects", r.warnings)

    def test_inline_flag(self):
        r = DETECTOR.detect("SKILLS: Python")
        self.assertTrue(r.metadata[0].inline)
        self.assertEqual(r.metadata[0].heading, "SKILLS:")
        self.assertEqual(r.metadata[0].text, "Python")

    def test_boundary_sections_in_metadata_but_not_dict(self):
        r = DETECTOR.detect("EDUCATION\nB.Tech\nHOBBIES\nChess")
        self.assertEqual(dict(r), {"education": "B.Tech"})
        other = [s for s in r.metadata if not s.target]
        self.assertEqual(len(other), 1)
        self.assertEqual((other[0].key, other[0].text), ("other", "Chess"))

    def test_preamble(self):
        r = DETECTOR.detect("RAHUL SHARMA\nrahul@x.com\n\nSKILLS\nPython")
        self.assertEqual(r.preamble, "RAHUL SHARMA\nrahul@x.com")
        self.assertEqual(dict(r), {"skills": "Python"})

    def test_internal_formatting_preserved(self):
        src = "PROJECTS\nApp\n    - bullet one\n    - bullet two\n"
        r = DETECTOR.detect(src)
        self.assertEqual(r["projects"], "App\n    - bullet one\n    - bullet two")

    def test_warnings(self):
        self.assertIn("empty_section:skills", DETECTOR.detect("SKILLS\nPROJECTS\nApp").warnings)
        self.assertIn("no_sections_detected", DETECTOR.detect("just some text").warnings)
        self.assertEqual(DETECTOR.detect("").warnings, [])

    def test_to_dict_json_serialisable(self):
        payload = DETECTOR.detect(SPEC_RESUME).to_dict()
        decoded = json.loads(json.dumps(payload))
        self.assertEqual(decoded["sections"]["education"], "B.Tech Computer Science")
        self.assertIn("heading_start", decoded["metadata"][0])

    def test_acts_like_plain_dict(self):
        r = DETECTOR.detect("EDUCATION\nB.Tech")
        self.assertIsInstance(r, SectionResult)
        self.assertEqual(r.get("projects", ""), "")
        self.assertEqual(r, {"education": "B.Tech"})


class TestConfigurationAndInvalidInput(unittest.TestCase):
    def test_invalid_input_types_raise_type_error(self):
        for bad in (None, 123, b"SKILLS\nPython", ["SKILLS"], {"a": 1}):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(TypeError):
                    DETECTOR.detect(bad)  # type: ignore[arg-type]

    def test_extra_aliases(self):
        d = SectionDetector(extra_aliases={"skills": ["tech arsenal"], "publications_x": ["papers"]})
        r = d.detect("TECH ARSENAL\nPython\nPAPERS\nOne paper")
        self.assertEqual(dict(r), {"skills": "Python", "publications_x": "One paper"})

    def test_inline_content_can_be_disabled(self):
        d = SectionDetector(allow_inline_content=False)
        self.assertEqual(dict(d.detect("SKILLS: Python")), {})
        self.assertEqual(dict(d.detect("SKILLS:\nPython")), {"skills": "Python"})

    def test_detector_is_reusable_and_deterministic(self):
        first = dict(DETECTOR.detect(SPEC_RESUME))
        for _ in range(3):
            self.assertEqual(dict(DETECTOR.detect(SPEC_RESUME)), first)


class TestSampleResume(unittest.TestCase):
    def test_sample_resume_end_to_end(self):
        text = (ROOT / "sample_resumes" / "sample_resume.txt").read_text(encoding="utf-8")
        r = DETECTOR.detect(text)
        self.assertEqual(
            list(r),
            ["summary", "skills", "projects", "experience", "education", "certifications", "achievements"],
        )
        self.assertTrue(r["projects"].startswith("1. E-commerce Application"))
        self.assertTrue(r["projects"].endswith("Scikit-learn."))
        self.assertNotIn("Chess", r["achievements"])      # Hobbies is a boundary
        self.assertNotIn("Java and Spring Boot to build", r["projects"])
        self.assertTrue(r.preamble.startswith("RAHUL SHARMA"))


# --------------------------------------------------------------------------- #
# Report mode
# --------------------------------------------------------------------------- #

def _short(value, limit=230):
    s = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else repr(value)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def report() -> int:
    passed = 0
    tp = fp = fn = 0
    for i, case in enumerate(CASES, 1):
        actual = _run(case)
        ok = actual == case["expected"]
        passed += ok
        for k, v in actual.items():
            if case["expected"].get(k) == v:
                tp += 1
            else:
                fp += 1
        fn += sum(1 for k, v in case["expected"].items() if actual.get(k) != v)
        print(f"[{i:02d}] {case['id']}  --  {case['desc']}")
        print(f"     Input   : {_short(case['input'])}")
        print(f"     Expected: {_short(case['expected'])}")
        print(f"     Actual  : {_short(actual)}")
        print(f"     Status  : {'PASS' if ok else 'FAIL'}\n")

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"Cases passed : {passed}/{len(CASES)}")
    print(
        f"Section-level (key AND exact text must match): "
        f"TP={tp} FP={fp} FN={fn}  precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}"
    )

    sample = (ROOT / "sample_resumes" / "sample_resume.txt").read_text(encoding="utf-8")
    for label, text in (("1 resume", sample), ("500 resumes concatenated", (sample + "\n") * 500)):
        runs = []
        for _ in range(15):
            t0 = time.perf_counter()
            DETECTOR.detect(text)
            runs.append((time.perf_counter() - t0) * 1000)
        print(
            f"Timing [{label}: {len(text):,} chars, {text.count(chr(10)):,} lines]: "
            f"median {statistics.median(runs):.3f} ms, min {min(runs):.3f} ms (15 runs)"
        )
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    if "--unittest" in sys.argv:
        sys.argv.remove("--unittest")
        unittest.main()
    sys.exit(report())
