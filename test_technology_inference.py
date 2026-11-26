"""Coverage of the hiring vocabulary and inferred ("implied") skills."""

import unittest

from technology_matcher import DictionaryError, TechnologyMatcher, build_dictionary

# Every skill on the hiring team's checklist must be recognised in a skills list.
CHECKLIST = {
    "AI/ML": "Machine Learning", "Deep Learning": "Deep Learning", "Computer Vision": "Computer Vision",
    "Image Processing": "Image Processing", "AI Automation": "AI Automation",
    "Multi-Agent Systems": "Multi-Agent Systems", "Data Visualization": "Data Visualization", "LLMOps": "LLMOps",
    "Manual Testing": "Manual Testing", "Automation Testing": "Automation Testing", "API Testing": "API Testing",
    "STLC": "STLC", "REST APIs": "REST API", "API Integration": "API Integration", "JSON": "JSON",
    "Microservices": "Microservices", "WebRTC": "WebRTC", "TCP/IP": "TCP/IP", "DNS": "DNS", "DHCP": "DHCP",
    "Routing": "Routing", "Switching": "Switching", "Subnetting": "Subnetting",
    "Network Monitoring": "Network Monitoring", "Operating Systems": "Operating Systems",
    "Linux Administration": "Linux Administration", "Troubleshooting": "Troubleshooting",
    "Production Monitoring": "Production Monitoring", "Ticketing Tools": "Ticketing Tools", "Cursor": "Cursor",
    "GitHub Copilot": "GitHub Copilot", "Codex": "Codex", "Web Scraping": "Web Scraping", "BullMQ": "BullMQ",
    "Django REST Framework": "Django REST Framework", "Machine Learning": "Machine Learning", "NLP": "NLP",
}


class VocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = TechnologyMatcher()

    def test_checklist_terms_in_skills_list(self):
        for term, name in CHECKLIST.items():
            with self.subTest(term=term):
                self.assertEqual(self.m.extract_names(f"Skills: {term}"), [name])

    def test_common_words_need_technology_context(self):
        for text in [
            "Moved the cursor to the next field.", "Routing requests to the right team was my job.",
            "Troubleshooting customer complaints at the store.", "I took the rest of the week off.",
            "Added 5 ML of reagent.", "The Codex of ancient law.", "Please find my CV attached.",
            "Watched CNN news daily.", "Switching careers was hard.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.m.extract_names(text, include_inferred=True), [])

    def test_copilot_is_not_reduced_to_github(self):
        self.assertEqual(self.m.extract_names("Used GitHub Copilot daily."), ["GitHub Copilot"])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = TechnologyMatcher()

    def test_off_by_default(self):
        self.assertEqual(self.m.extract_names("Trained models in PyTorch."), ["PyTorch"])

    def test_chain_and_evidence_point_at_written_term(self):
        text = "Projects\nTrained an image model in PyTorch."
        found = {x.name: x for x in self.m.extract(text, include_inferred=True)}
        self.assertEqual(set(found), {"PyTorch", "Deep Learning", "Machine Learning"})
        dl = found["Deep Learning"]
        self.assertEqual((dl.match_type, dl.inferred_from, dl.reason), ("inferred", "PyTorch", "inferred from PyTorch"))
        self.assertEqual(text[dl.start:dl.end], "PyTorch")          # offsets refer to the original text
        self.assertEqual(dl.evidence, "Trained an image model in PyTorch.")
        self.assertEqual(found["Machine Learning"].inferred_from, "PyTorch")

    def test_not_inferred_when_stated(self):
        found = self.m.find_all("Skills: Deep Learning, PyTorch", include_inferred=True)
        dl = [x for x in found if x.name == "Deep Learning"]
        self.assertEqual([x.match_type for x in dl], ["canonical"])

    def test_one_inferred_match_per_mention(self):
        text = "Used PyTorch.\nLater used PyTorch again."
        inferred = [x for x in self.m.find_all(text, include_inferred=True) if x.name == "Deep Learning"]
        self.assertEqual([text[x.start:x.end] for x in inferred], ["PyTorch", "PyTorch"])
        self.assertEqual(self.m.extract(text, include_inferred=True)[1].occurrences, 2)

    def test_narrow_rules(self):
        # Pandas/NumPy alone are not evidence of Machine Learning.
        self.assertEqual(self.m.extract_names("Skills: Pandas, NumPy", include_inferred=True), ["Pandas", "NumPy"])
        # On real resumes Postman (most developers) and Selenium/Playwright (also used for scraping)
        # are not evidence of a testing specialisation.
        self.assertEqual(self.m.extract_names("Tools: Postman, Selenium, Playwright", include_inferred=True),
                         ["Postman", "Selenium", "Playwright"])

    def test_new_vocabulary(self):
        cases = {
            "Skills: SDLC, Agile, Scrum, OOPs, DBMS, DSA":
                ["SDLC", "Agile", "Scrum", "OOP", "DBMS", "Data Structures and Algorithms"],
            "AWS (EC2, S3, Lambda), Docker": ["AWS", "EC2", "Amazon S3", "AWS Lambda", "Docker"],
            "Deployed backend on Render and frontend on Vercel": ["Render", "Vercel"],
            "CI/CD pipelines with GitHub Actions": ["CI/CD", "GitHub Actions"],
            "Regression, Smoke and Sanity testing": ["Regression Testing", "Smoke Testing", "Sanity Testing"],
            "Stack: React.js · Node.js · Express · MySQL": ["React", "Node.js", "Express.js", "MySQL"],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.m.extract_names(text), expected)

    def test_common_words_stay_words(self):
        for text in ["Render dynamic lists in React", "Agile teams move fast", "We ran a regression analysis",
                     "The Lambda function is anonymous", "A sanity check before release"]:
            with self.subTest(text=text):
                self.assertTrue(set(self.m.extract_names(text, include_inferred=True)) <= {"React"})

    def test_examples(self):
        cases = {
            "Skills: OpenCV": "Computer Vision", "Skills: spaCy": "NLP", "Skills: Tableau": "Data Visualization",
            "Skills: Cypress": "Automation Testing", "Skills: SoapUI": "API Testing",
            "Built a MERN application": "MongoDB", "Skills: n8n": "AI Automation",
            "Skills: CrewAI": "AI Agents", "Skills: Django REST Framework": "REST API",
            "Skills: Linux Administration": "Linux", "Skills: ServiceNow": "Ticketing Tools",
        }
        for text, implied in cases.items():
            with self.subTest(text=text):
                self.assertIn(implied, self.m.extract_names(text, include_inferred=True))


class ImpliesValidationTests(unittest.TestCase):
    BASE = {"categories": {"X": ["A", "B", "C"]}}

    def test_cycle_rejected(self):
        with self.assertRaisesRegex(DictionaryError, "cycle"):
            build_dictionary({**self.BASE, "implies": {"A": ["B"], "B": ["C"], "C": ["A"]}}, {})

    def test_unknown_target_rejected(self):
        with self.assertRaisesRegex(DictionaryError, "unknown"):
            build_dictionary({**self.BASE, "implies": {"A": ["Z"]}}, {})

    def test_closure(self):
        d = build_dictionary({**self.BASE, "implies": {"A": ["B"], "B": ["C"]}}, {})
        self.assertEqual(d.implies["A"], ("B", "C"))

    def test_extended_keeps_implications(self):
        d = build_dictionary({**self.BASE, "implies": {"A": ["B"]}}, {}).extended({"Y": ["D"]})
        self.assertEqual(d.implies["A"], ("B",))


if __name__ == "__main__":
    unittest.main()
