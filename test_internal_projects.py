"""Training-company projects (dictionaries/internal_projects.txt) are not counted as the candidate's work."""

import json
import unittest
from unittest.mock import patch

import app
from internal_projects import find_internal
from project_extractor import ProjectExtractor
from repository_filter import pre_analysis_exclusion
from section_detector import SectionDetector
from technology_matcher import TechnologyMatcher

RESUME = """Asha Kumar
Skills
Python, SQL

Projects
Nxt Trendz – E-commerce App | React JS, Redux
Built an e-commerce clone with product filters and cart.

Expense Tracker | Django, PostgreSQL
Tracks personal expenses with monthly reports.

Education
B.Tech, 2024
"""


class InternalProjectMatchingTests(unittest.TestCase):
    def test_variants_of_listed_names_match(self):
        for name in ("Nxt Trendz – E-commerce App | React JS", "nxt-trendz", "JobbyApp", "Jobby App - Job Portal",
                     "Todo Application", "IPL Dashboard App", "Netflix/Amazon Prime Clone", "Book Hub"):
            self.assertIsNotNone(find_internal(name), name)

    def test_own_projects_with_similar_words_do_not_match(self):
        for name in ("AI Chatbot with RAG", "Quiz Game Generator using LLM", "Expense Tracker",
                     "Instagram Clone using MERN", "Covid-19 Tracker", "", None):
            self.assertIsNone(find_internal(name), name)

    def test_internal_repositories_are_skipped_before_scanning(self):
        repo = {"full_name": "asha/nxt-trendz", "name": "nxt-trendz", "owner": {"login": "asha"}}
        self.assertEqual(pre_analysis_exclusion(repo), "internal_training_project")
        self.assertIsNone(pre_analysis_exclusion({**repo, "full_name": "asha/expense-tracker",
                                                  "name": "expense-tracker"}))


class ResumePipelineTests(unittest.TestCase):
    def test_internal_project_and_its_only_technologies_are_removed(self):
        matcher = TechnologyMatcher()
        with patch.object(app, "download", return_value=RESUME.encode("utf-8")):
            item = app.process_resume("", "https://example.com/r.txt", SectionDetector(), matcher,
                                      ProjectExtractor(matcher))
        row = item.row
        self.assertIn(row["status"], ("ok", "partial"))
        self.assertEqual(row["internal_projects"], "Nxt Trendz – E-commerce App")
        self.assertEqual(row["num_projects"], 1)
        self.assertEqual([p["project_name"] for p in json.loads(row["projects_json"])["projects"]],
                         ["Expense Tracker"])
        self.assertEqual(set(row["tech_projects"].split("; ")), {"Django", "PostgreSQL"})
        self.assertNotIn("React", row["technologies"])
        self.assertNotIn("Redux", row["technologies"])
        self.assertIn("Python", row["tech_skills"])
        matches = json.loads(row["technology_matches"])
        self.assertFalse([m for m in matches if m["name"] in ("React", "Redux")])


if __name__ == "__main__":
    unittest.main()
