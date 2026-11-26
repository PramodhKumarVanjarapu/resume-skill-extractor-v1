import csv
import io
import json
import unittest

import pandas as pd

import app
import candidate_ranker as ranker
from technology_matcher import TechnologyMatcher


class RankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = TechnologyMatcher()
        cls.requirements = ranker.normalize_requirements([
            ("Python", True), ("Django OR FastAPI OR Flask", True),
            ("PostgreSQL OR MySQL", True), ("Docker", True), ("Redis", False),
        ], cls.matcher)

    def row(self, **updates):
        row = {key: "" for key in app.ALL_COLUMNS}
        row.update(updates)
        if "projects_json" not in updates:
            techs = [tech.strip() for tech in updates.get("tech_projects", "").split(";") if tech.strip()]
            projects = [
                {"project_name": f"Project {i}", "technologies": techs, "start": i * 100, "end": i * 100 + 50}
                for i in range(2)
            ] if techs else []
            row["projects_json"] = json.dumps({"projects": projects})
            row["num_projects"] = len(projects)
        return row

    def test_alias_or_group_exact_matching_and_optional(self):
        requirements = ranker.normalize_requirements([("postgres OR MySQL", True), ("Java", True),
                                                     ("Docker", False)], self.matcher)
        result = ranker.score_candidate(self.row(tech_projects="PostgreSQL; JavaScript; Docker"),
                                        requirements, self.matcher)
        self.assertEqual(result["technology_match_percentage"], 50)
        self.assertEqual(result["matched_mandatory_groups"], 1)
        self.assertIn("PostgreSQL", result["matched_alternative_groups"])
        self.assertEqual(result["optional_matches"], "Docker: Docker")
        self.assertIn("Java", result["missing_requirements"])

    def test_coverage_evidence_and_priorities(self):
        common = dict(tech_projects="Python; FastAPI; PostgreSQL", github_tech_stack="Python; Docker",
                      tech_experience="PostgreSQL", github_status="done")
        p1 = ranker.score_candidate(self.row(**common), self.requirements, self.matcher)
        self.assertEqual((p1["technology_match_percentage"], p1["evidence_score"], p1["priority"]),
                         (100, 52.5, "P1"))
        p2 = ranker.score_candidate(self.row(**{**common, "tech_experience": ""}), self.requirements, self.matcher)
        self.assertEqual(p2["priority"], "P2")
        p3 = ranker.score_candidate(self.row(tech_projects="Python; FastAPI; PostgreSQL; Docker"),
                                    self.requirements, self.matcher)
        self.assertEqual(p3["priority"], "P3")
        self.assertIn("GitHub analysis unavailable", p3["ranking_warnings"])
        project_experience = ranker.score_candidate(
            self.row(tech_projects="Python; FastAPI", tech_experience="Docker; MySQL"),
            self.requirements, self.matcher,
        )
        self.assertEqual(project_experience["priority"], "P3")
        no_project = ranker.score_candidate(self.row(github_tech_stack="Python", tech_experience="Docker"),
                                            self.requirements, self.matcher, threshold=0)
        self.assertEqual(no_project["priority"], "Below Threshold")
        self.assertEqual(no_project["potential_priority"], "Unclassified")

    def test_threshold_missing_sources_and_stable_order(self):
        rows = [self.row(phone="first", tech_projects="Python"),
                self.row(phone="second", tech_projects="Python"),
                self.row(phone="third", tech_skills="Python; Docker")]
        ranked = ranker.rank_candidates(rows, self.requirements, self.matcher, threshold=60)
        self.assertEqual([r["phone"] for r in ranked], ["first", "second", "third"])
        self.assertTrue(all(r["priority"] == "Below Threshold" for r in ranked))
        self.assertEqual(ranked[2]["technology_match_percentage"], 0)
        self.assertEqual(ranker.score_candidate(rows[0], (), self.matcher)["potential_priority"], "Unclassified")

    def test_matching_project_thresholds_and_deduplication(self):
        single = self.row(
            tech_projects="Python; Docker", num_projects=3,
            projects_json=json.dumps({"projects": [
                {"project_name": "Relevant", "technologies": ["Python", "Docker"], "start": 0, "end": 50},
                {"project_name": "Unrelated", "technologies": ["Java"], "start": 51, "end": 90},
                {"project_name": "Relevant", "technologies": ["Python", "Docker"], "start": 0, "end": 50},
            ]}),
        )
        one = ranker.score_candidate(single, self.requirements, self.matcher, threshold=0)
        self.assertEqual(one["total_projects"], 3)
        self.assertEqual(one["matching_projects_count"], 1)
        self.assertEqual(one["matching_project_names"], "Relevant")
        self.assertEqual(one["project_threshold_met"], False)
        self.assertEqual(one["priority"], "Below Threshold")
        projects = json.loads(single["projects_json"])
        projects["projects"].append({"project_name": "Second", "technologies": ["Docker"], "start": 91, "end": 130})
        two = ranker.score_candidate(self.row(**{**single, "projects_json": json.dumps(projects)}),
                                     self.requirements, self.matcher, threshold=0)
        self.assertEqual(two["matching_projects_count"], 2)
        self.assertEqual(two["project_threshold_met"], True)
        self.assertEqual(two["priority"], "P3")
        empty = ranker.score_candidate(self.row(), self.requirements, self.matcher, threshold=0)
        self.assertEqual(empty["matching_projects_count"], 0)
        self.assertFalse(empty["shortlisting_eligible"])
        unknown = ranker.score_candidate(self.row(projects_json="", tech_projects="Python"),
                                         self.requirements, self.matcher, threshold=0)
        self.assertIsNone(unknown["project_threshold_met"])
        self.assertIn("unverified", unknown["ranking_reason"])

    def test_deferred_github_data_is_partial_not_final_rejection(self):
        row = self.row(tech_projects="Python", github_status="done", github_deferred_repos=4)
        scored = ranker.score_candidate(row, self.requirements, self.matcher)
        self.assertEqual(scored["priority"], "Pending Analysis")
        self.assertEqual(scored["screening_status"], "partial")
        self.assertIn("Docker", scored["unverified_requirements"])

    def test_priority_then_coverage_then_stable_input_order(self):
        rows = [
            self.row(phone="p3", tech_projects="Python; FastAPI; MySQL; Docker"),
            self.row(phone="p2", tech_projects="Python; FastAPI; MySQL", github_tech_stack="Docker"),
            self.row(phone="p1-first", tech_projects="Python; FastAPI; MySQL", github_tech_stack="Docker",
                     tech_experience="Python"),
            self.row(phone="p1-second", tech_projects="Python; FastAPI; MySQL", github_tech_stack="Docker",
                     tech_experience="Python"),
        ]
        ranked = ranker.rank_candidates(rows, self.requirements, self.matcher)
        self.assertEqual([row["phone"] for row in ranked], ["p1-first", "p1-second", "p2", "p3"])

    def test_export_appends_ranking_columns_and_keeps_phone(self):
        ranked = ranker.rank_candidates([self.row(phone="08810931580", resume_link="https://example.com/a.pdf",
                                                  tech_projects="Python")], self.requirements, self.matcher)
        frame = pd.DataFrame(ranked)
        columns = app.output_columns(frame)
        self.assertEqual(columns[:len(app.OUTPUT_COLUMNS)], app.OUTPUT_COLUMNS)
        self.assertIn("evidence_score", columns)
        rows = list(csv.DictReader(io.StringIO(app.output_csv_bytes(frame).decode("utf-8-sig"))))
        self.assertEqual(rows[0]["phone"], "08810931580")
        self.assertEqual(rows[0]["priority"], "Below Threshold")
        self.assertEqual(app.output_xlsx_bytes(frame)[:2], b"PK")


if __name__ == "__main__":
    unittest.main()
