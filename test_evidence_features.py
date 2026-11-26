import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

import analyzer
import app
import github_enricher
from experience_extractor import extract_experiences
from project_repo_matcher import match_projects
from technology_matcher import TechnologyMatcher
from project_extractor import Project, ProjectExtractionResult


class EvidenceTests(unittest.TestCase):
    def test_repository_provenance_from_manifest_source_and_filename(self):
        repo = {"full_name": "owner/api", "html_url": "https://github.com/owner/api", "clone_url": "unused",
                "language": "Python", "size": 1}
        with tempfile.TemporaryDirectory() as temp, patch.object(analyzer, "CLONE_ROOT", temp):
            def fake_git(args, cwd=None):
                if args[0] == "ls-tree":
                    return b"requirements.txt\0src/app.py\0Dockerfile\0"
                if args[0] == "read-tree":
                    Path(cwd, "requirements.txt").write_text("fastapi\n", encoding="utf-8")
                    Path(cwd, "src").mkdir(exist_ok=True)
                    Path(cwd, "src", "app.py").write_text("import fastapi\n", encoding="utf-8")
                    Path(cwd, "Dockerfile").write_text("FROM python:3\n", encoding="utf-8")
                return b""
            with patch.object(analyzer, "git", side_effect=fake_git):
                result = analyzer.analyze_repository(repo)
        self.assertNotIn("error", result)
        evidence = result["technology_evidence"]
        self.assertTrue(any(e["type"] == "dependency_manifest" and e["path"] == "requirements.txt"
                            for e in evidence))
        self.assertTrue(any(e["type"] == "source_import" and e["path"] == "src/app.py" for e in evidence))
        self.assertTrue(any(e["technology"] == "Docker" and e["type"] == "file_path" for e in evidence))

    def test_url_name_description_overlap_ambiguity_and_unmatched(self):
        repos = [
            {"repository": "alice/skill-bridge", "url": "https://github.com/alice/skill-bridge",
             "description": "skill exchange platform chat", "technologies": ["React", "MongoDB"]},
            {"repository": "alice/skill_bridge", "url": "https://github.com/alice/skill_bridge",
             "description": "skill exchange platform chat", "technologies": ["React", "MongoDB"]},
        ]
        project = {"project_name": "SkillBridge", "description": "skill exchange platform chat",
                   "technologies": ["React", "MongoDB"], "links": []}
        ambiguous = match_projects([project], repos)[0]
        self.assertTrue(ambiguous["requires_review"])
        self.assertEqual(ambiguous["confidence"], "low")
        direct = match_projects([{**project, "links": [repos[0]["url"]]}], repos)[0]
        self.assertEqual(direct["confidence"], "high")
        self.assertIn("explicit_repository_url", direct["matching_reasons"])
        unmatched = match_projects([{"project_name": "Unrelated", "description": "", "technologies": ["React"]}], repos)[0]
        self.assertEqual(unmatched["confidence"], "no_match")

    def test_experience_entries_keep_technology_in_its_job(self):
        text = """Backend Developer Intern
ABC Technologies
Jan 2025 - Jun 2025
- Developed REST APIs using FastAPI.
- Integrated PostgreSQL database.
Software Developer
XYZ Solutions
Jul 2025 - Present
- Built web applications using Django.
- Implemented Redis caching."""
        result = extract_experiences(text, TechnologyMatcher())
        jobs = result["experiences"]
        self.assertEqual(len(jobs), 2)
        self.assertEqual((jobs[0]["company"], jobs[0]["start_date"], jobs[0]["end_date"]),
                         ("ABC Technologies", "2025-01", "2025-06"))
        self.assertIn("FastAPI", jobs[0]["technologies"])
        self.assertNotIn("Django", jobs[0]["technologies"])
        self.assertIn("Django", jobs[1]["technologies"])
        self.assertIsNone(jobs[1]["end_date"])
        missing = extract_experiences("Software Developer\n- Used Docker.", TechnologyMatcher())
        self.assertIsNone(missing["experiences"][0]["company"])
        self.assertIsNone(missing["experiences"][0]["start_date"])

    def test_activity_boundaries_missing_and_archived_fork(self):
        now = datetime(2026, 11, 19, tzinfo=timezone.utc)
        repos = [
            {"repository": "recent", "pushed_at": (now - timedelta(days=90)).isoformat(), "archived": True},
            {"repository": "active", "pushed_at": (now - timedelta(days=91)).isoformat(), "fork": True},
            {"repository": "older", "pushed_at": (now - timedelta(days=181)).isoformat()},
            {"repository": "inactive", "pushed_at": (now - timedelta(days=366)).isoformat()},
            {"repository": "unknown", "updated_at": now.isoformat()},
        ]
        summary = github_enricher.activity_summary(repos, now)
        self.assertEqual([r["activity_status"] for r in summary["repositories"]],
                         ["recent", "active", "older", "inactive", "unknown"])
        self.assertTrue(summary["repositories"][0]["archived"])
        self.assertTrue(summary["repositories"][1]["fork"])
        self.assertEqual(summary["unknown_repositories_count"], 1)

    def test_github_integration_adds_evidence_without_changing_ranking(self):
        project = Project(name="SkillBridge", description="skill exchange platform", technologies=["React"],
                          technology_details=[], unrecognized_technologies=[], dates="", links=[],
                          title_style="plain", boundary_reasons=[], needs_review=False, review_reasons=[],
                          notes=[], start=0, end=40, name_start=0, name_end=11)
        row = {name: "" for name in app.ALL_COLUMNS}
        row.update(status="ok", tech_projects="React", projects_json=json.dumps({"projects": [project.to_dict()]}))
        item = app.Processed(row, project_result=ProjectExtractionResult([project], [], ""), profiles=["alice"])
        info = github_enricher.GitHubInfo(profiles=["alice"], status="done", repos_count=1,
                                         tech_stack=["React"], repositories=[{
                                             "repository": "alice/skillbridge", "url": "https://github.com/alice/skillbridge",
                                             "description": "skill exchange platform", "technologies": ["React"],
                                             "technology_evidence": [{"technology": "React", "type": "dependency_manifest",
                                                                      "path": "package.json", "strength": "medium"}],
                                             "pushed_at": "2026-10-15T00:00:00Z",
                                         }])
        with patch.object(github_enricher, "analyze_profiles", return_value=[info]):
            app.run_github([item])
        self.assertEqual(item.row["recent_repositories_count"], 1)
        self.assertIn("package.json", item.row["github_evidence_details"])
        self.assertIn("alice/skillbridge", item.row["verified_project_repositories"])
        self.assertEqual(item.row["tech_projects"], "React")


if __name__ == "__main__":
    unittest.main()
