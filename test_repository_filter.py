import tempfile
import logging
import json
import unittest
from unittest.mock import patch

import analyzer
import app
import github_enricher
from repository_filter import score_repository, select_for_candidate


def repo(number, name, language=None, size=10, description="", archived=False):
    return {"id": number, "full_name": f"alice/{name}", "name": name, "language": language,
            "size": size, "description": description, "topics": [], "archived": archived,
            "pushed_at": "2026-11-01T00:00:00Z", "html_url": f"https://github.com/alice/{name}"}


class RepositoryFilterTests(unittest.TestCase):
    def test_app_deferred_callback_uses_existing_screening_rules(self):
        row = {name: "" for name in app.ALL_COLUMNS}
        row.update(status="ok", tech_projects="Python", tech_experience="Python",
                   projects_json=json.dumps({"projects": [
                       {"project_name": "A", "technologies": ["Python"], "start": 0, "end": 10},
                       {"project_name": "B", "technologies": ["Python"], "start": 11, "end": 20},
                   ]}))
        item = app.Processed(row, profiles=["alice"])
        captured = {}
        def fake_profiles(*args, **kwargs):
            callback = kwargs["needs_more_evidence"]
            person = {"profiles": ["alice"]}
            captured["before"] = callback(person, [])
            captured["after"] = callback(person, [{"technologies": ["Python"]}])
            return [github_enricher.GitHubInfo(profiles=["alice"], status="done")]
        with patch.object(github_enricher, "analyze_profiles", side_effect=fake_profiles):
            app.run_github([item], required_groups=(("Python",),))
        self.assertEqual(captured, {"before": True, "after": False})

    def test_logging_handlers_not_duplicated_and_sensitive_token_not_written(self):
        analyzer.configure_logging(logging.DEBUG)
        count = len(analyzer.logger.handlers)
        analyzer.configure_logging(logging.INFO)
        self.assertEqual(len(analyzer.logger.handlers), count)
        self.assertTrue(any(handler.__class__.__name__ == "RotatingFileHandler"
                            for handler in analyzer.logger.handlers))

    def test_relevance_keeps_monorepos_missing_language_and_archived(self):
        groups = (("Django", "FastAPI", "Flask"), ("PostgreSQL",))
        backend = score_repository(repo(1, "backend-api", "Python", description="FastAPI PostgreSQL"), groups)
        monorepo = score_repository(repo(2, "frontend-backend", "TypeScript",
                                         description="React frontend and FastAPI backend"), groups)
        missing = score_repository(repo(3, "service", None, description="PostgreSQL service"), groups)
        archived = score_repository(repo(4, "django-app", "Python", archived=True), groups)
        self.assertGreater(backend.score, 0)
        self.assertGreater(monorepo.score, 0)
        self.assertGreater(missing.score, 0)
        self.assertGreater(archived.score, 0)
        self.assertEqual(score_repository(repo(5, "empty", size=0), groups).signals, ("empty_repository",))
        selected, deferred = select_for_candidate([repo(6, "misc"), repo(7, "fastapi")], groups, 1)
        self.assertEqual(selected[0]["name"], "fastapi")
        self.assertEqual(len(deferred), 1)

    def test_per_candidate_budget_cache_shared_repo_and_deferred_callback(self):
        shared = repo(1, "fastapi")
        other = repo(2, "portfolio")
        third = repo(3, "postgresql")
        accounts = {"alice": [shared, other, third], "bob": [shared, third]}
        people = [analyzer.new_person("") for _ in accounts]
        for person, user in zip(people, accounts):
            person["profiles"] = [user]
        analysed = []
        def fake_analyze(item):
            analysed.append(item["full_name"])
            return {"repository": item["full_name"], "technologies": [item["name"]], "categories": []}
        with tempfile.TemporaryDirectory() as temp:
            cache = analyzer.Cache(f"{temp}/cache.json")
            cache.set(other, {"repository": other["full_name"], "technologies": [], "categories": []})
            with patch.object(analyzer, "Cache", return_value=cache), \
                 patch.object(analyzer, "list_repositories", side_effect=lambda user, *_: accounts[user]), \
                 patch.object(analyzer, "analyze_repository", side_effect=fake_analyze):
                analyzer.run_report(people, None, required_groups=(("FastAPI",),), initial_repo_budget=1)
        self.assertEqual(analysed, [shared["full_name"]])
        self.assertEqual(people[0]["repositories_cached"], 1)
        self.assertEqual(people[0]["repositories_deferred"], 1)
        self.assertEqual(people[1]["repositories_deferred"], 1)
        self.assertEqual(people[0]["analysis_metrics"]["repositories_selected"], 1)
        self.assertEqual(people[0]["analysis_metrics"]["git_clones_avoided"], 2)
        self.assertEqual(people[0]["analysis_metrics"]["cache_hits"], 1)
        self.assertEqual(people[0]["analysis_metrics"]["successful_analyses"], 1)
        self.assertEqual(people[0]["analysis_metrics"]["api_requests"], 0)  # mocked listing
        self.assertEqual(people[0]["status"], "Done")

    def test_deferred_expansion_and_exhaustive_mode(self):
        repos = [repo(1, "fastapi"), repo(2, "postgresql"), repo(3, "other")]
        def run(exhaustive=False, callback=None):
            person = analyzer.new_person("")
            person["profiles"] = ["alice"]
            with tempfile.TemporaryDirectory() as temp:
                cache = analyzer.Cache(f"{temp}/cache.json")
                with patch.object(analyzer, "Cache", return_value=cache), \
                     patch.object(analyzer, "list_repositories", return_value=repos), \
                     patch.object(analyzer, "analyze_repository", side_effect=lambda r: {
                         "repository": r["full_name"], "technologies": [r["name"]], "categories": []}):
                    analyzer.run_report([person], None, required_groups=(("FastAPI",),), initial_repo_budget=1,
                                        exhaustive=exhaustive, needs_more_evidence=callback)
            return person
        partial = run(callback=lambda person, analysed: len(analysed) < 2)
        self.assertEqual(partial["repositories_selected"], 2)
        self.assertEqual(partial["repositories_deferred"], 1)
        complete = run(exhaustive=True)
        self.assertEqual(complete["repositories_selected"], 3)
        self.assertEqual(complete["repositories_deferred"], 0)

    def test_empty_repo_is_skipped_not_cached_as_analyzed(self):
        person = analyzer.new_person("")
        person["profiles"] = ["alice"]
        empty = repo(9, "empty", size=0)
        with tempfile.TemporaryDirectory() as temp:
            cache = analyzer.Cache(f"{temp}/cache.json")
            with patch.object(analyzer, "Cache", return_value=cache), \
                 patch.object(analyzer, "list_repositories", return_value=[empty]), \
                 patch.object(analyzer, "analyze_repository") as analyze:
                analyzer.run_report([person], "SECRET_TEST_TOKEN_123")
                analyze.assert_not_called()
        self.assertEqual(person["repositories_skipped_empty"], 1)
        self.assertEqual(person["repositories_selected"], 0)
        self.assertEqual(person["repository_results"], [])
        for handler in analyzer.logger.handlers:
            handler.flush()
            if hasattr(handler, "baseFilename"):
                with open(handler.baseFilename, encoding="utf-8") as log:
                    self.assertNotIn("SECRET_TEST_TOKEN_123", log.read())


if __name__ == "__main__":
    unittest.main()
