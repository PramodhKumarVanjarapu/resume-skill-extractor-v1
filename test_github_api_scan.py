import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests

import analyzer
import app
import github_enricher


REPO = {"id": 21, "full_name": "alice/demo", "name": "demo", "default_branch": "release/main",
        "size": 10, "language": "Python", "html_url": "https://github.com/alice/demo"}
API_URL = "https://api.github.com/repos/alice/demo/git"


def response(content, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
    return result


def blob(path, sha, size=25):
    return {"path": path, "sha": sha, "size": size, "mode": "100644", "type": "blob"}


class GitHubAPIScanTests(unittest.TestCase):
    def test_app_detects_evidence_without_git_and_reuses_cache_in_clone_mode(self):
        tree = {"truncated": False, "tree": [
            blob(".github/workflows/test.yml", "workflow"),
            blob("app.py", "source"),
            blob("requirements.txt", "manifest"),
            blob("node_modules/package.json", "ignored"),
            blob("large.py", "large", size=analyzer.MAX_CODE_SIZE + 1),
        ]}
        responses = {f"{API_URL}/trees/release%2Fmain": response(tree),
                     f"{API_URL}/blobs/manifest": response(b"fastapi==1.0\n"),
                     f"{API_URL}/blobs/source": response(b"from fastapi import FastAPI\n")}
        item = app.Processed({"status": "ok"}, profiles=["alice"])

        with tempfile.TemporaryDirectory() as folder:
            cache = analyzer.Cache(str(Path(folder) / "cache.json"))
            with patch.object(analyzer, "Cache", return_value=cache), \
                 patch.object(analyzer, "list_repositories", return_value=[REPO]), \
                 patch.object(analyzer, "git") as git, \
                 patch.object(requests.Session, "get", side_effect=lambda url, **_: responses[url]) as get:
                app.run_github([item], token="test-token", scan_method="api")
                self.assertIsNotNone(item.github)
                self.assertEqual(item.github.analysis_metrics["api_requests"], 3)
                self.assertEqual(item.github.analysis_metrics["git_clones_avoided"], 1)
                self.assertEqual(get.call_count, 3)
                self.assertEqual(get.call_args_list[0].kwargs["params"], {"recursive": "1"})
                for call in get.call_args_list:
                    self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer test-token")
                self.assertEqual(get.call_args.kwargs["headers"]["Accept"], "application/vnd.github.raw+json")

                self.assertEqual(set(item.github.tech_stack), {"FastAPI", "GitHub Actions", "Python"})
                evidence = json.loads(item.row["github_evidence_details"])
                self.assertEqual(
                    {(entry["path"], entry["type"], entry["strength"]) for entry in evidence
                     if entry["technology"] == "FastAPI"},
                    {("requirements.txt", "dependency_manifest", "strong"), ("app.py", "source_import", "strong")},
                )
                clone_mode = github_enricher.analyze_profiles([["alice"]])[0]
                self.assertEqual(clone_mode.cached_repos, 1)
                self.assertEqual(clone_mode.tech_stack, item.github.tech_stack)
                self.assertEqual(get.call_count, 3)
                git.assert_not_called()

    def test_truncated_tree_uses_nonrecursive_subtrees(self):
        responses = [
            response({"sha": "root", "truncated": True, "tree": []}),
            response({"truncated": False, "tree": [{"path": "backend", "type": "tree", "sha": "backend"}]}),
            response({"truncated": False, "tree": [blob("requirements.txt", "requirements")]}),
            response(b"django\n"),
        ]
        with patch.object(requests.Session, "get", side_effect=responses) as get:
            result = analyzer.analyze_repository(REPO, api=analyzer._GitHubRepositoryAPI("test-token"))

        self.assertNotIn("error", result)
        self.assertIn("Django", result["technologies"])
        self.assertTrue(any(entry["path"] == "backend/requirements.txt" for entry in result["technology_evidence"]))
        self.assertEqual([call.args[0] for call in get.call_args_list], [
            f"{API_URL}/trees/release%2Fmain", f"{API_URL}/trees/root", f"{API_URL}/trees/backend",
            f"{API_URL}/blobs/requirements",
        ])
        self.assertIsNone(get.call_args_list[1].kwargs["params"])

    def test_quota_failure_stops_requests_and_is_reported_without_caching(self):
        limited = response({"message": "You have exceeded a secondary rate limit"}, status=403)
        second = {**REPO, "id": 22, "name": "other", "full_name": "alice/other"}
        credentials = analyzer.GitHubCredentials(("test-token",), max_parallel_requests=1)
        with tempfile.TemporaryDirectory() as folder:
            cache = analyzer.Cache(str(Path(folder) / "cache.json"))
            with patch.object(analyzer, "Cache", return_value=cache), \
                 patch.object(analyzer, "list_repositories", return_value=[REPO, second]), \
                 patch.object(requests.Session, "get", return_value=limited) as get:
                infos = github_enricher.analyze_profiles([["alice"]], credentials, scan_method="api")

        self.assertEqual(get.call_count, 1)
        self.assertFalse(cache.data)
        self.assertEqual(infos[0].analysis_failures, 2)
        self.assertEqual(github_enricher.summarize(infos)["repository_rate_limited"], 1)

    def test_api_method_requires_a_token(self):
        with self.assertRaisesRegex(analyzer.GitHubError, "requires a token"):
            analyzer.run_report([], None, scan_method="api")


class GitHubCredentialsTests(unittest.TestCase):
    def test_parallel_limit_is_shared_by_listing_and_file_requests(self):
        credentials = analyzer.GitHubCredentials(("test-token",), max_parallel_requests=2)
        start = threading.Barrier(5, timeout=5)
        active_lock = threading.Lock()
        two_active = threading.Event()
        over_limit = threading.Event()
        release = threading.Event()
        active = 0

        def request(_url, **_kwargs):
            nonlocal active
            with active_lock:
                active += 1
                if active == 2:
                    two_active.set()
                if active > 2:
                    over_limit.set()
            try:
                self.assertTrue(release.wait(5))
                return response({})
            finally:
                with active_lock:
                    active -= 1

        def run_request(index):
            with requests.Session() as session:
                start.wait()
                return credentials.get(f"{API_URL}/{index}", session=session if index % 2 else None)

        with patch.object(requests, "get", side_effect=request), \
             patch.object(requests.Session, "get", side_effect=request), \
             ThreadPoolExecutor(4) as pool:
            futures = [pool.submit(run_request, index) for index in range(4)]
            try:
                start.wait()
                self.assertTrue(two_active.wait(5))
                self.assertFalse(over_limit.wait(0.1))
            finally:
                release.set()
            self.assertEqual([future.result().status_code for future in futures], [200] * 4)

    def test_streamlit_tokens_are_trimmed_deduplicated_and_prioritized(self):
        secrets = {"GITHUB_TOKENS": [" primary ", "", "backup", "typed", 5], "GITHUB_TOKEN": "legacy"}
        with patch.dict(os.environ, {"GITHUB_TOKEN": "environment"}):
            self.assertEqual(list(app._github_tokens(secrets, " typed ").items()), [
                ("typed", "the sidebar GitHub token"),
                ("primary", "GITHUB_TOKENS entry #1"),
                ("backup", "GITHUB_TOKENS entry #3"),
                ("legacy", "GITHUB_TOKEN in Streamlit secrets"),
                ("environment", "the GITHUB_TOKEN environment variable"),
            ])

    def test_credentials_rotate_during_preflight_and_repository_reads(self):
        credentials = analyzer.GitHubCredentials(
            ("expired", "primary", "backup"),
            token_sources={"expired": "GITHUB_TOKENS entry #1", "primary": "GITHUB_TOKENS entry #2"},
        )
        analyzer.configure_logging()
        quota = response({"resources": {"core": {"limit": 5000, "remaining": 4500, "reset": 1234}}})
        # A repository needs implementation files to count as a project; manage.py is Django's entry point.
        tree = response({"truncated": False, "tree": [blob("requirements.txt", "manifest"), blob("manage.py", "entry")]})

        with tempfile.TemporaryDirectory() as folder:
            cache = analyzer.Cache(str(Path(folder) / "cache.json"))
            with patch.object(analyzer, "Cache", return_value=cache), \
                 self.assertLogs(analyzer.logger, level="WARNING") as logs, \
                 patch.object(requests, "get", side_effect=[response({}, 401), quota, response([REPO])]) as listing, \
                 patch.object(requests.Session, "get", side_effect=[response({}, 401), tree, response(b"django\n"),
                                                                    response(b"django\n")]) as scan:
                rate = github_enricher.rate_limit_status(credentials)
                info = github_enricher.analyze_profiles([["alice"]], credentials, scan_method="api")[0]

        self.assertEqual(rate.remaining, 4500)
        self.assertEqual(info.status, "done")
        self.assertIn("Django", info.tech_stack)
        self.assertEqual([call.kwargs["headers"]["Authorization"] for call in listing.call_args_list],
                         ["Bearer expired", "Bearer primary", "Bearer primary"])
        self.assertEqual([call.kwargs["headers"]["Authorization"] for call in scan.call_args_list],
                         ["Bearer primary", "Bearer backup", "Bearer backup", "Bearer backup"])
        self.assertEqual(logs.output, [
            "WARNING:analyzer:GitHub rejected GITHUB_TOKENS entry #1 (HTTP 401)",
            "WARNING:analyzer:GitHub rejected GITHUB_TOKENS entry #2 (HTTP 401)",
        ])

    def test_retry_after_blocks_further_requests_without_switching_credentials(self):
        credentials = analyzer.GitHubCredentials(("primary", "backup"))
        limited = response({}, 429)
        limited.headers["Retry-After"] = "120"
        started = time.time()
        analyzer.configure_logging()
        with patch.object(requests, "get", return_value=limited) as get, \
             self.assertLogs(analyzer.logger, level="WARNING") as logs:
            self.assertIsNone(github_enricher.rate_limit_status(credentials))
            with self.assertRaisesRegex(analyzer.GitHubError, "rate limit reached"):
                analyzer.list_repositories("alice", credentials)

        self.assertGreaterEqual(credentials.retry_at, started + 120)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer primary")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("GitHub rate-limited GitHub token #1 (HTTP 429)", logs.output[0])
        self.assertNotIn("primary", logs.output[0])

    def test_primary_limit_uses_the_reset_time(self):
        credentials = analyzer.GitHubCredentials(("primary", "backup"))
        limited = response({}, 403)
        reset = int(time.time()) + 250
        limited.headers.update({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)})
        with patch.object(requests, "get", return_value=limited):
            with self.assertRaisesRegex(analyzer.GitHubError, "rate limit reached"):
                analyzer.list_repositories("alice", credentials)

        self.assertEqual(credentials.retry_at, reset)


if __name__ == "__main__":
    unittest.main()
