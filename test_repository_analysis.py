"""Repository technology extraction, project classification and filtering on synthetic repositories.

Each case builds a repository in memory and scans it through the real ``analyze_repository`` code path (clone mode
with ``git`` replaced, or API mode with HTTP replaced), so file selection, detection and assessment are all covered.
"""

import hashlib
import json
import os
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import requests

import analyzer
from repository_filter import practice_signal, pre_analysis_exclusion, score_repository, select_for_candidate


def meta(name, language=None, description="", owner="alice", size=10):
    return {"id": abs(hash(name)) % 10_000, "full_name": f"{owner}/{name}", "name": name, "language": language,
            "description": description, "topics": [], "size": size, "default_branch": "main",
            "html_url": f"https://github.com/{owner}/{name}", "clone_url": "unused", "owner": {"login": owner}}


def sha(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def scan(files: dict[str, str], repo: dict | None = None) -> dict:
    """Clone-mode scan of ``files`` with git replaced: ls-tree lists them, read-tree writes the sparse selection."""
    repo = repo or meta("project")
    listing = "".join(f"100644 blob {sha(text)} {len(text.encode())}\t{path}\0" for path, text in files.items())

    def fake_git(args, cwd=None):
        if args[0] == "ls-tree":
            return listing.encode("utf-8")
        if args[0] == "read-tree":
            wanted = Path(cwd, ".git", "info", "sparse-checkout").read_text(encoding="utf-8").split()
            for pattern in wanted:
                path = pattern.lstrip("/")
                target = Path(cwd, *path.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(files[path], encoding="utf-8")
        return b""

    with tempfile.TemporaryDirectory() as temp, patch.object(analyzer, "CLONE_ROOT", temp), \
         patch.object(analyzer, "git", side_effect=fake_git):
        return analyzer.analyze_repository(repo)


EXPRESS_SERVER = """const express = require('express');
const mongoose = require('mongoose');
const app = express();
app.get('/api/items', (req, res) => res.json([]));
app.listen(5000);
"""
REACT_APP = """import React, { useState } from 'react';
import axios from 'axios';
export default function App() { return <div />; }
"""


def package(deps=None, dev=None, peer=None, scripts=None):
    return json.dumps({"dependencies": deps or {}, "devDependencies": dev or {},
                       "peerDependencies": peer or {}, "scripts": scripts or {}})


class TechnologyExtractionTests(unittest.TestCase):
    def test_mern_monorepo_detects_node_express_and_classifies_full_stack(self):
        result = scan({
            "server/package.json": package({"express": "^4", "mongoose": "^8"}, scripts={"start": "node index.js"}),
            "server/index.js": EXPRESS_SERVER,
            "client/package.json": package({"react": "^18", "react-dom": "^18", "axios": "^1"}),
            "client/src/App.jsx": REACT_APP,
            "client/src/index.js": "import ReactDOM from 'react-dom/client';\n",
            "client/public/index.html": "<div id=root></div>",
        })
        self.assertNotIn("error", result)
        for tech in ("Node.js", "Express.js", "React", "MongoDB", "Mongoose", "Axios"):
            self.assertIn(tech, result["technologies"])
        self.assertEqual(result["technology_usage"]["Node.js"], "direct")
        self.assertEqual(result["technology_usage"]["Express.js"], "direct")
        express = [e for e in result["technology_evidence"] if e["technology"] == "Express.js"]
        self.assertEqual({e["type"] for e in express}, {"dependency_manifest", "source_import", "source_pattern"})
        self.assertTrue(all(e["strength"] == "strong" for e in express))
        self.assertIn("node", result["implementation"]["backend"])
        self.assertTrue(result["implementation"]["frontend"])
        self.assertIn("MERN", result["categories"])
        self.assertFalse(result["excluded"])

    def test_node_without_framework_from_core_modules_and_start_script(self):
        result = scan({
            "package.json": package(scripts={"start": "node server.js"}),
            "server.js": "const http = require('http');\nhttp.createServer((q, r) => r.end('ok')).listen(3000);\n",
        })
        self.assertEqual(result["technology_usage"]["Node.js"], "direct")
        self.assertIn("Node.js Backend", result["categories"])

    def test_es_module_express_import_without_package_json(self):
        result = scan({"src/server.mjs": "import express from 'express';\nconst app = express();\napp.listen(80);\n",
                       "src/routes/users.mjs": "import { Router } from 'express';\n"})
        self.assertIn("Express.js", result["technologies"])
        self.assertIn("Node.js", result["technologies"])
        self.assertIn("Node.js Backend", result["categories"])

    def test_react_app_with_express_dev_dependency_is_frontend_not_full_stack(self):
        result = scan({
            "package.json": package({"react": "^18"}, dev={"express": "^4", "vite": "^5"}),
            "vite.config.js": "import { defineConfig } from 'vite';\nimport path from 'path';\n",
            "src/index.js": "import ReactDOM from 'react-dom/client';\nimport App from './App';\n",
            "src/App.jsx": REACT_APP,
        })
        self.assertIn("React", result["technologies"])
        self.assertEqual(result["implementation"]["backend"], {})
        self.assertEqual(result["categories"], ["Frontend"])
        # The build config imports `path`, which is tooling, not a Node.js backend.
        self.assertEqual(result["technology_usage"]["Node.js"], "declared")

    def test_language_only_repository_gets_no_stack_category(self):
        result = scan({"main.js": "console.log(1)\n", "util.js": "export const a = 1;\n",
                       "lib/math.js": "export const b = 2;\n"}, meta("utils", "JavaScript"))
        self.assertIn("JavaScript", result["technologies"])
        self.assertEqual(result["categories"], [])

    def test_static_site_needs_html_files(self):
        result = scan({"index.html": "<html></html>", "style.css": "body{}", "script.js": "alert(1)"},
                      meta("portfolio", "HTML"))
        self.assertEqual(result["categories"], ["Static Website"])
        self.assertFalse(result["excluded"])

    def test_frozen_requirements_count_only_imported_packages(self):
        pins = [f"package{i}==1.{i}" for i in range(40)] + ["Django==4.2", "torch==2.1", "Flask==3.0"]
        result = scan({"requirements.txt": "\n".join(pins), "app.py": "from flask import Flask\napp = Flask(__name__)\n"})
        self.assertIn("Flask", result["technologies"])
        self.assertNotIn("Django", result["technologies"])
        self.assertNotIn("PyTorch", result["technologies"])
        self.assertIn("PyTorch", result["indirect_technologies"])
        self.assertIn("python", result["implementation"]["backend"])
        self.assertIn("Python Backend", result["categories"])

    def test_peer_dependencies_are_indirect(self):
        result = scan({"package.json": package({"clsx": "^2"}, peer={"react": ">=17"}),
                       "src/button.js": "export const Button = () => null;\n"})
        self.assertNotIn("React", result["technologies"])
        self.assertIn("React", result["indirect_technologies"])

    def test_docker_counts_image_references_only(self):
        result = scan({
            "docker-compose.yml": "services:\n  db:\n    image: postgres:16\n  # mongo is not used here\n",
            "Dockerfile": "FROM python:3.12\nRUN apt-get install -y redis-tools\n",
            "main.py": "import os\n",
        })
        self.assertIn("PostgreSQL", result["technologies"])
        self.assertNotIn("MongoDB", result["technologies"])
        self.assertNotIn("Redis", result["technologies"])

    def test_names_are_normalised_to_the_dictionary(self):
        self.assertEqual(analyzer.normalize_technology("NodeJS"), "Node.js")
        self.assertEqual(analyzer.normalize_technology("node"), "Node.js")
        self.assertEqual(analyzer.normalize_technology("Some Unknown Lib"), "Some Unknown Lib")
        result = scan({"requirements.txt": "scikit-learn\n", "train.py": "from sklearn.linear_model import Ridge\n"})
        self.assertIn("Scikit-learn", result["technologies"])
        self.assertNotIn("scikit-learn", result["technologies"])
        self.assertIn("Machine Learning", result["categories"])

    def test_generated_vendor_and_build_files_are_ignored(self):
        result = scan({
            "index.html": "<html></html>", "app.js": "document.title='x'", "style.css": "",
            "static/vendor/jquery.min.js": "", "dist/bundle.js": "require('express')",
            "node_modules/react/package.json": package({"react": "1"}), "package-lock.json": "{}",
        })
        self.assertNotIn("Express.js", result["technologies"])
        self.assertNotIn("React", result["technologies"])
        self.assertEqual(result["source_file_count"], 3)

    def test_angular_and_vue_need_no_runtime_import_scan(self):
        angular = scan({"angular.json": "{}", "package.json": package({"@angular/core": "^17"}),
                        "src/app/app.component.ts": "export class AppComponent {}\n"})
        self.assertEqual(angular["categories"], ["Frontend"])
        vue = scan({"package.json": package({"vue": "^3"}), "src/App.vue": "<template></template>"})
        self.assertEqual(vue["categories"], ["Frontend"])


class RepositoryFilterTests(unittest.TestCase):
    def test_dsa_practice_is_excluded_but_named_projects_with_implementation_are_kept(self):
        practice = scan({f"problem_{i}.cpp": "int main(){}" for i in range(6)}, meta("leetcode-solutions", "C++"))
        self.assertEqual(practice["exclusion_reason"], "coding_practice")
        unnamed = scan({f"{i}_two_sum.py": "print(1)" for i in range(1, 6)}, meta("python"))
        self.assertEqual(unnamed["exclusion_reason"], "coding_practice")
        visualizer = scan({"package.json": package({"react": "^18"}), "src/App.jsx": REACT_APP},
                          meta("dsa-visualizer", "JavaScript"))
        self.assertFalse(visualizer["excluded"])
        tutorial = scan({"index.html": "<html>", "style.css": "", "main.js": ""}, meta("js-assignment-3"))
        self.assertEqual(tutorial["exclusion_reason"], "tutorial_or_assignment")
        ml_course = scan({"requirements.txt": "pandas\n", "eda.ipynb": json.dumps({"cells": [
            {"cell_type": "code", "source": ["import pandas as pd"]}]})}, meta("ml-course-project"))
        self.assertFalse(ml_course["excluded"])

    def test_repositories_without_meaningful_code_are_excluded(self):
        self.assertEqual(scan({"README.md": "# notes", "docs/a.png": ""})["exclusion_reason"], "no_source_code")
        self.assertEqual(scan({"hello.py": "print('hi')"})["exclusion_reason"], "minimal_implementation")
        small_real = scan({"bot.py": "import discord\n", "requirements.txt": "discord.py\n"})
        self.assertFalse(small_real["excluded"])

    def test_profile_readme_and_practice_ordering(self):
        self.assertEqual(pre_analysis_exclusion(meta("Alice", owner="alice")), "profile_readme")
        self.assertIsNone(pre_analysis_exclusion(meta("alice.github.io", owner="alice")))
        self.assertEqual(practice_signal(meta("LeetCodeSolutions")), "coding_practice")
        self.assertEqual(practice_signal(meta("x", description="My DSA solutions")), "coding_practice")
        self.assertIsNone(practice_signal(meta("crop-yield", description="Machine learning algorithms for farms")))
        self.assertIsNone(practice_signal(meta("lab-management-system")))
        practice, project = meta("hackerrank"), meta("expense-tracker")
        self.assertLess(score_repository(practice).score, score_repository(project).score)
        first, later = select_for_candidate([practice, project], (), 1)
        self.assertEqual((first, later), ([project], [practice]))

    def test_duplicates_are_marked_once_per_candidate(self):
        files = {f"src/f{i}.js": f"export const v{i} = {i};" for i in range(6)}
        files["package.json"] = package({"react": "^18"})
        original = scan(files, meta("todo-app"))
        copy = scan({**files, "src/extra.js": "export const x = 1;"}, meta("todo-app-final"))
        other = scan({"src/a.js": "a", "src/b.js": "b", "src/c.js": "c", "package.json": package({"vue": "3"})},
                     meta("other"))
        analysed = [dict(original, created_at="2025-01-01"), dict(copy, created_at="2025-02-01"),
                    dict(other, created_at="2025-03-01")]
        analyzer.mark_duplicates(analysed)
        self.assertEqual([r.get("exclusion_reason") for r in analysed],
                         ["duplicate_of:alice/todo-app-final", None, None])

    def test_run_report_skips_excluded_repositories_in_the_stack(self):
        repos = [meta("alice", owner="alice"), meta("api", "JavaScript"), meta("leetcode", "Java")]
        results = {
            "alice/api": {"repository": "alice/api", "technologies": ["Express.js", "Node.js"], "categories": [],
                          "implementation": {"frontend": [], "backend": {"node": "x"}}, "excluded": False},
            "alice/leetcode": {"repository": "alice/leetcode", "technologies": ["Java"], "categories": [],
                               "excluded": True, "exclusion_reason": "coding_practice"},
        }
        person = analyzer.new_person("")
        person["profiles"] = ["alice"]
        with tempfile.TemporaryDirectory() as temp:
            cache = analyzer.Cache(os.path.join(temp, "cache.json"))
            with patch.object(analyzer, "Cache", return_value=cache), \
                 patch.object(analyzer, "list_repositories", return_value=repos), \
                 patch.object(analyzer, "analyze_repository", side_effect=lambda r: results[r["full_name"]]) as scan_:
                analyzer.run_report([person], None)
        self.assertEqual(scan_.call_count, 2)  # the profile README is never scanned
        self.assertEqual(person["technologies"], ["Express.js", "Node.js"])
        self.assertEqual(person["categories"], [("Node.js Backend", 1)])
        self.assertEqual({(e["repository"], e["reason"]) for e in person["repository_exclusions"]},
                         {("alice/alice", "profile_readme"), ("alice/leetcode", "coding_practice")})


class APIScanTests(unittest.TestCase):
    def tree(self, files):
        return {"truncated": False, "sha": "root", "tree": [
            {"path": path, "type": "blob", "mode": "100644", "sha": sha(text), "size": len(text)}
            for path, text in files.items()]}

    def response(self, payload, status=200):
        result = requests.Response()
        result.status_code = status
        result._content = json.dumps(payload).encode("utf-8")
        return result

    def test_graphql_reads_all_files_in_one_request(self):
        files = {"package.json": package({"express": "^4"}), "index.js": EXPRESS_SERVER,
                 "routes/users.js": "const router = require('express').Router();\n"}
        repo = meta("api")
        api = analyzer._GitHubRepositoryAPI("test-token")

        def graphql(url, json=None, **_):
            self.assertIn('object(oid: "', json["query"])
            self.assertEqual(json["variables"], {"owner": "alice", "name": "api"})
            by_sha = {sha(text): text for text in files.values()}
            oids = [part.split('"')[1] for part in json["query"].split("object(oid: ")[1:]]
            return self.response({"data": {"repository": {
                f"f{i}": {"text": by_sha[oid], "isBinary": False} for i, oid in enumerate(oids)}}})

        with patch.object(requests.Session, "get", return_value=self.response(self.tree(files))) as get, \
             patch.object(requests.Session, "post", side_effect=graphql) as post:
            result = analyzer.analyze_repository(repo, api=api)
        self.assertEqual((get.call_count, post.call_count), (1, 1))
        self.assertIn("Express.js", result["technologies"])
        self.assertIn("Node.js Backend", result["categories"])

    def test_graphql_failure_falls_back_to_rest_blobs(self):
        files = {"requirements.txt": "fastapi\n", "main.py": "from fastapi import FastAPI\n"}
        api = analyzer._GitHubRepositoryAPI("test-token")
        blobs = {sha(text): text for text in files.values()}

        def rest(url, **_):
            if "/git/trees/" in url:
                return self.response(self.tree(files))
            result = requests.Response()
            result.status_code = 200
            result._content = blobs[url.rsplit("/", 1)[-1]].encode("utf-8")
            return result

        limited = self.response({"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]})
        with patch.object(requests.Session, "get", side_effect=rest) as get, \
             patch.object(requests.Session, "post", return_value=limited):
            result = analyzer.analyze_repository(meta("svc"), api=api)
        self.assertFalse(api.graphql_enabled)
        self.assertEqual(get.call_count, 3)
        self.assertIn("FastAPI", result["technologies"])
        self.assertNotIn("error", result)

    def test_empty_and_inaccessible_repositories(self):
        api = analyzer._GitHubRepositoryAPI("test-token")
        with patch.object(requests.Session, "get", return_value=self.response({"message": "empty"}, 409)):
            empty = analyzer.analyze_repository(meta("new"), api=api)
        self.assertNotIn("error", empty)
        self.assertEqual(empty["exclusion_reason"], "empty_repository")
        with patch.object(requests.Session, "get", return_value=self.response({"message": "Not Found"}, 404)):
            gone = analyzer.analyze_repository(meta("gone"), api=api)
        self.assertIn("private, deleted or renamed", gone["error"])


class CacheTests(unittest.TestCase):
    def test_save_drops_old_versions_and_superseded_pushes(self):
        with tempfile.TemporaryDirectory() as temp:
            cache = analyzer.Cache(os.path.join(temp, "cache.json"))
            cache.data = {"v1:7:old": {"categories": []}}
            repo = {"id": 7, "pushed_at": "2026-01-01"}
            cache.set(repo, {"repository": "a/b", "categories": []})
            cache.set(dict(repo, pushed_at="2026-02-01"), {"repository": "a/b", "categories": []})
            cache.save()
            saved = json.loads(Path(temp, "cache.json").read_text(encoding="utf-8"))
        self.assertEqual(list(saved), [f"{analyzer.CACHE_VERSION}:7:2026-02-01"])


if __name__ == "__main__":
    unittest.main()
