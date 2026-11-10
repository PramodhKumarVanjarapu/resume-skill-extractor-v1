"""Tech stack detection for GitHub profiles.

How it works, per repository:
  1. Blobless clone: downloads file names only, no file contents.
  2. List every path locally and detect technologies from file names.
  3. Sparse checkout: download only the few files whose contents matter
     (package.json, requirements.txt, a few .py / .ipynb files, Dockerfiles).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable

import requests

from resume import read_resume

WORKERS = 15              # repositories analysed at the same time
LIST_WORKERS = 4          # profiles whose repo lists are fetched at the same time
RESUME_WORKERS = 4        # resumes downloaded and read at the same time
MAX_MANIFESTS = 15        # dependency files read per repo
MAX_CODE_FILES = 10       # .py / .ipynb files scanned for imports per repo
MAX_MANIFEST_SIZE = 200_000
MAX_CODE_SIZE = 5_000_000
GIT_TIMEOUT = 90          # seconds before a stuck git command is killed
CACHE_VERSION = "v5"      # bump when detection logic changes
CACHE_FILE = "tech-cache.json"

# =====================================================================
# Technology tables. Patterns ending in "*" are prefix matches.
# =====================================================================

JS_PACKAGES: dict[str, list[str]] = {
    "React": ["react"],
    "Next.js": ["next"],
    "Angular": ["@angular/core"],
    "Vue.js": ["vue"],
    "Svelte": ["svelte"],
    "Express.js": ["express"],
    "NestJS": ["@nestjs/core"],
    "Fastify": ["fastify"],
    "TypeScript": ["typescript"],
    "Tailwind CSS": ["tailwindcss"],
    "Prisma": ["prisma", "@prisma/client"],
    "MongoDB": ["mongodb", "mongoose"],
    "PostgreSQL": ["pg", "postgres"],
    "MySQL": ["mysql", "mysql2"],
    "Redis": ["redis", "ioredis"],
    "Zod": ["zod"],
    "BullMQ": ["bullmq"],
    "Socket.IO": ["socket.io"],
    "Drizzle ORM": ["drizzle-orm"],
    "Vite": ["vite"],
    "Redux": ["redux", "@reduxjs/toolkit"],
    "React Query": ["@tanstack/react-query"],
    "Jest": ["jest"],
    "Vitest": ["vitest"],
    "Playwright": ["playwright", "@playwright/test"],
    "React Native": ["react-native", "expo"],
    # AI / LLM
    "LangChain": ["langchain", "@langchain/*"],
    "LangGraph": ["@langchain/langgraph"],
    "LlamaIndex": ["llamaindex"],
    "OpenAI SDK": ["openai"],
    "Anthropic SDK": ["@anthropic-ai/sdk"],
    "Google Gemini SDK": ["@google/genai", "@google/generative-ai"],
    "Vercel AI SDK": ["ai"],
    "Groq": ["groq-sdk"],
    "Ollama": ["ollama"],
    "Transformers.js": ["@huggingface/transformers", "@xenova/transformers"],
    "Hugging Face Hub": ["@huggingface/inference"],
    "TensorFlow.js": ["@tensorflow/tfjs*"],
    # Vector databases
    "Pinecone": ["@pinecone-database/pinecone"],
    "ChromaDB": ["chromadb"],
    "Qdrant": ["@qdrant/js-client-rest"],
    "Weaviate": ["weaviate-client", "weaviate-ts-client"],
    "pgvector": ["pgvector"],
}

# technology: (pip / conda package names, Python import names)
PYTHON_TECH: dict[str, tuple[list[str], list[str]]] = {
    # Web / apps
    "FastAPI": (["fastapi"], ["fastapi"]),
    "Django": (["django"], ["django"]),
    "Flask": (["flask"], ["flask"]),
    "Streamlit": (["streamlit"], ["streamlit"]),
    "Gradio": (["gradio"], ["gradio"]),
    "SQLAlchemy": (["sqlalchemy"], ["sqlalchemy"]),
    "Pydantic": (["pydantic"], ["pydantic"]),
    # Data
    "Pandas": (["pandas"], ["pandas"]),
    "NumPy": (["numpy"], ["numpy"]),
    # Classical ML
    "scikit-learn": (["scikit-learn", "sklearn"], ["sklearn"]),
    "XGBoost": (["xgboost", "py-xgboost"], ["xgboost"]),
    "LightGBM": (["lightgbm"], ["lightgbm"]),
    "CatBoost": (["catboost"], ["catboost"]),
    # Deep learning
    "PyTorch": (["torch", "pytorch"], ["torch"]),
    "TensorFlow": (["tensorflow", "tensorflow-cpu", "tensorflow-gpu"], ["tensorflow"]),
    "Keras": (["keras"], ["keras", "tensorflow.keras"]),
    "JAX": (["jax", "flax"], ["jax", "flax"]),
    # Computer vision
    "OpenCV": (
        ["opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv"],
        ["cv2"],
    ),
    "torchvision": (["torchvision"], ["torchvision"]),
    "Ultralytics YOLO": (["ultralytics"], ["ultralytics"]),
    # NLP / Hugging Face
    "Transformers": (["transformers"], ["transformers"]),
    "Sentence Transformers": (["sentence-transformers"], ["sentence_transformers"]),
    "Hugging Face Hub": (["huggingface-hub"], ["huggingface_hub"]),
    "PEFT": (["peft"], ["peft"]),
    "spaCy": (["spacy"], ["spacy"]),
    "NLTK": (["nltk"], ["nltk"]),
    # LLM frameworks & SDKs
    "LangChain": (["langchain*"], ["langchain*"]),
    "LangGraph": (["langgraph*"], ["langgraph"]),
    "LlamaIndex": (["llama-index*"], ["llama_index"]),
    "OpenAI SDK": (["openai"], ["openai"]),
    "Anthropic SDK": (["anthropic"], ["anthropic"]),
    "Google Gemini SDK": (
        ["google-generativeai", "google-genai"],
        ["google.generativeai", "google.genai"],
    ),
    "Groq": (["groq"], ["groq"]),
    "Ollama": (["ollama"], ["ollama"]),
    # Agent frameworks
    "CrewAI": (["crewai*"], ["crewai"]),
    "AutoGen": (["pyautogen", "autogen*"], ["autogen", "autogen_agentchat", "autogen_core"]),
    "OpenAI Agents SDK": (["openai-agents"], []),
    # Vector databases
    "ChromaDB": (["chromadb"], ["chromadb"]),
    "FAISS": (["faiss", "faiss-cpu", "faiss-gpu"], ["faiss"]),
    "Pinecone": (["pinecone", "pinecone-client"], ["pinecone"]),
    "Qdrant": (["qdrant-client"], ["qdrant_client"]),
    "Weaviate": (["weaviate-client"], ["weaviate"]),
    "Milvus": (["pymilvus"], ["pymilvus"]),
    "pgvector": (["pgvector"], ["pgvector"]),
}

# =====================================================================
# Project categories
# =====================================================================

LLM = {
    "LangChain", "LangGraph", "LlamaIndex", "OpenAI SDK", "Anthropic SDK",
    "Google Gemini SDK", "Groq", "Ollama", "Vercel AI SDK", "CrewAI", "AutoGen",
    "OpenAI Agents SDK", "Semantic Kernel", "Azure OpenAI SDK",
}
RETRIEVAL = {
    "ChromaDB", "FAISS", "Pinecone", "Qdrant", "Weaviate", "Milvus", "pgvector",
    "Sentence Transformers",
}
AGENTS = {"LangGraph", "CrewAI", "AutoGen", "OpenAI Agents SDK"}
DEEP_LEARNING = {"PyTorch", "TensorFlow", "Keras", "JAX", "TensorFlow.js"}
CLASSICAL_ML = {"scikit-learn", "XGBoost", "LightGBM", "CatBoost", "ML.NET"}
VISION = {"OpenCV", "torchvision", "Ultralytics YOLO"}
NLP = {"Transformers", "Transformers.js", "spaCy", "NLTK", "PEFT"}


def categorize(technologies: list[str]) -> list[str]:
    t = set(technologies)
    cats: set[str] = set()
    llm = bool(t & LLM)
    retrieval = bool(t & RETRIEVAL)

    if "LlamaIndex" in t or (retrieval and (llm or "Transformers" in t)):
        cats.add("RAG")
    if llm:
        cats.add("GenAI / LLM")
    if t & AGENTS:
        cats.add("Agentic AI")
    if t & DEEP_LEARNING:
        cats.add("Deep Learning")
    if t & CLASSICAL_ML and not t & DEEP_LEARNING:
        cats.add("Machine Learning")
    if t & VISION:
        cats.add("Computer Vision")
    if t & NLP:
        cats.add("NLP")
    if not cats and ({"Pandas", "NumPy"} & t):
        cats.add("Data Analysis")
    cats.update(stack_categories(technologies))  # MERN, Java Full Stack, Frontend, ...
    if not cats and ({"HTML", "CSS"} & t):
        cats.add("Static Website")
    return sorted(cats)


# =====================================================================
# Stack categories (checked per repository, so frontend/ + backend/ folders combine)
# =====================================================================

FRONTEND = {"React", "Angular", "Vue.js", "Next.js", "Svelte"}
NODE_BACKEND = {"Express.js", "NestJS", "Fastify"}
JAVA_BACKEND = {"Spring Boot", "Servlets"}
PYTHON_WEB = {"Django", "Flask", "FastAPI"}
MOBILE = {"Flutter", "React Native", "Android"}


def stack_categories(technologies: list[str]) -> list[str]:
    t = set(technologies)
    cats: list[str] = []

    # A React Native app uses React, but it is a mobile app, not a web frontend.
    frontend = bool(FRONTEND & t) and "React Native" not in t
    has_backend = False

    if NODE_BACKEND & t:
        has_backend = True
        mern = frontend and "MongoDB" in t and bool({"React", "Next.js"} & t)
        mean = frontend and "MongoDB" in t and "Angular" in t
        if mern:
            cats.append("MERN")
        if mean:
            cats.append("MEAN")
        if not (mern or mean):
            cats.append("Node.js Full Stack" if frontend else "Node.js Backend")

    if JAVA_BACKEND & t:
        has_backend = True
        java_ui = frontend or bool({"Thymeleaf", "JSP", "HTML Templates"} & t)
        cats.append("Java Full Stack" if java_ui else "Java Backend")

    if PYTHON_WEB & t:
        has_backend = True
        python_ui = frontend or "HTML Templates" in t
        cats.append("Python Full Stack" if python_ui else "Python Backend")

    if "ASP.NET Core" in t:
        has_backend = True
        dotnet_ui = frontend or bool({"Blazor", "Razor Views"} & t)
        cats.append(".NET Full Stack" if dotnet_ui else ".NET Backend")

    if frontend and not has_backend:
        cats.append("Frontend")

    if MOBILE & t:
        cats.append("Mobile App")

    return cats


# =====================================================================
# File classification
# =====================================================================

IGNORED_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", ".next",
    ".venv", "venv", "env", "coverage", "site-packages", ".ipynb_checkpoints",
}

PATH_ONLY = {
    "go.mod": "Go",
    "cargo.toml": "Rust",
    "pom.xml": "Maven",
    "build.gradle": "Gradle",
    "build.gradle.kts": "Gradle",
    "composer.json": "PHP",
    "gemfile": "Ruby",
    "pubspec.yaml": "Flutter",
    "androidmanifest.xml": "Android",
    "chart.yaml": "Helm",
}

PYTHON_MANIFESTS = {
    "requirements.txt", "pyproject.toml", "pipfile", "setup.py",
    "environment.yml", "environment.yaml",
}
DOCKER_FILES = {
    "dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml",
}
JVM_BUILD_FILES = {"pom.xml", "build.gradle", "build.gradle.kts"}
DOTNET_PROJECT_EXT = (".csproj", ".fsproj", ".vbproj")
MANIFEST_FILES = {"package.json"} | PYTHON_MANIFESTS | DOCKER_FILES | JVM_BUILD_FILES


# =====================================================================
# Matching helpers
# =====================================================================

def matches_name(name: str, pattern: str) -> bool:
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def normalize_pip(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower())


PIP_INDEX = [
    (tech, [normalize_pip(p[:-1]) + "*" if p.endswith("*") else normalize_pip(p) for p in pips])
    for tech, (pips, _) in PYTHON_TECH.items()
]


def add_pip_name(raw: str, techs: set[str]) -> None:
    name = normalize_pip(raw)
    for tech, pips in PIP_INDEX:
        if any(matches_name(name, p) for p in pips):
            techs.add(tech)


def add_import(module: str, techs: set[str]) -> None:
    mod = module.lower()
    for tech, (_, imports) in PYTHON_TECH.items():
        for p in imports:
            hit = mod.startswith(p[:-1]) if p.endswith("*") else (mod == p or mod.startswith(p + "."))
            if hit:
                techs.add(tech)
                break


# =====================================================================
# Detectors
# =====================================================================

LINE_REQ = re.compile(r"""^\s*(?:-\s+)?["']?([A-Za-z0-9_.-]+)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~;"',]|$)""")
QUOTED_REQ = re.compile(r"""["']([A-Za-z0-9_.-]+)\s*(?:\[[^\]]*\])?\s*(?:[<>=!~;,]|["'])""")
PIP_LINE = re.compile(r"^[!%]\s*pip3?\s+install\s+(.+)$")
FROM_LINE = re.compile(r"^from\s+([A-Za-z_][\w.]*)\s+import\s+(.+)$")
IMPORT_LINE = re.compile(r"^import\s+(.+)$")


def detect_package_json(content: str, techs: set[str]) -> None:
    try:
        parsed = json.loads(content)
    except ValueError:
        return
    if not isinstance(parsed, dict):
        return
    deps: set[str] = set()
    for key in ("dependencies", "devDependencies", "peerDependencies"):
        if isinstance(parsed.get(key), dict):
            deps.update(parsed[key])
    for tech, patterns in JS_PACKAGES.items():
        if any(matches_name(d, p) for d in deps for p in patterns):
            techs.add(tech)


def detect_python_manifest(content: str, techs: set[str]) -> None:
    # One requirement per line: torch>=2 | - numpy=1.2 (conda) | fastapi = "^0.1" (poetry)
    for line in content.splitlines():
        m = LINE_REQ.match(line)
        if m:
            add_pip_name(m.group(1), techs)
    # Quoted requirements anywhere: dependencies = ["torch>=2", "numpy"] / install_requires=[...]
    for m in QUOTED_REQ.finditer(content):
        add_pip_name(m.group(1), techs)


def detect_docker(content: str, techs: set[str]) -> None:
    checks = {
        "PostgreSQL": r"postgres|postgresql",
        "MongoDB": r"mongodb|\bmongo\b",
        "Redis": r"\bredis\b",
        "Nginx": r"\bnginx\b",
        "Qdrant": r"qdrant",
        "ChromaDB": r"chroma",
        "Ollama": r"ollama",
    }
    for tech, pattern in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            techs.add(tech)


def detect_jvm_build(content: str, techs: set[str]) -> None:
    """pom.xml / build.gradle: Spring Boot and the libraries Java projects usually list."""
    checks = {
        "Spring Boot": r"spring-boot|org\.springframework\.boot",
        "Hibernate / JPA": r"hibernate|spring-boot-starter-data-jpa|jakarta\.persistence|javax\.persistence",
        "Thymeleaf": r"thymeleaf",
        "Servlets": r"javax\.servlet|jakarta\.servlet",
        "MySQL": r"mysql-connector|com\.mysql",
        "PostgreSQL": r"org\.postgresql|\bpostgresql\b",
        "MongoDB": r"mongodb",
        "Redis": r"\bredis\b",
        "Kafka": r"\bkafka\b",
        "JUnit": r"\bjunit\b",
    }
    for tech, pattern in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            techs.add(tech)


def detect_dotnet_project(content: str, techs: set[str]) -> None:
    """.csproj / .fsproj / .vbproj / packages.config"""
    techs.add(".NET")
    checks = {
        "ASP.NET Core": r"Microsoft\.NET\.Sdk\.Web|Microsoft\.AspNetCore",
        "Blazor": r"Microsoft\.NET\.Sdk\.BlazorWebAssembly|Microsoft\.AspNetCore\.Components",
        ".NET MAUI": r"<UseMaui>\s*true",
        "Entity Framework Core": r"Microsoft\.EntityFrameworkCore|EntityFramework",
        "SQL Server": r"EntityFrameworkCore\.SqlServer|Microsoft\.Data\.SqlClient|System\.Data\.SqlClient",
        "PostgreSQL": r"Npgsql",
        "MySQL": r"Pomelo\.EntityFrameworkCore\.MySql|MySql\.Data|MySqlConnector",
        "MongoDB": r"MongoDB\.Driver",
        "Redis": r"StackExchange\.Redis",
        "SignalR": r"SignalR",
        "Dapper": r"\bDapper\b",
        "xUnit": r"\bxunit\b",
        "NUnit": r"\bNUnit\b",
        "ML.NET": r"\bMicrosoft\.ML\b",
        "Semantic Kernel": r"Microsoft\.SemanticKernel",
        "Azure OpenAI SDK": r"Azure\.AI\.OpenAI",
    }
    for tech, pattern in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            techs.add(tech)


def detect_python_source(source: str, techs: set[str]) -> None:
    for raw in source.splitlines():
        line = raw.strip()

        pip = PIP_LINE.match(line)
        if pip:
            for token in pip.group(1).split():
                if token.startswith("-"):
                    continue
                name = re.split(r"[<>=!~\[;]", token.replace('"', "").replace("'", ""))[0]
                if name:
                    add_pip_name(name, techs)
            continue

        frm = FROM_LINE.match(line)
        if frm:
            module = frm.group(1)
            add_import(module, techs)
            # "from google import genai" -> also check "google.genai"
            for part in frm.group(2).replace("(", "").replace(")", "").split(","):
                words = part.split()
                if words and re.fullmatch(r"[A-Za-z_]\w*", words[0]):
                    add_import(f"{module}.{words[0]}", techs)
            continue

        imp = IMPORT_LINE.match(line)
        if imp:
            for part in imp.group(1).split(","):
                words = part.split()
                if words and re.fullmatch(r"[A-Za-z_][\w.]*", words[0]):
                    add_import(words[0], techs)


def notebook_to_source(content: str) -> str:
    try:
        nb = json.loads(content)
    except ValueError:
        return ""
    cells = nb.get("cells", []) if isinstance(nb, dict) else []
    parts = []
    for cell in cells:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            src = cell.get("source", "")
            parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


# =====================================================================
# git + GitHub
# =====================================================================

class GitHubError(Exception):
    pass


def git(args: list[str], cwd: str | None = None) -> bytes:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, env=env, capture_output=True, timeout=GIT_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"git {args[0]} timed out")
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(message or f"git {args[0]} failed")
    return result.stdout


def _remove_readonly(func, path, _exc):
    # Windows: git object files are read-only and block deletion.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_dir(path: str) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_remove_readonly)
    else:
        shutil.rmtree(path, onerror=_remove_readonly)


def list_repositories(username: str, token: str | None, skip_forks: bool = True) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repos: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"type": "owner", "per_page": 100, "page": page, "sort": "updated"},
            headers=headers,
            timeout=30,
        )
        if r.status_code == 404:
            raise GitHubError("GitHub user not found")
        if r.status_code == 401:
            raise GitHubError("GitHub token is invalid or expired")
        if r.status_code == 429 or (r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0"):
            reset = r.headers.get("x-ratelimit-reset")
            when = datetime.fromtimestamp(int(reset)).strftime("%I:%M %p") if reset else "later"
            raise GitHubError(f"GitHub rate limit reached. Add a GitHub token or try again after {when}")
        r.raise_for_status()

        batch = r.json()
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return [repo for repo in repos if not (skip_forks and repo.get("fork"))]


# GitHub's language names that duplicate a technology we detect ourselves.
LANGUAGE_ALIASES = {"Vue": "Vue.js"}


def sparse_pattern(path: str) -> str:
    return "/" + re.sub(r"([*?\[\]\\!#])", r"\\\1", path)


def depth(path: str) -> int:
    return path.count("/")


def analyze_repository(repo: dict) -> dict:
    techs: set[str] = set()
    if repo.get("language"):
        techs.add(LANGUAGE_ALIASES.get(repo["language"], repo["language"]))

    def finish(error: str | None = None) -> dict:
        technologies = sorted(techs)
        result = {
            "repository": repo["full_name"],
            "url": repo.get("html_url", ""),
            "primaryLanguage": repo.get("language"),
            "technologies": technologies,
            "categories": categorize(technologies),
        }
        if error:
            result["error"] = error
        return result

    if repo.get("size") == 0:
        return finish()

    tmp = tempfile.mkdtemp(prefix="tech-analyzer-")
    try:
        # 1. Blobless clone: file names only, no contents.
        git([
            "clone", "--depth", "1", "--single-branch",
            "--filter=blob:none", "--no-checkout",
            repo["clone_url"], tmp,
        ])

        # 2. List all paths locally.
        output = git(["ls-tree", "-r", "--name-only", "-z", "HEAD"], cwd=tmp)
        paths = [p for p in output.decode("utf-8", "replace").split("\0") if p]

        manifests: list[str] = []
        code: list[str] = []

        for file_path in paths:
            parts = file_path.split("/")
            if any(part in IGNORED_DIRS for part in parts):
                continue
            name = parts[-1].lower()

            if file_path.startswith(".github/workflows/"):
                techs.add("GitHub Actions")
            if name.endswith(".tf"):
                techs.add("Terraform")
            if name in PATH_ONLY:
                techs.add(PATH_ONLY[name])
            if name in PYTHON_MANIFESTS:
                techs.add("Python")
            if name in DOCKER_FILES:
                techs.add("Docker")
            if name.endswith(".ipynb"):
                techs.add("Jupyter Notebook")
            if name.endswith(".html") and "templates" in parts[:-1]:
                techs.add("HTML Templates")
            if name.endswith(".sln"):
                techs.add(".NET")
            if name.endswith(".jsp"):
                techs.add("JSP")
            if name.endswith(".cshtml"):
                techs.add("Razor Views")
            if name.endswith(".razor"):
                techs.add("Blazor")

            if name in MANIFEST_FILES or name.endswith(DOTNET_PROJECT_EXT) or name == "packages.config":
                manifests.append(file_path)
            elif name.endswith(".py") or name.endswith(".ipynb"):
                code.append(file_path)

        # Prefer files near the root.
        manifests.sort(key=depth)
        code.sort(key=depth)
        selected = manifests[:MAX_MANIFESTS] + code[:MAX_CODE_FILES]

        # 3. Sparse checkout: download only the selected files, in one batch.
        if selected:
            git(["config", "core.sparseCheckout", "true"], cwd=tmp)
            info_dir = os.path.join(tmp, ".git", "info")
            os.makedirs(info_dir, exist_ok=True)
            with open(os.path.join(info_dir, "sparse-checkout"), "w", encoding="utf-8") as f:
                f.write("\n".join(sparse_pattern(p) for p in selected) + "\n")
            git(["read-tree", "-mu", "HEAD"], cwd=tmp)

        for rel in selected:
            full = os.path.join(tmp, *rel.split("/"))
            if not os.path.isfile(full):
                continue
            name = rel.rsplit("/", 1)[-1].lower()
            is_code = name.endswith(".py") or name.endswith(".ipynb")
            limit = MAX_CODE_SIZE if is_code else MAX_MANIFEST_SIZE
            if os.path.getsize(full) > limit:
                continue

            with open(full, encoding="utf-8", errors="replace") as f:
                content = f.read()

            if name == "package.json":
                detect_package_json(content, techs)
            elif name in PYTHON_MANIFESTS:
                detect_python_manifest(content, techs)
                if name == "setup.py":
                    detect_python_source(content, techs)
            elif name in DOCKER_FILES:
                detect_docker(content, techs)
            elif name in JVM_BUILD_FILES:
                detect_jvm_build(content, techs)
            elif name.endswith(DOTNET_PROJECT_EXT) or name == "packages.config":
                detect_dotnet_project(content, techs)
            elif name.endswith(".ipynb"):
                detect_python_source(notebook_to_source(content), techs)
            elif name.endswith(".py"):
                detect_python_source(content, techs)

        return finish()
    except Exception as e:  # noqa: BLE001 - report any failure for this repo
        return finish(str(e))
    finally:
        remove_dir(tmp)


# =====================================================================
# Cache (shared format with the earlier TypeScript version)
# =====================================================================

class Cache:
    def __init__(self, path: str = CACHE_FILE):
        self.path = path
        self.lock = threading.Lock()
        try:
            with open(path, encoding="utf-8") as f:
                self.data: dict[str, dict] = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def key(repo: dict) -> str:
        return f"{CACHE_VERSION}:{repo['id']}:{repo.get('pushed_at') or ''}"

    def get(self, repo: dict) -> dict | None:
        with self.lock:
            hit = self.data.get(self.key(repo))
        return hit if hit and not hit.get("error") and "categories" in hit else None

    def set(self, repo: dict, result: dict) -> None:
        if not result.get("error"):
            with self.lock:
                self.data[self.key(repo)] = result

    def save(self) -> None:
        with self.lock:
            text = json.dumps(self.data, indent=2)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.path)


# =====================================================================
# People input
# =====================================================================

def to_username(raw: str) -> str | None:
    """Accepts: username, @username, github.com/username, https://github.com/username/repo"""
    s = raw.strip().lstrip("@")
    m = re.search(r"github\.com/([^/?#\s]+)", s, re.IGNORECASE)
    if m:
        s = m.group(1)
    return s if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", s) else None


def clean_phone(phone: str) -> str:
    return re.sub(r"[^\d+]", "", phone or "")


def looks_like_phone(value: str) -> bool:
    return bool(re.fullmatch(r"[+\d][\d\s().-]{6,}", value.strip()))


def to_resume_url(raw: str) -> str | None:
    """Accepts https://..., http://..., www.example.com/..., drive.google.com/..."""
    s = raw.strip()
    if re.match(r"(?i)^https?://\S+$", s):
        return s
    if re.match(r"(?i)^(www\.|[a-z0-9-]+(\.[a-z0-9-]+)+/)\S*$", s):
        return "https://" + s
    return None


def is_github_cell(cell: str) -> bool:
    return bool(re.search(r"(?i)github\.com/", cell)) or not re.search(r"[/.]", cell)


def parse_pasted(text: str) -> list[dict]:
    """Rows pasted from Excel / Google Sheets / CSV. Column order doesn't matter.

    Each cell is sorted by what it looks like: phone number, GitHub profile
    (github.com link or bare username) or resume link (any other link).
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in (line.split("\t") if "\t" in line else line.split(","))]
        cells = [c for c in cells if c]
        phone_index = next((i for i, c in enumerate(cells) if looks_like_phone(c)), None)
        phone = cells[phone_index] if phone_index is not None else ""
        others = [c for i, c in enumerate(cells) if i != phone_index]
        github = " ".join(c for c in others if is_github_cell(c))
        resume = " ".join(c for c in others if not is_github_cell(c))
        if not phone and re.search(r"(?i)\bphone\b|github|resume|\bcv\b", " ".join(others)) and not re.search(
            r"(?i)github\.com/|https?://", " ".join(others)
        ):
            continue  # header row
        rows.append({"phone": phone, "resume": resume, "github": github})
    return rows


def new_person(phone: str) -> dict:
    return {
        "phone": phone,
        "profiles": [],
        "resumes": [],
        "resume_projects": [],
        "resume_repo_links": [],
        "errors": [],
        "status": "Waiting",
        "done": 0,
        "total": 0,
        "repositories": 0,
        "failed_repos": 0,
        "categories": [],
        "technologies": [],
    }


def build_people(rows: list[dict]) -> list[dict]:
    """One person per phone number. Rows with the same phone are merged."""
    people: dict[str, dict] = {}

    for row in rows:
        phone = clean_phone(str(row.get("phone") or ""))
        tokens = [t for t in re.split(r"[\s,;]+", str(row.get("github") or "")) if t]
        resume_tokens = [t for t in re.split(r"\s+", str(row.get("resume") or "")) if t]
        if not phone and not tokens and not resume_tokens:
            continue

        profiles, resumes, errors = [], [], []
        for token in tokens:
            username = to_username(token)
            if username:
                profiles.append(username)
            else:
                errors.append(f"Not a GitHub profile: {token}")
        for token in resume_tokens:
            url = to_resume_url(token)
            if url:
                resumes.append(url)
            else:
                errors.append(f"Not a resume link: {token}")

        key = phone or "link:" + ",".join(profiles + resumes).lower()
        person = people.setdefault(key, new_person(phone))
        for p in profiles:
            if p.lower() not in (x.lower() for x in person["profiles"]):
                person["profiles"].append(p)
        for r in resumes:
            if r not in person["resumes"]:
                person["resumes"].append(r)
        person["errors"].extend(errors)

    for person in people.values():
        if not person["profiles"] and not person["resumes"]:
            person["status"] = "Failed"
            if not person["errors"]:
                person["errors"].append("No resume link or GitHub profile given")

    return list(people.values())


# =====================================================================
# Running a report
# =====================================================================

def count_by(lists: list[list[str]]) -> list[tuple[str, int]]:
    counter = Counter(item for items in lists for item in items)
    return counter.most_common()


def run_report(
    people: list[dict],
    token: str | None,
    skip_forks: bool = True,
    on_progress: Callable[[list[dict], int, int], None] | None = None,
) -> list[dict]:
    """Fills in each person's results. on_progress(people, repos_done, repos_total)."""
    cache = Cache()

    # 0. Read resumes: find GitHub profiles and the projects listed in the resume.
    resume_people = [p for p in people if p["resumes"]]
    if resume_people:
        for p in resume_people:
            p["status"] = "Reading resume"
        if on_progress:
            on_progress(people, 0, 0)

        urls = sorted({u for p in resume_people for u in p["resumes"]})
        found: dict[str, dict] = {}
        with ThreadPoolExecutor(RESUME_WORKERS) as pool:
            futures = {pool.submit(read_resume, u): u for u in urls}
            for future in as_completed(futures):
                u = futures[future]
                try:
                    found[u] = future.result()
                except Exception as e:  # noqa: BLE001
                    found[u] = {"profiles": [], "projects": [], "repo_links": [], "error": str(e)}

        for p in resume_people:
            from_resume: list[str] = []
            for u in p["resumes"]:
                info = found[u]
                if info.get("error"):
                    p["errors"].append(f"Resume: {info['error']}")
                for name in info.get("profiles", []):
                    if name.lower() not in (x.lower() for x in p["profiles"]):
                        p["profiles"].append(name)
                        from_resume.append(name)
                known = {x["name"].lower() for x in p["resume_projects"]}
                for project in info.get("projects", []):
                    if project["name"].lower() not in known:
                        p["resume_projects"].append(project)
                        known.add(project["name"].lower())
                for link in info.get("repo_links", []):
                    if link not in p["resume_repo_links"]:
                        p["resume_repo_links"].append(link)
            if len(from_resume) > 1:
                p["errors"].append(
                    "Several GitHub profiles found in resume: " + ", ".join(from_resume)
                )
            if not p["profiles"]:
                p["status"] = "No GitHub found"
                if not any(e.startswith("Resume:") for e in p["errors"]):
                    p["errors"].append("No GitHub profile found in resume")

    active = [p for p in people if p["profiles"]]
    usernames = sorted({u.lower(): u for p in active for u in p["profiles"]}.values())

    for p in active:
        p["status"] = "Finding repos"
    if on_progress:
        on_progress(people, 0, 0)

    # 1. List repositories for every profile.
    repos_by_user: dict[str, list[dict]] = {}
    user_errors: dict[str, str] = {}
    with ThreadPoolExecutor(LIST_WORKERS) as pool:
        futures = {pool.submit(list_repositories, u, token, skip_forks): u for u in usernames}
        for future in as_completed(futures):
            u = futures[future]
            try:
                repos_by_user[u.lower()] = future.result()
            except GitHubError as e:
                user_errors[u.lower()] = str(e)
            except Exception as e:  # noqa: BLE001
                user_errors[u.lower()] = f"Could not list repos: {e}"

    # Each person's repos, deduplicated by full name.
    person_repos: list[dict[str, dict]] = []
    for p in active:
        repos: dict[str, dict] = {}
        for u in p["profiles"]:
            if u.lower() in user_errors:
                p["errors"].append(f"{u}: {user_errors[u.lower()]}")
            for repo in repos_by_user.get(u.lower(), []):
                repos[repo["full_name"]] = repo
        person_repos.append(repos)
        p["total"] = len(repos)
        p["repositories"] = len(repos)
        p["status"] = "Analysing"

    # 2. Analyse every unique repository once (cache first).
    all_repos: dict[str, dict] = {}
    for repos in person_repos:
        all_repos.update(repos)

    results: dict[str, dict] = {}
    to_run = []
    for name, repo in all_repos.items():
        hit = cache.get(repo)
        if hit:
            results[name] = hit
        else:
            to_run.append(repo)

    def refresh():
        for p, repos in zip(active, person_repos):
            p["done"] = sum(1 for name in repos if name in results)
        if on_progress:
            on_progress(people, len(results), len(all_repos))

    refresh()
    last_update = time.monotonic()

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = {pool.submit(analyze_repository, repo): repo for repo in to_run}
        for future in as_completed(futures):
            repo = futures[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                result = {"repository": repo["full_name"], "technologies": [], "categories": [], "error": str(e)}
            results[repo["full_name"]] = result
            cache.set(repo, result)
            if time.monotonic() - last_update > 0.5:
                refresh()
                last_update = time.monotonic()

    cache.save()
    refresh()

    # 3. Combine results per person.
    for p, repos in zip(active, person_repos):
        analysed = [results[name] for name in repos if name in results]
        p["failed_repos"] = sum(1 for r in analysed if r.get("error"))
        # Recomputed from technologies, so older cached results get the new categories too.
        p["categories"] = count_by([categorize(r.get("technologies", [])) for r in analysed])
        p["technologies"] = [name for name, _ in count_by([r.get("technologies", []) for r in analysed])]
        failed_profiles = sum(1 for u in p["profiles"] if u.lower() in user_errors)
        p["status"] = "Failed" if failed_profiles == len(p["profiles"]) else "Done"

    if on_progress:
        on_progress(people, len(results), len(all_repos))
    return people


# =====================================================================
# Output
# =====================================================================

def notes(person: dict) -> str:
    items = list(person["errors"])
    if person["failed_repos"]:
        items.append(f"{person['failed_repos']} repos could not be read")
    return "; ".join(items)


def format_categories(categories: list[tuple[str, int]]) -> str:
    return "; ".join(f"{name} ({count})" for name, count in categories)


def people_to_rows(people: list[dict]) -> list[dict]:
    return [
        {
            "Phone": p["phone"],
            "Resume": " ".join(p["resumes"]),
            "GitHub profiles": " ".join(f"https://github.com/{u}" for u in p["profiles"]),
            "Projects in resume": "; ".join(x["name"] for x in p["resume_projects"]),
            "Repositories": p["repositories"],
            "Project categories": format_categories(p["categories"]),
            "Technologies": ", ".join(p["technologies"]),
            "Status": p["status"],
            "Notes": notes(p),
        }
        for p in people
    ]


def people_to_csv(people: list[dict]) -> bytes:
    buffer = io.StringIO()
    rows = people_to_rows(people)
    writer = csv.DictWriter(
        buffer,
        fieldnames=["Phone", "Resume", "GitHub profiles", "Projects in resume", "Repositories",
                    "Project categories", "Technologies", "Status", "Notes"],
        quoting=csv.QUOTE_ALL,
        lineterminator="\r\n",
    )
    writer.writeheader()
    for row in rows:
        # ="+9198..." stops Excel turning phone numbers into 9.19E+11
        if row["Phone"]:
            row["Phone"] = f'="{row["Phone"]}"'
        writer.writerow(row)
    # utf-8-sig adds a BOM so Excel opens the file as UTF-8
    return buffer.getvalue().encode("utf-8-sig")