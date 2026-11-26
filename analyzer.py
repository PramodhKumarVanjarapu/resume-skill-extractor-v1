"""Tech stack detection for GitHub profiles.

Repository scans use either a blobless clone with sparse checkout or GitHub's
authenticated Git Trees and Blobs API. Both use the same path and file content
detectors.
"""

from __future__ import annotations

import csv
import io
import json
import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from functools import lru_cache, partial
from typing import Callable, Iterator, Literal
from urllib.parse import quote

import requests

from resume import read_resume
from repository_filter import (
    practice_signal, pre_analysis_exclusion, score_repository, select_for_candidate,
)

WORKERS = 8               # concurrent Git clones (I/O bound: network + git subprocesses)
API_FILE_WORKERS = 4      # concurrent blob reads per repository (REST fallback only)
GRAPHQL_URL = "https://api.github.com/graphql"
GRAPHQL_MAX_FILES = 40    # files fetched per GraphQL request in API mode
GRAPHQL_MAX_BYTES = 2_000_000
DUPLICATE_SIMILARITY = 0.85  # Jaccard similarity of source blobs that marks a re-uploaded copy
DEFAULT_PARALLEL_GITHUB_REQUESTS = 20
MAX_PARALLEL_GITHUB_REQUESTS = 32
RepositoryScanMethod = Literal["clone", "api"]
MAX_INITIAL_REPOS_PER_CANDIDATE = 15
LOG_MAX_BYTES = 5_000_000
LOG_BACKUPS = 3
logger = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()
_API_LOCK = threading.Lock()
_API_REQUESTS = 0


def record_api_request() -> None:
    global _API_REQUESTS
    with _API_LOCK:
        _API_REQUESTS += 1


def configure_logging(level: int | None = None) -> None:
    """Install one console and one rotating file handler for extraction logs."""
    if level is None:
        level = getattr(logging, os.getenv("GITHUB_EXTRACTOR_LOG_LEVEL", "INFO").upper(), logging.INFO)
    with _LOG_LOCK:
        logger.setLevel(level)
        if getattr(logger, "_extractor_configured", False):
            return
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(module)s %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        try:
            log_dir = os.path.join(os.path.dirname(__file__), "logs")
            os.makedirs(log_dir, exist_ok=True)
            rotating = RotatingFileHandler(os.path.join(log_dir, "github_extractor.log"),
                                           maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8")
            rotating.setFormatter(formatter)
            logger.addHandler(rotating)
        except OSError:
            logger.warning("File logging unavailable")
        logger._extractor_configured = True
RESUME_WORKERS = 4        # resumes downloaded and read at the same time
MAX_MANIFESTS = 15        # dependency files read per repo
MAX_CODE_FILES = 24       # Python / JavaScript / TypeScript files scanned for imports per repo
MAX_MANIFEST_SIZE = 200_000
MAX_SOURCE_SIZE = 500_000     # .py / .js / .ts ... (imports sit at the top; larger files are generated)
MAX_CODE_SIZE = 5_000_000     # notebooks (outputs make them large)
GIT_TIMEOUT = 90          # seconds before a stuck git command is killed
CACHE_VERSION = "v6"      # bump when detection logic changes
CACHE_FILE = "tech-cache.json"
# All temporary clones go under this one folder, so it can be excluded from antivirus scanning
# (Windows Defender scans every file git writes, which slows clones down a lot).
CLONE_ROOT = os.path.join(tempfile.gettempdir(), "tech-analyzer")

# =====================================================================
# Technology tables. Patterns ending in "*" are prefix matches.
# =====================================================================

JS_PACKAGES: dict[str, list[str]] = {
    "React": ["react"],
    "Next.js": ["next"],
    "Angular": ["@angular/core"],
    "Vue.js": ["vue"],
    "Svelte": ["svelte"],
    "Nuxt.js": ["nuxt"],
    "Express.js": ["express"],
    "NestJS": ["@nestjs/core"],
    "Fastify": ["fastify"],
    "Koa.js": ["koa"],
    "TypeScript": ["typescript"],
    "Tailwind CSS": ["tailwindcss"],
    "Bootstrap": ["bootstrap", "react-bootstrap"],
    "Material UI": ["@mui/material", "@material-ui/core"],
    "Chakra UI": ["@chakra-ui/react"],
    "Ant Design": ["antd"],
    "Styled Components": ["styled-components"],
    "Framer Motion": ["framer-motion"],
    "Sass": ["sass", "node-sass"],
    "jQuery": ["jquery"],
    "Chart.js": ["chart.js", "react-chartjs-2"],
    "Three.js": ["three", "@react-three/fiber"],
    "React Router": ["react-router", "react-router-dom"],
    "Axios": ["axios"],
    "Zustand": ["zustand"],
    "Gatsby": ["gatsby"],
    "Electron": ["electron"],
    "Webpack": ["webpack"],
    "GraphQL": ["graphql", "@apollo/client", "@apollo/server", "apollo-server*"],
    "Prisma": ["prisma", "@prisma/client"],
    "Mongoose": ["mongoose"],
    "Sequelize": ["sequelize"],
    "TypeORM": ["typeorm"],
    "MongoDB": ["mongodb", "mongoose"],
    "PostgreSQL": ["pg", "postgres"],
    "MySQL": ["mysql", "mysql2"],
    "SQLite": ["sqlite3", "better-sqlite3"],
    "Redis": ["redis", "ioredis"],
    "Firebase": ["firebase", "firebase-admin"],
    "Supabase": ["@supabase/supabase-js"],
    "JWT": ["jsonwebtoken"],
    "Cloudinary": ["cloudinary"],
    "Zod": ["zod"],
    "BullMQ": ["bullmq"],
    "Socket.IO": ["socket.io", "socket.io-client"],
    "Drizzle ORM": ["drizzle-orm"],
    "Vite": ["vite"],
    "Redux": ["redux", "@reduxjs/toolkit"],
    "React Query": ["@tanstack/react-query"],
    "Jest": ["jest"],
    "Vitest": ["vitest"],
    "Mocha": ["mocha"],
    "Cypress": ["cypress"],
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
    "Django REST Framework": (["djangorestframework"], ["rest_framework"]),
    "Celery": (["celery"], ["celery"]),
    "Gunicorn": (["gunicorn"], []),
    "PyTest": (["pytest"], ["pytest"]),
    # Databases / cloud clients
    "PostgreSQL": (["psycopg2", "psycopg2-binary", "psycopg", "asyncpg"], ["psycopg2", "psycopg", "asyncpg"]),
    "MongoDB": (["pymongo", "motor", "mongoengine"], ["pymongo", "motor", "mongoengine"]),
    "MySQL": (["mysql-connector-python", "pymysql", "mysqlclient"], ["mysql.connector", "pymysql", "mysqldb"]),
    "SQLite": ([], ["sqlite3"]),
    "Redis": (["redis"], ["redis"]),
    "Firebase": (["firebase-admin", "pyrebase*"], ["firebase_admin", "pyrebase"]),
    "AWS": (["boto3"], ["boto3"]),
    # Scraping / automation
    "Selenium": (["selenium"], ["selenium"]),
    "BeautifulSoup": (["beautifulsoup4", "bs4"], ["bs4"]),
    "Scrapy": (["scrapy"], ["scrapy"]),
    # Data
    "Pandas": (["pandas"], ["pandas"]),
    "NumPy": (["numpy"], ["numpy"]),
    "SciPy": (["scipy"], ["scipy"]),
    "Matplotlib": (["matplotlib"], ["matplotlib"]),
    "Seaborn": (["seaborn"], ["seaborn"]),
    "Plotly": (["plotly"], ["plotly"]),
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
CLASSICAL_ML = {"scikit-learn", "Scikit-learn", "XGBoost", "LightGBM", "CatBoost", "ML.NET"}
VISION = {"OpenCV", "torchvision", "Ultralytics YOLO"}
NLP = {"Transformers", "Transformers.js", "spaCy", "NLTK", "PEFT"}


def categorize(technologies: list[str], implementation: Mapping | None = None) -> list[str]:
    """Project categories for one repository.

    ``implementation`` (from ``analyze_repository``) gates the web-stack categories on evidence that the frontend
    and backend are actually implemented. Without it (older cached results) the technology names alone decide.
    """
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
    cats.update(stack_categories(technologies, implementation))  # MERN, Java Full Stack, Frontend, ...
    # A static site is built from HTML files (not a language label) and uses no UI framework or backend.
    if not cats and not (FRONTEND | NODE_BACKEND | JAVA_BACKEND | PYTHON_WEB | DOTNET_BACKEND) & t and (
        "HTML" in t if implementation is not None else {"HTML", "CSS"} & t
    ):
        cats.add("Static Website")
    return sorted(cats)


# =====================================================================
# Stack categories (checked per repository, so frontend/ + backend/ folders combine)
# =====================================================================

FRONTEND = {"React", "Angular", "Vue.js", "Next.js", "Svelte", "Nuxt.js"}
NODE_BACKEND = {"Express.js", "NestJS", "Fastify", "Koa.js"}
JAVA_BACKEND = {"Spring Boot", "Servlets"}
PYTHON_WEB = {"Django", "Flask", "FastAPI"}
DOTNET_BACKEND = {"ASP.NET Core", "ASP.NET"}
MOBILE = {"Flutter", "React Native", "Android"}
BACKEND_STACKS = (("node", NODE_BACKEND), ("java", JAVA_BACKEND), ("python", PYTHON_WEB), ("dotnet", DOTNET_BACKEND))


def stack_categories(technologies: list[str], implementation: Mapping | None = None) -> list[str]:
    """Web/mobile stack categories.

    With ``implementation`` a stack counts only when its side is implemented (see ``_implementation``): frontend
    needs a UI framework that is imported/configured or backed by component files, backend needs framework code.
    Programming languages alone never produce a category.
    """
    t = set(technologies)
    cats: list[str] = []

    # A React Native app uses React, but it is a mobile app, not a web frontend.
    if implementation is not None:
        frontend = bool(implementation.get("frontend")) and "React Native" not in t
        backends = set(implementation.get("backend") or ())
    else:
        frontend = bool(FRONTEND & t) and "React Native" not in t
        backends = {stack for stack, names in BACKEND_STACKS if names & t}
    has_backend = False

    if "node" in backends:
        has_backend = True
        mern = frontend and "MongoDB" in t and bool({"React", "Next.js"} & t)
        mean = frontend and "MongoDB" in t and "Angular" in t
        if mern:
            cats.append("MERN")
        if mean:
            cats.append("MEAN")
        if not (mern or mean):
            cats.append("Node.js Full Stack" if frontend else "Node.js Backend")

    if "java" in backends:
        has_backend = True
        java_ui = frontend or bool({"Thymeleaf", "JSP", "HTML Templates"} & t)
        cats.append("Java Full Stack" if java_ui else "Java Backend")

    if "python" in backends:
        has_backend = True
        python_ui = frontend or "HTML Templates" in t
        cats.append("Python Full Stack" if python_ui else "Python Backend")

    if "dotnet" in backends:
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

# Build output, dependencies and tool caches: never evidence of what the candidate wrote (compared lowercase).
IGNORED_DIRS = {
    ".git", "node_modules", "bower_components", "jspm_packages", "vendor", "vendors", "third_party",
    "third-party", "thirdparty", "dist", "build", ".next", ".nuxt", ".svelte-kit", ".angular", ".expo",
    ".output", "out", "target", "obj", ".gradle", ".dart_tool", "pods", "deriveddata", ".venv", "venv",
    "env", "virtualenv", "site-packages", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".cache",
    ".parcel-cache", "coverage", "htmlcov", "storybook-static", ".ipynb_checkpoints", ".idea", ".vscode",
}
LOCK_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock", "pipfile.lock",
    "composer.lock", "gemfile.lock", "cargo.lock", "go.sum", "pubspec.lock",
}
GENERATED_FILE = re.compile(r"(\.min\.(js|css)|[.-]bundle\.js|\.chunk\.(js|css)|\.d\.ts|\.map)$")

# Implementation files by extension -> the language (or framework) they are written in.
SOURCE_EXTENSIONS = {
    ".py": "Python", ".ipynb": "Jupyter Notebook", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".java": "Java", ".kt": "Kotlin", ".go": "Go", ".rs": "Rust", ".dart": "Dart", ".php": "PHP",
    ".rb": "Ruby", ".cs": "C#", ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".c": "C", ".swift": "Swift",
    ".vue": "Vue.js", ".svelte": "Svelte", ".html": "HTML", ".css": "CSS", ".scss": "Sass", ".sass": "Sass",
    ".sol": "Solidity", ".sh": "Bash",
}
PY_SOURCE = (".py", ".ipynb")
JS_SOURCE = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".vue", ".svelte")
COMPONENT_FILE = re.compile(r"(\.(jsx|tsx|vue|svelte)|\.component\.ts)$")
# Build/test tooling scripts: they run under Node but are not backend code.
TOOLING_SCRIPT = re.compile(
    r"(^|[./])(config|conf)\.(js|cjs|mjs|ts|mts)$|^(webpack|vite|rollup|babel|jest|karma|gulpfile|gruntfile|"
    r"postcss|tailwind|craco|eslint|\.eslintrc|prettier|setuptests|reportwebvitals|serviceworker)\b"
    r"|\.(test|spec|stories)\.\w+$"
)
TEST_PATH = re.compile(r"(^|/)(__tests__|tests?|spec|e2e|cypress)/|\.(test|spec)\.\w+$|(^|/)test_[^/]+\.py$")
# Config files that only exist when the tool is actually set up (lowercase file name).
CONFIG_FILES = [
    (re.compile(r"angular\.json"), "Angular"),
    (re.compile(r"next\.config\.(js|mjs|cjs|ts)"), "Next.js"),
    (re.compile(r"nuxt\.config\.(js|mjs|ts)"), "Nuxt.js"),
    (re.compile(r"svelte\.config\.(js|mjs|cjs|ts)"), "Svelte"),
    (re.compile(r"vue\.config\.(js|mjs|cjs|ts)"), "Vue.js"),
    (re.compile(r"vite\.config\.(js|mjs|cjs|ts|mts)"), "Vite"),
    (re.compile(r"tailwind\.config\.(js|mjs|cjs|ts)"), "Tailwind CSS"),
    (re.compile(r"webpack\.config\.(js|mjs|cjs|ts)"), "Webpack"),
    (re.compile(r"gatsby-config\.(js|mjs|ts)"), "Gatsby"),
    (re.compile(r"nest-cli\.json"), "NestJS"),
    (re.compile(r"tsconfig\.json"), "TypeScript"),
    (re.compile(r"schema\.prisma"), "Prisma"),
    (re.compile(r"firebase\.json"), "Firebase"),
    (re.compile(r"manage\.py"), "Django"),
    (re.compile(r"jest\.config\.(js|mjs|cjs|ts)"), "Jest"),
    (re.compile(r"cypress\.config\.(js|mjs|cjs|ts)"), "Cypress"),
    (re.compile(r"playwright\.config\.(js|mjs|ts)"), "Playwright"),
]
# Entry points and folders that hold server code; used to rank files to read and as backend structure evidence.
ENTRY_STEMS = {"server", "app", "index", "main", "manage", "wsgi", "asgi", "run", "api", "application", "routes"}
BACKEND_DIRS = {"server", "backend", "api", "routes", "controllers", "middleware", "middlewares", "services",
                "models", "src", "app"}
SERVER_DIRS = {"server", "backend", "api", "routes", "controllers", "middleware", "middlewares"}

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


# Packages that only run on a Node.js server: declaring or importing one means Node.js is used for real.
NODE_SERVER_PACKAGES = {
    "express", "@nestjs/core", "fastify", "koa", "@hapi/hapi", "hapi", "restify", "mongoose", "sequelize",
    "typeorm", "nodemon", "body-parser", "cors", "jsonwebtoken", "bcrypt", "bcryptjs", "multer", "passport",
    "socket.io", "pg", "mysql", "mysql2", "mongodb", "ioredis", "express-session", "cookie-parser",
    "nodemailer", "firebase-admin", "ws", "morgan", "helmet", "express-validator",
}
# Node core modules that browser code cannot import (``node:*`` is always Node).
NODE_CORE_MODULES = {
    "http", "https", "http2", "net", "tls", "dgram", "dns", "fs", "child_process", "cluster",
    "worker_threads", "os", "readline",
}
NODE_START_SCRIPT = re.compile(r"(^|[\s;&|(])(node|nodemon|ts-node|ts-node-dev|pm2|forever)\s+[\w./-]")
JS_IMPORT = re.compile(
    r"""(?:\bimport\s+(?:type\s+)?(?:[\w*{}\s,$]+?\s+from\s+)?|\bexport\s+[\w*{}\s,$]+?\s+from\s+|"""
    r"""\brequire\s*\(\s*|\bimport\s*\(\s*)["']([^"'\s]+)["']"""
)
NODE_SERVER_CODE = re.compile(
    r"""\bhttp\.createServer\s*\(|\b(?:app|server|httpServer|fastify)\.listen\s*\(|\bexpress\.Router\s*\(|"""
    r"""\b(?:app|router)\.(?:get|post|put|patch|delete|use)\s*\(\s*["'`]/"""
)


@lru_cache(maxsize=1)
def _dictionary_names() -> dict[str, str]:
    """Normalised surface -> canonical name from the technology dictionaries (empty if they cannot be read)."""
    try:
        from technology_matcher import load_dictionary  # stdlib-only module; lazy to keep import order flexible
        return {surface: info.name for surface, info in load_dictionary().surfaces.items()}
    except Exception:  # noqa: BLE001 - normalisation is best effort; detection must not depend on it
        logger.debug("technology dictionary unavailable; names are not normalised")
        return {}


def normalize_technology(name: str) -> str:
    """Canonical dictionary name (``scikit-learn`` -> ``Scikit-learn``, ``NodeJS`` -> ``Node.js``)."""
    key = re.sub(r"[\s_-]+", " ", name.strip().lower())
    return _dictionary_names().get(key, name)


def inspect_package_json(content: str) -> dict[str, str]:
    """Technologies declared by one package.json, mapped to their scope: ``runtime``, ``dev`` or ``peer``.

    ``peer`` means "the host project must provide it": an indirect dependency, not usage by this project.
    ``Node.js`` is added when a server-only package is declared or a start script runs a file with node.
    """
    try:
        parsed = json.loads(content)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    scoped: dict[str, str] = {}
    for key, scope in (("dependencies", "runtime"), ("devDependencies", "dev"), ("peerDependencies", "peer"),
                       ("optionalDependencies", "runtime")):
        if isinstance(parsed.get(key), dict):
            for dep in parsed[key]:
                scoped.setdefault(str(dep), scope)  # earlier (stronger) scope wins

    found: dict[str, str] = {}
    rank = {"runtime": 0, "dev": 1, "peer": 2}

    def add(tech: str, scope: str) -> None:
        if tech not in found or rank[scope] < rank[found[tech]]:
            found[tech] = scope

    for dep, scope in scoped.items():
        for tech, patterns in JS_PACKAGES.items():
            if any(matches_name(dep, p) for p in patterns):
                add(tech, scope)
        if dep in NODE_SERVER_PACKAGES:
            add("Node.js", scope)
    scripts = parsed.get("scripts")
    if isinstance(scripts, dict):
        for key in ("start", "dev", "server", "serve", "prod", "backend", "start:dev", "start:prod"):
            if isinstance(scripts.get(key), str) and NODE_START_SCRIPT.search(scripts[key]):
                add("Node.js", "runtime")
    return found


def detect_package_json(content: str, techs: set[str]) -> None:
    """Technologies this package.json uses directly (runtime and dev dependencies; peer dependencies excluded)."""
    techs.update(tech for tech, scope in inspect_package_json(content).items() if scope != "peer")


def _js_package(specifier: str) -> str | None:
    """'@scope/pkg/sub' -> '@scope/pkg', 'pkg/sub' -> 'pkg'; relative paths and URLs -> None."""
    if specifier.startswith((".", "/", "~", "http:", "https:")) or specifier.startswith("@/"):
        return None
    parts = specifier.split("/")
    return "/".join(parts[:2]) if specifier.startswith("@") and len(parts) > 1 else parts[0]


def inspect_js_source(source: str, tooling: bool = False) -> tuple[set[str], set[str]]:
    """(technologies imported, technologies shown by server code) for one JS/TS/Vue/Svelte file.

    ``tooling`` files (vite.config.js, jest setup, tests) still count for the packages they import, but importing
    ``fs`` or ``path`` there is build tooling, not a Node.js backend.
    """
    imported: set[str] = set()
    patterns: set[str] = set()
    for specifier in JS_IMPORT.findall(source):
        if specifier.startswith("node:"):
            if not tooling:
                imported.add("Node.js")
            continue
        package = _js_package(specifier)
        if not package:
            continue
        for tech, names in JS_PACKAGES.items():
            if any(matches_name(package, p) for p in names):
                imported.add(tech)
        if not tooling and (package in NODE_CORE_MODULES or package in NODE_SERVER_PACKAGES):
            imported.add("Node.js")
    if not tooling and NODE_SERVER_CODE.search(source):
        patterns.add("Node.js")
        if "Express.js" in imported:
            patterns.add("Express.js")
    return imported, patterns


FROZEN_PIN = re.compile(r"^[A-Za-z0-9_.-]+(\[[^\]]*\])?\s*===?\s*[\w.+!-]+(\s*;.*)?$")
CONDA_PIN = re.compile(r"^-\s*[\w.-]+=[^=\s]+=\S+$")


def looks_frozen(content: str) -> bool:
    """A ``pip freeze`` / ``conda env export`` dump: lists every package in an environment, transitive ones included.

    Such a file cannot tell what the project itself uses, so its entries count only when the source imports them.
    """
    lines = [line.strip() for line in content.splitlines()]
    lines = [line for line in lines if line and not line.startswith(("#", "-r", "--", "name:", "channels:",
                                                                      "dependencies:", "prefix:", "- pip:"))]
    if len(lines) < 30:
        return False
    pinned = sum(bool(FROZEN_PIN.match(line) or CONDA_PIN.match(line)) for line in lines)
    return pinned / len(lines) >= 0.9


def detect_python_manifest(content: str, techs: set[str]) -> None:
    # One requirement per line: torch>=2 | - numpy=1.2 (conda) | fastapi = "^0.1" (poetry)
    for line in content.splitlines():
        m = LINE_REQ.match(line)
        if m:
            add_pip_name(m.group(1), techs)
    # Quoted requirements anywhere: dependencies = ["torch>=2", "numpy"] / install_requires=[...]
    for m in QUOTED_REQ.finditer(content):
        add_pip_name(m.group(1), techs)


CONTAINER_IMAGES = {
    "PostgreSQL": re.compile(r"^(postgres|postgresql|postgis|pgvector)$"),
    "MongoDB": re.compile(r"^(mongo|mongodb|mongodb-community-server)$"),
    "Redis": re.compile(r"^(redis|redis-stack|redis-stack-server)$"),
    "MySQL": re.compile(r"^(mysql|mysql-server)$"),
    "MariaDB": re.compile(r"^mariadb$"),
    "Nginx": re.compile(r"^(nginx|nginx-unprivileged)$"),
    "RabbitMQ": re.compile(r"^rabbitmq$"),
    "Elasticsearch": re.compile(r"^elasticsearch$"),
    "Apache Kafka": re.compile(r"^(kafka|cp-kafka)$"),
    "Qdrant": re.compile(r"^qdrant$"),
    "ChromaDB": re.compile(r"^(chroma|chromadb)$"),
    "Ollama": re.compile(r"^ollama$"),
}
IMAGE_REFERENCE = re.compile(r"^\s*(?:-\s*)?image\s*:\s*[\"']?([^\s\"'#]+)|^\s*FROM\s+(?:--\S+\s+)*([^\s#]+)",
                             re.IGNORECASE | re.MULTILINE)


def detect_docker(content: str, techs: set[str]) -> None:
    """Services a Dockerfile / compose file actually runs (``image: postgres:16``, ``FROM nginx``).

    Only image references count: a word such as "postgres" in a comment or a package list is not a service.
    """
    for match in IMAGE_REFERENCE.finditer(content):
        reference = (match.group(1) or match.group(2) or "").split("@")[0].lower()
        name = reference.rsplit("/", 1)[-1].split(":")[0]
        for tech, pattern in CONTAINER_IMAGES.items():
            if pattern.match(name):
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


class GitHubCredentials:
    """Share credentials, concurrency limits and cooldown across a GitHub batch."""

    def __init__(self, tokens: tuple[str, ...] = (), *,
                 token_sources: Mapping[str, str] | None = None,
                 max_parallel_requests: int = DEFAULT_PARALLEL_GITHUB_REQUESTS) -> None:
        if not 1 <= max_parallel_requests <= MAX_PARALLEL_GITHUB_REQUESTS:
            raise ValueError(f"GitHub concurrency must be between 1 and {MAX_PARALLEL_GITHUB_REQUESTS}")
        self._tokens: tuple[str | None, ...] = tuple(dict.fromkeys(tokens)) or (None,)
        self._token_sources = dict(token_sources or {})
        self._index = 0
        self._lock = threading.Lock()
        self.max_parallel_requests = max_parallel_requests
        self._request_slots = threading.BoundedSemaphore(max_parallel_requests)
        self.retry_at = 0.0

    @classmethod
    def from_token(cls, token: GitHubAuthorization) -> GitHubCredentials:
        return token if isinstance(token, cls) else cls((token,) if token else ())

    @property
    def authenticated(self) -> bool:
        return self._tokens[0] is not None

    @property
    def blocked(self) -> bool:
        with self._lock:
            return self.retry_at > time.time() or self._index >= len(self._tokens)

    def _rate_limit_error(self) -> GitHubError:
        when = time.strftime("%H:%M:%S", time.localtime(self.retry_at))
        return GitHubError(f"GitHub API rate limit reached. Try again after {when}.")

    def get(
        self, url: str, *, session: requests.Session | None = None, raw: bool = False,
        params: Mapping[str, str | int] | None = None, timeout: float | tuple[float, float] = 30,
        count: bool = True,
    ) -> requests.Response:
        return self.request("GET", url, session=session, raw=raw, params=params, timeout=timeout, count=count)

    def post(
        self, url: str, payload: Mapping, *, session: requests.Session | None = None,
        timeout: float | tuple[float, float] = 30, count: bool = True,
    ) -> requests.Response:
        """POST a JSON body (GraphQL) with the same token rotation, concurrency limit and rate-limit handling."""
        return self.request("POST", url, session=session, payload=payload, timeout=timeout, count=count)

    def request(
        self, method: str, url: str, *, session: requests.Session | None = None, raw: bool = False,
        params: Mapping[str, str | int] | None = None, payload: Mapping | None = None,
        timeout: float | tuple[float, float] = 30, count: bool = True,
    ) -> requests.Response:
        if method == "POST":
            post = requests.post if session is None else session.post
            requester = partial(post, json=payload)
        else:
            requester = requests.get if session is None else session.get
        while True:
            with self._request_slots:
                with self._lock:
                    if self.retry_at > time.time():
                        raise self._rate_limit_error()
                    index = self._index
                    if index >= len(self._tokens):
                        raise GitHubError("GitHub token is invalid or expired. Check the configured tokens.")
                    token = self._tokens[index]

                source = (self._token_sources.get(token, f"GitHub token #{index + 1}")
                          if token else "unauthenticated requests")
                accept = "application/vnd.github.raw+json" if raw else "application/vnd.github+json"
                headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                if count:
                    record_api_request()
                try:
                    response = requester(url, headers=headers, params=params, timeout=timeout)
                except requests.RequestException as exc:
                    raise GitHubError(f"GitHub API request failed ({type(exc).__name__})") from None

                if response.status_code == 401:
                    with self._lock:
                        if self._index == index:
                            self._index += 1
                            logger.warning("GitHub rejected %s (HTTP 401)", source)
                    continue

                retry_after = response.headers.get("retry-after", "")
                exhausted = response.headers.get("x-ratelimit-remaining") == "0"
                if response.status_code == 429 or (response.status_code == 403 and (
                    exhausted or retry_after or "rate limit" in response.text.lower()
                )):
                    now = time.time()
                    reset = response.headers.get("x-ratelimit-reset", "")
                    deadlines = []
                    if exhausted and reset.isdecimal() and int(reset) > now:
                        deadlines.append(float(reset))
                    if retry_after.isdecimal():
                        deadlines.append(now + max(int(retry_after), 1))
                    with self._lock:
                        already_paused = self.retry_at > now
                        self.retry_at = max(self.retry_at, max(deadlines, default=now + 60))
                        if not already_paused:
                            when = time.strftime("%H:%M:%S", time.localtime(self.retry_at))
                            logger.warning("GitHub rate-limited %s (HTTP %d). Requests paused until %s",
                                           source, response.status_code, when)
                        raise self._rate_limit_error()
                if response.status_code == 403:
                    logger.warning("GitHub denied access for %s (HTTP 403)", source)
                return response


GitHubAuthorization = str | GitHubCredentials | None


class EmptyRepositoryError(GitHubError):
    """The repository has no commits (GitHub answers HTTP 409 for its tree)."""


ACCESS_ERRORS = {
    403: "Repository access denied (HTTP 403)",
    404: "Repository is private, deleted or renamed (HTTP 404)",
    451: "Repository is unavailable for legal reasons (HTTP 451)",
}
GIT_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class _GitHubRepositoryAPI:
    def __init__(self, token: GitHubAuthorization) -> None:
        self.credentials = GitHubCredentials.from_token(token)
        self.graphql_enabled = True   # switched off for the run if GitHub rejects GraphQL (e.g. its own quota)

    def _get(self, session: requests.Session, url: str, *, raw: bool = False,
             params: dict[str, str] | None = None) -> requests.Response:
        response = self.credentials.get(url, session=session, raw=raw, params=params, timeout=(5, 25))
        if response.status_code == 409:
            raise EmptyRepositoryError("Repository is empty")
        if response.status_code in ACCESS_ERRORS:
            raise GitHubError(ACCESS_ERRORS[response.status_code])
        if response.status_code != 200:
            raise GitHubError(f"GitHub repository request returned HTTP {response.status_code}")
        return response

    def files(self, session: requests.Session, url: str, branch: str) -> dict[str, dict]:
        tree_url = f"{url}/git/trees"
        root = self._get(session, f"{tree_url}/{quote(branch, safe='')}", params={"recursive": "1"}).json()
        if not root["truncated"]:
            return {entry["path"]: entry for entry in root["tree"] if entry["type"] != "tree"}

        # Walk the same tree by SHA if GitHub cuts off the recursive response.
        files: dict[str, dict] = {}
        pending = [("", root["sha"])]
        while pending:
            prefix, sha = pending.pop()
            subtree = self._get(session, f"{tree_url}/{quote(sha, safe='')}").json()
            if subtree["truncated"]:
                raise GitHubError("GitHub returned an incomplete repository file tree")
            for entry in subtree["tree"]:
                path = prefix + entry["path"]
                if entry["type"] != "tree":
                    files[path] = entry
                elif entry["path"].lower() not in IGNORED_DIRS:
                    pending.append((path + "/", entry["sha"]))
        return files

    def _graphql_batch(self, session: requests.Session, owner: str, name: str,
                       batch: list[tuple[str, str, int]]) -> dict[str, str] | None:
        """Text of several blobs in ONE request; ``None`` when GraphQL cannot answer (the caller uses REST)."""
        fields = " ".join(f'f{i}: object(oid: "{sha}") {{ ... on Blob {{ text isBinary }} }}'
                          for i, (_path, sha, _limit) in enumerate(batch))
        query = ("query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) { "
                 + fields + " } }")
        response = self.credentials.post(GRAPHQL_URL, {"query": query, "variables": {"owner": owner, "name": name}},
                                         session=session, timeout=(5, 40))
        if response.status_code != 200:
            logger.debug("graphql_unavailable status=%d", response.status_code)
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        errors = payload.get("errors") or [] if isinstance(payload, dict) else []
        if any(isinstance(e, dict) and e.get("type") in ("RATE_LIMITED", "FORBIDDEN") for e in errors):
            self.graphql_enabled = False
            logger.warning("GitHub GraphQL refused file reads (%s); using REST for the rest of the run",
                           ",".join(sorted({str(e.get("type")) for e in errors if isinstance(e, dict)})))
            return None
        repository = (payload.get("data") or {}).get("repository") if isinstance(payload, dict) else None
        if not isinstance(repository, dict):
            return None
        texts: dict[str, str] = {}
        for i, (path, _sha, limit) in enumerate(batch):
            blob = repository.get(f"f{i}")
            if isinstance(blob, dict) and not blob.get("isBinary") and isinstance(blob.get("text"), str):
                if len(blob["text"]) <= limit:
                    texts[path] = blob["text"]
        return texts

    def contents(self, session: requests.Session, url: str, entries: dict[str, dict],
                 selected: list[str]) -> Iterator[tuple[str, str]]:
        files = []
        for path in selected:
            entry = entries[path]
            limit = _content_limit(path)
            if entry["type"] == "blob" and entry["mode"] != "120000" and entry["size"] <= limit:
                files.append((path, entry["sha"], limit))

        # GraphQL reads many files per request (1-2 requests per repository instead of one per file).
        owner, _, name = url.rsplit("/repos/", 1)[-1].partition("/")
        rest: list[tuple[str, str, int]] = [f for f in files if not GIT_SHA.fullmatch(f[1])]
        batches: list[list[tuple[str, str, int]]] = []
        current: list[tuple[str, str, int]] = []
        size = 0
        for item in (f for f in files if GIT_SHA.fullmatch(f[1])):
            item_size = int(entries[item[0]].get("size") or 0)
            if current and (len(current) >= GRAPHQL_MAX_FILES or size + item_size > GRAPHQL_MAX_BYTES):
                batches.append(current)
                current, size = [], 0
            current.append(item)
            size += item_size
        if current:
            batches.append(current)
        for batch in batches:
            texts = self._graphql_batch(session, owner, name, batch) if self.graphql_enabled else None
            if texts is None:
                rest.extend(batch)
                continue
            for path, _sha, _limit in batch:
                if path in texts:
                    yield path, texts[path]
        files = rest

        def read(connection: requests.Session, sha: str, limit: int) -> str | None:
            content = self._get(connection, f"{url}/git/blobs/{quote(sha, safe='')}", raw=True).content
            return content.decode("utf-8", "replace") if len(content) <= limit else None

        lanes = min(len(files), API_FILE_WORKERS, self.credentials.max_parallel_requests)
        if lanes <= 1:
            for path, sha, limit in files:
                content = read(session, sha, limit)
                if content is not None:
                    yield path, content
            return

        with ExitStack() as resources:
            connections = [session] + [resources.enter_context(requests.Session()) for _ in range(lanes - 1)]
            pool = resources.enter_context(ThreadPoolExecutor(lanes))
            for start in range(0, len(files), lanes):
                # Process one batch at a time to bound both requests and buffered file contents.
                batch = files[start:start + lanes]
                downloads = [(path, pool.submit(read, connection, sha, limit))
                             for connection, (path, sha, limit) in zip(connections, batch)]
                for path, future in downloads:
                    content = future.result()
                    if content is not None:
                        yield path, content


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


def list_repositories(username: str, token: GitHubAuthorization, skip_forks: bool = True) -> list[dict]:
    credentials = GitHubCredentials.from_token(token)

    repos: list[dict] = []
    page = 1
    while True:
        request_started = time.perf_counter()
        r = credentials.get(
            f"https://api.github.com/users/{username}/repos",
            params={"type": "owner", "per_page": 100, "page": page, "sort": "updated"},
        )
        logger.debug("api=list_repositories status=%d duration=%.3fs remaining=%s",
                     r.status_code, time.perf_counter() - request_started,
                     r.headers.get("x-ratelimit-remaining", "unknown"))
        if r.status_code == 404:
            raise GitHubError("GitHub user not found")
        r.raise_for_status()

        batch = r.json()
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    return [repo for repo in repos if not (skip_forks and repo.get("fork"))]


# GitHub's language names that duplicate a technology we detect ourselves.
LANGUAGE_ALIASES = {"Vue": "Vue.js", "Dockerfile": "Docker", "SCSS": "Sass", "Shell": "Bash", "HCL": "Terraform"}


def sparse_pattern(path: str) -> str:
    return "/" + re.sub(r"([*?\[\]\\!#])", r"\\\1", path)


def depth(path: str) -> int:
    return path.count("/")


def _content_limit(path: str) -> int:
    lower = path.lower()
    if lower.endswith(".ipynb"):
        return MAX_CODE_SIZE
    return MAX_SOURCE_SIZE if lower.endswith(PY_SOURCE + JS_SOURCE) else MAX_MANIFEST_SIZE


def _parse_ls_tree(output: bytes) -> dict[str, dict]:
    """``git ls-tree -r -z [-l]`` -> {path: entry} shaped like GitHub's tree API (plain path lists are accepted too)."""
    entries: dict[str, dict] = {}
    for item in output.decode("utf-8", "replace").split("\0"):
        if not item:
            continue
        if "\t" not in item:
            entries[item] = {"path": item, "type": "blob", "mode": "", "sha": "", "size": None}
            continue
        meta, path = item.split("\t", 1)
        fields = meta.split()
        if len(fields) < 3 or fields[1] != "blob":
            continue  # submodules have no content here
        size = int(fields[3]) if len(fields) > 3 and fields[3].isdigit() else None
        entries[path] = {"path": path, "type": "blob", "mode": fields[0], "sha": fields[2], "size": size}
    return entries


def _code_priority(path: str) -> tuple:
    """Entry points and server folders first, tests and build scripts last, then shallow before deep."""
    parts = path.lower().split("/")
    name = parts[-1]
    stem = name.split(".", 1)[0]
    demoted = bool(TOOLING_SCRIPT.search(name) or TEST_PATH.search(path.lower()))
    entry = stem in ENTRY_STEMS or name in ("app.jsx", "app.tsx", "app.vue")
    in_server_dir = any(part in BACKEND_DIRS for part in parts[:-1])
    return (demoted, not entry, not in_server_dir, len(parts), path)


def _select_code(py_files: list[str], js_files: list[str], limit: int = MAX_CODE_FILES) -> list[str]:
    """Up to ``limit`` files, balanced between Python and JavaScript/TypeScript when a repository has both."""
    py_files = sorted(py_files, key=_code_priority)
    js_files = sorted(js_files, key=_code_priority)
    half = limit // 2
    py_take = min(len(py_files), max(half, limit - len(js_files)))
    js_take = min(len(js_files), limit - py_take)
    return py_files[:py_take] + js_files[:js_take]


# Evidence kinds. Direct kinds show the project itself uses the technology.
DIRECT_KINDS = {"source_import", "source_pattern", "config_file", "source_file", "file_path"}
DECLARED_KINDS = {"dependency_manifest", "container_image"}
INDIRECT_SCOPES = {"peer", "frozen"}
# Not libraries: languages, build tools and markers. A repo with only these has no framework/library usage.
NON_LIBRARY = (set(SOURCE_EXTENSIONS.values()) - FRONTEND) | {
    "Jupyter Notebook", "Jupyter", "Python", "GitHub Actions", "Docker", "Maven", "Gradle", ".NET", "HTML Templates",
}
TOOLING = {"Jest", "Vitest", "Mocha", "Cypress", "Playwright", "JUnit", "xUnit", "NUnit", "PyTest", "Webpack", "Vite",
           "TypeScript", "Terraform"}
PRACTICE_FILE = re.compile(
    r"^(?:q|ques|question|problem|prob|p|ex|exercise|exp|experiment|day|week|lab|assignment|assign|task|"
    r"prog|program|pgm|lc|leetcode|hw|practical|practice|sol|solution|module)[-_ ]?\d+|^\d+[-_. ]",
    re.IGNORECASE,
)


def _evidence_summary(evidence: list[dict]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """(usage per technology, evidence kinds per technology).

    usage: ``direct`` (imported, configured, server code or implementation files), ``declared`` (dependency or
    container only), ``inferred`` (GitHub's language label only), ``indirect`` (peer dependency or frozen
    environment dump only; not counted as used).
    """
    kinds: dict[str, set[str]] = {}
    for item in evidence:
        kind = item["type"]
        if item.get("scope") in INDIRECT_SCOPES:
            kind = "indirect"
        kinds.setdefault(item["technology"], set()).add(kind)
    usage = {}
    for tech, found in kinds.items():
        usage[tech] = ("direct" if found & DIRECT_KINDS else "declared" if found & DECLARED_KINDS
                       else "inferred" if "repository_language" in found else "indirect")
    return usage, kinds


def _implementation(usage: Mapping[str, str], kinds: Mapping[str, set[str]], structure: Mapping) -> dict:
    """Evidence that the frontend / each backend stack is implemented, not only listed as a dependency."""
    frontend: list[str] = []
    for tech in sorted(FRONTEND & usage.keys()):
        direct = sorted(kinds[tech] & {"source_import", "config_file", "source_file"})
        if direct:
            frontend.append(f"{tech}: {', '.join(direct)}")
        elif usage[tech] == "declared" and structure["component_files"]:
            frontend.append(f"{tech}: dependency + {structure['component_files']} component files")

    backend: dict[str, str] = {}
    node = sorted(NODE_BACKEND & usage.keys())
    if any(kinds[t] & {"source_import", "source_pattern", "config_file"} for t in node):
        backend["node"] = ", ".join(t for t in node if kinds[t] & {"source_import", "source_pattern", "config_file"})
        backend["node"] += " used in source"
    elif "source_pattern" in kinds.get("Node.js", ()):
        backend["node"] = "Node.js server code"
    elif node and structure["node_server_files"]:
        backend["node"] = f"{', '.join(node)} dependency + server files"

    python = sorted(PYTHON_WEB & usage.keys())
    if any(kinds[t] & {"source_import", "config_file"} for t in python):
        backend["python"] = ", ".join(t for t in python if kinds[t] & {"source_import", "config_file"})
        backend["python"] += " used in source"
    elif python and structure["python_server_files"]:
        backend["python"] = f"{', '.join(python)} dependency + server files"

    java = sorted(JAVA_BACKEND & usage.keys())
    if java and structure["java_files"]:
        backend["java"] = f"{', '.join(java)} + {structure['java_files']} Java files"
    dotnet = sorted(DOTNET_BACKEND & usage.keys())
    if dotnet and structure["csharp_files"]:
        backend["dotnet"] = f"{', '.join(dotnet)} + {structure['csharp_files']} C# files"
    return {"frontend": frontend, "backend": backend}


def assess_repository(repo: Mapping, usage: Mapping[str, str], source_files: list[str], has_manifest: bool,
                      implementation: Mapping) -> str | None:
    """Why a repository is not a meaningful project (``None`` when it is one).

    Name-based signals (DSA, tutorials, assignments) only exclude a repository that also shows no application
    implementation and no library usage, so a real project is never dropped for its name, size or popularity.
    """
    if not source_files:
        return "no_source_code"
    used = {tech for tech, how in usage.items() if how in ("direct", "declared")}
    libraries = used - NON_LIBRARY
    application = bool(implementation.get("frontend") or implementation.get("backend"))
    signal = practice_signal(repo)
    if signal and not application and not libraries - TOOLING:
        return signal
    if application or libraries or has_manifest:
        return None
    names = [path.rsplit("/", 1)[-1] for path in source_files]
    practice_files = sum(bool(PRACTICE_FILE.match(name)) for name in names)
    if len(names) >= 4 and practice_files / len(names) >= 0.6:
        return "coding_practice"
    if len(names) <= 2:
        return "minimal_implementation"
    return None


def _checkout_contents(folder: str, selected: list[str]) -> Iterator[tuple[str, str]]:
    for path in selected:
        full_path = os.path.join(folder, *path.split("/"))
        if os.path.isfile(full_path) and os.path.getsize(full_path) <= _content_limit(path):
            with open(full_path, encoding="utf-8", errors="replace") as file:
                yield path, file.read()


def analyze_repository(repo: dict, *, api: _GitHubRepositoryAPI | None = None) -> dict:
    """Detect technologies, evidence, implementation and project relevance in one repository's default branch.

    Result keys (besides timings): ``technologies`` (used directly or declared; peer dependencies and entries of
    frozen environment dumps count only when the source imports them), ``technology_evidence`` (one item per
    technology, kind and file), ``technology_usage`` (direct / declared / inferred per technology),
    ``indirect_technologies``, ``implementation`` (frontend / backend evidence), ``categories``,
    ``excluded`` + ``exclusion_reason`` (not a meaningful project) and ``source_signature`` (duplicate detection).
    """
    analysis_started = time.perf_counter()
    clone_seconds = sparse_seconds = detection_seconds = api_seconds = 0.0
    evidence: list[dict] = []
    seen_paths: dict[tuple[str, str], dict] = {}
    structure = {"component_files": 0, "node_server_files": 0, "python_server_files": 0,
                 "java_files": 0, "csharp_files": 0}
    source_files: list[str] = []
    signature: list[str] = []
    has_manifest = False
    empty = False

    def record(names: set[str], kind: str, path: str, scope: str | None = None) -> None:
        for raw in sorted(names):
            name = normalize_technology(raw)
            if kind in ("file_path", "source_file", "config_file"):
                # One item per technology and kind; further files only raise the count.
                existing = seen_paths.get((name, kind))
                if existing is not None:
                    existing["files"] = existing.get("files", 1) + 1
                    continue
            item = {"technology": name, "type": kind, "path": path, "strength": "weak"}
            if scope:
                item["scope"] = scope
            evidence.append(item)
            if kind in ("file_path", "source_file", "config_file"):
                seen_paths[(name, kind)] = item

    if repo.get("language"):
        record({LANGUAGE_ALIASES.get(repo["language"], repo["language"])}, "repository_language", "")

    def finish(error: str | None = None) -> dict:
        usage, kinds = _evidence_summary(evidence)
        for item in evidence:
            found = kinds[item["technology"]]
            # Independent sources: dependency declaration, source import/code, config file, implementation files.
            sources = sum((bool(found & {"dependency_manifest", "container_image"}),
                           bool(found & {"source_import", "source_pattern"}),
                           bool(found & {"config_file"}), bool(found & {"source_file", "file_path"})))
            if item.get("scope") in INDIRECT_SCOPES or item["type"] == "repository_language":
                item["strength"] = "strong" if sources >= 2 else "medium" if sources else "weak"
            else:
                item["strength"] = "strong" if sources >= 2 else "medium"
        technologies = sorted(tech for tech, how in usage.items() if how != "indirect")
        implementation = _implementation(usage, kinds, structure)
        result = {
            "repository": repo["full_name"],
            "url": repo.get("html_url", ""),
            "primaryLanguage": repo.get("language"),
            "technologies": technologies,
            "technology_evidence": evidence,
            "technology_usage": {tech: usage[tech] for tech in technologies},
            "indirect_technologies": sorted(tech for tech, how in usage.items() if how == "indirect"),
            "implementation": implementation,
            "categories": categorize(technologies, implementation),
            "source_file_count": len(source_files),
            "source_signature": sorted(signature)[:500],
            "analysis_seconds": round(time.perf_counter() - analysis_started, 3),
            "clone_seconds": round(clone_seconds, 3),
            "sparse_seconds": round(sparse_seconds, 3),
            "detection_seconds": round(detection_seconds, 3),
            "repository_api_seconds": round(api_seconds, 3),
        }
        if error:
            result["error"] = error
        else:
            reason = "empty_repository" if empty else assess_repository(
                repo, usage, source_files, has_manifest, implementation)
            result["excluded"] = reason is not None
            if reason:
                result["exclusion_reason"] = reason
        return result

    if repo.get("size") == 0:
        empty = True
        return finish()

    resources = ExitStack()
    try:
        if api is not None:
            session = resources.enter_context(requests.Session())
            url = f"https://api.github.com/repos/{quote(repo['full_name'])}"
            started = time.perf_counter()
            try:
                entries = api.files(session, url, repo["default_branch"])
            finally:
                api_seconds = time.perf_counter() - started
        else:
            os.makedirs(CLONE_ROOT, exist_ok=True)
            tmp = tempfile.mkdtemp(prefix="repo-", dir=CLONE_ROOT)
            resources.callback(remove_dir, tmp)
            clone_started = time.perf_counter()
            logger.info("clone_started repo=%s", repo["full_name"])
            git([
                "clone", "--depth", "1", "--single-branch",
                "--filter=blob:none", "--no-checkout",
                repo["clone_url"], tmp,
            ])
            clone_seconds = time.perf_counter() - clone_started
            logger.info("clone_completed repo=%s duration=%.3fs", repo["full_name"], clone_seconds)
            try:
                # No "-l": sizes would make git download every blob of the blobless clone. Blob ids are free.
                output = git(["ls-tree", "-r", "-z", "HEAD"], cwd=tmp)
            except RuntimeError as exc:
                if "HEAD" in str(exc) or "not a tree" in str(exc).lower():
                    raise EmptyRepositoryError("Repository is empty") from None
                raise
            entries = _parse_ls_tree(output)

        manifests: list[str] = []
        py_code: list[str] = []
        js_code: list[str] = []

        for file_path in sorted(entries):
            parts = file_path.split("/")
            if any(part.lower() in IGNORED_DIRS for part in parts[:-1]):
                continue
            name = parts[-1].lower()
            if name in LOCK_FILES or GENERATED_FILE.search(name):
                continue
            lower_path = file_path.lower()
            dirs = [part.lower() for part in parts[:-1]]
            ext = os.path.splitext(name)[1]

            path_techs: set[str] = set()
            if file_path.startswith(".github/workflows/"):
                path_techs.add("GitHub Actions")
            if name.endswith(".tf"):
                path_techs.add("Terraform")
            if name in PATH_ONLY:
                path_techs.add(PATH_ONLY[name])
            if name in PYTHON_MANIFESTS:
                path_techs.add("Python")
            if name in DOCKER_FILES:
                path_techs.add("Docker")
            if name.endswith(".html") and "templates" in dirs:
                path_techs.add("HTML Templates")
            if name.endswith(".sln"):
                path_techs.add(".NET")
            if name.endswith(".jsp"):
                path_techs.add("JSP")
            if name.endswith(".cshtml"):
                path_techs.add("Razor Views")
            if name.endswith(".razor"):
                path_techs.add("Blazor")
            if name.endswith(".ipynb"):
                path_techs.add("Jupyter Notebook")
            record(path_techs, "file_path", file_path)
            record({tech for pattern, tech in CONFIG_FILES if pattern.fullmatch(name)}, "config_file", file_path)

            language = SOURCE_EXTENSIONS.get(ext)
            if language:
                source_files.append(file_path)
                if language != "Jupyter Notebook":  # already recorded as a file_path marker
                    record({language}, "source_file", file_path)
                sha = str(entries[file_path].get("sha") or "")
                if sha:
                    signature.append(sha[:12])
                if COMPONENT_FILE.search(name):
                    structure["component_files"] += 1
                stem = name.split(".", 1)[0]
                in_server_dir = bool(set(dirs) & SERVER_DIRS)
                # src/index.js is a React entry point; only server folders or a root-level entry file count for Node.
                if (ext in (".js", ".mjs", ".cjs", ".ts") and (in_server_dir or (stem in ENTRY_STEMS and not dirs))
                        and not TEST_PATH.search(lower_path)):
                    structure["node_server_files"] += 1
                if ext == ".py" and (in_server_dir or stem in ENTRY_STEMS | {"views", "urls"}):
                    structure["python_server_files"] += 1
                if ext == ".java":
                    structure["java_files"] += 1
                if ext == ".cs":
                    structure["csharp_files"] += 1

            if name in MANIFEST_FILES or name.endswith(DOTNET_PROJECT_EXT) or name == "packages.config":
                manifests.append(file_path)
                has_manifest = True
            elif ext in PY_SOURCE and not TEST_PATH.search(lower_path):
                py_code.append(file_path)
            elif ext in JS_SOURCE:
                js_code.append(file_path)

        # Manifests near the root first; code files by how likely they show the real stack (entry points, server).
        manifests.sort(key=depth)
        selected = manifests[:MAX_MANIFESTS] + _select_code(py_code, js_code)
        size_known = [p for p in selected if entries[p].get("size") is not None]
        selected = [p for p in selected if p not in size_known or entries[p]["size"] <= _content_limit(p)]
        logger.debug("repo=%s selected_files=%d", repo["full_name"], len(selected))

        # Sparse checkout: download only the selected files, in one batch.
        if selected and api is None:
            sparse_started = time.perf_counter()
            git(["config", "core.sparseCheckout", "true"], cwd=tmp)
            info_dir = os.path.join(tmp, ".git", "info")
            os.makedirs(info_dir, exist_ok=True)
            with open(os.path.join(info_dir, "sparse-checkout"), "w", encoding="utf-8") as f:
                f.write("\n".join(sparse_pattern(p) for p in selected) + "\n")
            git(["read-tree", "-mu", "HEAD"], cwd=tmp)
            sparse_seconds = time.perf_counter() - sparse_started

        if api is not None:
            contents = resources.enter_context(closing(api.contents(session, url, entries, selected)))
        else:
            contents = resources.enter_context(closing(_checkout_contents(tmp, selected)))
        content_started = time.perf_counter()
        for rel, content in contents:
            detection_started = time.perf_counter()
            name = rel.rsplit("/", 1)[-1].lower()
            detected: set[str] = set()
            if name == "package.json":
                for tech, scope in inspect_package_json(content).items():
                    record({tech}, "dependency_manifest", rel, scope)
            elif name in PYTHON_MANIFESTS:
                detect_python_manifest(content, detected)
                if name == "setup.py":
                    detect_python_source(content, detected)
                frozen = name != "setup.py" and looks_frozen(content)
                record(detected, "dependency_manifest", rel, "frozen" if frozen else None)
            elif name in DOCKER_FILES:
                detect_docker(content, detected)
                record(detected, "container_image", rel)
            elif name in JVM_BUILD_FILES:
                detect_jvm_build(content, detected)
                record(detected, "dependency_manifest", rel)
            elif name.endswith(DOTNET_PROJECT_EXT) or name == "packages.config":
                detect_dotnet_project(content, detected)
                record(detected, "dependency_manifest", rel)
            elif name.endswith(".ipynb"):
                detect_python_source(notebook_to_source(content), detected)
                record(detected, "source_import", rel)
            elif name.endswith(".py"):
                detect_python_source(content, detected)
                record(detected, "source_import", rel)
            elif name.endswith(JS_SOURCE):
                tooling = bool(TOOLING_SCRIPT.search(name) or TEST_PATH.search(rel.lower()))
                imported, server_code = inspect_js_source(content, tooling)
                record(imported, "source_import", rel)
                record(server_code, "source_pattern", rel)
            detection_seconds += time.perf_counter() - detection_started
        content_seconds = time.perf_counter() - content_started
        if api is not None:
            api_seconds += content_seconds - detection_seconds
        else:
            detection_seconds = content_seconds

        result = finish()
        logger.info("analysis_completed repo=%s technologies=%d excluded=%s duration=%.3fs",
                    repo["full_name"], len(result["technologies"]), result.get("exclusion_reason", "no"),
                    result["analysis_seconds"])
        return result
    except EmptyRepositoryError:
        empty = True
        return finish()
    except Exception as e:  # noqa: BLE001 - report any failure for this repo
        logger.warning("analysis_failed repo=%s error=%s", repo["full_name"], type(e).__name__)
        return finish(str(e))
    finally:
        resources.close()


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
            result["cache_refreshed_at"] = datetime.now(timezone.utc).isoformat()
            with self.lock:
                self.data[self.key(repo)] = result

    def save(self) -> None:
        """Write the cache, dropping entries no run can hit again: other detection versions and superseded pushes."""
        with self.lock:
            latest: dict[str, tuple[str, str]] = {}
            for key, value in self.data.items():
                version, _, rest = key.partition(":")
                if version != CACHE_VERSION:
                    continue
                repo_id = rest.partition(":")[0]
                stamp = str(value.get("cache_refreshed_at") or "")
                if repo_id not in latest or stamp >= latest[repo_id][1]:
                    latest[repo_id] = (key, stamp)
            keep = {key for key, _ in latest.values()}
            self.data = {key: value for key, value in self.data.items() if key in keep}
            text = json.dumps(self.data, separators=(",", ":"))
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
        "repositories_excluded": 0,
        "repository_exclusions": [],
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


def mark_duplicates(analysed: list[dict], threshold: float = DUPLICATE_SIMILARITY) -> None:
    """Flag repositories whose source files are (nearly) the same as another of the candidate's repositories.

    Compares blob hashes of implementation files (``source_signature``), so a project uploaded twice under two
    names is counted once, while two projects that only share boilerplate are not. The repository with more
    source files (then the older one) is kept. ``analysed`` holds per-candidate copies and is changed in place.
    """
    kept: list[tuple[dict, set[str]]] = []
    order = sorted(analysed, key=lambda r: (-len(r.get("source_signature") or ()), r.get("created_at") or "",
                                            r.get("repository", "")))
    for repo in order:
        signature = set(repo.get("source_signature") or ())
        if repo.get("excluded") or repo.get("error") or len(signature) < 3:
            continue
        for original, original_signature in kept:
            if len(signature & original_signature) / len(signature | original_signature) >= threshold:
                repo["excluded"] = True
                repo["exclusion_reason"] = f"duplicate_of:{original['repository']}"
                break
        else:
            kept.append((repo, signature))


def run_report(
    people: list[dict],
    token: GitHubAuthorization,
    skip_forks: bool = True,
    on_progress: Callable[[list[dict], int, int], None] | None = None,
    *,
    scan_method: RepositoryScanMethod = "clone",
    required_groups: tuple[tuple[str, ...], ...] = (),
    initial_repo_budget: int | None = MAX_INITIAL_REPOS_PER_CANDIDATE,
    exhaustive: bool = False,
    needs_more_evidence: Callable[[dict, list[dict]], bool] | None = None,
) -> list[dict]:
    """Fill results with per-candidate scan budgets and optional deferred expansion.

    ``scan_method="api"`` requires a token and scans repository files without Git.
    ``needs_more_evidence(person, analysed_repos)`` owns screening decisions;
    it is called after each stage and may request another deferred repo.
    """
    credentials = GitHubCredentials.from_token(token)
    if scan_method == "api" and not credentials.authenticated:
        raise GitHubError("GitHub API repository scanning requires a token")
    api = _GitHubRepositoryAPI(credentials) if scan_method == "api" else None
    scan_repository = partial(analyze_repository, api=api) if api is not None else analyze_repository
    configure_logging()
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:8]
    global _API_REQUESTS
    with _API_LOCK:
        _API_REQUESTS = 0
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
    listing_started = time.perf_counter()
    with ThreadPoolExecutor(credentials.max_parallel_requests) as pool:
        futures = {pool.submit(list_repositories, u, token, skip_forks): u for u in usernames}
        for future in as_completed(futures):
            u = futures[future]
            try:
                repos_by_user[u.lower()] = future.result()
            except GitHubError as e:
                user_errors[u.lower()] = str(e)
            except Exception as e:  # noqa: BLE001
                user_errors[u.lower()] = f"Could not list repos: {e}"
    listing_seconds = time.perf_counter() - listing_started

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
        identifier = ",".join(p["profiles"])
        p["_log_id"] = hashlib.sha256(identifier.encode()).hexdigest()[:10]
        logger.info("run=%s candidate=%s repos_discovered=%d", run_id, p["_log_id"], len(repos))

    # 2. Analyse every unique repository once (cache first).
    all_repos: dict[str, dict] = {}
    for repos in person_repos:
        all_repos.update(repos)

    results: dict[str, dict] = {}
    cache_started = time.perf_counter()
    for name, repo in all_repos.items():
        hit = cache.get(repo)
        if hit:
            results[name] = hit
            logger.debug("run=%s repo=%s cache=hit", run_id, name)
        else:
            logger.debug("run=%s repo=%s cache=miss", run_id, name)
    cache_seconds = time.perf_counter() - cache_started
    filtering_started = time.perf_counter()
    selected: dict[str, dict] = {}
    deferred: list[list[dict]] = []
    empty_count = 0
    # Repositories that are never projects (the profile README) are skipped before any request.
    pre_excluded: list[dict[str, str]] = []
    for p, repos in zip(active, person_repos):
        skipped = {name: reason for name, repo in repos.items() if (reason := pre_analysis_exclusion(repo))}
        pre_excluded.append(skipped)
        uncached = [repo for name, repo in repos.items() if name not in results and name not in skipped]
        empty = [repo for repo in uncached if repo.get("size") == 0]
        empty_count += len(empty)
        viable = [repo for repo in uncached if repo.get("size") != 0]
        first, later = select_for_candidate(viable, required_groups, None if exhaustive else initial_repo_budget)
        deferred.append(later)
        selected.update((repo["full_name"], repo) for repo in first)
        p["repositories_cached"] = sum(name in results for name in repos)
        p["repositories_selected"] = len(first)
        p["repositories_deferred"] = len(later)
        p["repositories_skipped_empty"] = len(empty)
        for repo in viable:
            relevance = score_repository(repo, required_groups)
            logger.debug("run=%s repo=%s language=%s score=%d priority=%s signals=%s decision=%s",
                         run_id, repo["full_name"], repo.get("language"), relevance.score, relevance.priority,
                         ",".join(relevance.signals), "selected" if repo in first else "deferred")
        logger.info("run=%s candidate=%s cached=%d selected=%d deferred=%d empty=%d",
                    run_id, p["_log_id"], p["repositories_cached"], len(first), len(later), len(empty))
    filtering_seconds = time.perf_counter() - filtering_started

    def refresh():
        for p, repos in zip(active, person_repos):
            p["done"] = sum(1 for name in repos if name in results)
        if on_progress:
            on_progress(people, len(results), len(set(results) | set(selected)))

    refresh()
    last_update = time.monotonic()
    # API scans need 2-3 sequential requests per repository (tree + batched GraphQL file reads), so every request
    # slot can serve its own repository; the shared semaphore in GitHubCredentials still bounds the total.
    scan_workers = WORKERS if api is None else credentials.max_parallel_requests

    def analyze_selected(batch: dict[str, dict]) -> None:
        nonlocal last_update
        with ThreadPoolExecutor(scan_workers) as pool:
            futures = {pool.submit(scan_repository, repo): repo for repo in batch.values() if repo["full_name"] not in results}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    result = future.result()
                except Exception as e:  # noqa: BLE001
                    logger.error("run=%s repo=%s analysis_failed=%s", run_id, repo["full_name"], type(e).__name__)
                    result = {"repository": repo["full_name"], "technologies": [], "categories": [], "error": str(e)}
                results[repo["full_name"]] = result
                cache.set(repo, result)
                if time.monotonic() - last_update > 0.5:
                    refresh()
                    last_update = time.monotonic()

    try:
        analyze_selected(selected)
        if needs_more_evidence is not None and not exhaustive:
            while api is None or not credentials.blocked:
                next_batch = {}
                for p, repos, remaining in zip(active, person_repos, deferred):
                    if not remaining:
                        continue
                    # Excluded repositories (practice, tutorials, no code) are not evidence.
                    analysed = [results[name] for name in repos if name in results
                                and not results[name].get("excluded")]
                    if needs_more_evidence(p, analysed):
                        next_repo = remaining.pop(0)
                        next_batch[next_repo["full_name"]] = next_repo
                        p["repositories_deferred"] = len(remaining)
                        p["repositories_selected"] += 1
                if not next_batch:
                    break
                selected.update(next_batch)
                analyze_selected(next_batch)
    finally:
        # Saved even when the run is interrupted, so finished scans are never repeated.
        try:
            cache.save()
        except OSError as exc:
            logger.error("run=%s cache_save_failed=%s", run_id, type(exc).__name__)
    refresh()
    failures = sum(bool(results[name].get("error")) for name in selected if name in results)
    deferred_count = sum(len(items) for items in deferred)
    metrics = {"profiles_processed": len(active), "repositories_discovered": sum(map(len, person_repos)),
               "scan_method": scan_method,
               "cache_hits": len(set(all_repos) & set(results)) - len(selected),
               "repositories_selected": len(selected), "repositories_deferred": deferred_count,
               "repositories_skipped_empty": empty_count, "successful_analyses": len(selected) - failures,
               "failed_analyses": failures,
               "git_clones_avoided": len(all_repos) if api is not None else len(all_repos) - len(selected),
               "listing_seconds": listing_seconds, "cache_seconds": cache_seconds,
               "filtering_seconds": filtering_seconds,
               "clone_seconds": sum(float(results[name].get("clone_seconds", 0)) for name in selected if name in results),
               "sparse_seconds": sum(float(results[name].get("sparse_seconds", 0)) for name in selected if name in results),
               "repository_api_seconds": sum(float(results[name].get("repository_api_seconds", 0))
                                             for name in selected if name in results),
               "detection_seconds": sum(float(results[name].get("detection_seconds", 0)) for name in selected if name in results),
               "analysis_seconds": sum(float(results[name].get("analysis_seconds", 0)) for name in selected if name in results),
               "api_requests": _API_REQUESTS,
               "total_seconds": time.perf_counter() - started}
    for p in active:
        p["analysis_metrics"] = metrics
        logger.info("run=%s candidate=%s completed duration=%.2fs", run_id, p["_log_id"], metrics["total_seconds"])
    logger.info("run=%s summary method=%s profiles=%d discovered=%d cached=%d selected=%d deferred=%d empty=%d "
                "success=%d failed=%d clones_avoided=%d reduction=%.1f%% api_requests=%d duration=%.2fs", run_id, scan_method,
                metrics["profiles_processed"], metrics["repositories_discovered"], metrics["cache_hits"],
                metrics["repositories_selected"], metrics["repositories_deferred"], metrics["repositories_skipped_empty"],
                metrics["successful_analyses"], metrics["failed_analyses"], metrics["git_clones_avoided"],
                100 * metrics["git_clones_avoided"] / max(len(all_repos), 1), metrics["api_requests"],
                metrics["total_seconds"])

    # 3. Combine results per person.
    for p, repos, remaining, skipped in zip(active, person_repos, deferred, pre_excluded):
        analysed = [dict(results[name], **{
            "url": repos[name].get("html_url", results[name].get("url", "")),
            "description": repos[name].get("description") or "",
            "created_at": repos[name].get("created_at"), "updated_at": repos[name].get("updated_at"),
            "pushed_at": repos[name].get("pushed_at"), "last_commit_at": None,
            "archived": bool(repos[name].get("archived")), "fork": bool(repos[name].get("fork")),
        }) for name in repos if name in results and name not in skipped]
        mark_duplicates(analysed)
        included = [r for r in analysed if not r.get("excluded") and not r.get("error")]
        p["repository_results"] = analysed
        p["deferred_repository_names"] = [repo["full_name"] for repo in remaining]
        p["failed_repos"] = sum(1 for r in analysed if r.get("error"))
        p["repository_exclusions"] = (
            [{"repository": name, "reason": reason} for name, reason in skipped.items()]
            + [{"repository": r["repository"], "reason": r["exclusion_reason"]} for r in analysed if r.get("excluded")]
        )
        p["repositories_excluded"] = len(p["repository_exclusions"])
        # Recomputed from technologies, so older cached results get the new categories too.
        p["categories"] = count_by([categorize(r.get("technologies", []), r.get("implementation"))
                                    for r in included])
        p["technologies"] = [name for name, _ in count_by([r.get("technologies", []) for r in included])]
        failed_profiles = sum(1 for u in p["profiles"] if u.lower() in user_errors)
        p["status"] = "Failed" if failed_profiles == len(p["profiles"]) else "Done"

    if on_progress:
        on_progress(people, len(results), len(set(results) | set(selected)))
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
