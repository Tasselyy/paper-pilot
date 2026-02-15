#!/usr/bin/env python3
"""
Environment readiness check for PaperPilot development.

Run before starting any dev work:
    python scripts/check_env.py

Checks:
    1. Python version >= 3.10
    2. Virtual environment is active (.venv)
    3. Core dependencies are importable
    4. RAG_PROJECT_ROOT env var (optional, warns if missing)
    5. config/settings.yaml exists (optional, warns if missing)
"""

import importlib
import os
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────
MIN_PYTHON = (3, 10)
CORE_PACKAGES = [
    ("pydantic", "pydantic"),
    ("pyyaml", "yaml"),
    ("rich", "rich"),
]
AGENT_PACKAGES = [
    ("langgraph", "langgraph"),
    ("langchain", "langchain"),
    ("langchain-core", "langchain_core"),
    ("langchain-openai", "langchain_openai"),
    ("langchain-mcp-adapters", "langchain_mcp_adapters"),
]
DEV_PACKAGES = [
    ("pytest", "pytest"),
    ("ruff", None),  # CLI only, no Python import
]

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"


def check_python_version() -> bool:
    ver = sys.version_info[:2]
    ok = ver >= MIN_PYTHON
    tag = PASS if ok else FAIL
    print(f"  {tag} Python {ver[0]}.{ver[1]} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})")
    return ok


def check_venv() -> bool:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    exe = Path(sys.executable)
    is_project_venv = ".venv" in exe.parts
    tag = PASS if (in_venv and is_project_venv) else (WARN if in_venv else FAIL)
    detail = f"({exe})"
    if not in_venv:
        detail += " — activate with: .venv\\Scripts\\Activate.ps1"
    elif not is_project_venv:
        detail += " — active but not project .venv"
    print(f"  {tag} Virtual environment {detail}")
    return in_venv


def check_packages(name: str, packages: list) -> int:
    """Check a group of packages. Returns count of missing."""
    missing = 0
    for pip_name, import_name in packages:
        if import_name is None:
            # CLI-only package, skip import check
            continue
        try:
            importlib.import_module(import_name)
            print(f"  {PASS} {pip_name}")
        except ImportError:
            print(f"  {FAIL} {pip_name} — pip install {pip_name}")
            missing += 1
    return missing


def check_rag_project_root() -> bool:
    val = os.environ.get("RAG_PROJECT_ROOT")
    if not val:
        print(f"  {WARN} RAG_PROJECT_ROOT not set (needed for MCP integration)")
        return False
    p = Path(val)
    if not p.exists():
        print(f"  {WARN} RAG_PROJECT_ROOT={val} — path does not exist")
        return False
    print(f"  {PASS} RAG_PROJECT_ROOT={val}")
    return True


def check_config_file() -> bool:
    repo = Path(__file__).resolve().parent.parent
    cfg = repo / "config" / "settings.yaml"
    if cfg.exists():
        print(f"  {PASS} config/settings.yaml found")
        return True
    print(f"  {WARN} config/settings.yaml not found (will be created in task A2)")
    return False


def main():
    print("\n=== PaperPilot Environment Check ===\n")
    errors = 0

    print("1. Python version")
    if not check_python_version():
        errors += 1

    print("\n2. Virtual environment")
    if not check_venv():
        errors += 1

    print("\n3. Core packages")
    errors += check_packages("core", CORE_PACKAGES)

    print("\n4. Agent packages")
    agent_missing = check_packages("agent", AGENT_PACKAGES)
    if agent_missing:
        print(f"  {INFO} Install all: pip install -e .")
    errors += agent_missing

    print("\n5. Dev packages")
    dev_missing = check_packages("dev", DEV_PACKAGES)
    if dev_missing:
        print(f"  {INFO} Install all: pip install -e '.[dev]'")
    errors += dev_missing

    print("\n6. Environment variables")
    check_rag_project_root()  # warn only, not an error

    print("\n7. Config files")
    check_config_file()  # warn only, not an error

    print("\n" + "=" * 40)
    if errors == 0:
        print(f"{PASS} All checks passed! Ready to develop.")
    else:
        print(f"{FAIL} {errors} issue(s) found. Fix them before proceeding.")
    print()

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
