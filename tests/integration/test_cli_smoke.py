"""CLI smoke tests for ``main.py`` dry-run execution (H6).

Validates the CLI path can be exercised automatically in pytest without
manual terminal execution:

- Subprocess run of ``main.py --question ... --dry-run`` exits with code 0.
- CLI stdout contains a rendered answer panel and answer text.
- Programmatic ``run_agent(...)`` returns a state containing ``final_answer``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from main import run_agent


def _project_root() -> Path:
    """Return repository root from this test file location."""
    return Path(__file__).resolve().parents[2]


def test_main_cli_dry_run_exits_zero_and_prints_answer() -> None:
    """Dry-run CLI invocation should succeed and print an answer."""
    root = _project_root()
    cmd = [
        sys.executable,
        "main.py",
        "--question",
        "What is LoRA?",
        "--dry-run",
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, combined_output
    assert "Answer" in combined_output, combined_output
    assert "No RAG/LLM configured yet." in combined_output, combined_output


async def test_run_agent_dry_run_returns_final_answer() -> None:
    """``run_agent`` should return a state with non-empty ``final_answer``."""
    result = await run_agent(
        question="What is LoRA?",
        config_path="config/settings.yaml",
        dry_run=True,
        verbose=False,
    )

    final_answer = result.get("final_answer", "")
    assert final_answer, "Expected non-empty final_answer from dry-run execution"
    assert "No RAG/LLM configured yet." in final_answer, final_answer
