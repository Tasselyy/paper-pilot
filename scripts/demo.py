#!/usr/bin/env python3
"""Interview demo runner for Paper Pilot.

This script defines four curated demo questions, one per strategy:
`simple`, `comparative`, `multi_hop`, and `exploratory`.

Usage examples:
    python scripts/demo.py
    python scripts/demo.py --run --dry-run
    python scripts/demo.py --run --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoCase:
    """A single interview demo case."""

    question: str
    expected_strategy: str
    highlight: str


DEMO_CASES: tuple[DemoCase, ...] = (
    DemoCase(
        question="What is LoRA?",
        expected_strategy="simple",
        highlight="Single factual concept with concise answer path.",
    ),
    DemoCase(
        question="Compare LoRA and QLoRA in terms of memory and accuracy.",
        expected_strategy="comparative",
        highlight="Entity + dimension comparison with side-by-side synthesis.",
    ),
    DemoCase(
        question="First explain attention in Transformers, then how FlashAttention improves it.",
        expected_strategy="multi_hop",
        highlight="Two-step dependency that benefits from decomposition.",
    ),
    DemoCase(
        question="What are the latest trends in efficient fine-tuning?",
        expected_strategy="exploratory",
        highlight="Open-ended survey requiring broader exploration.",
    ),
)

_STRATEGY_PATTERN = re.compile(r"strategy=([a-z_]+)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="paper-pilot-demo",
        description="Run interview demo cases for all four strategies.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute each question against main.py (otherwise print plan only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to main.py while executing demo cases.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Config path forwarded to main.py when --run is used.",
    )
    return parser.parse_args(argv)


def print_demo_plan() -> None:
    """Print the 4-question interview plan with expected strategies."""
    print("Paper Pilot Interview Demo Plan\n")
    for idx, case in enumerate(DEMO_CASES, start=1):
        print(f"{idx}. Q: {case.question}")
        print(f"   Expected strategy: {case.expected_strategy}")
        print(f"   Highlight: {case.highlight}\n")
    print("Tip: use live mode to verify real intent routing behavior.")


def _extract_strategy(output: str) -> str | None:
    """Extract strategy tag from CLI output."""
    match = _STRATEGY_PATTERN.search(output)
    return match.group(1) if match else None


def run_demo_cases(*, repo_root: Path, dry_run: bool, config_path: str) -> int:
    """Run all demo cases and print expected vs observed strategies."""
    print_demo_plan()
    print("\nExecuting demo cases...\n")

    if dry_run:
        print("[NOTE] --dry-run enabled: placeholder routing may not reflect full live strategy selection.\n")

    failures = 0
    info_mismatches = 0
    for idx, case in enumerate(DEMO_CASES, start=1):
        cmd = [sys.executable, "main.py", "--question", case.question, "--config", config_path]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[Case {idx}] {case.expected_strategy}: {case.question}")
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        strategy = _extract_strategy(completed.stdout)
        print(f"  exit_code={completed.returncode}")
        print(f"  expected={case.expected_strategy} observed={strategy or 'N/A'}")
        if completed.returncode != 0:
            failures += 1
            print("  status=FAILED (non-zero exit)")
        elif strategy != case.expected_strategy:
            if dry_run:
                info_mismatches += 1
                print("  status=INFO (dry-run placeholder routing)")
            else:
                failures += 1
                print("  status=MISMATCH")
        else:
            print("  status=OK")
        print()

    if failures:
        print(f"Demo completed with {failures} mismatch/error case(s).")
        return 1
    if info_mismatches:
        print(
            "Demo completed with placeholder-routing notices in dry-run mode; "
            "use live mode for strict strategy validation."
        )
        return 0
    print("Demo completed: all cases matched expected strategies.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for demo script."""
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    if not args.run:
        print_demo_plan()
        return 0
    return run_demo_cases(
        repo_root=repo_root,
        dry_run=args.dry_run,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
