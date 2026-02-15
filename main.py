"""Paper Pilot — CLI entry point.

Usage::

    python main.py --question "What is LoRA?"
    python main.py --question "Compare BERT and GPT" --dry-run
"""

from __future__ import annotations

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="paper-pilot",
        description="Multi-strategy research Agent powered by LangGraph",
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        required=True,
        help="The research question to answer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without calling real LLM/MCP (for testing).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Paper Pilot agent.

    Args:
        argv: CLI argument list.
    """
    args = parse_args(argv)
    print(f"[paper-pilot] question: {args.question}")
    print(f"[paper-pilot] config:   {args.config}")
    print(f"[paper-pilot] dry-run:  {args.dry_run}")

    # Placeholder — full implementation in task E4
    from src.agent import build_main_graph  # noqa: F811

    graph = build_main_graph()
    print(f"[paper-pilot] graph built: {graph}")
    print("[paper-pilot] (end — full pipeline not yet wired)")


if __name__ == "__main__":
    main()
