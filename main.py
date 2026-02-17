"""Paper Pilot — CLI entry point.

Provides a fully functional CLI for invoking the multi-strategy research
agent.  Supports configurable MCP/LLM connections and a ``--dry-run``
mode that exercises the graph with placeholder nodes (no external
services required).

Usage::

    # Full run (requires MCP RAG Server + OpenAI API key)
    python main.py --question "What is LoRA?"

    # Dry-run (placeholders — no external services needed)
    python main.py --question "What is LoRA?" --dry-run

    # Custom config file
    python main.py -q "Compare BERT and GPT" --config path/to/settings.yaml

    # Verbose output
    python main.py -q "Explain attention" --dry-run --verbose

Design reference: DEV_SPEC E4.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

from rich.console import Console

logger = logging.getLogger(__name__)


def _suppress_pydantic_serializer_noise() -> None:
    """Suppress known non-actionable serializer warnings in CLI output.

    LangChain structured output can attach parsed objects to transient message
    payloads, and Pydantic may emit "Pydantic serializer warnings" while
    serializing those internals. This is noisy but does not affect behavior.
    """
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pydantic\.main",
        message=r"(?s)Pydantic serializer warnings:.*",
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


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
        help="Run without calling real LLM/MCP (uses placeholder nodes).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose / debug output.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------


async def run_agent(
    *,
    question: str,
    config_path: str,
    dry_run: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run the Paper Pilot agent and return the final state.

    In **dry-run** mode the graph is built with placeholder nodes so
    that no external services (MCP RAG Server, Cloud LLM) are needed.
    In **live** mode the graph is wired to real MCP tools and a Cloud
    LLM via the configuration in *config_path*.

    Args:
        question: The user's research question.
        config_path: Path to the YAML settings file.
        dry_run: If ``True``, skip real LLM/MCP dependencies.
        verbose: Enable verbose Rich output.

    Returns:
        The final agent state dict.
    """
    from src.agent.graph import build_main_graph
    from src.config import load_settings
    from src.tracing.rich_output import RichStreamOutput
    from src.tracing.tracer import AgentTrace

    console = Console()

    # ── Load configuration ─────────────────────────────────
    try:
        settings = load_settings(config_path)
    except FileNotFoundError:
        if dry_run:
            # Dry-run can proceed without a config file
            settings = None
            console.print(
                f"[yellow]Config not found ({config_path}) — "
                "proceeding with defaults (dry-run).[/yellow]"
            )
        else:
            console.print(
                f"[red]Error: Config file not found: {config_path}[/red]"
            )
            sys.exit(1)
    except ValueError as exc:
        if dry_run:
            settings = None
            console.print(
                f"[yellow]Config validation issue — "
                f"proceeding with defaults (dry-run): {exc}[/yellow]"
            )
        else:
            console.print(f"[red]Config error: {exc}[/red]")
            sys.exit(1)

    # ── Build graph ────────────────────────────────────────
    rag = None
    llm = None

    if not dry_run and settings is not None:
        from src.llm.client import create_llm
        from src.tools.mcp_client import MCPClientManager
        from src.tools.tool_wrapper import RAGToolWrapper

        llm = create_llm(settings.llm)
        mcp_manager = MCPClientManager(settings.mcp)
        server_names = mcp_manager.get_server_names()
        if not server_names:
            console.print("[red]No MCP servers configured.[/red]")
            sys.exit(1)
        # Keep one MCP session open for the whole run so RAG uses one process
        # instead of spawning a new one per search (avoids ~90s cold start each time).
        server_name = server_names[0]
        console.print(
            f"[dim]Opening MCP session for {server_name!r} (spawn RAG process)...[/dim]"
        )
        async with mcp_manager.session(server_name) as _mcp_session:
            console.print("[dim]MCP session ready, loading tools...[/dim]")
            mcp_tools = await mcp_manager.get_tools_using_session(
                _mcp_session, server_name
            )
            rag = RAGToolWrapper(mcp_tools)
            console.print(
                f"[dim]MCP tools: {', '.join(rag.available_tools())}[/dim]"
            )

            rag_default_collection = getattr(
                settings.mcp, "rag_default_collection", None
            )
            graph = build_main_graph(
                rag=rag,
                llm=llm,
                rag_default_collection=rag_default_collection,
            )
            thread_id = uuid.uuid4().hex[:12]
            graph_config: dict[str, Any] = {
                "configurable": {"thread_id": thread_id},
            }
            rich_output = RichStreamOutput(console=console, verbose=verbose)
            run_start = time.time()
            final_state = await rich_output.stream_graph(
                graph,
                {"question": question},
                config=graph_config,
            )
            run_duration_ms = (time.time() - run_start) * 1000
            intent = final_state.get("intent")
            strategy = (
                intent.to_strategy()
                if intent is not None and hasattr(intent, "to_strategy")
                else (str(intent) if intent is not None else "")
            )
            retry_count = final_state.get("retry_count", 0)
            rich_output.display_summary(
                strategy=strategy,
                total_ms=run_duration_ms,
                retry_count=retry_count,
            )
            trace_dir = Path(settings.tracing.trace_dir)
            trace_file = trace_dir / settings.tracing.trace_file
            try:
                from src.agent.state import AgentState
                state_obj = _build_agent_state(final_state)
                trace = AgentTrace.from_state(
                    state_obj,
                    duration_ms=run_duration_ms,
                )
                trace.flush_to_jsonl(trace_file)
                console.print(f"[dim]Trace saved to {trace_file}[/dim]")
            except Exception as exc:
                logger.warning("Failed to persist trace: %s", exc)
                if verbose:
                    console.print(
                        f"[yellow]Trace persistence failed: {exc}[/yellow]"
                    )
            return final_state

    rag_default_collection = (
        getattr(settings.mcp, "rag_default_collection", None)
        if settings
        else None
    )
    graph = build_main_graph(
        rag=rag,
        llm=llm,
        rag_default_collection=rag_default_collection,
    )

    # ── Configure run ──────────────────────────────────────
    thread_id = uuid.uuid4().hex[:12]
    graph_config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
    }

    # ── Stream with Rich output ────────────────────────────
    rich_output = RichStreamOutput(console=console, verbose=verbose)
    run_start = time.time()

    final_state = await rich_output.stream_graph(
        graph,
        {"question": question},
        config=graph_config,
    )

    run_duration_ms = (time.time() - run_start) * 1000

    # ── Execution summary ──────────────────────────────────
    intent = final_state.get("intent")
    strategy = ""
    if intent is not None:
        strategy = (
            intent.to_strategy()
            if hasattr(intent, "to_strategy")
            else str(intent)
        )

    retry_count = final_state.get("retry_count", 0)

    rich_output.display_summary(
        strategy=strategy,
        total_ms=run_duration_ms,
        retry_count=retry_count,
    )

    # ── Trace persistence ──────────────────────────────────
    if settings is not None:
        trace_dir = Path(settings.tracing.trace_dir)
        trace_file = trace_dir / settings.tracing.trace_file
    else:
        trace_dir = Path("data/traces")
        trace_file = trace_dir / "trace.jsonl"

    try:
        # Build AgentTrace from raw state dict
        from src.agent.state import AgentState

        # Reconstruct AgentState from the accumulated dict when possible
        state_obj = _build_agent_state(final_state)
        trace = AgentTrace.from_state(
            state_obj,
            duration_ms=run_duration_ms,
        )
        trace.flush_to_jsonl(trace_file)
        console.print(f"[dim]Trace saved to {trace_file}[/dim]")
    except Exception as exc:
        logger.warning("Failed to persist trace: %s", exc)
        if verbose:
            console.print(f"[yellow]Trace persistence failed: {exc}[/yellow]")

    return final_state


def _build_agent_state(state_dict: dict[str, Any]) -> Any:
    """Best-effort reconstruction of ``AgentState`` from a raw dict.

    If the dict contains Pydantic model instances (e.g. ``Intent``,
    ``CriticVerdict``) they are used directly.  Scalar fields are
    passed through.  Fields that cannot be set are silently ignored.

    Args:
        state_dict: The accumulated state dict from graph execution.

    Returns:
        An ``AgentState`` instance (or a minimal stand-in).
    """
    from src.agent.state import AgentState

    # Filter to only known AgentState fields
    known_fields = set(AgentState.model_fields.keys())
    filtered = {k: v for k, v in state_dict.items() if k in known_fields}
    try:
        return AgentState(**filtered)
    except Exception:
        # Fallback: minimal state with just the essentials
        return AgentState(
            question=state_dict.get("question", ""),
            final_answer=state_dict.get("final_answer", ""),
            intent=state_dict.get("intent"),
            critic_verdict=state_dict.get("critic_verdict"),
            retry_count=state_dict.get("retry_count", 0),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Run the Paper Pilot agent.

    Args:
        argv: CLI argument list.
    """
    args = parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    _suppress_pydantic_serializer_noise()

    # Run the async agent
    final_state = asyncio.run(
        run_agent(
            question=args.question,
            config_path=args.config,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    )

    # Exit with error code if no answer was produced
    final_answer = final_state.get("final_answer", "")
    if not final_answer:
        sys.exit(1)


if __name__ == "__main__":
    main()
