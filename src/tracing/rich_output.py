"""Rich streaming output for LangGraph agent execution.

Provides ``RichStreamOutput`` — a callback-driven display layer that uses
the ``rich`` library to render real-time progress as graph nodes execute.

Features:

- Real-time node completion display with timing and key output summaries.
- Question header and final answer panels.
- Compact execution summary (strategy, tokens, retries).
- Async ``stream_graph()`` integration via ``graph.astream()``.
- Configurable verbosity.

Usage modes:

1. **Callback mode** — call ``on_node_start`` / ``on_node_end`` from node
   wrappers or graph hooks::

       out = RichStreamOutput()
       out.display_header("What is LoRA?")
       out.on_node_start("route")
       # ... node executes ...
       out.on_node_end("route", output={"intent": intent})
       out.display_final_answer(state.final_answer)
       out.display_summary(strategy="simple", total_ms=450)

2. **Async streaming mode** — wrap ``graph.astream()`` for automatic
   Rich output::

       out = RichStreamOutput()
       final = await out.stream_graph(compiled_graph, {"question": "..."})

Design reference: DEV_SPEC E3.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node display configuration
# ---------------------------------------------------------------------------

_NODE_DISPLAY: dict[str, tuple[str, str]] = {
    # node_name -> (display_label, rich_style)
    "load_memory": ("Load Memory", "dim"),
    "route": ("Router", "bold cyan"),
    "slot_fill": ("Slot Filling", "bold cyan"),
    "simple": ("Simple Strategy", "bold green"),
    "multi_hop": ("Multi-hop Strategy", "bold green"),
    "comparative": ("Comparative Strategy", "bold green"),
    "exploratory": ("Exploratory Strategy", "bold green"),
    "critic": ("Critic", "bold yellow"),
    "retry_refine": ("Retry / Refine", "bold red"),
    "save_memory": ("Save Memory", "dim"),
    "format_output": ("Format Output", "bold magenta"),
}

_STRATEGY_NODES = frozenset({"simple", "multi_hop", "comparative", "exploratory"})


def _get_node_display(node_name: str) -> tuple[str, str]:
    """Return ``(display_label, rich_style)`` for a graph node.

    Falls back to the raw *node_name* with ``"bold"`` style when the
    node is not in the predefined display map.

    Args:
        node_name: Graph node identifier.

    Returns:
        A two-tuple of (display label, Rich markup style).
    """
    return _NODE_DISPLAY.get(node_name, (node_name, "bold"))


# ---------------------------------------------------------------------------
# Helper — attribute / key access
# ---------------------------------------------------------------------------


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Retrieve a value from *obj* by attribute or dict key.

    Works transparently with Pydantic models and plain dicts.

    Args:
        obj: Object to read from.
        name: Attribute or key name.
        default: Fallback value when the name is not found.

    Returns:
        The resolved value, or *default*.
    """
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


# ---------------------------------------------------------------------------
# Per-node output summarisers
# ---------------------------------------------------------------------------


def _summarize_node_output(node_name: str, output: dict[str, Any]) -> str:
    """Extract a concise summary from a node's partial-state output.

    The summary is a short, human-readable string highlighting the most
    important fields produced by each node type.

    Args:
        node_name: The graph node name.
        output: The partial state dict returned by the node.

    Returns:
        A short summary string, or ``""`` if nothing notable.
    """
    if node_name == "route":
        return _summarize_route(output)
    if node_name == "slot_fill":
        return _summarize_slot_fill(output)
    if node_name in _STRATEGY_NODES:
        return _summarize_strategy(output)
    if node_name == "critic":
        return _summarize_critic(output)
    if node_name == "retry_refine":
        count = output.get("retry_count")
        return f"retry #{count}" if count is not None else ""
    if node_name == "format_output":
        answer = output.get("final_answer", "")
        return f"{len(answer)} chars" if isinstance(answer, str) and answer else ""
    return ""


def _summarize_route(output: dict[str, Any]) -> str:
    intent = output.get("intent")
    if intent is None:
        return ""
    itype = _attr_or_key(intent, "type", "")
    conf = _attr_or_key(intent, "confidence", None)
    if not itype:
        return ""
    conf_str = f" ({conf:.0%})" if conf is not None else ""
    return f"intent={itype}{conf_str}"


def _summarize_slot_fill(output: dict[str, Any]) -> str:
    intent = output.get("intent")
    if intent is None:
        return ""
    query: str = _attr_or_key(intent, "reformulated_query", "") or ""
    if not query:
        return ""
    truncated = (query[:50] + "...") if len(query) > 50 else query
    return f'query="{truncated}"'


def _summarize_strategy(output: dict[str, Any]) -> str:
    contexts = output.get("retrieved_contexts", [])
    n = len(contexts) if isinstance(contexts, list) else 0
    draft = output.get("draft_answer", "")
    draft_len = len(draft) if isinstance(draft, str) else 0
    parts: list[str] = []
    if n:
        parts.append(f"{n} context(s)")
    if draft_len:
        parts.append(f"draft={draft_len} chars")
    return ", ".join(parts)


def _summarize_critic(output: dict[str, Any]) -> str:
    verdict = output.get("critic_verdict")
    if verdict is None:
        return ""
    passed = _attr_or_key(verdict, "passed", None)
    score = _attr_or_key(verdict, "score", 0)
    if passed is None:
        return ""
    status = "passed" if passed else "failed"
    return f"{status} ({score:.1f}/10)"


# ---------------------------------------------------------------------------
# RichStreamOutput
# ---------------------------------------------------------------------------


class RichStreamOutput:
    """Rich-powered streaming display for agent graph execution.

    Provides callback hooks for per-node events plus an async
    ``stream_graph()`` helper that wraps ``graph.astream()`` with
    automatic Rich rendering.

    Attributes:
        console: The ``rich.Console`` used for rendering.
        verbose: Whether verbose output is enabled.
    """

    def __init__(
        self,
        *,
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialise the streaming output handler.

        Args:
            console: Optional Rich Console (creates default if omitted).
            verbose: Show additional output detail when ``True``.
        """
        self._console = console or Console()
        self._verbose = verbose
        self._start_times: dict[str, float] = {}
        self._completed: list[tuple[str, float, dict[str, Any]]] = []
        self._run_start: float = 0.0

    # -- Properties ----------------------------------------------------------

    @property
    def console(self) -> Console:
        """The Rich Console instance."""
        return self._console

    @property
    def verbose(self) -> bool:
        """Whether verbose output is enabled."""
        return self._verbose

    @property
    def completed_nodes(self) -> list[tuple[str, float, dict[str, Any]]]:
        """Completed nodes as ``(node_name, duration_ms, output)`` tuples.

        Returns a *copy* so mutations do not affect internal state.
        """
        return list(self._completed)

    # -- Lifecycle -----------------------------------------------------------

    def display_header(self, question: str) -> None:
        """Display the question header panel and reset internal state.

        Args:
            question: The user's research question.
        """
        self._run_start = time.time()
        self._completed.clear()
        self._start_times.clear()

        panel = Panel(
            Text(question, style="bold white"),
            title="[bold]Paper Pilot[/bold]",
            subtitle="[dim]Multi-strategy Research Agent[/dim]",
            border_style="blue",
            padding=(0, 2),
        )
        self._console.print(panel)
        self._console.print()

    def on_node_start(self, node_name: str) -> None:
        """Record the start of a node execution.

        Internally stores the start timestamp for duration computation.
        In verbose mode an "in-progress" indicator is printed.

        Args:
            node_name: Name of the graph node starting execution.
        """
        self._start_times[node_name] = time.time()
        if self._verbose:
            label, style = _get_node_display(node_name)
            self._console.print(
                f"  [dim]...[/dim] [{style}]{label}[/{style}]",
            )
        logger.debug("Node started: %s", node_name)

    def on_node_end(
        self,
        node_name: str,
        *,
        output: dict[str, Any] | None = None,
    ) -> None:
        """Record node completion and print the result line.

        Displays a check mark, the node label, elapsed time, and a
        concise summary of key output fields.

        Args:
            node_name: Name of the completed node.
            output: The node's partial state update dict.
        """
        start = self._start_times.pop(node_name, None)
        duration_ms = (time.time() - start) * 1000 if start else 0.0

        self._completed.append((node_name, duration_ms, output or {}))

        label, style = _get_node_display(node_name)
        summary = _summarize_node_output(node_name, output or {})

        time_str = f"[dim]{duration_ms:>7.0f}ms[/dim]"
        summary_str = f"  [dim italic]{summary}[/dim italic]" if summary else ""

        self._console.print(
            f"  [green]v[/green] [{style}]{label:<22s}[/{style}]"
            f" {time_str}{summary_str}",
        )
        logger.debug("Node completed: %s (%.1fms)", node_name, duration_ms)

    def on_node_error(self, node_name: str, error: Exception) -> None:
        """Display a node execution error.

        Args:
            node_name: Name of the failed node.
            error: The exception that occurred.
        """
        start = self._start_times.pop(node_name, None)
        duration_ms = (time.time() - start) * 1000 if start else 0.0

        label, style = _get_node_display(node_name)
        error_msg = str(error)[:80]

        self._console.print(
            f"  [red]x[/red] [{style}]{label:<22s}[/{style}]"
            f" [dim]{duration_ms:>7.0f}ms[/dim]  [red]{error_msg}[/red]",
        )
        logger.error("Node error: %s - %s", node_name, error)

    # -- Output displays -----------------------------------------------------

    def display_final_answer(
        self,
        answer: str,
        *,
        sources: list[str] | None = None,
    ) -> None:
        """Display the final answer in a styled panel.

        Args:
            answer: The formatted final answer text.
            sources: Optional list of source citations to append.
        """
        self._console.print()

        body = answer
        if sources:
            source_lines = "\n".join(f"  - {s}" for s in sources)
            body += f"\n\nSources:\n{source_lines}"

        panel = Panel(
            body,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        self._console.print(panel)

    def display_summary(
        self,
        *,
        strategy: str = "",
        total_ms: float | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        retry_count: int = 0,
    ) -> None:
        """Display a compact execution summary.

        Args:
            strategy: The strategy that was executed.
            total_ms: Total duration in ms (auto-computed if omitted).
            tokens_in: Total input tokens consumed.
            tokens_out: Total output tokens consumed.
            retry_count: Number of critic retries.
        """
        if total_ms is None:
            total_ms = (
                (time.time() - self._run_start) * 1000
                if self._run_start
                else 0.0
            )

        parts: list[str] = [f"[bold]{total_ms:.0f}ms[/bold] total"]
        if strategy:
            parts.append(f"strategy=[cyan]{strategy}[/cyan]")
        if tokens_in or tokens_out:
            parts.append(f"tokens={tokens_in:,} in / {tokens_out:,} out")
        if retry_count:
            parts.append(f"retries={retry_count}")

        self._console.print()
        self._console.print(f"  {' | '.join(parts)}")
        self._console.print()

    def display_trace_summary(self, trace: Any) -> None:
        """Display execution summary extracted from an ``AgentTrace``.

        Uses ``getattr`` to read fields, so any object with the expected
        attributes (or an ``AgentTrace`` instance) will work.

        Args:
            trace: A completed trace object with ``strategy_executed``,
                ``duration_ms``, ``tokens_used``, and ``retry_count``.
        """
        tokens = getattr(trace, "tokens_used", {}) or {}
        self.display_summary(
            strategy=getattr(trace, "strategy_executed", ""),
            total_ms=getattr(trace, "duration_ms", None),
            tokens_in=tokens.get("input", 0),
            tokens_out=tokens.get("output", 0),
            retry_count=getattr(trace, "retry_count", 0),
        )

    # -- Async streaming integration -----------------------------------------

    async def stream_graph(
        self,
        graph: Any,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Stream graph execution with real-time Rich output.

        Wraps LangGraph's ``astream(stream_mode="updates")`` to display
        per-node completion as events arrive.  A Rich spinner is shown
        between events to indicate the agent is working.

        Args:
            graph: A compiled LangGraph ``StateGraph``.
            input_state: Initial state dict (should include ``question``).
            config: Optional LangGraph run config dict.

        Returns:
            The final accumulated state dict after graph execution.
        """
        question = input_state.get("question", "")
        self.display_header(question)

        final_state: dict[str, Any] = dict(input_state)
        prev_time = time.time()

        with self._console.status(
            "[bold blue]Starting...[/bold blue]",
            spinner="dots",
        ) as status:
            async for event in graph.astream(
                input_state,
                config=config or {},
                stream_mode="updates",
            ):
                if not isinstance(event, dict):
                    continue

                for node_name, state_update in event.items():
                    if node_name.startswith("__"):
                        continue

                    # Track approximate duration from inter-event timing
                    now = time.time()
                    self._start_times[node_name] = prev_time

                    output = (
                        state_update if isinstance(state_update, dict) else {}
                    )
                    self.on_node_end(node_name, output=output)
                    prev_time = now

                    # Merge into accumulated state
                    if isinstance(state_update, dict):
                        final_state.update(state_update)

                    label, _ = _get_node_display(node_name)
                    status.update(
                        f"[bold blue]{label} done — processing...[/bold blue]",
                    )

        # Display final answer if available
        answer = final_state.get("final_answer", "")
        if answer:
            self.display_final_answer(answer)

        return final_state


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_rich_output(
    *,
    console: Console | None = None,
    verbose: bool = False,
) -> RichStreamOutput:
    """Create a ``RichStreamOutput`` instance.

    Convenience factory for consistent construction.

    Args:
        console: Optional Rich Console instance.
        verbose: Enable verbose output.

    Returns:
        A configured ``RichStreamOutput``.
    """
    return RichStreamOutput(console=console, verbose=verbose)
