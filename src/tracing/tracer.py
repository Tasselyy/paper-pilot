"""AgentTrace structure and JSONL trace output.

Provides the ``AgentTrace`` Pydantic model that captures a complete decision
trace for a single agent run, plus ``NodeTraceEntry`` for per-node execution
records and ``AgentTracer`` — a stateful builder that tracks node-level
timing during graph execution.

The module also exposes ``flush_trace_to_jsonl`` for appending a single
``AgentTrace`` record to a JSONL file.  Each line is a self-contained JSON
object containing intent, strategy, critic verdict, final answer, and the
full per-node trace — suitable for offline analysis and replay.

Design reference: PAPER_PILOT_DESIGN.md §10.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.agent.state import (
    AgentState,
    CriticVerdict,
    Intent,
    ReasoningStep,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-node trace entry
# ---------------------------------------------------------------------------


class NodeTraceEntry(BaseModel):
    """Execution record for a single graph node.

    Attributes:
        node_name: Name of the graph node (e.g. ``"router"``, ``"critic"``).
        duration_ms: Wall-clock execution time of the node in milliseconds.
        output: Summary of the node's output (key fields only).
        timestamp: Epoch timestamp when the node started executing.
    """

    node_name: str = Field(description="Name of the graph node")
    duration_ms: float = Field(
        ge=0,
        description="Node execution duration in milliseconds",
    )
    output: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of the node's output",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Epoch timestamp when node execution started",
    )


# ---------------------------------------------------------------------------
# Full agent trace
# ---------------------------------------------------------------------------


class AgentTrace(BaseModel):
    """Complete agent decision trace for a single question.

    Captures everything needed to understand and replay an agent's reasoning:
    intent classification, strategy selection, retrieval queries, per-node
    execution timing, critic evaluation, and the final answer.

    Attributes:
        trace_id: Unique identifier for this trace (UUID4).
        question: The original user question.
        timestamp: Epoch timestamp when the agent run started.
        duration_ms: Total wall-clock duration of the agent run in ms.
        intent: Fully populated intent (after slot filling).
        strategy_executed: Which strategy was executed (e.g. ``"simple"``).
        reasoning_steps: Ordered list of reasoning steps from the agent state.
        retrieval_queries: All queries sent to RAG during the run.
        contexts_retrieved: Total number of retrieved context chunks.
        tokens_used: Token usage breakdown (e.g. ``{"input": N, "output": M}``).
        critic_verdict: Final critic evaluation result.
        retry_count: Number of critic retries that occurred.
        final_answer: The formatted final answer.
        node_traces: Per-node execution records with timing and output.
        router_latency_ms: Router node latency in ms.
        retrieval_latency_ms: Total retrieval latency in ms.
        llm_latency_ms: Total LLM call latency in ms.
        critic_latency_ms: Critic node latency in ms.
    """

    trace_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique trace identifier (UUID4 hex)",
    )
    question: str = Field(default="", description="Original user question")
    timestamp: float = Field(
        default_factory=time.time,
        description="Epoch timestamp of the agent run start",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total agent run duration in milliseconds",
    )

    # Intent & strategy
    intent: Intent | None = Field(
        default=None,
        description="Fully populated intent (from slot filling)",
    )
    strategy_executed: str = Field(
        default="",
        description="Strategy node that was executed",
    )

    # Reasoning
    reasoning_steps: list[ReasoningStep] = Field(
        default_factory=list,
        description="Ordered reasoning steps from the agent state",
    )

    # Retrieval
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="All queries sent to RAG",
    )
    contexts_retrieved: int = Field(
        default=0,
        ge=0,
        description="Total number of retrieved context chunks",
    )
    tokens_used: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage breakdown",
    )

    # Evaluation
    critic_verdict: CriticVerdict | None = Field(
        default=None,
        description="Final critic evaluation result",
    )
    retry_count: int = Field(default=0, ge=0, description="Number of critic retries")

    # Output
    final_answer: str = Field(default="", description="Formatted final answer")

    # Per-node execution records
    node_traces: list[NodeTraceEntry] = Field(
        default_factory=list,
        description="Per-node execution records with timing",
    )

    # Latency breakdown
    router_latency_ms: float = Field(default=0.0, ge=0, description="Router latency")
    retrieval_latency_ms: float = Field(
        default=0.0, ge=0, description="Total retrieval latency"
    )
    llm_latency_ms: float = Field(default=0.0, ge=0, description="Total LLM latency")
    critic_latency_ms: float = Field(default=0.0, ge=0, description="Critic latency")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_jsonl_record(self) -> dict[str, Any]:
        """Serialize the trace to a JSON-compatible dictionary.

        Uses Pydantic's ``model_dump`` with ``mode="json"`` so that all
        nested models (Intent, CriticVerdict, ReasoningStep, etc.) are
        recursively converted to plain dicts/lists suitable for
        ``json.dumps``.

        Returns:
            A plain dict that can be written as a single JSONL line.
        """
        return self.model_dump(mode="json")

    def flush_to_jsonl(self, path: str | Path) -> Path:
        """Append this trace as a single JSON line to *path*.

        Creates parent directories if they do not exist.  The file is
        opened in **append** mode so that multiple traces accumulate in
        a single JSONL file.

        Args:
            path: Destination JSONL file path.  Can be relative (resolved
                against cwd) or absolute.

        Returns:
            The resolved ``Path`` that was written to.
        """
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        record = self.to_jsonl_record()
        line = json.dumps(record, ensure_ascii=False, default=str)

        with resolved.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        logger.debug("Trace %s flushed to %s", self.trace_id, resolved)
        return resolved

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_state(
        cls,
        state: AgentState,
        *,
        trace_id: str | None = None,
        duration_ms: float = 0.0,
        timestamp: float | None = None,
        node_traces: list[NodeTraceEntry] | None = None,
        tokens_used: dict[str, int] | None = None,
    ) -> AgentTrace:
        """Build an ``AgentTrace`` from a completed ``AgentState``.

        Extracts intent, strategy, reasoning steps, retrieval info, critic
        verdict, and final answer from the state object.

        Args:
            state: The completed agent state after graph execution.
            trace_id: Optional trace ID (auto-generated if omitted).
            duration_ms: Total run duration in milliseconds.
            timestamp: Epoch timestamp of the run start.
            node_traces: Pre-collected per-node trace entries.
            tokens_used: Token usage dict.

        Returns:
            A populated ``AgentTrace`` instance.
        """
        strategy = ""
        if state.intent is not None:
            strategy = state.intent.to_strategy()

        # Compute latency breakdown from node_traces if available
        router_latency = 0.0
        retrieval_latency = 0.0
        llm_latency = 0.0
        critic_latency = 0.0
        if node_traces:
            for entry in node_traces:
                name = entry.node_name
                if name == "route":
                    router_latency += entry.duration_ms
                elif name in ("simple", "multi_hop", "comparative", "exploratory"):
                    retrieval_latency += entry.duration_ms
                elif name == "slot_fill":
                    llm_latency += entry.duration_ms
                elif name == "critic":
                    critic_latency += entry.duration_ms

        return cls(
            trace_id=trace_id or uuid.uuid4().hex,
            question=state.question,
            timestamp=timestamp or time.time(),
            duration_ms=duration_ms,
            intent=state.intent,
            strategy_executed=strategy,
            reasoning_steps=list(state.reasoning_trace),
            retrieval_queries=list(state.retrieval_queries),
            contexts_retrieved=len(state.retrieved_contexts),
            tokens_used=tokens_used or {},
            critic_verdict=state.critic_verdict,
            retry_count=state.retry_count,
            final_answer=state.final_answer,
            node_traces=node_traces or [],
            router_latency_ms=router_latency,
            retrieval_latency_ms=retrieval_latency,
            llm_latency_ms=llm_latency,
            critic_latency_ms=critic_latency,
        )


# ---------------------------------------------------------------------------
# AgentTracer — stateful builder for node-level timing
# ---------------------------------------------------------------------------

# Mapping from node names to latency categories
_LATENCY_CATEGORY: dict[str, str] = {
    "route": "router",
    "simple": "retrieval",
    "multi_hop": "retrieval",
    "comparative": "retrieval",
    "exploratory": "retrieval",
    "slot_fill": "llm",
    "critic": "critic",
}


class AgentTracer:
    """Stateful builder that tracks per-node timing during graph execution.

    Usage::

        tracer = AgentTracer(question="What is LoRA?")

        tracer.start_node("route")
        # ... node executes ...
        tracer.end_node("route", output={"intent_type": "factual"})

        trace = tracer.build_trace(state)

    Attributes:
        question: The user question being traced.
        node_traces: Collected node trace entries.
    """

    def __init__(self, question: str = "") -> None:
        self._question = question
        self._start_time = time.time()
        self._node_traces: list[NodeTraceEntry] = []
        self._active_nodes: dict[str, float] = {}
        self._tokens_used: dict[str, int] = {"input": 0, "output": 0}

    @property
    def question(self) -> str:
        """The user question being traced."""
        return self._question

    @property
    def node_traces(self) -> list[NodeTraceEntry]:
        """Collected node trace entries (read-only copy)."""
        return list(self._node_traces)

    def start_node(self, node_name: str) -> None:
        """Record the start of a node's execution.

        Args:
            node_name: Name of the graph node starting execution.
        """
        self._active_nodes[node_name] = time.time()

    def end_node(
        self,
        node_name: str,
        *,
        output: dict[str, Any] | None = None,
    ) -> NodeTraceEntry:
        """Record the end of a node's execution and create a trace entry.

        Args:
            node_name: Name of the graph node that finished.
            output: Summary of the node's output (key fields).

        Returns:
            The created ``NodeTraceEntry``.

        Raises:
            ValueError: If ``start_node`` was not called for *node_name*.
        """
        start_ts = self._active_nodes.pop(node_name, None)
        if start_ts is None:
            raise ValueError(
                f"end_node('{node_name}') called without a matching start_node"
            )

        duration = (time.time() - start_ts) * 1000  # ms

        entry = NodeTraceEntry(
            node_name=node_name,
            duration_ms=duration,
            output=output or {},
            timestamp=start_ts,
        )
        self._node_traces.append(entry)
        return entry

    def add_token_usage(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Accumulate token usage counters.

        Args:
            input_tokens: Number of input (prompt) tokens consumed.
            output_tokens: Number of output (completion) tokens consumed.
        """
        self._tokens_used["input"] += input_tokens
        self._tokens_used["output"] += output_tokens

    def build_trace(
        self,
        state: AgentState,
        *,
        trace_id: str | None = None,
    ) -> AgentTrace:
        """Finalize and build the ``AgentTrace`` from the completed state.

        Computes total duration from the tracer's start time to now, and
        delegates to ``AgentTrace.from_state`` with collected node traces
        and token usage.

        Args:
            state: The completed agent state after graph execution.
            trace_id: Optional trace ID override.

        Returns:
            A fully populated ``AgentTrace``.
        """
        total_duration = (time.time() - self._start_time) * 1000  # ms

        return AgentTrace.from_state(
            state,
            trace_id=trace_id,
            duration_ms=total_duration,
            timestamp=self._start_time,
            node_traces=list(self._node_traces),
            tokens_used=dict(self._tokens_used),
        )

    def flush_to_jsonl(
        self,
        state: AgentState,
        path: str | Path,
        *,
        trace_id: str | None = None,
    ) -> AgentTrace:
        """Build the trace from *state* and append it to a JSONL file.

        Convenience wrapper that combines ``build_trace`` and
        ``AgentTrace.flush_to_jsonl`` in a single call.

        Args:
            state: The completed agent state.
            path: Destination JSONL file path.
            trace_id: Optional trace ID override.

        Returns:
            The finalized ``AgentTrace`` that was written.
        """
        trace = self.build_trace(state, trace_id=trace_id)
        trace.flush_to_jsonl(path)
        return trace


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def flush_trace_to_jsonl(trace: AgentTrace, path: str | Path) -> Path:
    """Append *trace* as a single JSON line to *path*.

    This is a thin module-level convenience wrapper around
    ``AgentTrace.flush_to_jsonl``.

    Args:
        trace: The ``AgentTrace`` to persist.
        path: Destination JSONL file path.

    Returns:
        The resolved ``Path`` that was written to.
    """
    return trace.flush_to_jsonl(path)
