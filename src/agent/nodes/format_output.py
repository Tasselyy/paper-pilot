"""Format output node: produce final_answer with citations.

Reads ``draft_answer`` and ``retrieved_contexts`` from state, appends a
**Sources** section listing each unique source referenced during retrieval,
and writes the result to ``final_answer``.

Design reference: PAPER_PILOT_DESIGN.md §5 (format_output), DEV_SPEC B4.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import ReasoningStep, RetrievedContext

logger = logging.getLogger(__name__)


def _build_sources_section(contexts: list[RetrievedContext]) -> str:
    """Build a de-duplicated **Sources** section from retrieved contexts.

    Args:
        contexts: Retrieved contexts with ``source`` and ``doc_id`` fields.

    Returns:
        A formatted string listing unique sources, or an empty string if
        no contexts are available.
    """
    if not contexts:
        return ""

    seen: set[str] = set()
    lines: list[str] = []
    for ctx in contexts:
        # Prefer source path/title as the dedupe key. Multiple chunks from the
        # same document can have different doc_id values, which would otherwise
        # produce repeated entries in the final Sources section.
        source = (ctx.source or "").strip()
        key = source.lower() if source else (ctx.doc_id or "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {ctx.source}")

    if not lines:
        return ""

    return "\n\n**Sources**:\n" + "\n".join(lines)


def format_output_node(state: Any) -> dict[str, Any]:
    """Format the draft answer into a final answer with source citations.

    Combines ``state.draft_answer`` with a de-duplicated list of sources
    from ``state.retrieved_contexts`` to produce ``final_answer``.

    Args:
        state: The current AgentState (Pydantic model or dict-like).

    Returns:
        Partial state update with ``final_answer`` and a reasoning trace
        step recording the formatting action.
    """
    draft: str = (
        state.draft_answer
        if hasattr(state, "draft_answer")
        else state.get("draft_answer", "")
    )
    contexts: list[RetrievedContext] = (
        state.retrieved_contexts
        if hasattr(state, "retrieved_contexts")
        else state.get("retrieved_contexts", [])
    )

    unique_sources = {
        ((ctx.source or "").strip().lower() or (ctx.doc_id or "").strip().lower())
        for ctx in contexts
    }
    unique_sources.discard("")

    if not draft:
        logger.warning("format_output_node: draft_answer is empty")
        final = "No answer generated."
    else:
        sources_section = _build_sources_section(contexts)
        final = draft + sources_section

    trace_step = ReasoningStep(
        step_type="action",
        content=(
            f"Formatted final answer ({len(final)} chars) "
            f"with {len(unique_sources)} unique source(s)"
        ),
        metadata={
            "num_sources": len(contexts),
            "num_unique_sources": len(unique_sources),
        },
    )

    logger.info(
        "format_output_node: final_answer=%d chars, sources=%d (unique=%d)",
        len(final),
        len(contexts),
        len(unique_sources),
    )

    return {
        "final_answer": final,
        "reasoning_trace": [trace_step],
    }
