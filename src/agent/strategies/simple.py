"""Simple strategy: single retrieval + synthesis.

Placeholder implementation — returns a stub draft_answer.
Real implementation with RAG retrieval is wired in task B3.
"""

from __future__ import annotations


def simple_strategy_node(state) -> dict:
    """Execute simple strategy: retrieve once, synthesize answer.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer`` and ``retrieved_contexts``.
    """
    return {"draft_answer": "[simple placeholder] No retrieval performed yet."}
