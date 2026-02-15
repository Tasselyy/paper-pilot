"""Comparative strategy: parallel retrieval by entities/dimensions (ReWOO).

Placeholder implementation — returns a stub draft_answer.
Real parallel retrieval is wired in task C5.
"""

from __future__ import annotations


def comparative_strategy_node(state) -> dict:
    """Execute comparative strategy with parallel retrieval.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[comparative placeholder] No retrieval performed yet."}
