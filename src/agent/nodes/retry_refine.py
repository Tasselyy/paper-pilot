"""Retry-refine node: rewrite question or plan based on critic feedback.

Placeholder implementation — increments retry_count.
Real refinement logic is wired in task D3.
"""

from __future__ import annotations


def retry_refine_node(state) -> dict:
    """Refine the question or plan based on critic feedback.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with incremented ``retry_count``.
    """
    count = state.retry_count if hasattr(state, "retry_count") else state.get("retry_count", 0)
    return {"retry_count": count + 1}
