"""Format output node: produce final_answer with citations.

Placeholder implementation — copies draft_answer to final_answer.
Real formatting with citations is wired in task B4.
"""

from __future__ import annotations


def format_output_node(state) -> dict:
    """Format the draft answer into a final answer with source citations.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``final_answer``.
    """
    draft = state.draft_answer if hasattr(state, "draft_answer") else state.get("draft_answer", "")
    return {"final_answer": draft or "No answer generated."}
