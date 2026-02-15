"""Slot Filling node: extract entities, dimensions, and reformulated query.

Placeholder implementation — passes intent through unchanged.
Real slot filling is wired in task C2.
"""

from __future__ import annotations


def slot_filling_node(state) -> dict:
    """Fill intent slots using Cloud LLM.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update (no-op placeholder).
    """
    # Placeholder — real implementation in task C2
    return {}
