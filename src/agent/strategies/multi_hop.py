"""Multi-hop strategy: Plan-and-Execute sub-graph.

Placeholder implementation — returns a stub draft_answer.
Real Plan-and-Execute sub-graph is wired in task C4.
"""

from __future__ import annotations


def multi_hop_strategy_node(state) -> dict:
    """Execute multi-hop strategy via plan / execute / replan / synthesize.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[multi_hop placeholder] No retrieval performed yet."}
