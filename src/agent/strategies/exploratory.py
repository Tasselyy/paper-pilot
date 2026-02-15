"""Exploratory strategy: ReAct sub-graph (think/act/observe/synthesize).

Placeholder implementation — returns a stub draft_answer.
Real ReAct sub-graph is wired in task C6.
"""

from __future__ import annotations


def exploratory_strategy_node(state) -> dict:
    """Execute exploratory strategy via ReAct loop.

    Args:
        state: The current AgentState.

    Returns:
        Partial state update with ``draft_answer``.
    """
    return {"draft_answer": "[exploratory placeholder] No retrieval performed yet."}
