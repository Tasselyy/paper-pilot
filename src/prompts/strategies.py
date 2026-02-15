"""Prompt templates for strategy nodes (simple, multi_hop, etc.).

Each strategy defines a system prompt and a user prompt template.
Templates use Python ``str.format()`` placeholders for runtime values.

Design reference: PAPER_PILOT_DESIGN.md §6.3 (simple), §6.4–6.6 (others).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Simple strategy
# ---------------------------------------------------------------------------

SIMPLE_SYSTEM_PROMPT = """\
You are PaperPilot, an AI research assistant specializing in academic papers \
and machine-learning literature.  Your job is to answer a user's question \
based **only** on the retrieved knowledge-base excerpts provided below.

Guidelines:
- Synthesize a clear, concise, and accurate answer from the provided contexts.
- Cite sources by mentioning the paper/document name where the information \
  comes from (e.g., "According to [LoRA: Low-Rank Adaptation]...").
- If the retrieved contexts do not contain enough information to answer the \
  question, state that clearly rather than fabricating an answer.
- Keep the answer focused; avoid unnecessary tangents.
- Use technical terminology appropriately for an academic audience.
"""

SIMPLE_USER_TEMPLATE = """\
Question: {question}

Retrieved contexts:
{contexts}

Please provide a well-structured answer based on the above contexts.\
"""


def format_simple_contexts(
    contexts: list[dict[str, str]],
) -> str:
    """Format retrieved contexts into a numbered text block for the prompt.

    Args:
        contexts: List of dicts with ``content`` and ``source`` keys.

    Returns:
        A formatted string with each context numbered and labelled.
    """
    if not contexts:
        return "(No relevant contexts were retrieved.)"

    parts: list[str] = []
    for idx, ctx in enumerate(contexts, 1):
        source = ctx.get("source", "Unknown")
        content = ctx.get("content", "")
        parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Shared context formatter (works with RetrievedContext objects or dicts)
# ---------------------------------------------------------------------------


def format_retrieved_contexts(
    contexts: list[Any],
) -> str:
    """Format context objects or dicts into a numbered text block.

    Accepts both ``RetrievedContext`` Pydantic model instances (with
    ``.content`` / ``.source`` attributes) and plain dicts.

    Args:
        contexts: List of context objects or dicts with ``content`` and
            ``source`` keys/attributes.

    Returns:
        A formatted string with each context numbered and labelled.
    """
    if not contexts:
        return "(No relevant contexts were retrieved.)"

    parts: list[str] = []
    for idx, ctx in enumerate(contexts, 1):
        if hasattr(ctx, "content") and not isinstance(ctx, dict):
            source = getattr(ctx, "source", "Unknown")
            content = ctx.content
        else:
            source = ctx.get("source", "Unknown") if isinstance(ctx, dict) else "Unknown"
            content = ctx.get("content", "") if isinstance(ctx, dict) else ""
        parts.append(f"[{idx}] Source: {source}\n{content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Multi-hop strategy (Plan-and-Execute)
# ---------------------------------------------------------------------------

MULTI_HOP_DECOMPOSE_SYSTEM_PROMPT = """\
You are a research assistant that decomposes complex multi-hop questions \
into a sequence of simpler sub-questions. Each sub-question should be \
answerable by a single knowledge-base search. Order them logically so \
that earlier answers can inform later questions. Return 2–4 sub-questions.\
"""

MULTI_HOP_DECOMPOSE_USER_TEMPLATE = """\
Original question: {question}

Constraints: {constraints}

Decompose this into 2–4 focused sub-questions that, when answered \
sequentially, fully address the original question.\
"""

MULTI_HOP_REPLAN_SYSTEM_PROMPT = """\
You are a research assistant revising a multi-hop investigation plan \
based on results obtained so far. Review what has been answered and \
what remains, then produce an updated complete plan.

Rules:
- Keep already-answered sub-questions as the first items (unchanged).
- You may add, remove, or modify only the remaining sub-questions.
- The total plan should have 2–5 sub-questions.\
"""

MULTI_HOP_REPLAN_USER_TEMPLATE = """\
Original question: {question}

Original plan:
{original_plan}

Queries executed so far:
{executed_queries}

Results obtained so far:
{results_so_far}

Constraints: {constraints}

Produce a revised complete plan (keep executed sub-questions as the first \
items, then update the remaining ones).\
"""

MULTI_HOP_DECIDE_SYSTEM_PROMPT = """\
You are deciding the next action in a multi-hop research investigation. \
Based on the plan progress and latest results, choose one action:

- "next_step": continue executing the next sub-question as planned.
- "replan": results suggest the plan needs revision (add/remove/modify \
sub-questions).
- "synthesize": enough information has been gathered to answer the original \
question.\
"""

MULTI_HOP_DECIDE_USER_TEMPLATE = """\
Original question: {question}

Full plan:
{plan}

Progress: {executed_count}/{total_steps} steps executed

Latest retrieved results:
{latest_results}

Remaining sub-questions:
{remaining}

Choose the next action and explain briefly.\
"""

MULTI_HOP_SYNTHESIS_SYSTEM_PROMPT = """\
You are PaperPilot, an AI research assistant specializing in academic papers \
and machine-learning literature. Synthesize a comprehensive answer to the \
original multi-hop question based on all sub-question results.

Guidelines:
- Integrate findings from all sub-questions into a coherent narrative.
- Cite sources by mentioning the paper/document name.
- If some sub-questions could not be fully answered, note the gaps.
- Structure the answer logically, following the reasoning chain.\
"""

MULTI_HOP_SYNTHESIS_USER_TEMPLATE = """\
Original question: {question}

Sub-questions investigated:
{sub_questions}

All retrieved contexts:
{contexts}

Synthesize a comprehensive answer that addresses the original question \
by connecting the findings from all sub-questions.\
"""
