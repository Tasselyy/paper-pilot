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


# ---------------------------------------------------------------------------
# Comparative strategy (ReWOO — parallel retrieval + structured comparison)
# ---------------------------------------------------------------------------

COMPARATIVE_EXTRACT_SYSTEM_PROMPT = """\
You are a research assistant that extracts entities and comparison \
dimensions from a comparative question about academic papers and \
machine-learning techniques.

Return:
- **entities**: the items being compared (e.g. "LoRA", "QLoRA", \
"Full Fine-tuning").
- **dimensions**: the aspects/criteria to compare (e.g. "memory usage", \
"accuracy", "training speed").\
"""

COMPARATIVE_EXTRACT_USER_TEMPLATE = """\
Question: {question}

Extract the entities being compared and the dimensions of comparison.\
"""

COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT = """\
You are PaperPilot, an AI research assistant specializing in academic papers \
and machine-learning literature. You are performing a **structured comparison** \
of multiple entities based on retrieved knowledge-base excerpts.

Guidelines:
- Organize the answer by comparing entities across the given dimensions.
- Use a structured format (e.g., a table or dimension-by-dimension breakdown).
- Cite sources by mentioning the paper/document name.
- If information for a certain entity–dimension pair is not available in the \
  retrieved contexts, state that clearly.
- Be objective and factual; avoid speculation.\
"""

COMPARATIVE_SYNTHESIS_USER_TEMPLATE = """\
Question: {question}

Entities being compared: {entities}

Comparison dimensions: {dimensions}

Retrieved contexts per entity:
{entity_contexts}

Please synthesize a structured comparison addressing the question.\
"""


# ---------------------------------------------------------------------------
# Exploratory strategy (ReAct — think / act / observe / synthesize)
# ---------------------------------------------------------------------------

EXPLORATORY_THINK_SYSTEM_PROMPT = """\
You are a research assistant performing an exploratory investigation using \
a ReAct (Reasoning + Acting) loop. At each step you must decide what to \
do next based on the question and observations so far.

You have access to a knowledge-base search tool. At each step you must \
either:
- **search**: formulate a focused search query to investigate a specific \
  aspect of the question.
- **synthesize**: you have gathered enough information to produce a final \
  answer.

Guidelines:
- Each search query should target a different aspect or follow-up angle.
- Avoid repeating searches you have already performed.
- Synthesize as soon as you have sufficient information; do not search \
  unnecessarily.\
"""

EXPLORATORY_THINK_USER_TEMPLATE = """\
Original question: {question}

Steps completed so far: {step_count}/{max_steps}

Previous queries executed:
{previous_queries}

Observations so far:
{observations}

Decide your next action: search for more information or synthesize the \
final answer.\
"""

EXPLORATORY_SYNTHESIS_SYSTEM_PROMPT = """\
You are PaperPilot, an AI research assistant specializing in academic papers \
and machine-learning literature. Synthesize a comprehensive answer to the \
user's exploratory question based on all observations gathered during your \
investigation.

Guidelines:
- Integrate findings from all search observations into a coherent answer.
- Cite sources by mentioning the paper/document name.
- If some aspects could not be fully explored, note the gaps.
- Provide a well-structured, thorough answer suitable for researchers.\
"""

EXPLORATORY_SYNTHESIS_USER_TEMPLATE = """\
Original question: {question}

Queries investigated:
{queries}

All observations:
{observations}

Synthesize a comprehensive answer that addresses the original question \
based on the investigation above.\
"""


def format_observations(
    queries: list[str],
    contexts_per_query: list[list],
) -> str:
    """Format per-query observations into a labelled text block.

    Args:
        queries: Ordered list of search queries executed.
        contexts_per_query: Parallel list of context lists (one per query).

    Returns:
        A formatted string with each query's results labelled.
    """
    if not queries:
        return "(No observations yet.)"

    parts: list[str] = []
    for idx, (query, contexts) in enumerate(
        zip(queries, contexts_per_query), 1,
    ):
        header = f"--- Step {idx}: {query} ---"
        if not contexts:
            parts.append(f"{header}\n(No results returned.)")
            continue
        ctx_parts: list[str] = []
        for cidx, ctx in enumerate(contexts, 1):
            if hasattr(ctx, "content") and not isinstance(ctx, dict):
                source = getattr(ctx, "source", "Unknown")
                content = ctx.content
            elif isinstance(ctx, dict):
                source = ctx.get("source", "Unknown")
                content = ctx.get("content", "")
            else:
                source = "Unknown"
                content = str(ctx)
            ctx_parts.append(f"  [{cidx}] Source: {source}\n  {content}")
        parts.append(f"{header}\n" + "\n\n".join(ctx_parts))
    return "\n\n".join(parts)


def format_entity_contexts(
    entity_contexts: dict[str, list],
) -> str:
    """Format per-entity retrieved contexts into a labelled text block.

    Args:
        entity_contexts: Mapping from entity name to a list of context
            objects (``RetrievedContext`` or dicts with ``content``/``source``).

    Returns:
        A formatted string with each entity's contexts numbered.
    """
    if not entity_contexts:
        return "(No entity contexts available.)"

    parts: list[str] = []
    for entity, contexts in entity_contexts.items():
        header = f"=== {entity} ==="
        if not contexts:
            parts.append(f"{header}\n(No relevant contexts retrieved.)")
            continue
        ctx_parts: list[str] = []
        for idx, ctx in enumerate(contexts, 1):
            if hasattr(ctx, "content") and not isinstance(ctx, dict):
                source = getattr(ctx, "source", "Unknown")
                content = ctx.content
            elif isinstance(ctx, dict):
                source = ctx.get("source", "Unknown")
                content = ctx.get("content", "")
            else:
                source = "Unknown"
                content = str(ctx)
            ctx_parts.append(f"  [{idx}] Source: {source}\n  {content}")
        parts.append(f"{header}\n" + "\n\n".join(ctx_parts))
    return "\n\n".join(parts)
