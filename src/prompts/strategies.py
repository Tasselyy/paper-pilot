"""Prompt templates for strategy nodes (simple, multi_hop, etc.).

Each strategy defines a system prompt and a user prompt template.
Templates use Python ``str.format()`` placeholders for runtime values.

Design reference: PAPER_PILOT_DESIGN.md §6.3 (simple), §6.4–6.6 (others).
"""

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
