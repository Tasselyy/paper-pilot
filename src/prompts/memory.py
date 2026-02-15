"""Prompt templates for memory-related operations.

Contains the system and user prompts used by ``save_memory_node`` to
extract notable facts from a completed Q&A turn via Cloud LLM.

Design reference: PAPER_PILOT_DESIGN.md §8.2.
"""

EXTRACT_FACTS_SYSTEM_PROMPT = """\
You are a fact extraction assistant for an academic paper Q&A system.

Your job is to extract **key factual claims** from a completed question-answer \
pair that would be useful to remember for future conversations.

Guidelines:
- Extract concrete, self-contained factual statements.
- Focus on information that would help answer similar or related questions later.
- Include paper names, technique names, quantitative results, and relationships \
between concepts when present.
- Each fact should be a single, clear sentence.
- Do NOT extract trivial or overly generic statements.
- If the answer is low-quality, vague, or empty, return an empty list.
- Return between 0 and 5 facts (only extract what is genuinely worth remembering).
"""

EXTRACT_FACTS_USER_TEMPLATE = """\
## Question
{question}

## Answer
{answer}

Extract the key facts worth remembering from this Q&A pair. \
Return only the list of facts (0–5 items). If nothing is worth remembering, \
return an empty list.
"""
