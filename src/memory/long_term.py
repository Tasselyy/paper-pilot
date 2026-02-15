"""Long-term fact memory: recall / memorize with JSONL persistence.

Provides ``LongTermMemory`` — a keyword-based fact store that persists
to a JSONL file.  Each fact is stored with keywords extracted from its
content, the source question, and a timestamp.

At the start of each agent turn, ``recall`` returns the most relevant
historical facts.  At the end, ``memorize`` extracts and persists new
facts from the completed Q&A pair.

Design reference: PAPER_PILOT_DESIGN.md §8.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Fact(BaseModel):
    """A single fact stored in long-term memory.

    Attributes:
        content: The fact text.
        keywords: Keywords extracted from the fact for retrieval.
        source_question: The user question that produced this fact.
        timestamp: Unix epoch when the fact was stored.
    """

    content: str
    keywords: list[str] = Field(default_factory=list)
    source_question: str = Field(default="")
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Keyword extraction (simple, dependency-free)
# ---------------------------------------------------------------------------

# Common English stop words to exclude from keyword extraction.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out",
        "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up", "its",
        "it", "this", "that", "these", "those", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them",
        "their", "what", "which", "who", "whom",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_\-]+")


def extract_keywords(text: str) -> list[str]:
    """Extract keywords from *text* for simple keyword-overlap retrieval.

    Tokenizes on word boundaries, lowercases, removes stop words and
    very short tokens (< 2 chars), and returns unique keywords in
    occurrence order.

    Args:
        text: Input text (question, fact, etc.).

    Returns:
        Deduplicated list of keyword strings.
    """
    tokens = _TOKEN_PATTERN.findall(text.lower())
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in tokens:
        if len(tok) < 2 or tok in _STOP_WORDS or tok in seen:
            continue
        seen.add(tok)
        keywords.append(tok)
    return keywords


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Keyword-based long-term fact memory with JSONL persistence.

    Args:
        memory_file: Path to the JSONL file that stores facts.
            Created on first write if it does not exist.
        top_k: Default number of facts to return from ``recall``.
    """

    def __init__(
        self,
        memory_file: str = "data/long_term_memory.jsonl",
        top_k: int = 5,
    ) -> None:
        self.memory_file = Path(memory_file)
        self.top_k = top_k
        self.facts: list[Fact] = self._load_facts()
        logger.info(
            "LongTermMemory initialized: file=%s, existing_facts=%d",
            self.memory_file,
            len(self.facts),
        )

    # -- Persistence --------------------------------------------------------

    def _load_facts(self) -> list[Fact]:
        """Load existing facts from the JSONL file.

        Returns:
            List of ``Fact`` instances.  Returns empty list if the file
            does not exist or is empty.
        """
        if not self.memory_file.exists():
            return []

        facts: list[Fact] = []
        try:
            with self.memory_file.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        facts.append(Fact(**data))
                    except (json.JSONDecodeError, Exception) as exc:
                        logger.warning(
                            "Skipping malformed fact on line %d: %s",
                            line_no,
                            exc,
                        )
        except OSError as exc:
            logger.error("Failed to read memory file %s: %s", self.memory_file, exc)

        return facts

    def _append_to_file(self, fact: Fact) -> None:
        """Append a single fact as a JSON line to the memory file.

        Creates parent directories if they do not exist.
        """
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            with self.memory_file.open("a", encoding="utf-8") as fh:
                fh.write(fact.model_dump_json() + "\n")
        except OSError as exc:
            logger.error("Failed to write fact to %s: %s", self.memory_file, exc)

    # -- Public API ---------------------------------------------------------

    async def recall(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[str]:
        """Retrieve the most relevant facts for *question*.

        Uses simple keyword overlap scoring: each fact is scored by the
        number of overlapping keywords with *question*.  Facts with zero
        overlap are excluded.

        Args:
            question: The user's current question.
            top_k: Maximum number of facts to return.  Defaults to the
                instance-level ``top_k``.

        Returns:
            List of fact content strings, ordered by relevance (highest
            overlap first).
        """
        if not self.facts:
            return []

        k = top_k if top_k is not None else self.top_k
        question_keywords = set(extract_keywords(question))
        if not question_keywords:
            return []

        scored: list[tuple[int, float, str]] = []
        for fact in self.facts:
            fact_keywords = set(fact.keywords)
            overlap = len(question_keywords & fact_keywords)
            if overlap > 0:
                # Secondary sort by recency (higher timestamp = more recent)
                scored.append((overlap, fact.timestamp, fact.content))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [content for _, _, content in scored[:k]]

    async def memorize(
        self,
        question: str,
        answer: str,
        facts_to_store: list[str],
    ) -> list[Fact]:
        """Store new facts extracted from the current Q&A turn.

        The caller is responsible for extracting fact strings (typically
        done by the ``save_memory_node`` via LLM or heuristic).  This
        method creates ``Fact`` objects, appends them to the in-memory
        list and JSONL file.

        Args:
            question: The user's original question.
            answer: The agent's draft answer.
            facts_to_store: List of fact content strings to persist.

        Returns:
            List of newly created ``Fact`` instances.
        """
        if not facts_to_store:
            logger.debug("memorize: no facts to store")
            return []

        new_facts: list[Fact] = []
        for fact_text in facts_to_store:
            fact_text = fact_text.strip()
            if not fact_text:
                continue
            fact = Fact(
                content=fact_text,
                keywords=extract_keywords(fact_text),
                source_question=question,
                timestamp=time.time(),
            )
            self.facts.append(fact)
            self._append_to_file(fact)
            new_facts.append(fact)

        logger.info(
            "memorize: stored %d new facts (total=%d)",
            len(new_facts),
            len(self.facts),
        )
        return new_facts

    def clear(self) -> None:
        """Remove all facts from memory and delete the JSONL file.

        Intended for testing and administrative purposes.
        """
        self.facts.clear()
        if self.memory_file.exists():
            self.memory_file.unlink()
            logger.info("Cleared memory file: %s", self.memory_file)

    @property
    def fact_count(self) -> int:
        """Number of facts currently in memory."""
        return len(self.facts)

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Export all facts as a list of dicts (for debugging / inspection).

        Returns:
            List of serialized ``Fact`` dictionaries.
        """
        return [f.model_dump() for f in self.facts]
