"""Generate synthetic DPO preference pairs for Critic model training.

Each row is a triple (prompt, chosen, rejected) in TRL DPO format:
- prompt: instruction + question + draft answer (model input).
- chosen: preferred verdict JSON (high score).
- rejected: dispreferred verdict JSON (low score).

Output JSONL can be loaded by ``datasets.load_dataset("json", data_files=...)``
or by ``training/dpo_critic.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.models.inference import build_critic_evaluation_prompt

DEFAULT_PAIRS = 500
DEFAULT_SEED = 42

CRITIC_INSTRUCTION = (
    "Evaluate the answer quality for the given question and retrieved context.\n"
    "Score rubric:\n"
    "- score: overall quality in [0, 10]\n"
    "- completeness: coverage in [0, 1]\n"
    "- faithfulness: grounding in context in [0, 1]\n"
    "Return ONLY valid JSON with keys: score, completeness, faithfulness, feedback.\n"
    '{"score":7.8,"completeness":0.82,"faithfulness":0.86,'
    '"feedback":"Add a concrete example."}'
)


def _verdict_json(
    *,
    score: float,
    completeness: float,
    faithfulness: float,
    feedback: str,
) -> str:
    """Serialize a critic verdict to a compact JSON string."""
    obj = {
        "score": round(score, 1),
        "completeness": round(completeness, 2),
        "faithfulness": round(faithfulness, 2),
        "feedback": feedback,
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def generate_dpo_dataset(
    *,
    num_pairs: int = DEFAULT_PAIRS,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate a synthetic DPO dataset for Critic (prompt, chosen, rejected).

    Args:
        num_pairs: Number of preference pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with keys prompt, chosen, rejected.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    questions = _sample_questions(rng, size=num_pairs)
    draft_answers = _sample_draft_answers(rng, size=num_pairs)
    contexts_blocks = _sample_context_blocks(rng, size=num_pairs)
    strategies = _sample_strategies(rng, size=num_pairs)

    for i in range(num_pairs):
        question = questions[i]
        draft = draft_answers[i]
        contexts = contexts_blocks[i]
        strategy = strategies[i]
        prompt = _build_prompt(
            question=question,
            draft_answer=draft,
            contexts=contexts,
            strategy=strategy,
        )
        chosen = _build_chosen_verdict(rng=rng, idx=i)
        rejected = _build_rejected_verdict(rng=rng, idx=i)
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return rows


def _build_prompt(*, question: str, draft_answer: str, contexts: str, strategy: str) -> str:
    """Build the Critic input prompt (no verdict)."""
    return build_critic_evaluation_prompt(
        question=question,
        draft_answer=draft_answer,
        contexts=contexts,
        strategy=strategy,
    )


def _build_chosen_verdict(*, rng: random.Random, idx: int) -> str:
    """Build a preferred verdict (pass, score >= 7)."""
    score = 7.0 + rng.uniform(0.5, 2.9)
    score = min(10.0, round(score, 1))
    completeness = round(0.75 + rng.uniform(0, 0.24), 2)
    faithfulness = round(0.75 + rng.uniform(0, 0.24), 2)
    feedback = _pick(
        rng,
        (
            "Answer is complete and well-grounded.",
            "Covers the question and stays faithful to sources.",
            "Good coverage; minor polish could help.",
        ),
        idx,
    )
    return _verdict_json(
        score=score,
        completeness=completeness,
        faithfulness=faithfulness,
        feedback=feedback,
    )


def _build_rejected_verdict(*, rng: random.Random, idx: int) -> str:
    """Build a dispreferred verdict (fail, score < 7)."""
    score = 2.0 + rng.uniform(0, 4.5)
    score = min(6.9, round(score, 1))
    completeness = round(rng.uniform(0.2, 0.65), 2)
    faithfulness = round(rng.uniform(0.2, 0.65), 2)
    feedback = _pick(
        rng,
        (
            "Answer misses key aspects of the question.",
            "Some claims are not supported by the contexts.",
            "Add more detail from the retrieved sources.",
            "Clarify the main point and avoid unsupported claims.",
        ),
        idx + 1,
    )
    return _verdict_json(
        score=score,
        completeness=completeness,
        faithfulness=faithfulness,
        feedback=feedback,
    )


def _sample_questions(rng: random.Random, size: int) -> list[str]:
    """Sample synthetic questions."""
    templates = (
        "What is LoRA and when would you use it?",
        "Compare QLoRA and full fine-tuning for memory usage.",
        "How does RAG improve factual consistency in LLM outputs?",
        "Explain the role of attention in Transformers.",
        "What are the trade-offs of 4-bit quantization?",
        "How does Plan-and-Execute differ from ReAct?",
        "What is DPO and how does it relate to RLHF?",
        "When should you use retrieval augmentation?",
        "Describe FlashAttention in simple terms.",
        "What are the main challenges in long-context modeling?",
    )
    return [rng.choice(templates) for _ in range(size)]


def _sample_draft_answers(rng: random.Random, size: int) -> list[str]:
    """Sample short draft answers (good and bad mix)."""
    answers = (
        "LoRA is a parameter-efficient method that trains low-rank adapters.",
        "QLoRA uses quantized base models and typically reduces memory vs full fine-tuning.",
        "RAG retrieves relevant documents and conditions the model on them to reduce hallucination.",
        "Attention allows the model to weigh different input positions when producing output.",
        "4-bit quantization saves memory but can slightly reduce accuracy.",
        "Plan-and-Execute first plans steps then executes; ReAct interleaves thought and action.",
        "DPO is a direct preference optimization method that avoids explicit reward models.",
        "Use RAG when you need up-to-date or domain-specific knowledge.",
        "FlashAttention speeds up attention by reducing memory reads and improving GPU utilization.",
        "Long-context modeling faces cost, attention complexity, and retrieval trade-offs.",
    )
    return [rng.choice(answers) for _ in range(size)]


def _sample_context_blocks(rng: random.Random, size: int) -> list[str]:
    """Sample synthetic context blocks formatted like runtime retrieval text."""
    contexts = (
        "[1] Source: LoRA Paper (2021)\nLoRA injects low-rank adapters into attention layers.\n\n"
        "[2] Source: QLoRA (2023)\nQLoRA combines 4-bit quantization with LoRA fine-tuning.",
        "[1] Source: RAG Survey\nRAG improves factual grounding by conditioning on retrieved passages.\n\n"
        "[2] Source: Faithfulness Benchmarks\nGrounded generation reduces unsupported claims.",
        "[1] Source: FlashAttention\nFlashAttention optimizes memory reads during attention.\n\n"
        "[2] Source: Transformer Primer\nAttention complexity scales with sequence length.",
        "(no retrieved contexts)",
    )
    return [rng.choice(contexts) for _ in range(size)]


def _sample_strategies(rng: random.Random, size: int) -> list[str]:
    """Sample agent strategy labels for critic prompts."""
    strategies = ("simple", "comparative", "multi_hop", "exploratory")
    return [rng.choice(strategies) for _ in range(size)]


def _pick(rng: random.Random, choices: tuple[str, ...], idx: int) -> str:
    """Deterministic choice for variety."""
    return choices[(idx + rng.randint(0, 10)) % len(choices)]


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows as UTF-8 JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs for Critic training."
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=DEFAULT_PAIRS,
        help="Number of (prompt, chosen, rejected) pairs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/dpo_train.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save DPO training data."""
    args = _parse_args()
    rows = generate_dpo_dataset(num_pairs=args.num_pairs, seed=args.seed)
    write_jsonl(rows=rows, output_path=args.output)
    print(f"Generated {len(rows)} DPO pairs at: {args.output}")


if __name__ == "__main__":
    main()
