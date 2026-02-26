"""Generate synthetic Critic SFT training data in Alpaca JSONL format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.models.inference import build_critic_evaluation_prompt

DEFAULT_NUM_SAMPLES = 800
DEFAULT_SEED = 42

CRITIC_INSTRUCTION = (
    "Evaluate the answer quality for the given question and retrieved context.\n"
    "Score rubric:\n"
    "- score: overall quality in [0, 10]\n"
    "- completeness: coverage in [0, 1]\n"
    "- faithfulness: grounding in context in [0, 1]\n"
    "Return ONLY valid JSON with keys: score, completeness, faithfulness, feedback."
)


def generate_critic_sft_dataset(
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Generate synthetic Critic SFT rows in Alpaca format."""
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        quality_band = idx % 4
        question = _sample_question(rng)
        draft_answer = _sample_answer(rng, quality_band=quality_band)
        contexts = _sample_context_block(rng)
        strategy = _sample_strategy(rng)

        input_text = build_critic_evaluation_prompt(
            question=question,
            draft_answer=draft_answer,
            contexts=contexts,
            strategy=strategy,
        )
        output_text = _build_verdict_json(rng, quality_band=quality_band)
        rows.append(
            {
                "instruction": CRITIC_INSTRUCTION,
                "input": input_text,
                "output": output_text,
            }
        )

    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write rows as UTF-8 JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_verdict_json(rng: random.Random, *, quality_band: int) -> str:
    if quality_band == 0:
        score = rng.uniform(1.5, 4.2)
        completeness = rng.uniform(0.2, 0.5)
        faithfulness = rng.uniform(0.2, 0.5)
        feedback = rng.choice(
            (
                "Answer misses key points and is weakly grounded in provided contexts.",
                "Several claims are unsupported; add concrete details from sources.",
            )
        )
    elif quality_band == 1:
        score = rng.uniform(4.3, 6.8)
        completeness = rng.uniform(0.45, 0.72)
        faithfulness = rng.uniform(0.45, 0.72)
        feedback = rng.choice(
            (
                "Partially correct but lacks complete coverage of the question.",
                "Reasonable answer with gaps; improve evidence use and specificity.",
            )
        )
    elif quality_band == 2:
        score = rng.uniform(7.0, 8.6)
        completeness = rng.uniform(0.72, 0.9)
        faithfulness = rng.uniform(0.72, 0.9)
        feedback = rng.choice(
            (
                "Good answer overall with minor room for clarification.",
                "Mostly complete and faithful; a brief concrete example would help.",
            )
        )
    else:
        score = rng.uniform(8.7, 9.8)
        completeness = rng.uniform(0.9, 0.99)
        faithfulness = rng.uniform(0.9, 0.99)
        feedback = rng.choice(
            (
                "Excellent answer: comprehensive, accurate, and well-grounded.",
                "High-quality response with strong coverage and source faithfulness.",
            )
        )

    payload = {
        "score": round(score, 1),
        "completeness": round(completeness, 2),
        "faithfulness": round(faithfulness, 2),
        "feedback": feedback,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _sample_question(rng: random.Random) -> str:
    pool = (
        "What is LoRA and when is it preferable to full fine-tuning?",
        "Compare LoRA and QLoRA in terms of memory usage and quality.",
        "How does retrieval augmentation improve factual consistency?",
        "Explain why attention is important in Transformers.",
        "What are the trade-offs of 4-bit quantization for inference?",
        "How does DPO differ from SFT in training objectives?",
    )
    return rng.choice(pool)


def _sample_answer(rng: random.Random, *, quality_band: int) -> str:
    bad = (
        "LoRA is just a dataset and usually slower than all methods.",
        "RAG mainly increases hallucinations because of extra context.",
        "Quantization always improves accuracy and has no downsides.",
    )
    mid = (
        "LoRA adds trainable low-rank adapters and reduces fine-tuning memory.",
        "RAG retrieves related documents and can reduce unsupported claims.",
        "4-bit quantization lowers memory cost but may slightly affect quality.",
    )
    good = (
        "LoRA inserts low-rank trainable matrices while freezing base weights, reducing trainable parameters and memory.",
        "QLoRA combines quantized base weights with LoRA adapters, typically lowering VRAM while preserving quality for many tasks.",
        "RAG conditions generation on retrieved context, which improves factual grounding when retrieval quality is high.",
    )
    if quality_band == 0:
        return rng.choice(bad)
    if quality_band == 1:
        return rng.choice(mid)
    if quality_band == 2:
        return rng.choice(mid + good)
    return rng.choice(good)


def _sample_context_block(rng: random.Random) -> str:
    pool = (
        "[1] Source: LoRA Paper\nLoRA reduces trainable parameters via low-rank updates.\n\n"
        "[2] Source: QLoRA\nQLoRA uses quantized base weights with LoRA adapters.",
        "[1] Source: RAG Survey\nRAG improves grounding by conditioning on retrieved passages.\n\n"
        "[2] Source: Hallucination Study\nGrounded generation lowers unsupported claims.",
        "[1] Source: Quantization Overview\n4-bit quantization cuts memory significantly.\n\n"
        "[2] Source: Accuracy Trade-offs\nLower precision can slightly reduce quality in some cases.",
        "(no retrieved contexts)",
    )
    return rng.choice(pool)


def _sample_strategy(rng: random.Random) -> str:
    return rng.choice(("simple", "comparative", "multi_hop", "exploratory"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Critic SFT Alpaca JSONL dataset.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/critic_sft_train.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_critic_sft_dataset(num_samples=args.num_samples, seed=args.seed)
    write_jsonl(rows=rows, output_path=args.output)
    print(f"Generated {len(rows)} Critic SFT samples at: {args.output}")


if __name__ == "__main__":
    main()
