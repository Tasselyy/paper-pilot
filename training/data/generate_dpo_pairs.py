"""生成 Critic DPO 训练用的偏好对，TRL 标准格式。

每行一个三元组：prompt（指令+问题+草稿+上下文+策略，即模型输入）、chosen（偏好评判 JSON，高分）、
rejected（非偏好评判 JSON，低分）。DPOTrainer 会学习拉大 chosen 与 rejected 的 log 差异。
输出可由 training/dpo_critic.py 直接加载。
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

# prompt 中会包含的指令头，与 Critic 推理/ SFT 一致
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
    """将单条评判序列化为紧凑 JSON 字符串（无空格、ASCII 不转义）。"""
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
    """生成 num_pairs 条 (prompt, chosen, rejected)；prompt 用合成问题/草稿/上下文/策略拼成，chosen 高分、rejected 低分。"""
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
    """拼出 Critic 的输入 prompt（不含评判结果），与推理接口一致。"""
    return build_critic_evaluation_prompt(
        question=question,
        draft_answer=draft_answer,
        contexts=contexts,
        strategy=strategy,
    )


def _build_chosen_verdict(*, rng: random.Random, idx: int) -> str:
    """生成「偏好」评判：分数 >= 7，completeness/faithfulness 偏高，正向 feedback。"""
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
    """生成「非偏好」评判：分数 < 7，completeness/faithfulness 偏低，改进向 feedback。"""
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
    """从固定问题模板中随机抽 size 个问题。"""
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
    """从预定义的短答案池中随机抽 size 个作为草稿回答。"""
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
    """随机抽 size 段「检索上下文」文本，格式与运行时一致。"""
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
    """随机抽 size 个策略标签，与 Router 输出一致。"""
    strategies = ("simple", "comparative", "multi_hop", "exploratory")
    return [rng.choice(strategies) for _ in range(size)]


def _pick(rng: random.Random, choices: tuple[str, ...], idx: int) -> str:
    """按 idx 与随机偏移从 choices 中取一项，兼顾可复现与多样性。"""
    return choices[(idx + rng.randint(0, 10)) % len(choices)]


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """将行按 UTF-8 写入 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    """解析命令行：偏好对数量、随机种子、输出 JSONL 路径。"""
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
    """入口：生成 DPO 偏好对并写入 JSONL，打印条数与路径。"""
    args = _parse_args()
    rows = generate_dpo_dataset(num_pairs=args.num_pairs, seed=args.seed)
    write_jsonl(rows=rows, output_path=args.output)
    print(f"Generated {len(rows)} DPO pairs at: {args.output}")


if __name__ == "__main__":
    main()
