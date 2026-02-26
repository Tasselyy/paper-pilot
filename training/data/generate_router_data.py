"""生成 Router 训练用的合成数据，Alpaca JSONL 格式。

在 5 类意图上均衡采样：factual / comparative / multi_hop / exploratory / follow_up。
每行格式：instruction、input（用户问题）、output（JSON：type + confidence）、intent（标签）。
可用于 training/sft_router.py 的 --dataset。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# Router 支持的 5 类意图，与下游 SFT 的 output 格式一致
INTENT_TYPES: tuple[str, ...] = (
    "factual",
    "comparative",
    "multi_hop",
    "exploratory",
    "follow_up",
)

DEFAULT_SAMPLES_PER_INTENT = 160
DEFAULT_SEED = 42

# 模型需要根据 input 输出 type + confidence 的 JSON，与此说明一致
_INSTRUCTION = (
    "Classify the user question into one intent type. "
    "Allowed types: factual, comparative, multi_hop, exploratory, follow_up. "
    "Return strict JSON with keys: type, confidence."
)


def generate_router_dataset(
    *,
    samples_per_intent: int = DEFAULT_SAMPLES_PER_INTENT,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """按意图类别均衡生成合成 Router 数据；每类 samples_per_intent 条，最后打乱顺序。"""
    if samples_per_intent <= 0:
        raise ValueError("samples_per_intent must be > 0")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for intent in INTENT_TYPES:
        for idx in range(samples_per_intent):
            question = _build_question(intent=intent, idx=idx, rng=rng)
            confidence = _confidence_for_intent(intent=intent, idx=idx)  # 按意图给一个 [0.8, 0.99] 的置信度
            output = json.dumps(
                {"type": intent, "confidence": confidence},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            rows.append(
                {
                    "instruction": _INSTRUCTION,
                    "input": question,
                    "output": output,
                    "intent": intent,
                }
            )

    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    """将生成的行按 UTF-8 写入 JSONL，每行一个 JSON 对象。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_question(*, intent: str, idx: int, rng: random.Random) -> str:
    """根据意图类型从模板池中选一句合成问题（实体、指标、话题等用固定池子按 idx/rng 选取）。"""
    entity_a = _pick(
        rng,
        (
            "LoRA",
            "QLoRA",
            "Adapter",
            "Prefix Tuning",
            "FlashAttention",
            "Transformer",
            "RAG",
            "MoE",
            "RLHF",
            "DPO",
            "SFT",
            "Knowledge Distillation",
        ),
        idx,
    )
    entity_b = _pick(
        rng,
        (
            "BERT",
            "GPT-4",
            "Llama-3",
            "Mistral",
            "DeepSeek",
            "Longformer",
            "RoPE",
            "ALiBi",
            "AdaFactor",
            "AdamW",
            "PPO",
            "GRPO",
        ),
        idx + 3,
    )
    metric = _pick(
        rng,
        (
            "memory usage",
            "latency",
            "throughput",
            "inference cost",
            "hallucination rate",
            "factual consistency",
            "sample efficiency",
            "stability",
        ),
        idx + 7,
    )
    topic = _pick(
        rng,
        (
            "efficient fine-tuning",
            "retrieval augmentation",
            "long-context modeling",
            "agentic reasoning",
            "alignment methods",
            "tool-using agents",
            "evaluation benchmarks",
            "multimodal adaptation",
        ),
        idx + 11,
    )

    if intent == "factual":
        templates = (
            "What is {entity_a} in simple terms?",
            "Define {entity_a} and its core purpose.",
            "How does {entity_a} work at a high level?",
            "When should practitioners use {entity_a}?",
            "Give a concise explanation of {entity_a}.",
        )
    elif intent == "comparative":
        templates = (
            "Compare {entity_a} and {entity_b} for {metric}.",
            "What are the key differences between {entity_a} and {entity_b}?",
            "Which is better for production, {entity_a} or {entity_b}, and why?",
            "Contrast {entity_a} vs {entity_b} from a trade-off perspective.",
            "How do {entity_a} and {entity_b} differ in real-world usage?",
        )
    elif intent == "multi_hop":
        templates = (
            "How does {entity_a} influence {entity_b}, and what impact does that have on {metric}?",
            "Explain how {entity_a} relates to {topic}, then infer implications for deployment.",
            "If a system adopts {entity_a}, what changes in {entity_b} and downstream evaluation?",
            "Connect {entity_a} with {entity_b} and derive practical recommendations.",
            "Trace the chain from {entity_a} design choices to outcomes in {metric}.",
        )
    elif intent == "exploratory":
        templates = (
            "What are current research trends in {topic}?",
            "Survey recent directions around {topic} and open problems.",
            "What should I read to understand the landscape of {topic}?",
            "Give a broad overview of methods, benchmarks, and challenges in {topic}.",
            "Where is {topic} heading in the next 1-2 years?",
        )
    elif intent == "follow_up":
        templates = (
            "Can you elaborate on that with a practical example?",
            "What did you mean by the previous point about {metric}?",
            "Could you explain that again in simpler language?",
            "How does that apply to a real project using {entity_a}?",
            "Can you break the last answer into actionable steps?",
        )
    else:
        raise ValueError(f"Unsupported intent: {intent}")

    template = _pick(rng, templates, idx + 17)
    return template.format(entity_a=entity_a, entity_b=entity_b, metric=metric, topic=topic)


def _confidence_for_intent(*, intent: str, idx: int) -> float:
    """按意图给一个基准置信度，再根据 idx 做小幅偏移，结果限制在 [0.80, 0.99]。"""
    base_by_intent = {
        "factual": 0.94,
        "comparative": 0.90,
        "multi_hop": 0.87,
        "exploratory": 0.86,
        "follow_up": 0.92,
    }
    base = base_by_intent[intent]
    drift = (idx % 7) * 0.01
    score = base - drift
    return max(0.8, min(round(score, 2), 0.99))


def _pick(rng: random.Random, choices: tuple[str, ...], idx: int) -> str:
    """从 choices 中按 (idx + 随机偏移) 取一个，保证可复现又有变化。"""
    offset = rng.randint(0, len(choices) - 1)
    return choices[(idx + offset) % len(choices)]


def _parse_args() -> argparse.Namespace:
    """解析命令行：每类样本数、随机种子、输出 JSONL 路径。"""
    parser = argparse.ArgumentParser(description="Generate Router training dataset JSONL.")
    parser.add_argument(
        "--samples-per-intent",
        type=int,
        default=DEFAULT_SAMPLES_PER_INTENT,
        help="Number of samples to generate per intent class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/data/router_train.jsonl"),
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main() -> None:
    """入口：生成 Router 数据并写入 JSONL，打印样本数与路径。"""
    args = _parse_args()
    rows = generate_router_dataset(
        samples_per_intent=args.samples_per_intent,
        seed=args.seed,
    )
    write_jsonl(rows=rows, output_path=args.output)
    print(
        f"Generated {len(rows)} samples "
        f"({args.samples_per_intent} per intent) at: {args.output}"
    )


if __name__ == "__main__":
    main()
