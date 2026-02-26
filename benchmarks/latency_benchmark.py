"""Latency benchmark for cloud and vLLM inference modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from main import run_agent

DEFAULT_OUTPUT_CLOUD = Path("benchmarks/results/baseline_cloud.json")
DEFAULT_OUTPUT_VLLM = Path("benchmarks/results/vllm_local.json")

BENCHMARK_QUESTIONS: list[str] = [
    "What is LoRA?",
    "Explain attention in Transformers.",
    "What is DPO in preference learning?",
    "What is retrieval-augmented generation?",
    "What are trade-offs of 4-bit quantization?",
    "Compare LoRA and QLoRA for memory usage.",
    "Compare DPO and PPO for alignment.",
    "Compare RAG and long-context prompting.",
    "Compare FlashAttention and vanilla attention.",
    "Compare AWQ and GPTQ quantization.",
    "How does FlashAttention affect long-context scaling?",
    "How does LoRA interact with quantization in QLoRA?",
    "How does retrieval quality impact faithfulness metrics?",
    "How does DPO depend on SFT initialization quality?",
    "How does intent routing affect agent latency?",
    "What are current trends in efficient fine-tuning?",
    "What are open challenges in trustworthy RAG systems?",
    "What papers should I read for preference optimization?",
    "What is the outlook for Multi-LoRA serving?",
    "What are trends in local model deployment for agents?",
    "Can you elaborate on the LoRA rank trade-off?",
    "Can you give a practical example of RAG failure modes?",
    "Can you clarify when to use comparative strategy?",
    "Can you break down DPO training steps?",
    "Can you summarize AWQ deployment tips?",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "mean_ms": round(mean(values), 1),
        "p50_ms": round(_percentile(values, 0.50), 1),
        "p95_ms": round(_percentile(values, 0.95), 1),
        "max_ms": round(max(values), 1),
    }


def _extract_node_llm_ms(reasoning_trace: list[Any], *, step_type: str) -> float | None:
    for step in reasoning_trace:
        if getattr(step, "step_type", None) == step_type:
            metadata = getattr(step, "metadata", {}) or {}
            value = metadata.get("llm_call_duration_ms")
            if isinstance(value, (int, float)):
                return float(value)
        elif isinstance(step, dict) and step.get("step_type") == step_type:
            metadata = step.get("metadata", {}) or {}
            value = metadata.get("llm_call_duration_ms")
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _build_mode_config(*, base_config: str, mode: str) -> str:
    with Path(base_config).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("vllm", {})
    if mode == "vllm":
        data["vllm"]["enabled"] = True
    else:
        data["vllm"]["enabled"] = False

    fd, temp_path = tempfile.mkstemp(suffix=".yaml", prefix=f"benchmark-{mode}-")
    os.close(fd)
    temp_config = Path(temp_path)
    temp_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(temp_config)


async def run_benchmark(*, mode: str, config_path: str) -> dict[str, Any]:
    bench_config = _build_mode_config(base_config=config_path, mode=mode)
    router_latencies: list[float] = []
    critic_latencies: list[float] = []
    e2e_latencies: list[float] = []
    tokens_input = 0
    tokens_output = 0

    try:
        for question in BENCHMARK_QUESTIONS:
            t0 = time.perf_counter()
            result = await run_agent(
                question=question,
                config_path=bench_config,
                dry_run=False,
                verbose=False,
            )
            e2e_ms = (time.perf_counter() - t0) * 1000
            e2e_latencies.append(e2e_ms)

            trace = result.get("reasoning_trace", [])
            router_ms = _extract_node_llm_ms(trace, step_type="route")
            critic_ms = _extract_node_llm_ms(trace, step_type="critique")
            if router_ms is not None:
                router_latencies.append(router_ms)
            if critic_ms is not None:
                critic_latencies.append(critic_ms)

            tokens = result.get("tokens_used", {})
            if isinstance(tokens, dict):
                in_tok = tokens.get("input")
                out_tok = tokens.get("output")
                if isinstance(in_tok, int):
                    tokens_input += in_tok
                if isinstance(out_tok, int):
                    tokens_output += out_tok
    finally:
        Path(bench_config).unlink(missing_ok=True)

    return {
        "mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(BENCHMARK_QUESTIONS),
        "router": _stats(router_latencies),
        "critic": _stats(critic_latencies),
        "e2e": _stats(e2e_latencies),
        "tokens": {"input": tokens_input, "output": tokens_output},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Paper Pilot latency.")
    parser.add_argument("--mode", choices=("cloud", "vllm"), required=True)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output: Path
    if args.output is not None:
        output = args.output
    elif args.mode == "vllm":
        output = DEFAULT_OUTPUT_VLLM
    else:
        output = DEFAULT_OUTPUT_CLOUD

    payload = asyncio.run(run_benchmark(mode=args.mode, config_path=args.config))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark finished. Results saved to: {output}")


if __name__ == "__main__":
    main()
