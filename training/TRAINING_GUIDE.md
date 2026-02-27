# Training Guide (A800 80GB)

This guide covers the full pipeline: data generation, Router/Critic SFT, Critic DPO, AWQ quantization, vLLM Multi-LoRA serving, and benchmark verification.

**Config file:** Defaults for all training steps live in `training/training_config.yaml`. You can edit that file once; then run the scripts with no (or few) CLI args. Any CLI argument overrides the config file.

**Running scripts:** Always run from the **project root** (where `pyproject.toml` lives). After `pip install -e ".[training]"`, `python training/sft_critic.py` works. If you see `ModuleNotFoundError: No module named 'training'`, either reinstall with `pip install -e ".[training]"` or run as a module: `python -m training.sft_critic` (same for other scripts under `training/`).

## 0) Environment

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[training]"
pip install vllm
```

Optional wandb:

```bash
echo "WANDB_API_KEY=your_key_here" >> .env
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name())"
```

## 1) Generate Data（全 LLM）

训练数据统一由 LLM 生成（题目/草稿/上下文与 verdict 均来自 LLM）。需配置 `OPENAI_API_KEY` 并安装 `pip install openai`（或 `pip install -e ".[vllm]"`）。

**一键生成（推荐）：**

```bash
# 从项目根目录执行；输出: training/data/router_train.jsonl, critic_sft_train.jsonl, dpo_train.jsonl
python training/data/generate_all_with_llm.py
```

默认：`--full-scenario`（题目/草稿/上下文由 LLM 生成），`--scenarios 100` → 100 条 Critic SFT + 100 对 DPO；Router 每意图 40 条。可选：`--router-per-intent`、`--output-dir`、`--model`、`--api-base`。使用 `--no-full-scenario` 时仅 verdict 用 LLM、场景用固定池（更快），此时可用 `--critic-samples`、`--dpo-pairs` 指定条数。

**按数据集单独生成（格式与上面一致）：**

```bash
python training/data/generate_router_data_with_llm.py --samples-per-intent 40 --output training/data/router_train.jsonl
python training/data/generate_critic_sft_with_llm.py --num-samples 200 --output training/data/critic_sft_train.jsonl
python training/data/generate_dpo_pairs_with_llm.py --num-pairs 200 --output training/data/dpo_train.jsonl
```

## 2) Train Router LoRA (SFT)

Using defaults from `training/training_config.yaml` (optional overrides with `--run-name`, `--report-to`, etc.):

```bash
python training/sft_router.py
```

Example with overrides (e.g. wandb and run name):

```bash
python training/sft_router.py --report-to wandb --run-name router-sft
```

## 3) Train Critic LoRA (SFT Stage 1)

```bash
python training/sft_critic.py
```

With wandb: `python training/sft_critic.py --report-to wandb --run-name critic-sft`

## 4) Train Critic LoRA (DPO Stage 2)

Use the Critic SFT adapter as base (override in config or CLI):

```bash
python training/dpo_critic.py --model-name training/artifacts/critic_sft_adapter
```

All other defaults (dataset, output-dir, lr, etc.) come from `training_config.yaml`. With wandb: add `--report-to wandb --run-name critic-dpo`.

## 5) Quantize Base Model to AWQ INT4

**Note:** AutoAWQ requires Transformers 4.51.3, which conflicts with TRL (which needs ≥4.56.2). Run step 5 in a **separate virtualenv** with the `quantize` extra only:

```bash
python -m venv .venv-quantize
. .venv-quantize/bin/activate   # Windows: .venv-quantize\Scripts\Activate.ps1
pip install -U pip && pip install -e ".[quantize]"
python training/quantize_base_model.py
```

```bash
python training/quantize_base_model.py
```

Defaults (model, output-dir, bits, group-size, calib-samples) are in `training_config.yaml`; override with CLI if needed. On failure, run with `--no-fallback` to see the full traceback.

## 6) Serve with vLLM Multi-LoRA

**If you see** `ImportError: cannot import name 'Keys' from 'gguf.constants'`, **upgrade gguf:** `pip install "gguf>=0.17.0"` (vLLM requires this for GGUF support).

```bash
vllm serve training/artifacts/qwen2.5-7b-awq \
  --enable-lora \
  --lora-modules router-lora=training/artifacts/router_lora_adapter \
                 critic-lora=training/artifacts/critic_dpo_model \
  --max-lora-rank 16 \
  --quantization awq \
  --port 8000
```

## 7) Verify Endpoints

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"router-lora","prompt":"Classify: What is LoRA?","max_tokens":64}'
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"critic-lora","prompt":"Evaluate the answer quality...","max_tokens":128}'
```

## 8) Run Benchmark

```bash
python benchmarks/latency_benchmark.py --mode cloud --output benchmarks/results/baseline_cloud.json
python benchmarks/latency_benchmark.py --mode vllm --output benchmarks/results/vllm_awq.json
```

