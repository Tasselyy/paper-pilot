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

## 1) Generate Data

```bash
python training/data/generate_router_data.py --samples-per-intent 200 --output training/data/router_train.jsonl
python training/data/generate_critic_sft_data.py --num-samples 800 --output training/data/critic_sft_train.jsonl
python training/data/generate_dpo_pairs.py --num-pairs 500 --output training/data/dpo_train.jsonl
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

```bash
python training/quantize_base_model.py
```

Defaults (model, output-dir, bits, group-size, calib-samples) are in `training_config.yaml`; override with CLI if needed.

## 6) Serve with vLLM Multi-LoRA

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

