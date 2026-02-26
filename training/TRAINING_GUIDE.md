# Training Guide (A800 80GB)

This guide covers the full pipeline: data generation, Router/Critic SFT, Critic DPO, AWQ quantization, vLLM Multi-LoRA serving, and benchmark verification.

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

```bash
python training/sft_router.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --dataset training/data/router_train.jsonl \
  --output-dir training/artifacts/router_lora_adapter \
  --max_steps 200 --epochs 3 \
  --batch-size 16 --grad-accum-steps 1 \
  --lora-r 16 --lora-alpha 32 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-seq-len 256 --bf16 \
  --report-to wandb --wandb-project paper-pilot --run-name router-sft
```

## 3) Train Critic LoRA (SFT Stage 1)

```bash
python training/sft_critic.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --dataset training/data/critic_sft_train.jsonl \
  --output-dir training/artifacts/critic_sft_adapter \
  --max_steps 200 --epochs 3 \
  --batch-size 8 --grad-accum-steps 2 \
  --lora-r 16 --lora-alpha 32 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-seq-len 512 --bf16 \
  --report-to wandb --wandb-project paper-pilot --run-name critic-sft
```

## 4) Train Critic LoRA (DPO Stage 2)

```bash
python training/dpo_critic.py \
  --model-name training/artifacts/critic_sft_adapter \
  --dataset training/data/dpo_train.jsonl \
  --output-dir training/artifacts/critic_dpo_model \
  --max_steps 200 --epochs 2 \
  --batch-size 4 --grad-accum-steps 2 \
  --learning-rate 5e-5 --beta 0.1 \
  --max-length 512 \
  --lora-r 16 --lora-alpha 32 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --bf16 \
  --report-to wandb --wandb-project paper-pilot --run-name critic-dpo
```

## 5) Quantize Base Model to AWQ INT4

```bash
python training/quantize_base_model.py \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir training/artifacts/qwen2.5-7b-awq \
  --bits 4 --group-size 128 --calib-samples 128
```

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

