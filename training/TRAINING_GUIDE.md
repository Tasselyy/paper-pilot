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
# 重要：vLLM 及其依赖必须从官方 PyPI 安装，见下方「根因与一次性方案」
pip install vllm -i https://pypi.org/simple/
```

### 为什么一直报错？根因与一次性方案

**根因：** 使用国内 PyPI 镜像（如腾讯、阿里、清华）时，容易出现：

1. **镜像同步滞后或 repackage**：同一版本号下拿到的是旧构建或改过的包，缺少新 API（如 `gguf.constants.Keys`、`cachetools.LRUCache`、`llguidance.LLMatcher`）。
2. **元数据不完整**：镜像里的包有时缺少正确 `dist-info`，导致 `importlib.metadata.version()` 为 `None`（如 email-validator），进而触发 Pydantic 的 `NoneType.partition`。
3. **版本解析不一致**：vLLM 对 gguf、cachetools、llguidance 等有明确版本要求；镜像可能解析出错误版本组合。

**一次性解决（二选一）：**

- **方案 A（推荐）：** 单独建一个只跑 vLLM 的虚拟环境，全程用官方源安装，避免和 training 环境混用：
  ```bash
  python -m venv .venv-vllm
  . .venv-vllm/bin/activate   # Windows: .venv-vllm\Scripts\Activate.ps1
  pip install -U pip
  pip install vllm -i https://pypi.org/simple/
  # 之后只用此环境跑 vllm serve ...
  ```
- **方案 B：** 在当前 `.venv` 里强制用官方源重装 vLLM 及依赖（会覆盖镜像装过的相关包）：
  ```bash
  pip install --force-reinstall --no-cache-dir vllm -i https://pypi.org/simple/
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

**Troubleshooting（若仍用镜像，可先按上方「一次性方案」重装 vLLM；或单独修）：**
- `ImportError: cannot import name 'Keys' from 'gguf.constants'` → 见一次性方案，或 `pip install --force-reinstall "gguf>=0.17.0" -i https://pypi.org/simple/`
- `AttributeError: module 'cachetools' has no attribute 'LRUCache'` → 见一次性方案，或 `pip install --force-reinstall "cachetools>=5.0.0" -i https://pypi.org/simple/`
- `AttributeError: 'NoneType' object has no attribute 'partition'`（pydantic/networks.py）→ 见一次性方案，或 `pip install --force-reinstall --no-cache-dir "email-validator>=2.0" -i https://pypi.org/simple/`
- `AttributeError: module 'llguidance' has no attribute 'LLMatcher'` → 镜像可能装了不兼容的 llguidance（如 1.4+）；见一次性方案，或 `pip install --force-reinstall "llguidance>=1.3.0,<1.4.0" -i https://pypi.org/simple/`

```bash
vllm serve training/artifacts/qwen2.5-7b-awq \
  --enable-lora \
  --lora-modules router-lora=training/artifacts/router_lora_adapter \
                 critic-lora=training/artifacts/critic_dpo_model \
  --max-lora-rank 16 \
  --quantization awq \
  --dtype float16 \
  --port 8000
```

**若报错 `torch.bfloat16 is not supported for quantization method awq`**：必须加 `--dtype float16`（AWQ 仅支持 float16）。

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

在**项目根目录**执行。`--mode cloud` 用云 API，任意机器可跑；`--mode vllm` 会请求配置里的 vLLM 地址（默认 `http://localhost:8000/v1`）。

**vLLM 在云端、benchmark 在本地跑时**：先设环境变量再跑 vllm 模式，例如（把 `YOUR_CLOUD_IP` 换成实际 IP 或域名）：

```bash
export VLLM_BASE_URL="http://YOUR_CLOUD_IP:8000/v1"
python benchmarks/latency_benchmark.py --mode vllm --output benchmarks/results/vllm_awq.json
```

Windows PowerShell: `$env:VLLM_BASE_URL="http://YOUR_CLOUD_IP:8000/v1"`，再执行上面第二条命令。

```bash
python benchmarks/latency_benchmark.py --mode cloud --output benchmarks/results/baseline_cloud.json
python benchmarks/latency_benchmark.py --mode vllm --output benchmarks/results/vllm_awq.json
```

