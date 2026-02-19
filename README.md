# Paper Pilot

Paper Pilot is a multi-strategy research agent built with LangGraph, LangChain, and MCP-based RAG tools.
It routes user questions to different reasoning strategies (`simple`, `comparative`, `multi_hop`, `exploratory`), critiques the draft answer, and outputs a final response with sources.

## Features

- Intent routing with structured `Intent` state
- Four strategy paths:
  - `simple`: single-shot retrieval + synthesis
  - `comparative`: parallel retrieval for entity/dimension comparison
  - `multi_hop`: plan-execute-replan style decomposition
  - `exploratory`: iterative ReAct-style exploration
- Critic + retry/refine loop
- Long-term memory persistence (JSONL)
- Trace logging to JSONL + Rich streaming CLI output
- MCP integration for RAG tools:
  - `query_knowledge_hub`
  - `list_collections`
  - `get_document_summary`

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- Optional for live runs:
  - OpenAI-compatible API key
  - Running RAG MCP server (HTTP or stdio transport)

## Quick Start

### 1) Clone and install

```bash
git clone <your-repo-url>
cd paper-pilot
python -m venv .venv
```

Activate virtualenv:

- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- bash/zsh:

```bash
source .venv/bin/activate
```

Install package and dev dependencies:

```bash
pip install -e ".[dev]"
```

### 2) Configure environment variables

Copy `.env.example` to `.env`, then set at least `OPENAI_API_KEY`:

```bash
cp .env.example .env
```

`.env` example:

```dotenv
OPENAI_API_KEY=your_key_here
```

`src/config.py` loads `.env` automatically and resolves `${VAR}` placeholders in YAML.

### 3) Configure MCP + agent settings

Default config file: `config/settings.yaml`.

Two supported MCP transport modes:

- `streamable-http` (recommended if your RAG server is already running)
- `stdio` (Paper Pilot launches the RAG process as a subprocess)

#### Option A: streamable-http

```yaml
mcp:
  rag_default_collection: essays
  connections:
    rag_server:
      transport: streamable-http
      url: http://127.0.0.1:8000/mcp
```

#### Option B: stdio

```yaml
mcp:
  rag_default_collection: null
  connections:
    rag_server:
      transport: stdio
      command: C:/path/to/rag/.venv/Scripts/python.exe
      args: ["-m", "src.mcp_server.server"]
      cwd: C:/path/to/rag
      mcp_stderr: devnull
      env:
        OPENAI_API_KEY: ${OPENAI_API_KEY}
```

### 4) Run

Dry-run (no live MCP/LLM dependency):

```bash
python main.py --question "What is LoRA?" --dry-run
```

Live run (requires valid LLM + MCP config):

```bash
python main.py --question "Compare LoRA and QLoRA in terms of memory and accuracy."
```

Verbose mode:

```bash
python main.py -q "Explain attention" --dry-run --verbose
```

## Strategy Examples

Use these prompts to quickly validate routing behavior:

- `simple`: `What is LoRA?`
- `comparative`: `Compare LoRA and QLoRA in terms of memory and accuracy.`
- `multi_hop`: `First explain attention in Transformers, then how FlashAttention improves it.`
- `exploratory`: `What are the latest trends in efficient fine-tuning?`

## Output and Traces

- CLI prints final answer and sources.
- Trace records are written to:
  - `tracing.trace_dir` + `tracing.trace_file` from `config/settings.yaml`
  - default: `data/traces/trace.jsonl`
- Long-term memory is written to:
  - `memory.memory_file` (default: `data/long_term_memory.jsonl`)

## Testing

Run all tests:

```bash
pytest
```

Useful targeted test commands:

```bash
pytest tests/unit/test_config.py
pytest tests/integration/test_simple_path.py
pytest tests/integration/test_all_strategies.py
```

Optional E2E tests (requires running RAG server):

```bash
pytest tests/e2e -m e2e
```

## Project Layout

Key paths:

- `main.py`: CLI entry
- `config/settings.yaml`: runtime configuration
- `src/agent/graph.py`: main graph builder
- `src/agent/nodes/`: router, slot filling, critic, memory, output nodes
- `src/agent/strategies/`: `simple`, `comparative`, `multi_hop`, `exploratory`
- `src/tools/`: MCP client and RAG wrapper
- `src/tracing/`: trace model and Rich output
- `tests/`: unit, integration, e2e

## Troubleshooting

- Config validation fails:
  - Ensure required `llm` and `mcp` sections exist in `config/settings.yaml`.
- No MCP servers configured:
  - Confirm `mcp.connections.rag_server` exists and transport fields are valid.
- API key issues:
  - Confirm `.env` exists and `OPENAI_API_KEY` is set.
- `stdio` hangs:
  - Set `mcp_stderr: devnull` or switch to `streamable-http`.

## Notes

- Local Router/Critic model paths are optional (`agent.router_model_path`, `agent.critic_model_path`).
- If local models are not configured, cloud fallback is used.
