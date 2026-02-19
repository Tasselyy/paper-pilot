# PaperPilot Agent — Developer Specification (DEV_SPEC)

> 版本：0.1 — 开发规格说明  
> 详细架构与数据模型见：[PAPER_PILOT_DESIGN.md](./docs/PAPER_PILOT_DESIGN.md)

## 目录

- 项目概述
- 核心特点
- 技术选型
- 测试方案
- 系统架构与模块设计
- 项目排期

---

## 1. 项目概述

PaperPilot 是一个**对接 Modular-RAG MCP Server 的多策略研究 Agent**，面向 AI/ML 论文阅读场景：用户提问后，Agent 通过意图识别选择推理模式（Direct / Plan-and-Execute / ReWOO / ReAct），调用 RAG 知识库检索并合成回答，经 Critic（Reflexion）评估后输出或带反馈重试。

### 设计理念 (Design Philosophy)

- **意图驱动、多推理范式**：不同复杂度的问题走不同执行路径，而非单一 ReAct 循环；每种策略对应一种已命名推理模式，便于面试讲解。
- **模型分层**：高频轻量任务（意图分类、质量评估）用本地 LoRA/DPO 小模型；重推理与合成用 Cloud LLM，兼顾成本与能力。
- **MCP 标准化对接**：以 MCP Client 身份调用现有 RAG Server 的 `query_knowledge_hub` / `list_collections` / `get_document_summary`，不侵入 RAG 代码，便于独立迭代与演示。

### 与 Modular-RAG 的关系

| 角色 | 项目 | 职责 |
|------|------|------|
| **知识提供方** | Modular-RAG MCP Server | 提供混合检索、集合列表、文档摘要等 MCP Tools |
| **调用方** | PaperPilot Agent | 作为 MCP Client 连接 RAG Server，在 LangGraph 工作流中调用上述工具，完成意图理解、多策略执行、Critic、记忆与输出 |

部署时需先启动 RAG Server（或配置其进程/URL），再启动 PaperPilot；Agent 通过 MCP 协议（如 stdio/streamable-http）与 RAG 通信。

### 对接的 RAG 项目（开发时必填）

在**新文件夹**下开发 PaperPilot 时，需在 Agent 配置中填入下述 RAG 项目信息，以便 MCP Client 正确启动或连接 RAG Server。

| 项 | 值 |
|----|-----|
| **项目名称** | MODULAR-RAG-MCP-SERVER |
| **入口模块** | `src.mcp_server.server`（在 RAG 项目根目录执行：`python -m src.mcp_server.server`） |
| **暴露的 Tools** | `query_knowledge_hub`、`list_collections`、`get_document_summary` |
| **传输** | stdio（子进程方式，Agent 启动时由 MCP Client 拉起 RAG 进程） |

**Agent 项目中的配置示例**（`config/settings.yaml`）：

```yaml
mcp:
  connections:
    rag_server:
      command: "python"
      args: ["-m", "src.mcp_server.server"]
      # 使用环境变量，便于不同环境/本机路径一致
      cwd: "${RAG_PROJECT_ROOT}"
      transport: "stdio"
      env:
        PYTHONPATH: "${RAG_PROJECT_ROOT}"
```

**环境变量约定**：在运行 PaperPilot 前设置 `RAG_PROJECT_ROOT` 为 RAG 项目根目录的绝对路径（例如 `C:/code/python/MODULAR-RAG-MCP-SERVER` 或 `/path/to/MODULAR-RAG-MCP-SERVER`）。若未设置，实现时可回退到默认相对路径或报错提示。

**联调步骤**：1）在 RAG 项目根目录安装依赖并完成数据摄取；2）设置 `RAG_PROJECT_ROOT` 指向该根目录；3）在 PaperPilot 项目下运行 `python main.py`（或等价入口），Agent 会按配置启动 RAG 子进程并调用其 Tools。

更多细节见 [PAPER_PILOT_DESIGN.md §7.0](./docs/PAPER_PILOT_DESIGN.md#70-对接的-rag-项目-upstream-rag-project)。

---

## 2. 核心特点

### 意图识别 (Intent Understanding)

- **Step1 — Router（LoRA 分类）**：本地 Qwen2.5-1.5B + LoRA，5 类意图：`factual` / `comparative` / `multi_hop` / `exploratory` / `follow_up`，输出 type + confidence。
- **Step2 — Slot Filling（LLM 槽位填充）**：Cloud LLM 根据 type 与原始问题，填充 `entities`、`dimensions`、`constraints`、`reformulated_query`，产出完整 `Intent` 对象供下游策略消费。
- **Fallback**：本地 Router 不可用时，Router 与 Slot Filling 均可退化为 Cloud LLM 结构化输出。

### 多推理范式 (4 策略 × 4 模式)

| 策略 | 推理模式 | 说明 |
|------|---------|------|
| simple | Direct Execution | 单次 RAG 查询 + 合成，使用 `intent.reformulated_query` |
| multi_hop | Plan-and-Execute | 子图：plan → execute → replan 循环，支持执行中修订计划 |
| comparative | ReWOO | 一次性规划查询，并行检索，统一合成；优先用 `intent.entities` / `intent.dimensions` |
| exploratory | ReAct | think → act → observe 循环，工具为 RAG 检索 |
| （跨策略） | Reflexion | Critic 评估 + 带反馈重试（最多 2 次） |

### 模型分层

- **Cloud LLM**：主推理、问题分解、槽位填充、各策略合成、ReAct think/observe、Reflexion 重试时的改进计划。
- **本地模型（4-bit 量化）**：Router（LoRA SFT）、Critic（DPO），用于分类与质量评估，降低延迟与 API 成本。

### 记忆与可观测性

- **短期记忆**：LangGraph Checkpointer（如 MemorySaver），对话消息与当前会话状态。
- **长期记忆**：自定义事实累积（如 JSONL + 关键词/embedding 检索），在 load_memory / save_memory 节点读写。
- **Trace**：记录 intent、strategy、reasoning_steps、retrieval_queries、critic_verdict、耗时等，支持 JSONL 与流式展示。

---

## 3. 技术选型

### 3.1 核心依赖（与设计文档一致）

- **Agent 框架**：LangGraph ≥1.0.8，LangChain ≥1.2.10，langchain-core ≥1.2.12，langchain-openai ≥1.1.9，langchain-mcp-adapters ≥0.2.1。
- **类型与配置**：Pydantic ≥2，PyYAML ≥6，Rich ≥13（CLI/流式输出）。
- **可选 — 本地模型**：transformers，bitsandbytes，accelerate，torch（推理时 4-bit 加载）。
- **可选 — 训练**：peft，trl，datasets（LoRA SFT + DPO）。

### 3.2 与 RAG Server 的对接方式

- **传输**：stdio（子进程启动 RAG Server）或 streamable-http（远程 RAG Server），由 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 配置。
- **工具**：仅依赖 RAG 暴露的 `query_knowledge_hub`、`list_collections`、`get_document_summary`；在 Agent 侧封装为统一 RAG 工具层（错误处理、参数归一化、结果解析）。

### 3.3 配置驱动

- 主配置（如 `config/settings.yaml`）包含：Cloud LLM 端点与模型、MCP 连接参数、本地 Router/Critic 模型路径、各策略参数（如 max_react_steps、max_retries）、记忆与 Trace 路径等。
- 无本地模型时仅使用 Cloud LLM Fallback，不影响主流程跑通。

---

## 4. 测试方案

### 4.1 设计理念

- **TDD**：关键节点与边先写测试（含 Mock），再实现；状态与 Pydantic 模型变更时同步更新测试。
- **分层**：单元测试（单节点、单函数、Mock LLM/MCP）→ 集成测试（子图或主图 + Fake MCP）→ 端到端（真实 MCP 连接 RAG Server，可选）。

### 4.2 单元测试重点

| 模块 | 测试重点 |
|------|---------|
| **State / Intent** | Intent.to_strategy() 映射、AgentState 字段与 reducer |
| **Router Node** | 给定 question（+ 可选 recent_turn），Mock 本地模型返回 type/confidence，断言写入 state.intent 的 type/confidence |
| **Slot Filling Node** | Mock Cloud LLM 返回 SlotFillingOutput，断言 intent 被完整填充 |
| **route_by_intent** | 各 intent.type 对应 correct strategy 分支 |
| **Simple Node** | Mock MCP 返回固定 chunks，断言使用 intent.reformulated_query 检索、draft_answer 非空 |
| **Comparative Node** | 优先使用 intent.entities/dimensions，缺失时 Mock LLM 抽取；断言并行检索与合成调用 |
| **Plan-and-Execute 子图** | plan → execute → replan 分支（next_step / replan / synthesize），Mock RAG 与 LLM |
| **ReAct 子图** | think → act → observe 循环与终止条件，Mock 工具与 LLM |
| **Critic / Retry** | Mock Critic 返回 pass/fail，断言 retry 时 state.retry_count 递增、重试后再次进入 critic |
| **Memory** | load_memory 写入 accumulated_facts，save_memory 调用长期记忆写入（Mock 存储） |

### 4.3 集成测试重点

- **主图 + Fake MCP**：从 START 到 END，Fake 实现 `query_knowledge_hub` 等返回固定内容，验证 simple / comparative 两条路径的端到端 state 与 final_answer。
- **MCP Client 集成**：在测试环境中启动 RAG Server 子进程或 Mock Server，验证 Agent 能正确发现并调用 tools、解析返回格式。
- **长期记忆集成**：主图接入真实 LongTermMemory，多轮问答下断言第二轮能 recall 第一轮写入的事实，或 JSONL 中有对应记录。
- **Trace 集成**：跑通一条路径后读取 trace 文件，断言 JSONL 记录包含 intent/strategy/critic/final_answer 等字段，保证可观测链路可用。
- **配置驱动建图**：加载测试用 config → 构建 graph（可 Mock RAG/LLM）→ invoke 一次，断言 state 含预期键，验证配置与建图链不被破坏。
- **MCP 子进程集成（可选）**：测试内启动 RAG 子进程或最小 stub → MCP Client 建连 → 调用一次 query_knowledge_hub → 断言返回可解析；可标 `@pytest.mark.slow` 或可选执行。
- **Checkpointer / 多会话（可选）**：同一 thread_id 多轮 invoke 断言状态累积，不同 thread_id 断言隔离；在实现多轮对话或会话隔离时补充。

### 4.4 端到端（可选）

- 依赖已运行的 RAG Server 与灌入的测试文档；执行若干条真实问题，检查意图分类、检索调用次数、最终回答与引用是否符合预期；可标记为 `@pytest.mark.e2e` 并在 CI 中可选执行。
- **CLI 冒烟**：自动化执行 `main.py --question ... --dry-run`（或指定测试 config），断言 exit 0 且输出中含 final_answer 与 sources 结构，将 E4 验收从手动改为可回归。

---

## 5. 系统架构与模块设计

### 5.1 整体架构图

（与 [PAPER_PILOT_DESIGN.md §2](./docs/PAPER_PILOT_DESIGN.md#2-整体架构) 一致，此处仅摘要。）

```
User Question
     → 意图理解 (Router LoRA + Slot Filling LLM) → Intent
     → LangGraph 主图: load_memory → route → slot_fill
     → 条件边 route_by_intent → simple | multi_hop | comparative | exploratory
     → 各策略子图/节点执行（调用 MCP RAG）
     → critic (Reflexion)
     → pass → save_memory → format_output → END
     → retry → retry_refine → 回到 route
```

### 5.2 目录结构

（与设计文档 §11 对齐，作为交付清单。）

```
paper-pilot/
├── pyproject.toml
├── config/
│   └── settings.yaml
├── src/
│   ├── agent/
│   │   ├── graph.py              # 主图 build_main_graph()
│   │   ├── state.py              # AgentState, Intent, 子类型
│   │   ├── nodes/
│   │   │   ├── router.py         # Router 节点 (LoRA 意图分类)
│   │   │   ├── slot_filling.py   # Slot Filling 节点
│   │   │   ├── critic.py         # Critic 节点 (Reflexion)
│   │   │   ├── memory_nodes.py   # load_memory / save_memory
│   │   │   └── format_output.py
│   │   ├── strategies/
│   │   │   ├── simple.py
│   │   │   ├── multi_hop.py      # Plan-and-Execute 子图
│   │   │   ├── comparative.py    # ReWOO
│   │   │   └── exploratory.py    # ReAct 子图
│   │   └── edges.py              # route_by_intent, critic_gate 等
│   ├── tools/
│   │   ├── mcp_client.py         # MCP Client 封装
│   │   └── tool_wrapper.py       # RAGToolWrapper
│   ├── models/
│   │   ├── loader.py             # LocalModelManager (量化加载)
│   │   └── inference.py          # Router/Critic 推理封装
│   ├── memory/
│   │   ├── short_term.py         # Checkpointer 封装
│   │   └── long_term.py          # 长期事实记忆
│   ├── llm/
│   │   └── client.py             # Cloud LLM 封装
│   ├── prompts/
│   │   ├── router.py
│   │   ├── slot_filling.py
│   │   ├── strategies.py
│   │   ├── critic.py
│   │   └── memory.py
│   └── tracing/
│       └── tracer.py             # AgentTrace + JSONL
├── training/                     # 可选
│   ├── data/
│   │   ├── generate_router_data.py
│   │   └── generate_dpo_pairs.py
│   ├── sft_router.py
│   ├── dpo_critic.py
│   └── ...
├── main.py                       # CLI 入口
└── tests/
    ├── unit/
    ├── integration/
    └── e2e/                      # 可选
```

### 5.3 模块说明（与设计文档对应）

| 模块 | 职责 | 设计文档章节 |
|------|------|-------------|
| **agent/state.py** | AgentState、Intent、RouterDecision/CriticVerdict/ReasoningStep 等 Pydantic 定义；Intent.to_strategy() | §4 |
| **agent/graph.py** | 主图节点注册、边与条件边、compile(checkpointer) | §5.1, 5.2 |
| **agent/nodes/router.py** | LoRA 分类，输出部分 Intent；Fallback Cloud LLM | §6.1 |
| **agent/nodes/slot_filling.py** | LLM 槽位填充，输出完整 Intent | §6.2 |
| **agent/strategies/simple.py** | 使用 intent.reformulated_query 单次检索 + 合成 | §6.3 |
| **agent/strategies/multi_hop.py** | Plan-and-Execute 子图：plan / execute_step / replan / pe_synthesize | §5.5, §6.4 |
| **agent/strategies/comparative.py** | ReWOO：优先 intent.entities/dimensions，并行检索 + 合成 | §6.5 |
| **agent/strategies/exploratory.py** | ReAct 子图：think / act / observe / synthesize | §5.6, §6.6 |
| **agent/nodes/critic.py** | Critic 评估（本地 DPO 或 Cloud LLM），写 critic_verdict | §6.7 |
| **agent/nodes/.../retry_refine** | Reflexion 重试：根据 feedback 改写 question 或计划 | §6.8 |
| **tools/mcp_client.py** | MultiServerMCPClient 连接 RAG Server，get_tools() | §7 |
| **tools/tool_wrapper.py** | RAGToolWrapper：search / list_collections / get_doc_info，错误处理 | §7.2 |
| **memory/long_term.py** | 长期记忆 recall / memorize，JSONL 持久化 | §8 |
| **tracing/tracer.py** | AgentTrace 结构与 JSONL 记录 | §10 |

### 5.4 数据流摘要

- **输入**：用户 question（+ 可选 messages 用于 follow_up 与记忆）。
- **意图层**：route 写入 intent（部分）→ slot_fill 写入 intent（完整）。
- **策略层**：各策略读写 state.retrieved_contexts、retrieval_queries、sub_questions、draft_answer、reasoning_trace。
- **质量层**：critic 写入 critic_verdict；retry_refine 可能更新 question、retry_count。
- **输出**：final_answer、reasoning_trace、sources（引用）；Trace 落盘与流式展示。

### 5.5 配置项要点

- **llm**：cloud 端点、模型名、api_key 等。
- **mcp**：connections（rag_server 的 command/args 或 url、transport）。
- **agent**：max_retries、max_react_steps、router_model_path、critic_model_path（可选）。
- **memory**：memory_file（长期记忆路径）。
- **tracing**：trace_dir、trace_file。

---

## 6. 项目排期

> **排期原则**
> - 以本 DEV_SPEC 与 [PAPER_PILOT_DESIGN.md](./docs/PAPER_PILOT_DESIGN.md) 为唯一设计依据，按阶段交付可运行、可测试的增量。
> - 每个子任务尽量在约 1 小时内可验收，并配有验收标准或测试。
> - 先打通主闭环（意图 → 单策略 → Critic → 输出），再补齐四种策略与本地模型。

### 阶段总览

| 阶段 | 目的 |
|------|------|
| **A：工程骨架与配置** | 目录树、pyproject.toml、config、State 与 Intent 定义、可 import 的占位节点 |
| **B：MCP 对接与 Simple 策略** | MCP Client 连接 RAG Server，RAGToolWrapper，Simple 策略节点，主图骨架（route→slot_fill→simple→critic→output） |
| **C：意图理解与四策略** | Router + Slot Filling 节点与 Prompt，multi_hop（Plan-and-Execute）、comparative（ReWOO）、exploratory（ReAct）子图/节点，条件边 route_by_intent |
| **D：Reflexion 与记忆** | Critic 节点，retry_refine 节点，load_memory/save_memory，长期记忆接口 |
| **E：可观测与 CLI** | AgentTrace、JSONL 记录、Rich 流式输出、main.py CLI |
| **F：本地模型（可选）** | LocalModelManager，Router/Critic 本地推理，Fallback 逻辑 |
| **G：微调 Pipeline（可选）** | Router 训练数据合成，LoRA SFT，DPO 数据与训练，量化导出 |
| **H：验收与文档** | 集成/E2E 测试收口，README，面试 Demo 脚本 |

### 任务列表（auto-coder 兼容格式）

**状态**：`[ ]` 未开始 | `[~]` 进行中 | `[x]` 已完成

> **每条任务格式**：`[状态] ID 标题 — 说明`，下方缩进行为 **产物** 与 **验收**。

- [x] A1 初始化目录树与 pyproject.toml — 与 §5.2 一致，pip install -e . 可安装
  - 产物: `pyproject.toml`, `src/` 目录树（含各级 `__init__.py`）, `tests/unit/`, `tests/integration/`, `main.py`
  - 验收: `pip install -e .` 成功 → `python -c "from src.agent import graph"` 无报错
- [x] A2 config/settings.yaml 与加载逻辑 — 能读取 llm/mcp/agent 等配置项
  - 产物: `config/settings.yaml`, `src/config.py`
  - 验收: `pytest tests/unit/test_config.py` 全部通过；缺失必填项时抛 `ValueError`
- [x] A3 state.py：AgentState、Intent、子类型、to_strategy() — 单元测试覆盖 Intent 映射与 State 字段
  - 产物: `src/agent/state.py`, `tests/unit/test_state.py`
  - 验收: `pytest tests/unit/test_state.py` 全部通过；覆盖 5 种 intent→strategy 映射
- [x] A4 主图骨架与占位节点（空实现） — build_main_graph() 可 compile，invoke 单轮不报错
  - 产物: `src/agent/graph.py`, `src/agent/nodes/` 各占位文件, `src/agent/edges.py`
  - 验收: `pytest tests/unit/test_graph_skeleton.py` 通过；graph.compile() 成功且 invoke 返回 state
- [x] B1 MCP Client 连接 RAG Server（stdio） — get_tools() 返回 query_knowledge_hub 等，可 ainvoke
  - 产物: `src/tools/mcp_client.py`
  - 验收: `pytest tests/unit/test_mcp_client.py` 通过（Mock stdio 传输）
- [x] B2 RAGToolWrapper：search/list_collections 封装 — 单元测试 Mock MCP，断言参数与解析结果
  - 产物: `src/tools/tool_wrapper.py`, `tests/unit/test_tool_wrapper.py`
  - 验收: `pytest tests/unit/test_tool_wrapper.py` 全部通过；断言参数归一化与错误处理
- [x] B3 Simple 策略节点：检索 + 合成（Cloud LLM） — 给定 state.intent.reformulated_query，写入 draft_answer、retrieved_contexts
  - 产物: `src/agent/strategies/simple.py`, `src/prompts/strategies.py`（simple 部分）, `tests/unit/test_simple.py`
  - 验收: `pytest tests/unit/test_simple.py` 通过；Mock MCP+LLM，断言 draft_answer 非空且 retrieved_contexts 有值
- [x] B4 主图：load_memory → route → slot_fill → simple → critic → format_output — route/slot_fill 暂用 Cloud LLM 占位，critic 占位 pass；集成测试跑通 simple 路径
  - 产物: `src/agent/graph.py`（更新）, `src/agent/nodes/format_output.py`, `tests/integration/test_simple_path.py`
  - 验收: `pytest tests/integration/test_simple_path.py` 通过；从 START→END 完整跑通 simple 路径，final_answer 非空
- [x] C1 Router 节点：Cloud LLM 输出 type+confidence，写部分 Intent — 单元测试 Mock LLM，断言 intent.type 与 to_strategy()
  - 产物: `src/agent/nodes/router.py`, `src/prompts/router.py`, `tests/unit/test_router.py`
  - 验收: `pytest tests/unit/test_router.py` 通过；5 种 question 各返回正确 intent.type
- [x] C2 Slot Filling 节点与 SlotFillingOutput、Prompt — 单元测试 Mock LLM，断言 intent 被完整填充
  - 产物: `src/agent/nodes/slot_filling.py`, `src/prompts/slot_filling.py`, `tests/unit/test_slot_filling.py`
  - 验收: `pytest tests/unit/test_slot_filling.py` 通过；断言 entities/dimensions/reformulated_query 均非空
- [x] C3 route_by_intent 条件边，连接 slot_fill 与四策略 — 各 intent.type 正确进入对应策略节点
  - 产物: `src/agent/edges.py`（更新）, `tests/unit/test_edges.py`
  - 验收: `pytest tests/unit/test_edges.py` 通过；5 种 intent.type→正确策略分支
- [x] C4 Multi-hop：Plan-and-Execute 子图（plan/execute/replan/synthesize） — 单元/集成测试 Mock RAG+LLM，验证 replan 分支
  - 产物: `src/agent/strategies/multi_hop.py`, `tests/unit/test_multi_hop.py`
  - 验收: `pytest tests/unit/test_multi_hop.py` 通过；plan→execute→replan 循环可终止且 draft_answer 非空
- [x] C5 Comparative 节点：intent.entities/dimensions 优先，并行检索 — 单元测试断言并行调用与合成
  - 产物: `src/agent/strategies/comparative.py`, `tests/unit/test_comparative.py`
  - 验收: `pytest tests/unit/test_comparative.py` 通过；断言并行检索调用次数 = len(entities)
- [x] C6 Exploratory：ReAct 子图（think/act/observe/synthesize） — 单元测试断言步数上限与终止条件
  - 产物: `src/agent/strategies/exploratory.py`, `tests/unit/test_exploratory.py`
  - 验收: `pytest tests/unit/test_exploratory.py` 通过；断言 max_react_steps 后强制终止
- [x] D1 Critic 节点：Cloud LLM 输出 CriticVerdict — 单元测试 Mock LLM，断言 passed/score/feedback
  - 产物: `src/agent/nodes/critic.py`, `src/prompts/critic.py`, `tests/unit/test_critic.py`
  - 验收: `pytest tests/unit/test_critic.py` 通过；passed=True 时 score≥0.7；passed=False 时 feedback 非空
- [x] D2 critic_gate 条件边：pass → save_memory，retry → retry_refine — 集成测试 retry 路径 state.retry_count 递增
  - 产物: `src/agent/edges.py`（更新）, `tests/integration/test_retry_path.py`
  - 验收: `pytest tests/integration/test_retry_path.py` 通过；retry_count 从 0→1→2 后终止
- [x] D3 retry_refine 节点：根据 feedback 改写 question 或计划 — 单元测试断言 refinement 写入 state
  - 产物: `src/agent/nodes/retry_refine.py`, `tests/unit/test_retry_refine.py`
  - 验收: `pytest tests/unit/test_retry_refine.py` 通过；断言 state.question 被更新且含 feedback 关键词
- [x] D4 load_memory / save_memory 节点，长期记忆 recall/memorize 接口 — 单元测试 Mock 长期记忆，断言调用与 state.accumulated_facts
  - 产物: `src/agent/nodes/memory_nodes.py`, `src/memory/long_term.py`, `src/memory/short_term.py`, `tests/unit/test_memory.py`
  - 验收: `pytest tests/unit/test_memory.py` 通过；recall 返回相关事实；memorize 写入 JSONL
- [x] E1 AgentTrace 结构与 reasoning_steps 写入 — 单轮运行后 state.reasoning_trace 与 Trace 文件一致
  - 产物: `src/tracing/tracer.py`（AgentTrace 类）, `tests/unit/test_tracer.py`
  - 验收: `pytest tests/unit/test_tracer.py` 通过；trace 记录包含 node_name、duration、output
- [x] E2 JSONL Trace 落盘（tracer.py） — 可配置路径，单条记录包含 intent/strategy/critic/final_answer
  - 产物: `src/tracing/tracer.py`（更新 flush_to_jsonl）
  - 验收: `pytest tests/unit/test_tracer.py` 通过；JSONL 文件可被 `json.loads` 逐行解析
- [x] E3 Rich 流式输出（astream_events 或等价） — CLI 能实时打印节点名与关键输出
  - 产物: `src/tracing/rich_output.py`
  - 验收: 手动运行 `python main.py --question "test"` 观察节点流式输出（或 Mock 测试）
- [x] E4 main.py CLI：输入 question，输出 final_answer + 引用 — 可配置 MCP 与 LLM，本地一键运行
  - 产物: `main.py`（更新）
  - 验收: `python main.py --question "What is LoRA?" --dry-run` 输出 final_answer 与 sources
- [x] F1 LocalModelManager：Router 4-bit 加载与 classify_question — 无 GPU 时跳过或 Mock；有 GPU 时返回 (IntentType, confidence)
  - 产物: `src/models/loader.py`, `src/models/inference.py`, `tests/unit/test_local_model.py`
  - 验收: `pytest tests/unit/test_local_model.py` 通过（无 GPU 时标记 `@pytest.mark.skipif`）
- [x] F2 Router 节点优先调用本地模型，Fallback Cloud LLM — 集成测试可切换本地/Cloud
  - 产物: `src/agent/nodes/router.py`（更新 fallback 逻辑）
  - 验收: `pytest tests/unit/test_router.py -k fallback` 通过
- [x] F3 Critic 本地 DPO 模型加载与 evaluate — 同上，Fallback 逻辑
  - 产物: `src/models/inference.py`（更新 critic 推理）
  - 验收: `pytest tests/unit/test_local_model.py -k critic` 通过
- [x] F4 Critic 节点优先调用本地模型，Fallback Cloud LLM — 集成测试可切换
  - 产物: `src/agent/nodes/critic.py`（更新 fallback 逻辑）
  - 验收: `pytest tests/unit/test_critic.py -k fallback` 通过
- [x] G1 Router 训练数据合成（5 类，~800 条） — 格式符合 Alpaca/instruction-output，可加载
  - 产物: `training/data/generate_router_data.py`, `training/data/router_train.jsonl`
  - 验收: JSONL 可被 `datasets.load_dataset("json", ...)` 加载；5 类各 ≥100 条
- [x] G2 LoRA SFT 训练脚本（sft_router.py） — 可跑通 1 epoch，保存 adapter
  - 产物: `training/sft_router.py`
  - 验收: `python training/sft_router.py --max_steps 10` 不报错，输出 adapter 目录
- [x] G3 DPO 偏好对合成与 DPO 训练脚本 — 可跑通 1 epoch，保存模型
  - 产物: `training/data/generate_dpo_pairs.py`, `training/dpo_critic.py`
  - 验收: `python training/dpo_critic.py --max_steps 10` 不报错，输出模型目录
- [x] G4 量化导出与 LocalModelManager 集成 — 4-bit 加载，推理延迟与设计文档一致量级
  - 产物: `training/export_quantized.py`
  - 验收: 导出的 4-bit 模型可被 `LocalModelManager.load()` 加载
- [x] H1 集成测试覆盖：simple / comparative / multi_hop / exploratory 各一条路径 — 全部通过（可 Mock RAG）
  - 产物: `tests/integration/test_all_strategies.py`
  - 验收: `pytest tests/integration/test_all_strategies.py` 4 条路径全部通过
- [x] H2 E2E（可选）：真实 RAG Server + 测试集合，2～3 条问答 — 回答与引用合理
  - 产物: `tests/e2e/test_e2e.py`
  - 验收: `pytest tests/e2e/ -m e2e` 通过（需 RAG Server 运行）
- [x] H3 README：环境、配置、运行方式、与 RAG 的对接说明 — 新克隆可按文档跑通（2026-02-18：已补充完整 Quick Start 与 MCP 对接说明）
  - 产物: `README.md`
  - 验收: 按 README 步骤在干净环境可成功运行 demo
- [x] H4 面试 Demo 脚本（3～4 个问题对应 4 种模式） — 文档或脚本内注明每个问题的预期策略与亮点（2026-02-18：新增 `scripts/demo.py`，覆盖 4 个问题、预期策略与演示亮点）
  - 产物: `scripts/demo.py` 或 `docs/DEMO.md`
  - 验收: 4 个问题各触发对应策略，输出含 strategy 标签
- [x] H5 长期记忆集成/端到端 — 主图接入 LongTermMemory + create_load_memory_node/create_save_memory_node，多轮问答断言第二轮 accumulated_facts 或 JSONL 含第一轮写入事实（2026-02-18：新增 memory e2e 集成测试并完成主图接线）
  - 产物: `tests/integration/test_memory_e2e.py`（或扩展现有 integration）
  - 验收: pytest 通过；两轮问答后 recall 能返回首轮 memorize 的事实
- [ ] H6 CLI 冒烟自动化 — 用 pytest 或子进程执行 main.py --question "..." --dry-run（或测试 config），断言 exit 0 且 stdout/返回含 final_answer 与 sources
  - 产物: `tests/integration/test_cli_smoke.py` 或 `tests/e2e/test_cli_smoke.py`
  - 验收: 无需手动跑 main.py 即可在 CI 中回归 CLI 行为
- [ ] H7 Trace 集成 — 跑通 simple 路径后读取配置的 trace 文件，断言至少一条 JSONL 记录且含 intent/strategy/critic/final_answer
  - 产物: `tests/integration/test_trace_integration.py`
  - 验收: pytest 通过；trace 落盘格式与字段符合预期
- [ ] H8 配置驱动建图集成（可选） — 加载测试用 settings 或 fixture yaml，build_main_graph 后 invoke 一次，断言 state 含 final_answer 或预期键
  - 产物: `tests/integration/test_config_graph.py`
  - 验收: 配置或建图方式变更时自动化发现断裂
- [ ] H9 MCP 子进程集成（可选） — 测试内启动 RAG Server 子进程或最小 stub，MCP Client 建连后调用 query_knowledge_hub 一次，断言返回可解析；可标 @pytest.mark.slow
  - 产物: `tests/integration/test_mcp_subprocess.py`
  - 验收: 不依赖人工启动 RAG 即可验证 MCP 连接与工具调用
- [ ] H10 Checkpointer 多会话（可选） — 同一 thread_id 连续 invoke 两次断言状态累积，或两 thread_id 各 invoke 一次断言隔离；可在实现多轮对话时补充
  - 产物: `tests/integration/test_checkpointer_sessions.py`
  - 验收: 短期记忆/会话隔离行为符合预期

---

### 总体进度（模板）

| 阶段 | 总任务数 | 已完成 | 进度 |
|------|---------|--------|------|
| A | 4 | 0 | 0% |
| B | 4 | 0 | 0% |
| C | 6 | 0 | 0% |
| D | 4 | 0 | 0% |
| E | 4 | 0 | 0% |
| F | 4 | 0 | 0% |
| G | 4 | 0 | 0% |
| H | 10 | 0 | 0% |
| **总计** | **40** | **0** | **0%** |

---

## 附录：与设计文档的对应关系

| DEV_SPEC 章节 | 设计文档章节 |
|---------------|-------------|
| §1 项目概述 | §1 项目定位与核心价值 |
| §2 核心特点 | §1.3 技术亮点、§2 整体架构、§5.4 推理模式矩阵 |
| §3 技术选型 | §3 技术选型、§7 MCP Client |
| §4 测试方案 | 设计文档未单独成章，本 DEV_SPEC 补充 |
| §5 架构与模块 | §4 State、§5 LangGraph、§6 各节点、§11 项目结构 |
| §6 排期 | §12 实施路线图（本处拆分为可勾选任务表） |

完成每个子任务后，请将对应行的状态更新为 `[x]` 并填写完成日期与备注，便于追踪总体进度。
