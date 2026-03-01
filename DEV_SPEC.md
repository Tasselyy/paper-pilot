# PaperPilot Agent — Developer Specification (DEV_SPEC)

> 版本：0.1 — 开发规格说明  
> 详细架构与数据模型见：[PAPER_PILOT_DESIGN.md](./docs/PAPER_PILOT_DESIGN.md)

## 目录

- 项目概述
- 核心特点
- 技术选型
- 测试方案
- 系统架构与模块设计（含短期记忆与 Follow-up 实现方案）
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
| （跨策略） | Reflexion | Critic 评估 + 带反馈重试（最多 2 次）；重试用尽时可选 **Web Search 兜底**（见阶段 K） |

### 模型分层

- **Cloud LLM**：主推理、问题分解、槽位填充、各策略合成、ReAct think/observe、Reflexion 重试时的改进计划。
- **本地模型（4-bit 量化）**：Router（LoRA SFT）、Critic（DPO），用于分类与质量评估，降低延迟与 API 成本。

### 记忆与可观测性

- **短期记忆**：LangGraph Checkpointer（如 MemorySaver），对话消息与当前会话状态。短期记忆的**真正用法**分四层：
  - **消息积累**：`state.messages` 用 `add_messages` reducer 追加；`format_output_node` 每轮写回 `AIMessage(content=final_answer)`，调用方每轮传入 `HumanMessage(content=question)`，使消息历史在同一 thread_id 下自动完整积累（`[HumanMessage(Q1), AIMessage(A1), HumanMessage(Q2), ...]`）。
  - **历史格式化 helper**：共享工具函数 `format_conversation_history(messages, max_turns=3)` 把最近 N 轮消息转为可读文本摘要（`User: …\nAssistant: …`），供 Router、Slot Filling、synthesis 节点复用；截断单条消息至合理长度以控制 token 开销。
  - **Router 感知历史**：Router 的 user prompt 注入最近 N 轮历史摘要，使 `follow_up` 能被正确识别（如「那和 full fine-tuning 比呢」，没有历史时无法分类）；仅 Cloud LLM 路径注入，本地 LoRA Router 保持原接口不变（单问题分类）。
  - **Slot Filling 消解指代**：当 `intent.type == "follow_up"` 或历史非空时，把历史摘要注入 Slot Filling user prompt，使 `reformulated_query` 能正确展开指代（「那」→「LoRA」），从而提升 RAG 检索质量；普通 factual/comparative 请求不注入，避免增加 token 成本。
  - **synthesis 带历史（follow_up）**：各策略节点的 synthesis 步骤在 `follow_up` 意图下将最近 N 轮 Q&A 注入 LLM prompt，保持多轮回答连贯、避免重复解释基础概念；其他意图不注入。
- **长期记忆**：自定义事实累积（如 JSONL + 关键词/embedding 检索），在 load_memory / save_memory 节点读写；**事实的利用**见下条。
- **长期记忆事实的利用（Fact Usage）**：
  - **写入与召回**：load_memory 用当前问题 recall 相关事实，写入 `state.accumulated_facts`；save_memory 在 Critic 通过后从本轮 Q&A 抽取并 memorize 新事实。
  - **使用位置**：仅在**回答合成（synthesis）**阶段使用 fact，不参与路由或槽填充。
  - **使用方式**：各策略（simple、multi_hop、comparative、exploratory）在合成 draft_answer 时，将 `accumulated_facts` 作为 **「Prior knowledge from past conversations」** 单独段落注入 LLM prompt，与 Retrieved contexts 明确区分。
  - **使用规则**（在 prompt 中写明）：回答**主要依据** Retrieved contexts 并尽量引用文献；Prior knowledge 仅用于补充背景、保持与以往说法一致或在不与检索冲突时略作延伸；**不得**将 Prior knowledge 当作论文/文档引用；若与检索内容矛盾，以检索为准。
  - **策略差异**：simple / comparative 在唯一一次 synthesis 时注入；multi_hop / exploratory 仅在**最终 synthesis** 注入，不在 plan/execute 步骤使用；follow_up 意图下可对 recall 使用略大的 top_k，以便上一轮相关结论更多进入合成。
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
| **Critic / Retry** | Mock Critic 返回 pass/fail，断言 retry 时 state.retry_count 递增、重试后再次进入 critic；critic_gate 在 retry 用尽且启用 Web 时返回 web_fallback（阶段 K） |
| **Memory** | load_memory 写入 accumulated_facts，save_memory 调用长期记忆写入（Mock 存储）；各策略 synthesis 在 state 含 accumulated_facts 时，传给 LLM 的 prompt 包含「Prior knowledge」段（见 I0） |

### 4.3 集成测试重点

- **主图 + Fake MCP**：从 START 到 END，Fake 实现 `query_knowledge_hub` 等返回固定内容，验证 simple / comparative 两条路径的端到端 state 与 final_answer。
- **MCP Client 集成**：在测试环境中启动 RAG Server 子进程或 Mock Server，验证 Agent 能正确发现并调用 tools、解析返回格式。
- **长期记忆集成**：主图接入真实 LongTermMemory，多轮问答下断言第二轮能 recall 第一轮写入的事实，或 JSONL 中有对应记录。
- **Trace 集成**：跑通一条路径后读取 trace 文件，断言 JSONL 记录包含 intent/strategy/critic/final_answer 等字段，保证可观测链路可用。
- **配置驱动建图**：加载测试用 config → 构建 graph（可 Mock RAG/LLM）→ invoke 一次，断言 state 含预期键，验证配置与建图链不被破坏。
- **MCP 子进程集成（可选）**：测试内启动 RAG 子进程或最小 stub → MCP Client 建连 → 调用一次 query_knowledge_hub → 断言返回可解析；可标 `@pytest.mark.slow` 或可选执行。
- **Checkpointer / 多会话（可选）**：同一 thread_id 多轮 invoke 断言状态累积，不同 thread_id 断言隔离；在实现多轮对话或会话隔离时补充。
- **短期记忆多轮集成**（阶段 J 补充）：
  - **消息积累**：同一 thread_id 连续 invoke 两次（Q1/A1 → Q2），断言第二轮运行时 `state.messages` 包含第一轮的 `HumanMessage(Q1)` 与 `AIMessage(A1)`；验证 `format_output_node` 写回 AIMessage 的机制。
  - **follow_up 识别**：第一轮问「什么是 LoRA？」，第二轮问「那和 full fine-tuning 比呢」，断言 Router 在读取历史后将第二轮分类为 `follow_up` 或 `comparative`（而非 `factual`）。
  - **指代消解**：在 `follow_up` 场景下，断言 Slot Filling 输出的 `reformulated_query` 中「那/it/that」已被展开为完整实体名（如「LoRA」），而不是原样保留指代词。
  - **synthesis 连贯性**：follow_up 意图下，synthesis 的 LLM messages 包含对话历史摘要段落；可 Mock LLM 捕获 messages 列表并断言其中含历史内容。
  - **会话隔离**：两个不同 thread_id 各 invoke 一次，断言互不干扰（第二个 thread 的 state.messages 不含第一个 thread 的消息）。

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
     → pass         → save_memory → format_output → END
     → retry         → retry_refine → 回到 route
     → web_fallback  → web_search_fallback（可选，重试用尽且配置启用）→ save_memory → format_output → END
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
│   │   │   ├── retry_refine.py   # Reflexion 重试：改写 question/计划
│   │   │   ├── web_search_fallback.py  # Web Search 兜底（阶段 K，可选）
│   │   │   └── format_output.py
│   │   ├── strategies/
│   │   │   ├── simple.py
│   │   │   ├── multi_hop.py      # Plan-and-Execute 子图
│   │   │   ├── comparative.py    # ReWOO
│   │   │   └── exploratory.py    # ReAct 子图
│   │   └── edges.py              # route_by_intent, critic_gate 等
│   ├── tools/
│   │   ├── mcp_client.py         # MCP Client 封装
│   │   ├── tool_wrapper.py       # RAGToolWrapper
│   │   └── web_search.py         # Web Search 客户端（阶段 K，可选）
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
│   │   ├── retry_refine.py
│   │   ├── web_search_fallback.py  # 阶段 K
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
| **agent/nodes/router.py** | LoRA 分类，输出部分 Intent；Fallback Cloud LLM；Cloud LLM 路径下注入 `format_conversation_history(state.messages)` 以正确识别 `follow_up` | §6.1 |
| **agent/nodes/slot_filling.py** | LLM 槽位填充，输出完整 Intent；`intent.type == follow_up` 时注入历史摘要以消解指代、改写完整 `reformulated_query` | §6.2 |
| **agent/strategies/simple.py** | 使用 intent.reformulated_query 单次检索 + 合成；synthesis 时注入 state.accumulated_facts 作为 Prior knowledge；`follow_up` 时追加对话历史段落 | §6.3 |
| **agent/strategies/multi_hop.py** | Plan-and-Execute 子图：plan / execute_step / replan / pe_synthesize；仅在最终 synthesis 注入 accumulated_facts；`follow_up` 时最终 synthesis 追加历史 | §5.5, §6.4 |
| **agent/strategies/comparative.py** | ReWOO：优先 intent.entities/dimensions，并行检索 + 合成；synthesis 时注入 accumulated_facts | §6.5 |
| **agent/strategies/exploratory.py** | ReAct 子图：think / act / observe / synthesize；仅在最终 synthesis 注入 accumulated_facts | §5.6, §6.6 |
| **agent/nodes/critic.py** | Critic 评估（本地 DPO 或 Cloud LLM），写 critic_verdict | §6.7 |
| **agent/nodes/retry_refine.py** | Reflexion 重试：根据 feedback 改写 question 或计划 | §6.8 |
| **agent/nodes/web_search_fallback.py** | Web Search 兜底：retry 用尽时用开放网络检索重写 draft_answer，写 used_web_fallback；详见 [WEB_SEARCH_FALLBACK_DESIGN.md](./docs/WEB_SEARCH_FALLBACK_DESIGN.md) | 阶段 K |
| **agent/nodes/format_output.py** | 格式化最终回答；写回 `AIMessage(content=final_answer)` 到 `state.messages`；Sources 段支持 Web 来源 URL（ctx.url） | §5 |
| **tools/mcp_client.py** | MultiServerMCPClient 连接 RAG Server，get_tools() | §7 |
| **tools/tool_wrapper.py** | RAGToolWrapper：search / list_collections / get_doc_info，错误处理 | §7.2 |
| **tools/web_search.py** | Web Search 客户端（Tavily/Serper 等）；仅 retry 用尽兜底时使用 | 阶段 K |
| **memory/long_term.py** | 长期记忆 recall / memorize，JSONL 持久化 | §8 |
| **memory/conversation.py** | `format_conversation_history(messages, max_turns)` 共享工具函数，把 `state.messages` 转为 Router/Slot Filling/synthesis 可用的文本摘要；控制 token 开销 | 阶段 J |
| **prompts/strategies.py** | 各策略 synthesis 模板；含「Prior knowledge」段及占位，约定 RAG 为主、fact 为辅；`follow_up` 时含「Conversation history」段 | §6.3–6.6 |
| **tracing/tracer.py** | AgentTrace 结构与 JSONL 记录 | §10 |

### 5.4 数据流摘要

- **输入**：调用方每轮传入 `question`（原始问题字符串）和 `messages: [HumanMessage(content=question)]`；同一 thread_id 下 LangGraph 自动加载上一轮 checkpoint 并按 reducer 合并，使本轮初始 state 含完整历史。
- **消息积累**：`state.messages` 通过 `add_messages` reducer 在每轮追加；`format_output_node` 每轮写回 `AIMessage(content=final_answer)`，使 checkpoint 里的 messages 逐轮完整积累（`[HumanMessage(Q1), AIMessage(A1), HumanMessage(Q2), …]`）。
- **记忆层**：
  - 短期：`state.messages` 通过 checkpointer 跨轮持久化；Router/Slot Filling/synthesis 按需读取并通过 `format_conversation_history()` 格式化为文本。
  - 长期：load_memory 根据 question 做 recall，将相关事实写入 `state.accumulated_facts`；各策略在**合成回答**时将该列表作为「Prior knowledge」注入 synthesis prompt，与 RAG 检索结果共同参与生成；save_memory 在 pass 后从本轮 Q&A 抽取并持久化新事实。
- **意图层**：route（Cloud LLM 路径注入历史摘要）写入 intent（部分）→ slot_fill（`follow_up` 时注入历史摘要消解指代）写入 intent（完整，含展开后的 `reformulated_query`）。
- **策略层**：各策略读写 state.retrieved_contexts、retrieval_queries、sub_questions、draft_answer、reasoning_trace；synthesis 步骤统一使用 state.accumulated_facts 作为 prior knowledge 段落；`follow_up` 意图下额外注入对话历史段落。
- **质量层**：critic 写入 critic_verdict；critic_gate 在 retry 用尽且启用 Web 时可路由至 web_search_fallback；retry_refine 可能更新 question、retry_count；web_search_fallback 写入 draft_answer、retrieved_contexts（含 url）、used_web_fallback。
- **输出**：final_answer（同时写回 AIMessage 到 messages）、reasoning_trace、sources（引用）；Trace 落盘与流式展示。

### 5.5 配置项要点

- **llm**：cloud 端点、模型名、api_key 等。
- **mcp**：connections（rag_server 的 command/args 或 url、transport）。
- **agent**：max_retries、max_react_steps、router_model_path、critic_model_path（可选）；**web_search_fallback**（可选）：enabled、provider（如 tavily）、api_key、max_results，详见 [WEB_SEARCH_FALLBACK_DESIGN.md §4](./docs/WEB_SEARCH_FALLBACK_DESIGN.md)。
- **memory**：memory_file（长期记忆路径）；可选 recall_top_k（默认 5），follow_up 时可配置略大的 top_k 以多召回上一轮相关事实。
- **tracing**：trace_dir、trace_file。

### 5.6 短期记忆与 Follow-up 实现方案

> 本节对应「阶段 J」的详细分析与设计依据；任务列表见 §6 阶段 J（J1–J6）。

#### 现状与问题诊断

| 缺口 | 表现 | 影响 |
|------|------|------|
| messages 从不写入 | 无节点在 return 中写 `messages`；format_output 只写 final_answer。 | 历史无法积累，同一 thread_id 下 checkpoint 的 messages 仍为空或仅本轮一条。 |
| 助手回复未回写 | 图只产出 final_answer，不产出 AIMessage。 | 下一轮 Router/Slot Filling 看不到「上一轮答了什么」，无法判断是否延续上一轮。 |
| Router 无上下文 | 只读 state.question，prompt 仅 `Question: {question}`。 | 「那和 full fine-tuning 比呢」无前文，无法可靠识别为 follow_up。 |
| Slot Filling 无上下文 | 只读 question 与 intent，模板无历史。 | 指代（它/那）无法消解，reformulated_query 不完整，RAG 检索质量差。 |
| 调用方每次新会话 | main 每次新 thread_id，且只传 question 不传 messages。 | checkpoint 从未被复用；多轮等同于多个独立单轮。 |

结论：类型、state 与 checkpointer 已支持多轮，但缺「谁写 messages、谁读、怎么用」，故当前**无法可靠判断用户何时在 follow-up 提问**。

#### 设计目标与约束

- **目标**：同一 thread 内能识别 follow_up、Slot Filling 消解指代、follow_up 时 synthesis 回答连贯；首轮或未传 history 时行为与单轮一致。
- **约束**：历史以「最近 N 轮 + 单条截断」注入，token 可控；本地 Router 接口不变，历史仅注入 Cloud LLM 路径；Slot Filling / synthesis 仅在「有历史」或 intent==follow_up 时注入；不传 `--session` 时保持现有单轮行为。

#### 实施顺序与依赖

| 步骤 | 任务 | 依赖 | 说明 |
|------|------|------|------|
| 1 | J1：format_output 写回 AIMessage | 无 | 助手回复进入 state，否则后续「读历史」无内容。 |
| 2 | J2：format_conversation_history 工具 | 无 | Router/Slot Filling/J6 均依赖，优先实现并单测。 |
| 3 | J3：调用方传 HumanMessage + --session | 无 | 与 J1 共同保证同一 thread 内消息完整积累。 |
| 4 | J4：Router Cloud 路径注入历史 | J2 | 实现 follow_up 识别的关键。 |
| 5 | J5：Slot Filling 条件注入 + 指代消解规则 | J2 | 实现指代消解与检索质量提升。 |
| 6 | J6：follow_up 时 synthesis 带历史 | J2 | 提升回答连贯性，依赖 J4 产出 follow_up。 |

建议严格按 1→2→3→4→5→6 执行：1+2+3 解决「存」与工具，4+5 解决「用」与 follow-up 判断/改写，6 再增强连贯性。

#### 风险与取舍

- **本地 Router 暂不喂历史**：避免改接口与训练数据；follow_up 识别先由 Cloud 路径承担。
- **历史长度与截断**：max_turns=3、max_chars_per_msg=200 为折中，可做成配置项后续调优。
- **误判 follow_up**：可结合 confidence 阈值或「仅当历史非空才允许 follow_up」等规则做保守策略。
- **多轮 retry**：retry_refine 更新 question 后回到 route，messages 中已有本轮 HumanMessage，历史仍在，无需特殊处理。

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
| **I：记忆模块优化** | 语义向量检索、事实去重、低质量答案过滤、启发式兜底改进、记忆上限与淘汰、短期记忆持久化 |
| **J：短期记忆真正用上** | 消息积累（format_output 写回 AIMessage、调用方传 messages+session）、format_conversation_history、Router/Slot Filling 带历史识别 follow_up 与指代消解、synthesis 带历史；详见 §5.6 |
| **K：Web Search 兜底** | Critic 不通过且 retry 用尽时可选分支：Web 检索 + 重写回答；配置开关、Tavily/Serper 客户端、used_web_fallback 状态、save_memory 与 Trace 配合；详见 [WEB_SEARCH_FALLBACK_DESIGN.md](./docs/WEB_SEARCH_FALLBACK_DESIGN.md) |

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
- [x] H6 CLI 冒烟自动化 — 用 pytest 或子进程执行 main.py --question "..." --dry-run（或测试 config），断言 exit 0 且 stdout/返回含 final_answer 与 sources（2026-02-18：新增 `tests/integration/test_cli_smoke.py`，覆盖 subprocess CLI 与 run_agent 返回值冒烟）
  - 产物: `tests/integration/test_cli_smoke.py` 或 `tests/e2e/test_cli_smoke.py`
  - 验收: 无需手动跑 main.py 即可在 CI 中回归 CLI 行为
- [x] H7 Trace 集成 — 跑通 simple 路径后读取配置的 trace 文件，断言至少一条 JSONL 记录且含 intent/strategy/critic/final_answer（2026-02-18：新增 `tests/integration/test_trace_integration.py`，断言 trace JSONL 至少一条且含 intent/strategy_executed/critic_verdict/final_answer）
  - 产物: `tests/integration/test_trace_integration.py`
  - 验收: pytest 通过；trace 落盘格式与字段符合预期
- [x] H8 配置驱动建图集成（可选） — 加载测试用 settings 或 fixture yaml，build_main_graph 后 invoke 一次，断言 state 含 final_answer 或预期键
  - 产物: `tests/integration/test_config_graph.py`
  - 验收: 配置或建图方式变更时自动化发现断裂
- [x] H9 MCP 子进程集成（可选） — 测试内启动 RAG Server 子进程或最小 stub，MCP Client 建连后调用 query_knowledge_hub 一次，断言返回可解析；可标 @pytest.mark.slow
  - 产物: `tests/integration/test_mcp_subprocess.py`
  - 验收: 不依赖人工启动 RAG 即可验证 MCP 连接与工具调用
- [x] H10 Checkpointer 多会话（可选） — 同一 thread_id 连续 invoke 两次断言状态累积，或两 thread_id 各 invoke 一次断言隔离；多轮对话的详细完整实现见**阶段 J**，本任务可作为阶段 J 集成测试的收口
  - 产物: `tests/integration/test_checkpointer_sessions.py`
  - 验收: 短期记忆/会话隔离行为符合预期；阶段 J 完成后可扩展此测试文件覆盖 follow_up 识别、指代消解等多轮场景

### 阶段 I：记忆模块优化

> 对 D4 已实现的记忆模块进行质量与健壮性提升，并完成「事实在合成阶段的利用」设计；各任务相互独立，可按优先级单独交付。

- [ ] I0 长期记忆事实在合成阶段的注入 — 在各策略的**回答合成（synthesis）**步骤中，将 `state.accumulated_facts` 以「Prior knowledge from past conversations」段落注入 LLM prompt；prompt 中明确：回答主要依据 Retrieved contexts，Prior knowledge 仅用于补充与连贯，不得将 prior knowledge 当作文献引用，与检索冲突时以检索为准；follow_up 意图下 load_memory 可选用略大的 top_k（配置或写死）。产物含各策略 synthesis 模板占位与调用处传入 accumulated_facts；单元测试断言有 accumulated_facts 时 prompt 包含该段。
  - 产物: `src/prompts/strategies.py`（各 synthesis 模板增加 prior_knowledge 段及使用说明）, `src/agent/strategies/simple.py`, `src/agent/strategies/multi_hop.py`, `src/agent/strategies/comparative.py`, `src/agent/strategies/exploratory.py`（synthesis 处传入并格式化 accumulated_facts）, 可选 `src/agent/nodes/memory_nodes.py` 或 config（follow_up 时 top_k）
  - 验收: 各策略 synthesis 收到非空 accumulated_facts 时，传给 LLM 的 prompt 包含 Prior knowledge 段；pytest 通过；多轮场景下第二轮回答可延续首轮事实（可选 E2E 断言）
- [ ] I1 语义向量检索替代关键词匹配 — 用 embedding（如 sentence-transformers）替换 `LongTermMemory.recall` 的关键词重叠评分；引入向量索引（faiss 或 chromadb），保持 `recall(question, top_k)` 接口不变
  - 产物: `src/memory/long_term.py`（更新 recall 实现），可选 `src/memory/vector_store.py`，`tests/unit/test_long_term_memory.py`（更新断言）
  - 验收: 同义词/缩写场景（如 "self-attention" vs "Transformer attention"）能被正确召回；关键词不重叠时不返回零结果；pytest 通过
- [ ] I2 强制 pass 后跳过 save_memory — 在 `run_save_memory` 中检查 `state.critic_verdict`；若 `passed=False`（重试耗尽被强制放行），跳过 memorize 并在 reasoning_trace 中记录跳过原因，避免低质量事实污染长期记忆
  - 产物: `src/agent/nodes/memory_nodes.py`（更新 `run_save_memory`），`tests/unit/test_memory.py`（补充强制 pass 场景）
  - 验收: Mock critic_verdict.passed=False + retry_count≥max_retries，断言 memorize 未被调用；正常 passed=True 时行为不变
- [ ] I3 事实去重与 upsert — 存入前先 recall，若新事实与现有事实语义相似度超过阈值（如余弦 ≥0.92），跳过存储或替换旧事实（upsert），而非盲目追加
  - 产物: `src/memory/long_term.py`（更新 `memorize`），`tests/unit/test_long_term_memory.py`（补充重复事实场景）
  - 验收: 写入内容相近的两条事实后，存储总数不翻倍；不同事实正常追加；pytest 通过
- [ ] I4 改进启发式兜底策略 — 将 `_heuristic_extract` 从"取第一句 >20 字符"改为综合评分（句子长度 + 是否含数字/专有名词/关键术语），返回评分最高的 1～2 句，而非固定取首句
  - 产物: `src/agent/nodes/memory_nodes.py`（更新 `_heuristic_extract`），`tests/unit/test_memory.py`（补充多种 answer 格式的断言）
  - 验收: 铺垫句靠后、核心结论居中的 answer 能正确提取核心句；pytest 通过
- [ ] I5 记忆上限与淘汰策略 — 在 `LongTermMemory` 中新增 `max_facts` 配置项（默认 1000）；超限时按 LRU（最旧时间戳优先淘汰）或 LFU（`Fact` 模型新增 `recall_count` 字段，按召回次数淘汰）删除事实，并同步重写 JSONL 文件
  - 产物: `src/memory/long_term.py`（更新 `memorize` 与 `Fact` 模型），`src/config.py`（`MemoryConfig` 增加 `max_facts` 字段），`tests/unit/test_long_term_memory.py`
  - 验收: 写入超过 max_facts 条事实后，总数不超限；JSONL 内容与内存列表一致；pytest 通过
- [ ] I6 短期记忆持久化（SQLite backend） — 在 `create_checkpointer` 中新增 `"sqlite"` backend，使用 LangGraph 内置 `SqliteSaver`；路径通过 `MemoryConfig` 配置；进程重启后对话历史可恢复
  - 产物: `src/memory/short_term.py`（更新 `create_checkpointer`），`src/config.py`（`MemoryConfig` 增加 `checkpointer_backend` 与 `checkpointer_db` 字段），`tests/unit/test_short_term.py`
  - 验收: backend="sqlite" 时返回 `SqliteSaver` 实例；重建 checkpointer 后能从 DB 恢复已有 checkpoint；pytest 通过

---

### 阶段 J：短期记忆真正用上（多轮对话）

> **目标**：让「同一 thread_id 多轮 invoke」真正发挥作用。当前问题：① messages 从不积累（无节点写回 AIMessage）；② Router/Slot Filling 只用 `state.question`，无法识别 follow_up 或消解指代；③ 调用方每次新建 thread_id，checkpoint 形同虚设。本阶段四层递进：**消息积累 → 历史工具 → Router/Slot Filling 感知历史 → synthesis 连贯 → CLI session**。
>
> **设计约束**：
> - `format_conversation_history(messages, max_turns=3)` 为核心工具函数，截断单条至 200 字，控制 token；最多取最近 N 轮（2N 条消息）。
> - Router 本地 LoRA 分类器接口不变（单 question 输入），历史注入仅在 Cloud LLM 路径；本地路径如分类为 `factual` 而历史非空，节点记录 warning 日志但不强制覆盖。
> - Slot Filling 仅在 `intent.type == follow_up` **或** `len(state.messages) > 2`（有历史）时注入历史段，普通 factual/comparative 首轮无额外开销。
> - synthesis 历史注入仅在 `follow_up` 意图下，不影响其他意图的 prompt 结构与现有测试。
> - 调用方（main.py）的 `--session` 参数可选，不传时行为与现在一致（每次新 thread_id，单轮模式）。

- [ ] J1 format_output_node 写回 AIMessage — 在 `format_output_node` 的 return dict 里追加 `"messages": [AIMessage(content=final)]`；AIMessage 通过 `add_messages` reducer 自动追加入 state，不覆盖已有消息。
  - 产物: `src/agent/nodes/format_output.py`（更新 return，增加 `messages` 键），`tests/unit/test_format_output.py`（补充断言 messages 含 AIMessage）
  - 验收: 单元测试断言 format_output_node 返回的 dict 中 `messages` 包含 `AIMessage`，内容与 `final_answer` 一致；现有 test 不破坏

- [ ] J2 共享历史工具函数 `format_conversation_history` — 新建 `src/memory/conversation.py`，实现 `format_conversation_history(messages: list[BaseMessage], max_turns: int = 3, max_chars_per_msg: int = 200) -> str`；空历史返回空字符串；单元测试覆盖 0/1/多轮及截断场景。
  - 产物: `src/memory/conversation.py`，`tests/unit/test_conversation_history.py`
  - 验收: 0 条消息返回 `""`；1 轮（1 Human + 1 AI）返回 `"User: …\nAssistant: …"`；超过 max_turns 时只保留最近 N 轮；单条超过 max_chars_per_msg 时末尾截断；pytest 全部通过

- [ ] J3 调用方每轮传 HumanMessage — `main.py` 的 `stream_graph` 调用处，在 `input_state` 中加入 `"messages": [HumanMessage(content=question)]`；`--session <id>` 参数可选，传入则复用指定 thread_id，不传则维持现有行为（新 thread_id，单轮）；`graph_config` 中使用该 thread_id。
  - 产物: `main.py`（更新 `input_state` 构造与 `--session` argparse 参数）
  - 验收: `--session abc` 时 thread_id 固定为 `abc`；两次以相同 session 运行后，第二次日志中的 state.messages 含第一次的 HumanMessage（可在 rich_output 或 tracer 的 verbose 模式下观察）；不传 `--session` 时行为不变

- [ ] J4 Router 带历史分类 follow_up — 在 `run_router` 的 Cloud LLM 分支中，用 `format_conversation_history(state.messages)` 生成历史摘要，非空时将其作为额外段落注入 `ROUTER_USER_TEMPLATE`（更新模板，增加可选 `{history}` 占位；历史为空时占位为空字符串，prompt 无多余换行）；本地 LoRA 分类器路径不变。
  - 产物: `src/prompts/router.py`（`ROUTER_USER_TEMPLATE` 增加 `{history}` 段），`src/agent/nodes/router.py`（Cloud LLM 路径构造 history 并格式化 prompt），`tests/unit/test_router.py`（补充：有历史且问题含指代时，Mock LLM 返回 `follow_up`，断言 intent.type）
  - 验收: 历史为空时 prompt 与现在结构一致（不含 history 段）；历史非空时 prompt 含最近 N 轮摘要；Mock LLM 断言：「什么是 LoRA？」→ AIMessage(LoRA 解释) → 「那和 full fine-tuning 比呢」被分类为 `follow_up`；pytest 全部通过

- [ ] J5 Slot Filling 利用历史消解指代 — 在 `run_slot_filling` 中，当 `intent.type == "follow_up"` 或 `len(state.messages) > 2` 时，用 `format_conversation_history(state.messages)` 生成历史摘要，注入 `SLOT_FILLING_USER_TEMPLATE`（增加可选 `{history}` 占位，历史为空时为空字符串）；`reformulated_query` 的 system prompt 规则补充：「对 follow_up 问题，必须将指代词（那/it/that/这/其）替换为前轮实际实体名，使 reformulated_query 自包含且可独立用于检索」。
  - 产物: `src/prompts/slot_filling.py`（system prompt 补充 follow_up 解指代规则；`SLOT_FILLING_USER_TEMPLATE` 增加 `{history}` 段），`src/agent/nodes/slot_filling.py`（构造 history 并注入），`tests/unit/test_slot_filling.py`（补充 follow_up 场景：有历史时断言 `reformulated_query` 不含「那/it」等指代词）
  - 验收: `intent.type != follow_up` 且 messages ≤ 2 条时，prompt 不含 history 段；`follow_up` 时 `reformulated_query` 不含裸指代词（通过 `assertNotIn("那", ...)` 或正则检查）；pytest 通过

- [ ] J6 follow_up synthesis 带对话历史 — 在 `simple`（及 `multi_hop` 最终 synthesis）策略的 synthesis 步骤中，当 `state.intent.type == "follow_up"` 时，将 `format_conversation_history(state.messages)` 作为「Conversation history」段落追加到 synthesis user prompt（在 Retrieved contexts 之前，Prior knowledge 之后）；prompt 段落顺序：`Conversation history → Prior knowledge → Retrieved contexts → Question`；其他意图不注入，不影响现有 prompt 结构。
  - 产物: `src/prompts/strategies.py`（simple/multi_hop 模板增加可选 `{conversation_history}` 占位），`src/agent/strategies/simple.py`（follow_up 时传入 history），`src/agent/strategies/multi_hop.py`（最终 synthesis 时传入 history），`tests/unit/test_simple.py`（补充 follow_up 场景：Mock LLM 捕获 messages，断言含 Conversation history 段）
  - 验收: `follow_up` 意图时 synthesis 的 user prompt 含「Conversation history」段且内容来自 state.messages；其他意图 prompt 不含该段；现有 test 不破坏；pytest 通过

---

### 阶段 K：Web Search 兜底

> **目标**：当 Critic 不通过且 retry 已达上限时，在「强制 pass」之外增加可选分支：使用开放网络搜索重写回答并标注 Web 来源。主检索仍仅用 MCP RAG；Web 仅作兜底。详细设计见 [WEB_SEARCH_FALLBACK_DESIGN.md](./docs/WEB_SEARCH_FALLBACK_DESIGN.md)。

- [ ] K1 配置模型与加载 — `AgentConfig` 增加 `web_search_fallback`（enabled、provider、api_key、max_results）；`load_settings` 解析；无 key 时 client 不创建
  - 产物: `src/config.py`（WebSearchFallbackConfig / AgentConfig 扩展）, `config/settings.yaml` 示例
  - 验收: 单元测试加载 yaml 得到 enabled/provider/max_results；无 api_key 时工厂返回 None

- [ ] K2 State 与 RetrievedContext 扩展 — `RetrievedContext` 增加可选 `url: str | None`；`AgentState` 增加 `used_web_fallback: bool = False`
  - 产物: `src/agent/state.py`
  - 验收: 单元测试序列化/反序列化；旧 trace 无 url 仍兼容

- [ ] K3 Web Search 客户端 — `WebSearchResult`、`WebSearchClient` 接口、`TavilyWebSearchClient` 及工厂 `create_web_search_client(config)`
  - 产物: `src/tools/web_search.py`
  - 验收: 单元测试 Mock HTTP 或 Tavily，断言 search() 返回列表且字段完整

- [ ] K4 web_search_fallback 节点 — 取 query → 调用 client → 转 RetrievedContext（含 url）→ LLM 合成 → 写 state（draft_answer、retrieved_contexts、used_web_fallback、reasoning_trace）
  - 产物: `src/agent/nodes/web_search_fallback.py`, `src/prompts/web_search_fallback.py`
  - 验收: 单元测试 Mock client + LLM，断言 state 更新正确且 used_web_fallback=True

- [ ] K5 critic_gate 增加 web_fallback 分支 — 当 retry_count >= max_retries 且 verdict.passed=False 且 web_search 启用且可用时返回 "web_fallback"，否则 "pass"
  - 产物: `src/agent/edges.py`
  - 验收: 单元测试：retry 用尽 + enabled + 有 key → web_fallback；未启用或无 key → pass

- [ ] K6 主图注册 web_search_fallback — 仅当配置启用且 client 创建成功时注册节点与 critic 条件边；web_search_fallback → save_memory
  - 产物: `src/agent/graph.py`
  - 验收: 集成测试：启用 Web + Mock client，critic 不通过且 retry 用尽后进入 web_fallback → save_memory → format_output

- [ ] K7 save_memory 与 used_web_fallback — 当 used_web_fallback=True 时执行 memorize；当强制 pass（未用 Web）且 verdict.passed=False、retry_count>=max_retries 时跳过 memorize（与 I2 一致）
  - 产物: `src/agent/nodes/memory_nodes.py`
  - 验收: 单元/集成：两路径分别断言 memorize 调用/未调用

- [ ] K8 format_output 支持 Web 来源 — 对 `RetrievedContext.url` 非空时 Sources 段输出为「source (url)」或可点击链接
  - 产物: `src/agent/nodes/format_output.py`
  - 验收: 单元测试：contexts 含 url 时 Sources 段含 URL

- [ ] K9 Trace 记录 used_web_fallback — AgentTrace 或 reasoning_steps 含 used_web_fallback、web_search_query、web_result_count 等
  - 产物: `src/tracing/tracer.py`
  - 验收: 集成测试走 Web 兜底后 trace 文件含对应字段

- [ ] K10 文档与配置说明 — README 或设计文档说明 Web 兜底触发条件、配置项、Tavily/Serper 申请方式
  - 产物: `README.md` 或 `docs/WEB_SEARCH_FALLBACK_DESIGN.md` 补充
  - 验收: 按文档可开启并跑通一次

---

### 总体进度
| 阶段 | 总任务数 | 已完成 | 进度 |
|------|---------|--------|------|
| A | 4 | 4 | 100% |
| B | 4 | 4 | 100% |
| C | 6 | 6 | 100% |
| D | 4 | 4 | 100% |
| E | 4 | 4 | 100% |
| F | 4 | 4 | 100% |
| G | 4 | 4 | 100% |
| H | 10 | 10 | 100% |
| I | 7 | 0 | 0% |
| **总计** | **47** | **40** | **85%** |

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
| 阶段 K（Web Search 兜底） | [WEB_SEARCH_FALLBACK_DESIGN.md](./docs/WEB_SEARCH_FALLBACK_DESIGN.md) |

完成每个子任务后，请将对应行的状态更新为 `[x]` 并填写完成日期与备注，便于追踪总体进度。
