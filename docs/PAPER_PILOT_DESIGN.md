# PaperPilot — AI Research Agent 技术设计文档

> 基于 LangGraph 的多策略研究 Agent，通过 MCP 协议对接 Modular-RAG 知识库，帮助研究者跨论文分析、对比方案、梳理技术脉络。Router 和 Critic 使用 LoRA 微调 + DPO 对齐的本地小模型，兼顾推理效率与评估质量。

---

## 目录

1. [项目定位与核心价值](#1-项目定位与核心价值)
2. [整体架构](#2-整体架构)
3. [技术选型](#3-技术选型)
4. [Agent State 设计](#4-agent-state-设计)
5. [LangGraph 工作流设计](#5-langgraph-工作流设计)
6. [各节点详细设计](#6-各节点详细设计)
7. [MCP Client 集成](#7-mcp-client-集成)
8. [记忆系统设计](#8-记忆系统设计)
9. [微调 Pipeline 设计](#9-微调-pipeline-设计)
10. [可观测性设计](#10-可观测性设计)
11. [项目结构](#11-项目结构)
12. [实施路线图](#12-实施路线图)
13. [面试叙事指南](#13-面试叙事指南)

---

## 1. 项目定位与核心价值

### 1.1 问题背景

AI/ML 研究者在阅读论文时面临三个痛点：

1. **信息分散** — 关键知识散落在几十篇论文中，手动交叉引用效率极低
2. **对比困难** — "LoRA vs QLoRA vs Full Fine-tuning" 需要翻阅多篇论文逐项对比
3. **脉络模糊** — 技术演进的脉络（Attention → Transformer → GPT → InstructGPT → RLHF → DPO）需要大量阅读才能建立

### 1.2 解决方案

PaperPilot 是一个 **多策略研究 Agent**，能够：

- **简单查询**：直接从论文中检索答案（如 "LoRA 的默认 rank 是多少？"）
- **多跳推理**：跨论文追踪技术链条（如 "QLoRA 是如何在 LoRA 的基础上引入量化的？"）
- **对比分析**：并行检索并结构化对比（如 "对比 DPO、RLHF、KTO 的训练流程"）
- **探索梳理**：迭代深挖并构建知识图谱（如 "梳理 2023-2025 年 Agent 架构的演进"）

### 1.3 核心技术亮点（面试卖点）

| 技术维度 | 具体展示 |
|----------|----------|
| **意图识别** | LoRA 分类 (意图类型) + LLM 槽位填充 (实体/约束/查询改写) |
| Agent 工作流 | LangGraph StateGraph，条件路由，循环控制，子图组合 |
| **多推理范式** | 4 种策略 = 4 种推理模式: Direct / Plan-and-Execute / ReWOO / ReAct |
| 自我反思 | Critic 评估 + 带反馈重试 (Reflexion 模式，跨策略通用) |
| 自适应策略 | 意图驱动，不同复杂度问题自动选择最优推理模式 |
| 工具使用 | MCP 协议标准化工具调用 |
| 记忆管理 | 短期对话记忆 + 长期事实累积 |
| 模型微调 | LoRA SFT (Router 意图分类) + DPO (Critic) |
| 模型优化 | 4-bit NF4 量化部署 |
| 结构化输出 | Pydantic 强类型约束 |
| 流式输出 | 实时推理过程展示 |
| 可观测性 | 决策链路追踪 |

### 1.4 "自指性" 叙事

> "我用 LoRA 微调了一个 Router 模型——而 LoRA 本身就是我用这个 Agent 研究过的论文之一。"

这个项目的知识库内容（AI 论文）和项目使用的技术（LoRA、DPO、RAG、Agent）高度重合，形成了优雅的自指闭环。面试时这条叙事线非常有说服力。

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PaperPilot Agent System                      │
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────┐                          │
│  │ 意图理解 (Intent Understanding)       │                          │
│  │  Step1: Router — LoRA 分类 (~50ms)   │  ← intent_type            │
│  │  Step2: Slot Filling — Cloud LLM       │  ← entities, constraints, │
│  │         实体/约束/查询改写            │     reformulated_query    │
│  └──────────────────┬───────────────────┘                          │
│                     │  Intent 对象                                   │
│                     ▼  strategy=?                                   │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │            LangGraph StateGraph (主工作流)                │       │
│  │                                                          │       │
│  │  "simple"       → [Retrieve] → [Synthesize]  (Direct)    │       │
│  │  "multi_hop"    → [Plan→Execute→Replan] loop (Plan&Exec) │       │
│  │  "comparative"  → [ParallelRetrieve] → [Compare] (ReWOO) │       │
│  │  "exploratory"  → [Think→Act→Observe] loop    (ReAct)    │       │
│  │                                                          │       │
│  │  所有检索节点通过 MCP Client → RAG Server                 │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         │                                           │
│                         ▼                                           │
│                ┌─────────────────┐                                   │
│                │  Critic Agent    │  ← Qwen2.5-1.5B + DPO (4-bit)  │
│                │  Reflexion 评估   │     Output: {pass, score, feedback}│
│                └────────┬────────┘                                   │
│                         │                                           │
│                 pass ←──┼──→ fail (retry, max 2 times)              │
│                  │                                                   │
│                  ▼                                                   │
│           Final Response                                            │
│           ├── answer (with citations)                               │
│           ├── reasoning_trace                                       │
│           └── sources                                               │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Short-term  │  │  Long-term   │  │   Tracer     │               │
│  │ Memory      │  │  Memory      │  │   (JSONL)    │               │
│  │ (对话上下文) │  │ (事实累积)   │  │  (决策链路)   │               │
│  └─────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
         │  MCP Protocol (stdio)
         ▼
┌─────────────────────────────────────┐
│  Modular-RAG MCP Server             │
│  ├── query_knowledge_hub (hybrid)   │
│  ├── list_collections               │
│  └── get_document_summary           │
└─────────────────────────────────────┘
```

### 2.1 模型分层策略

```
┌─────────────────────────────────────────────────────┐
│              模型调度层级                              │
│                                                      │
│  Layer 1 — Cloud LLM (GPT-4o / DeepSeek-V3)        │
│    用途: 主推理、回答合成、问题分解                    │
│    特点: 能力强，成本高，延迟高                       │
│    场景: Synthesize, Decompose, ReAct reasoning      │
│                                                      │
│  Layer 2 — Local Fine-tuned Model (Qwen2.5-1.5B)   │
│    用途: 路由分类、质量评估                           │
│    特点: 成本零，延迟低 (<50ms)，专项能力强           │
│    场景: Router (LoRA SFT), Critic (DPO)            │
│    部署: 4-bit NF4 量化，单 GPU ~1GB 显存            │
└─────────────────────────────────────────────────────┘
```

**为什么分层？** 在生产环境中，Agent 的每次交互可能触发 5-10 次 LLM 调用。如果全部使用 Cloud API：
- Router 分类（1 次）+ 检索决策（2-3 次）+ 合成（1 次）+ 评估（1 次）= 5-6 次 API 调用
- 成本：~$0.05/次交互（GPT-4o 计费）
- 延迟：累积 3-5 秒

分层后，Router 和 Critic 使用本地模型，仅推理和合成使用 Cloud API：
- API 调用减少到 2-3 次
- 成本降低 ~50%
- Router 延迟从 ~800ms 降到 <50ms

---

## 3. 技术选型

### 3.1 核心依赖

```toml
[project]
name = "paper-pilot"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Agent Framework (2026.02 最新稳定版)
    "langgraph>=1.0.8",                # v1.0 LTS, StateGraph API 稳定
    "langchain>=1.2.10",               # v1.x LTS
    "langchain-core>=1.2.12",          # 2026-02-12 发布
    "langchain-openai>=1.1.9",         # OpenAI LLM (适配 langchain-core 1.x)
    "langchain-mcp-adapters>=0.2.1",   # MCP Client (支持 streamable-http + stdio)

    # Structured Output
    "pydantic>=2.0",

    # Configuration
    "pyyaml>=6.0",

    # CLI
    "rich>=13.0",                       # Rich terminal output
]

[project.optional-dependencies]
training = [
    # Fine-tuning (2026.02 最新稳定版)
    "transformers>=4.48.0",
    "peft>=0.18.1",                    # LoRA (2026-01-09, 兼容 transformers v5)
    "trl>=0.28.0",                     # DPO/GRPO Trainer (2026-02-10)
    "datasets>=2.21.0",
    "bitsandbytes>=0.45.0",            # 4-bit quantization
    "accelerate>=1.3.0",
    "torch>=2.5.0",
]

local-models = [
    # Local inference
    "transformers>=4.48.0",
    "bitsandbytes>=0.45.0",
    "accelerate>=1.3.0",
    "torch>=2.5.0",
]
```

> **版本说明 (2026-02-14 更新):**
> - LangChain 生态已进入 **v1.x LTS**，API 稳定，不会有破坏性变更
> - LangGraph **v1.0** 相比 0.x 核心 StateGraph API 不变，主要废弃了 `create_react_agent` (改用 `langchain.create_agent`)
> - TRL **v0.28.0** 新增了 GRPO、async tool calls 支持，DPOTrainer 的 `tokenizer` 参数已改名为 `processing_class`
> - PEFT **v0.18.1** 增加了对 transformers v5 的兼容性

### 3.2 选型决策记录

| 决策 | 选择 | 版本 | 原因 | 备选 |
|------|------|------|------|------|
| Agent 框架 | LangGraph | v1.0.8 | v1 LTS, StateGraph API 稳定, 原生循环/条件路由/子图 | 自研 (更灵活但开发周期长) |
| LLM 框架 | LangChain | v1.2.10 | v1 LTS, 生态成熟, structured output 好 | LlamaIndex |
| LLM | GPT-4o / DeepSeek-V3 | — | Function calling 成熟，Structured Output 好 | Claude (MCP 原生但 FC 较弱) |
| 微调基座 | Qwen2.5-1.5B | — | 中文好，体积小，社区活跃 | Llama-3.2-1B (英文更强) |
| LoRA 库 | PEFT | v0.18.1 | HuggingFace 官方，兼容 transformers v5 | Unsloth (更快但兼容性一般) |
| DPO 训练 | TRL DPOTrainer | v0.28.0 | 官方实现，新增 GRPO/async tool calls | 自己实现 DPO loss |
| 量化 | bitsandbytes NF4 | v0.45+ | 推理时量化，无需预处理，精度损失小 | GPTQ/AWQ (更快但需预量化) |
| MCP 集成 | langchain-mcp-adapters | v0.2.1 | 官方 LangChain ↔ MCP 桥接, 支持 stdio+http | 自己写 MCP Client |

---

## 4. Agent State 设计

这是整个系统的数据骨架。LangGraph 的每个节点读写这个 State。

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# ── 子结构 ──────────────────────────────────────────

# 意图类型: LoRA 分类器的输出 (封闭集，5 类)
IntentType = Literal["factual", "comparative", "multi_hop", "exploratory", "follow_up"]

# 图策略节点名: 与 IntentType 映射，follow_up 走 simple 并带上对话上下文
StrategyName = Literal["simple", "multi_hop", "comparative", "exploratory"]

class Intent(BaseModel):
    """
    意图理解模块的完整输出 (方法2: LoRA 分类 + LLM 槽位填充)。
    - type/confidence: 来自本地 LoRA 分类
    - entities, dimensions, constraints, reformulated_query: 来自 Cloud LLM 槽位填充
    """
    type: IntentType = Field(description="意图类型 (LoRA 分类结果)")
    confidence: float = Field(ge=0, le=1, description="分类置信度")
    # 以下由 Slot Filling (Cloud LLM) 填充
    entities: list[str] = Field(default_factory=list, description="关键实体，如论文名、技术名")
    dimensions: list[str] = Field(default_factory=list, description="对比维度 (仅 comparative 时非空)")
    constraints: dict = Field(default_factory=dict, description="隐含约束，如 model_scale, time_range")
    reformulated_query: str = Field(description="改写后的清晰查询，供检索与合成使用")

    def to_strategy(self) -> StrategyName:
        """映射到 LangGraph 策略节点名。follow_up 按 simple 处理 (结合对话历史)。"""
        if self.type == "factual" or self.type == "follow_up":
            return "simple"
        return self.type  # comparative, multi_hop, exploratory 与节点名一致

class RetrievedContext(BaseModel):
    """单条检索结果"""
    content: str
    source: str               # 论文标题/文档名
    doc_id: str
    relevance_score: float
    chunk_index: int | None = None

class CriticVerdict(BaseModel):
    """Critic Agent 的评估结果"""
    passed: bool
    score: float = Field(ge=0, le=10, description="回答质量评分 0-10")
    completeness: float = Field(ge=0, le=1, description="完整性")
    faithfulness: float = Field(ge=0, le=1, description="忠实度 (无幻觉)")
    feedback: str = Field(description="改进建议 (当 passed=False 时)")

class ReasoningStep(BaseModel):
    """推理链路中的一步 (用于 ReAct 和 Tracing)"""
    step_type: Literal["thought", "action", "observation", "critique", "route"]
    content: str
    timestamp: float
    metadata: dict = Field(default_factory=dict)

# ── 主状态 ──────────────────────────────────────────

class AgentState(BaseModel):
    """PaperPilot Agent 的完整状态 — LangGraph StateGraph 的核心"""

    # 对话
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    question: str = ""                  # 当前用户问题 (原始)

    # 意图理解 (Router 分类 + Slot Filling 槽位填充)
    intent: Intent | None = None

    # 检索
    sub_questions: list[str] = Field(default_factory=list)          # multi_hop 分解后的子问题
    retrieved_contexts: list[RetrievedContext] = Field(default_factory=list)
    retrieval_queries: list[str] = Field(default_factory=list)      # 实际发送给 RAG 的查询

    # 生成
    draft_answer: str = ""              # 当前草稿回答
    final_answer: str = ""              # 最终回答

    # 评估
    critic_verdict: CriticVerdict | None = None

    # 控制流
    retry_count: int = 0
    max_retries: int = 2
    current_react_step: int = 0         # ReAct 当前步数
    max_react_steps: int = 5            # ReAct 最大步数

    # 推理链路 (可观测性)
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)

    # 记忆
    accumulated_facts: list[str] = Field(default_factory=list)      # 长期记忆中提取的相关事实
```

### 4.1 State 设计要点

1. **`intent`** — 意图理解结果；`intent.to_strategy()` 得到图节点名 (simple/multi_hop/comparative/exploratory)
2. **`messages` 使用 `add_messages` reducer** — LangGraph 的消息追加语义，多个节点可以向消息列表追加而不覆盖
3. **子结构全部是 Pydantic Model** — 保证类型安全，同时方便 LLM Structured Output 直接输出
4. **控制流变量 (`retry_count`, `current_react_step`)** — 让图的条件边可以基于这些值做路由决策
5. **`reasoning_trace`** — 记录每一步决策，用于调试和面试演示

---

## 5. LangGraph 工作流设计

### 5.1 主工作流 (Main Graph)

```python
from langgraph.graph import StateGraph, START, END

def build_main_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # ── 节点注册 ─────────────────────────
    graph.add_node("load_memory",     load_memory_node)
    graph.add_node("route",           router_node)           # Step1: LoRA 意图分类
    graph.add_node("slot_fill",       slot_filling_node)    # Step2: LLM 槽位填充
    graph.add_node("simple",          simple_strategy_node)
    graph.add_node("multi_hop",       multi_hop_strategy_node)      # 子图
    graph.add_node("comparative",     comparative_strategy_node)    # 子图
    graph.add_node("exploratory",     exploratory_strategy_node)    # 子图 (ReAct)
    graph.add_node("critic",          critic_node)
    graph.add_node("retry_refine",    retry_refine_node)
    graph.add_node("save_memory",     save_memory_node)
    graph.add_node("format_output",   format_output_node)

    # ── 边 ───────────────────────────────

    # 入口: 加载记忆 → 路由 (意图分类) → 槽位填充
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "route")
    graph.add_edge("route", "slot_fill")

    # 槽位填充后 → 按意图策略分支 (条件边)
    graph.add_conditional_edges(
        "slot_fill",
        route_by_intent,                # 使用 state.intent.to_strategy()
        {
            "simple":      "simple",
            "multi_hop":   "multi_hop",
            "comparative": "comparative",
            "exploratory": "exploratory",
        }
    )

    # 所有策略 → Critic
    graph.add_edge("simple",      "critic")
    graph.add_edge("multi_hop",   "critic")
    graph.add_edge("comparative", "critic")
    graph.add_edge("exploratory", "critic")

    # Critic → 通过/重试 (条件边)
    graph.add_conditional_edges(
        "critic",
        critic_gate,                      # 条件函数
        {
            "pass":  "save_memory",
            "retry": "retry_refine",
        }
    )

    # 重试 → 重新做意图理解
    graph.add_edge("retry_refine", "route")

    # 保存记忆 → 格式化输出 → 结束
    graph.add_edge("save_memory", "format_output")
    graph.add_edge("format_output", END)

    # LangGraph v1.0: MemorySaver 用于开发/测试 (内存态，重启丢失)
    # 生产环境可替换为 SqliteSaver / PostgresSaver
    from langgraph.checkpoint.memory import MemorySaver
    return graph.compile(checkpointer=MemorySaver())
```

### 5.2 条件边函数

```python
def route_by_intent(state: AgentState) -> str:
    """根据意图理解结果选择策略节点。follow_up 映射为 simple (由 reformulated_query 携带上下文)。"""
    return state.intent.to_strategy()

def critic_gate(state: AgentState) -> str:
    """Critic 评估后决定是通过还是重试"""
    if state.critic_verdict.passed:
        return "pass"
    if state.retry_count >= state.max_retries:
        return "pass"    # 超过重试次数，强制通过 (避免无限循环)
    return "retry"
```

### 5.3 工作流可视化

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ load_memory │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   route     │  LoRA 意图分类
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ slot_fill   │  LLM 槽位填充
                    └──────┬──────┘
                           │
                  ┌────────▼────────┐
             ┌────│ intent.strategy │────┐
             │    └────────────────┘    │
         ┌───▼──┐ ┌───▼───┐ ┌▼──────┐ ┌▼──────────┐
         │simple│ │m_hop  │ │compar │ │exploratory│
         └───┬──┘ └───┬───┘ └┬──────┘ └┬──────────┘
             │        │      │         │
             └────┬───┘──────┘─────────┘
                  │
           ┌──────▼──────┐
           │   critic     │◄──────────┐
           └──────┬──────┘           │
                  │                   │
            pass ─┤── retry ──► retry_refine
                  │                   │
           ┌──────▼──────┐           │
           │ save_memory │     (回到 route)
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │format_output│
           └──────┬──────┘
                  │
              ┌───▼───┐
              │  END  │
              └───────┘
```

### 5.4 推理模式矩阵 — 4 策略 × 4 范式

每种策略对应一种学术界已有名称的推理模式，而非全部用 ReAct：

| 策略 | 推理模式 | 内部结构 | 为什么用这个模式 |
|------|---------|---------|----------------|
| **simple** | **Direct Execution** | 单次检索 → 合成，无循环 | 事实性问题一次检索即可，不需要迭代 |
| **multi_hop** | **Plan-and-Execute** | 先规划子问题 → 逐步执行 → 执行中可修订计划 | 多跳推理需要全局规划，但中途可能需要调整 |
| **comparative** | **ReWOO** | 一次性规划所有查询 → 并行执行 → 统一合成 | 对比维度已知，无需中间 LLM 调用，省成本省延迟 |
| **exploratory** | **ReAct** | think → act → observe 循环，边做边想 | 事先无法规划步数和方向，需要逐步探索 |
| *(跨策略)* | **Reflexion** | Critic 评估 → 带反馈重试 (最多 2 次) | 所有策略共享的质量保障层 |

> **为什么不用 `create_agent`？** LangGraph 的 `create_agent` 默认是 ReAct 模式——所有问题都走同一个 think→act→observe 循环。我们需要按意图选择不同推理模式，只有 exploratory 需要 ReAct，因此用自定义子图。

> **面试叙事**："我没有把所有问题都交给同一个 ReAct 循环。不同复杂度的问题用不同的推理模式——事实查询直接执行、多跳推理用 Plan-and-Execute、对比分析用 ReWOO、开放探索用 ReAct——最后统一用 Reflexion 做质量保障。"

### 5.5 Multi-Hop 子图 (Plan-and-Execute)

相比旧版的"分解→固定执行"，升级为 **Plan-and-Execute**：执行过程中观察结果，可以跳过已回答的子问题、追加新子问题、或修订计划。

```python
def build_plan_and_execute_subgraph() -> StateGraph:
    """multi_hop 策略: Plan-and-Execute 模式子图"""
    graph = StateGraph(AgentState)

    graph.add_node("plan",       plan_node)            # 制定/修订子问题计划
    graph.add_node("execute",    execute_step_node)    # 执行当前子问题 (RAG 检索)
    graph.add_node("replan",     replan_node)          # 观察结果，决定下一步
    graph.add_node("synthesize", pe_synthesize_node)   # 合成最终回答

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "replan")

    # replan 决定: 继续执行下一步 / 需要修订计划 / 信息足够可合成
    graph.add_conditional_edges(
        "replan",
        pe_should_continue,
        {
            "next_step":  "execute",    # 按计划继续执行下一个子问题
            "replan":     "plan",       # 结果偏差，回到 plan 修订
            "synthesize": "synthesize", # 信息足够，合成回答
        }
    )

    graph.add_edge("synthesize", END)
    return graph.compile()
```

**Plan-and-Execute 流程图:**

```
┌──────┐    ┌─────────┐    ┌────────┐
│ plan │───→│ execute  │───→│ replan │
└──────┘    └─────────┘    └───┬────┘
   ▲                           │
   │  replan (计划需修订)       │ next_step (继续执行)
   └───────────────────────────┤
                               │ synthesize (信息足够)
                         ┌─────▼──────┐
                         │ synthesize  │
                         └─────────────┘
```

**Plan-and-Execute 条件函数:**

```python
def pe_should_continue(state: AgentState) -> str:
    """判断 Plan-and-Execute 的下一步动作"""
    plan = state.sub_questions
    executed = len(state.retrieval_queries)  # 已执行的子问题数

    # 所有子问题都执行完 → 合成
    if executed >= len(plan):
        return "synthesize"

    # replan_node 判断结果偏差 → 需要修订计划
    last_trace = state.reasoning_trace[-1] if state.reasoning_trace else None
    if last_trace and "REPLAN" in last_trace.content:
        return "replan"

    # 默认: 继续执行下一个子问题
    return "next_step"
```

**与 ReAct 的关键区别:**

| 维度 | Plan-and-Execute (multi_hop) | ReAct (exploratory) |
|------|------------------------------|---------------------|
| 计划 | 先有全局计划 (子问题列表) | 无计划，每步临时决定 |
| 步骤数 | 计划决定 (通常 2-4 步) | 不确定 (上限 5 步) |
| 中间决策 | 只在结果偏差时修订计划 | 每步都要 LLM 思考下一步 |
| LLM 调用 | 更少 (不需要每步 think) | 更多 (每步 think + observe) |
| 适用场景 | 问题可分解，步骤可预估 | 问题开放，无法预估方向和深度 |

### 5.6 Exploratory 子图 (ReAct Loop)

这是最复杂的策略，内部是一个 ReAct 循环：

```python
def build_react_subgraph() -> StateGraph:
    """探索策略的 ReAct 子图"""
    graph = StateGraph(AgentState)

    graph.add_node("think",    react_think_node)     # 思考下一步行动
    graph.add_node("act",      react_act_node)       # 执行工具调用 (RAG 检索)
    graph.add_node("observe",  react_observe_node)   # 观察结果，更新状态
    graph.add_node("synthesize", react_synthesize_node)  # 合成最终回答

    graph.add_edge(START, "think")
    graph.add_edge("think", "act")
    graph.add_edge("act", "observe")

    # 观察后决定: 继续思考 or 合成回答
    graph.add_conditional_edges(
        "observe",
        react_should_continue,
        {
            "continue": "think",       # 信息不足，继续循环
            "synthesize": "synthesize", # 信息足够，合成回答
        }
    )

    graph.add_edge("synthesize", END)
    return graph.compile()

def react_should_continue(state: AgentState) -> str:
    """判断 ReAct 是否应该继续"""
    if state.current_react_step >= state.max_react_steps:
        return "synthesize"  # 达到最大步数，强制合成
    # LLM 在 think 阶段会输出 "FINISH" 信号
    last_thought = state.reasoning_trace[-1].content if state.reasoning_trace else ""
    if "FINISH" in last_thought or "足够" in last_thought:
        return "synthesize"
    return "continue"
```

---

## 6. 各节点详细设计

### 6.1 Router Node — LoRA 意图分类 (Step1)

只做**封闭集分类**，输出意图类型 + 置信度；槽位由下一节点填充。

```python
async def router_node(state: AgentState) -> dict:
    """
    使用本地 LoRA 微调模型对用户问题进行意图分类 (5 类)。
    输出仅含 type + confidence，其余槽位由 slot_filling_node 填充。
    Fallback: 本地模型不可用时用 Cloud LLM 做分类。
    """
    question = state.question
    # 可选: 多轮时传入最近一轮对话，便于识别 follow_up
    recent_turn = get_last_exchange(state.messages) if state.messages else ""

    try:
        intent_type, confidence = await local_router_model.classify(question, recent_turn=recent_turn)
    except ModelNotAvailableError:
        result = await cloud_llm.with_structured_output(IntentTypeOutput).ainvoke(
            ROUTER_PROMPT.format(question=question, recent_turn=recent_turn)
        )
        intent_type, confidence = result.type, result.confidence

    # 先写入部分 Intent，供 slot_fill 补全
    partial_intent = Intent(
        type=intent_type,
        confidence=confidence,
        entities=[],
        dimensions=[],
        constraints={},
        reformulated_query=question,  # 占位，slot_fill 会覆盖
    )

    trace_step = ReasoningStep(
        step_type="route",
        content=f"Intent: {intent_type} (confidence: {confidence:.2f})",
        timestamp=time.time(),
        metadata={},
    )

    return {
        "intent": partial_intent,
        "reasoning_trace": [trace_step],
    }
```

**IntentTypeOutput (Cloud Fallback 用，仅 type + confidence):**

```python
class IntentTypeOutput(BaseModel):
    type: IntentType
    confidence: float = Field(ge=0, le=1)
```

**Router 分类体系 (LoRA 训练标签，5 类):**

| 标签 | 含义 | 映射到图节点 |
|------|------|--------------|
| `factual` | 单一事实查询 | simple |
| `comparative` | 对比分析 | comparative |
| `multi_hop` | 多跳推理 | multi_hop |
| `exploratory` | 开放探索 | exploratory |
| `follow_up` | 追问/延续上一轮 | simple (由 reformulated_query 带上下文) |

**Router Prompt (Cloud LLM Fallback 用，输出 IntentTypeOutput: type + confidence):**

```
你是一个研究问题意图分类专家。将用户问题分类为以下之一：
- factual: 单一事实性问题
- comparative: 需要对比多个主题
- multi_hop: 需要跨多源追踪信息链
- exploratory: 开放探索/梳理脉络
- follow_up: 对上一轮回答的追问或补充

用户问题: {question}
最近一轮对话: {recent_turn}
```

### 6.2 Slot Filling Node — LLM 槽位填充 (Step2)

根据已得到的 `intent.type` 和原始问题，用 Cloud LLM 一次性填充实体、维度、约束、改写查询。

```python
async def slot_filling_node(state: AgentState) -> dict:
    """
    根据 state.intent.type 与 state.question，调用 Cloud LLM 做槽位填充。
    输出: entities, dimensions, constraints, reformulated_query。
    仅分类由 LoRA 负责，开放集抽取全部交给 LLM，保证质量。
    """
    intent = state.intent
    question = state.question
    # follow_up 时把上一轮问答放进 prompt，便于改写为完整查询
    context = get_last_exchange(state.messages) if intent.type == "follow_up" and state.messages else ""

    slot_result = await cloud_llm.with_structured_output(SlotFillingOutput).ainvoke(
        SLOT_FILLING_PROMPT.format(
            intent_type=intent.type,
            question=question,
            conversation_context=context,
        )
    )

    full_intent = Intent(
        type=intent.type,
        confidence=intent.confidence,
        entities=slot_result.entities,
        dimensions=slot_result.dimensions,
        constraints=slot_result.constraints,
        reformulated_query=slot_result.reformulated_query,
    )

    return {
        "intent": full_intent,
        "reasoning_trace": [
            ReasoningStep(
                step_type="route",
                content=f"Slot filled: {len(slot_result.entities)} entities, query rewritten",
                timestamp=time.time(),
                metadata={"reformulated_query": slot_result.reformulated_query[:80] + "..."},
            )
        ],
    }
```

**SlotFillingOutput (Pydantic，供 LLM structured output):**

```python
class SlotFillingOutput(BaseModel):
    entities: list[str] = Field(description="关键实体，如论文名、方法名、技术术语")
    dimensions: list[str] = Field(default_factory=list, description="对比维度，仅 comparative 时填写")
    constraints: dict = Field(default_factory=dict, description="隐含约束，如 model_scale, time_range, language")
    reformulated_query: str = Field(description="改写后的清晰、无歧义查询，供检索使用")
```

**Slot Filling Prompt 要点:** 根据 `intent_type` 强调不同重点：comparative 时必填 dimensions；multi_hop 时在 reformulated_query 中保留逻辑链；follow_up 时结合 conversation_context 把代词/省略补全。

### 6.3 Simple Strategy Node — 直接检索 + 生成

使用意图模块产出的 **reformulated_query** 做检索，语义更清晰（尤其对 follow_up）。

```python
async def simple_strategy_node(state: AgentState) -> dict:
    """直接 RAG 查询 → 生成回答。检索使用 intent.reformulated_query。"""
    intent = state.intent
    query = intent.reformulated_query or state.question

    # 1. 调用 RAG (via MCP)
    results = await mcp_tools["query_knowledge_hub"].ainvoke({
        "query": query,
        "top_k": 5,
    })
    contexts = parse_rag_results(results)

    # 2. 合成回答 (面向用户时仍用原始 question 表述)
    answer = await cloud_llm.ainvoke(
        SIMPLE_SYNTHESIS_PROMPT.format(
            question=state.question,
            contexts=format_contexts(contexts),
            accumulated_facts=state.accumulated_facts,
        )
    )

    return {
        "retrieved_contexts": contexts,
        "retrieval_queries": [query],
        "draft_answer": answer.content,
        "reasoning_trace": [
            ReasoningStep(step_type="action", content=f"RAG query: {query}", ...),
            ReasoningStep(step_type="observation", content=f"Retrieved {len(contexts)} chunks", ...),
        ],
    }
```

### 6.4 Multi-Hop Strategy — Plan-and-Execute 节点实现

升级为 Plan-and-Execute 子图 (§5.5)，支持执行中观察结果并修订计划。

#### plan_node — 制定/修订子问题计划

```python
async def plan_node(state: AgentState) -> dict:
    """
    Plan-and-Execute Step1: 制定（或修订）子问题计划。
    首次: 根据 intent 分解子问题。
    修订: 根据已有结果 + replan 反馈，调整计划（增删改子问题）。
    """
    intent = state.intent
    question = intent.reformulated_query or state.question
    is_replan = len(state.retrieval_queries) > 0  # 非首次说明是修订

    if is_replan:
        # 已有部分结果，让 LLM 根据已有信息修订计划
        plan_result = await cloud_llm.with_structured_output(SubQuestions).ainvoke(
            REPLAN_PROMPT.format(
                original_question=question,
                original_plan=state.sub_questions,
                executed_so_far=state.retrieval_queries,
                results_so_far=format_contexts(state.retrieved_contexts),
                constraints=intent.constraints,
            )
        )
    else:
        # 首次: 分解子问题
        plan_result = await cloud_llm.with_structured_output(SubQuestions).ainvoke(
            DECOMPOSE_PROMPT.format(question=question, constraints=intent.constraints)
        )

    return {
        "sub_questions": plan_result.questions,
        "reasoning_trace": [
            ReasoningStep(
                step_type="thought",
                content=f"{'Replan' if is_replan else 'Plan'}: {plan_result.questions}",
                timestamp=time.time(),
            )
        ],
    }
```

#### execute_step_node — 执行当前子问题

```python
async def execute_step_node(state: AgentState) -> dict:
    """Plan-and-Execute Step2: 执行当前子问题的 RAG 检索。"""
    executed = len(state.retrieval_queries)
    if executed >= len(state.sub_questions):
        return {}  # 所有子问题已执行

    current_sub_q = state.sub_questions[executed]

    results = await mcp_tools["query_knowledge_hub"].ainvoke({
        "query": current_sub_q,
        "top_k": 3,
    })
    new_contexts = parse_rag_results(results)

    return {
        "retrieved_contexts": new_contexts,
        "retrieval_queries": [current_sub_q],  # 追加到已执行列表
        "reasoning_trace": [
            ReasoningStep(step_type="action", content=f"Execute step {executed+1}: {current_sub_q}", ...),
            ReasoningStep(step_type="observation", content=f"Retrieved {len(new_contexts)} chunks", ...),
        ],
    }
```

#### replan_node — 观察结果，决定下一步

```python
async def replan_node(state: AgentState) -> dict:
    """
    Plan-and-Execute Step3: 观察最新执行结果，决定:
    - 按计划继续 (next_step)
    - 结果偏差需修订 (REPLAN)
    - 信息已足够 (synthesize)
    """
    decision = await cloud_llm.with_structured_output(ReplanDecision).ainvoke(
        REPLAN_DECISION_PROMPT.format(
            original_question=state.question,
            plan=state.sub_questions,
            executed=state.retrieval_queries,
            latest_result=format_contexts(state.retrieved_contexts[-3:]),
            remaining_steps=state.sub_questions[len(state.retrieval_queries):],
        )
    )
    # decision.action: "next_step" | "REPLAN" | "synthesize"
    # decision.reason: 简要说明

    return {
        "reasoning_trace": [
            ReasoningStep(
                step_type="thought",
                content=f"Replan decision: {decision.action} — {decision.reason}",
                timestamp=time.time(),
            )
        ],
    }
```

#### pe_synthesize_node — 合成最终回答

```python
async def pe_synthesize_node(state: AgentState) -> dict:
    """Plan-and-Execute 最终合成: 基于所有子问题结果生成完整回答。"""
    answer = await cloud_llm.ainvoke(
        MULTI_HOP_SYNTHESIS_PROMPT.format(
            original_question=state.question,
            sub_questions=state.sub_questions,
            all_contexts=format_contexts(state.retrieved_contexts),
        )
    )
    return {"draft_answer": answer.content}
```

### 6.5 Comparative Strategy Node — ReWOO 模式 (并行检索 + 结构化对比)

**ReWOO 模式**：一次性规划所有查询 → 并行执行 → 统一合成，全程无中间 LLM 调用。优先使用意图识别已填充的 `intent.entities` / `intent.dimensions`。

```python
async def comparative_strategy_node(state: AgentState) -> dict:
    """
    1. 实体与维度: 优先用 intent.entities / intent.dimensions，缺失时再 LLM 抽取
    2. 并行检索每个实体
    3. 结构化对比
    """
    intent = state.intent
    question = intent.reformulated_query or state.question

    # Step 1: 实体与维度 (意图识别已填充则直接用，否则 Fallback LLM)
    if intent.entities and (intent.dimensions or intent.type != "comparative"):
        entities, dimensions = intent.entities, intent.dimensions
    else:
        extracted = await cloud_llm.with_structured_output(CompareEntities).ainvoke(
            EXTRACT_ENTITIES_PROMPT.format(question=question)
        )
        entities, dimensions = extracted.entities, extracted.dimensions

    # Step 2: 并行检索 (asyncio.gather)
    retrieval_tasks = [
        mcp_tools["query_knowledge_hub"].ainvoke({
            "query": f"{entity}: {question}",
            "top_k": 3,
        })
        for entity in entities
    ]
    all_results = await asyncio.gather(*retrieval_tasks)

    # Step 3: 结构化对比 (Cloud LLM)
    answer = await cloud_llm.ainvoke(
        COMPARATIVE_SYNTHESIS_PROMPT.format(
            question=state.question,
            entities=entities,
            dimensions=dimensions,
            entity_contexts={
                entity: parse_rag_results(result)
                for entity, result in zip(entities, all_results)
            },
        )
    )

    return {
        "retrieved_contexts": flatten_all_contexts(all_results),
        "draft_answer": answer.content,
    }
```

### 6.6 Exploratory Strategy Node — ReAct 循环

```python
async def react_think_node(state: AgentState) -> dict:
    """ReAct: Think — 分析当前信息，决定下一步行动"""

    think_result = await cloud_llm.ainvoke(
        REACT_THINK_PROMPT.format(
            question=state.question,
            steps_taken=state.current_react_step,
            max_steps=state.max_react_steps,
            current_knowledge=format_contexts(state.retrieved_contexts),
            previous_reasoning=[
                s for s in state.reasoning_trace if s.step_type in ("thought", "observation")
            ],
        )
    )
    # LLM 输出示例:
    # "我已经了解了 Agent 架构的基本概念，但还缺少 2024 年的最新发展。
    #  我需要搜索 'Agent architecture 2024 developments' 来补充信息。
    #  Action: search('2024年Agent架构的最新发展')"

    # 解析出搜索查询
    action_query = extract_search_query(think_result.content)

    return {
        "reasoning_trace": [
            ReasoningStep(step_type="thought", content=think_result.content, ...)
        ],
        "retrieval_queries": [action_query] if action_query else [],
    }

async def react_act_node(state: AgentState) -> dict:
    """ReAct: Act — 执行检索"""
    if not state.retrieval_queries:
        return {}

    query = state.retrieval_queries[-1]
    results = await mcp_tools["query_knowledge_hub"].ainvoke({
        "query": query,
        "top_k": 5,
    })
    new_contexts = parse_rag_results(results)

    return {
        "retrieved_contexts": new_contexts,  # 追加到已有上下文
        "reasoning_trace": [
            ReasoningStep(step_type="action", content=f"Search: {query}", ...)
        ],
    }

async def react_observe_node(state: AgentState) -> dict:
    """ReAct: Observe — 分析检索结果"""
    latest_contexts = state.retrieved_contexts[-5:]  # 最近一次检索的结果

    observation = await cloud_llm.ainvoke(
        REACT_OBSERVE_PROMPT.format(
            question=state.question,
            new_results=format_contexts(latest_contexts),
            all_knowledge_so_far=len(state.retrieved_contexts),
        )
    )

    return {
        "current_react_step": state.current_react_step + 1,
        "reasoning_trace": [
            ReasoningStep(step_type="observation", content=observation.content, ...)
        ],
    }
```

### 6.7 Critic Node — 质量评估 (Reflexion 模式)

```python
async def critic_node(state: AgentState) -> dict:
    """
    使用 DPO 对齐的本地模型评估回答质量。
    评估维度: 完整性、忠实度 (有无幻觉)、引用质量。
    Fallback: Cloud LLM。
    """
    try:
        verdict = await local_critic_model.evaluate(
            question=state.question,
            contexts=state.retrieved_contexts,
            answer=state.draft_answer,
        )
    except ModelNotAvailableError:
        verdict = await cloud_llm.with_structured_output(CriticVerdict).ainvoke(
            CRITIC_PROMPT.format(
                question=state.question,
                contexts=format_contexts(state.retrieved_contexts),
                answer=state.draft_answer,
            )
        )

    return {
        "critic_verdict": verdict,
        "reasoning_trace": [
            ReasoningStep(
                step_type="critique",
                content=f"Score: {verdict.score}/10 | Pass: {verdict.passed} | {verdict.feedback}",
                timestamp=time.time(),
            )
        ],
    }
```

**Critic 评估 Prompt:**

```
你是一个严格的学术回答评估专家。请评估以下回答的质量。

评估维度:
1. 完整性 (0-1): 回答是否覆盖了问题的所有方面？
2. 忠实度 (0-1): 回答是否忠实于检索到的上下文？是否存在幻觉（编造未出现在上下文中的信息）？
3. 引用质量: 回答是否正确引用了来源？

通过标准: score >= 7 且 faithfulness >= 0.8

问题: {question}
检索上下文: {contexts}
回答: {answer}
```

### 6.8 Retry Refine Node — Reflexion 反馈重试

```python
async def retry_refine_node(state: AgentState) -> dict:
    """
    收到 Critic 的否定反馈后，调整策略重试。
    可能的调整: 更换检索查询、补充检索、调整合成 prompt。
    """
    feedback = state.critic_verdict.feedback

    # 让 LLM 根据反馈生成改进计划
    refinement = await cloud_llm.with_structured_output(RefinementPlan).ainvoke(
        REFINEMENT_PROMPT.format(
            question=state.question,
            current_answer=state.draft_answer,
            critic_feedback=feedback,
        )
    )

    return {
        "retry_count": state.retry_count + 1,
        "question": refinement.refined_question or state.question,  # 可能改写问题
        "reasoning_trace": [
            ReasoningStep(
                step_type="thought",
                content=f"Retry #{state.retry_count + 1}: {refinement.plan}",
                ...
            )
        ],
    }
```

---

## 7. MCP Client 集成

### 7.0 对接的 RAG 项目 (Upstream RAG Project)

PaperPilot 作为独立仓库开发时，需在配置中明确「对接的 RAG 项目」路径与启动方式，以便 MCP Client 以 stdio 子进程方式启动 RAG Server。以下为当前对接项目的约定信息，**在 Agent 项目（新文件夹）中开发时请按此配置**。

| 项 | 说明 |
|----|------|
| **项目名称** | MODULAR-RAG-MCP-SERVER（Modular RAG MCP Server） |
| **仓库/路径** | 与 PaperPilot 同级目录，或任意本地路径。建议用环境变量 `RAG_PROJECT_ROOT` 表示 RAG 项目根目录，便于不同机器/CI 复用。 |
| **入口模块** | `src.mcp_server.server`（从 RAG 项目**根目录**执行 `python -m src.mcp_server.server`） |
| **运行要求** | 必须在 RAG 项目根目录下运行，且 `PYTHONPATH` 包含该根目录（或已 `pip install -e .` 安装为包），否则 `src.mcp_server.server` 无法解析。 |
| **暴露的 Tools** | `query_knowledge_hub`（混合检索）、`list_collections`（集合列表）、`get_document_summary`（文档摘要） |
| **传输方式** | stdio（Agent 以子进程方式启动 RAG Server，通过标准输入/输出通信） |

**Agent 侧配置示例**（在 PaperPilot 的 `config/settings.yaml` 或等价配置中）：

```yaml
# 使用环境变量 RAG_PROJECT_ROOT，未设置时回退到相对路径或占位
mcp:
  connections:
    rag_server:
      command: "python"
      args: ["-m", "src.mcp_server.server"]
      cwd: "${RAG_PROJECT_ROOT}"   # 或写死: C:/code/python/MODULAR-RAG-MCP-SERVER
      transport: "stdio"
      env:
        PYTHONPATH: "${RAG_PROJECT_ROOT}"
```

**联调时**：先在本机启动 RAG Server（在 RAG 项目根目录执行 `python -m src.mcp_server.server`），或由 Agent 按上表配置在运行时自动拉起 RAG 子进程。RAG 项目自身的环境、依赖与数据（如 Chroma、BM25 索引、已摄入文档）需已在 RAG 侧准备好。

### 7.1 连接方式

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def create_mcp_tools():
    """
    通过 MCP 协议连接 Modular-RAG Server。
    使用 langchain-mcp-adapters 将 MCP tools 转换为 LangChain tools。

    v0.2.1 API 说明:
    - 默认 stateless 模式: 每次工具调用创建新 session，调用完自动清理
    - 支持 stdio / streamable-http 两种 transport
    - tool_name_prefix=True 时自动加 "rag_server_" 前缀避免命名冲突
    """
    client = MultiServerMCPClient(
        connections={
            "rag_server": {
                "command": "python",
                "args": ["-m", "src.mcp_server.server"],
                "cwd": "/path/to/MODULAR-RAG-MCP-SERVER",
                "transport": "stdio",    # 本地子进程通信
                "env": {
                    "PYTHONPATH": "/path/to/MODULAR-RAG-MCP-SERVER",
                },
            }
        },
        # tool_name_prefix=False,  # 不加服务名前缀 (只有一个 server 时)
    )

    tools = client.get_tools()
    # tools 包含 (LangChain Tool 对象，可直接用于 LangGraph):
    # - query_knowledge_hub(query, top_k?, collection?)
    # - list_collections(include_stats?)
    # - get_document_summary(doc_id, collection?)

    return tools, client

# 也可以用 async context manager (推荐，自动清理):
async def run_with_mcp():
    async with MultiServerMCPClient(
        connections={"rag_server": {"command": "python", "args": [...], "transport": "stdio"}}
    ) as client:
        tools = client.get_tools()
        # ... 使用 tools ...
```

### 7.2 工具使用封装

```python
class RAGToolWrapper:
    """封装 MCP 工具调用，添加错误处理和日志"""

    def __init__(self, mcp_tools: dict):
        self.tools = {t.name: t for t in mcp_tools}

    async def search(self, query: str, top_k: int = 5, collection: str | None = None) -> list[RetrievedContext]:
        """查询知识库"""
        try:
            params = {"query": query, "top_k": top_k}
            if collection:
                params["collection"] = collection

            result = await self.tools["query_knowledge_hub"].ainvoke(params)
            return self._parse_results(result)
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            return []  # Graceful degradation

    async def list_collections(self) -> list[str]:
        """列出可用的知识集合"""
        result = await self.tools["list_collections"].ainvoke({"include_stats": True})
        return self._parse_collections(result)

    async def get_doc_info(self, doc_id: str) -> dict:
        """获取文档摘要"""
        result = await self.tools["get_document_summary"].ainvoke({"doc_id": doc_id})
        return self._parse_doc_info(result)
```

---

## 8. 记忆系统设计

### 8.1 双层记忆架构

```
┌─────────────────────────────────────────────────┐
│                Memory System                     │
│                                                  │
│  ┌─────────────────────────────┐                │
│  │ Short-term Memory           │                │
│  │ (LangGraph Checkpointer)   │                │
│  │                             │                │
│  │ • 当前对话的消息历史         │                │
│  │ • 当前 session 的 AgentState│                │
│  │ • 自动管理 by LangGraph     │                │
│  └─────────────────────────────┘                │
│                                                  │
│  ┌─────────────────────────────┐                │
│  │ Long-term Memory            │                │
│  │ (Custom JSON Store)         │                │
│  │                             │                │
│  │ • 从历史对话中提取的关键事实  │                │
│  │ • 用户研究兴趣 profile       │                │
│  │ • 论文间的已知关联           │                │
│  │                             │                │
│  │ 存储: facts.jsonl           │                │
│  │ 检索: embedding similarity  │                │
│  │ 更新: 每次对话结束后提取     │                │
│  └─────────────────────────────┘                │
└─────────────────────────────────────────────────┘
```

### 8.2 长期记忆实现

```python
class LongTermMemory:
    """基于事实的长期记忆"""

    def __init__(self, memory_file: str = "data/memory/facts.jsonl"):
        self.memory_file = Path(memory_file)
        self.facts: list[Fact] = self._load_facts()

    async def recall(self, question: str, top_k: int = 5) -> list[str]:
        """
        从长期记忆中检索与当前问题相关的事实。
        使用简单的关键词匹配 + embedding 相似度。
        """
        if not self.facts:
            return []

        # 简单实现: 关键词匹配 (可以升级为 embedding 检索)
        question_keywords = extract_keywords(question)
        scored_facts = []
        for fact in self.facts:
            overlap = len(set(fact.keywords) & set(question_keywords))
            if overlap > 0:
                scored_facts.append((overlap, fact.content))

        scored_facts.sort(reverse=True)
        return [f for _, f in scored_facts[:top_k]]

    async def memorize(self, question: str, answer: str, contexts: list[RetrievedContext]):
        """
        从本次对话中提取值得记住的事实。
        使用 LLM 提取，而不是存储整个对话。
        """
        extraction = await cloud_llm.with_structured_output(ExtractedFacts).ainvoke(
            EXTRACT_FACTS_PROMPT.format(
                question=question,
                answer=answer,
            )
        )

        for fact_text in extraction.facts:
            fact = Fact(
                content=fact_text,
                keywords=extract_keywords(fact_text),
                source_question=question,
                timestamp=time.time(),
            )
            self.facts.append(fact)
            self._append_to_file(fact)
```

### 8.3 记忆在工作流中的位置

```python
async def load_memory_node(state: AgentState) -> dict:
    """工作流开始时加载相关记忆"""
    relevant_facts = await long_term_memory.recall(state.question)
    return {"accumulated_facts": relevant_facts}

async def save_memory_node(state: AgentState) -> dict:
    """工作流结束时保存新知识到长期记忆"""
    await long_term_memory.memorize(
        question=state.question,
        answer=state.draft_answer,
        contexts=state.retrieved_contexts,
    )
    return {"final_answer": state.draft_answer}
```

---

## 9. 微调 Pipeline 设计

### 9.1 训练数据合成

#### Router 训练数据

```python
# training/data/generate_router_data.py

ROUTER_DATA_GEN_PROMPT = """
你是一个训练数据生成专家。请为"研究问题意图分类器"生成训练样本。

分类体系 (5 类，与下游策略映射一致):
- factual: 单一事实性问题 (只需查一个来源即可回答)
- comparative: 需要对比多个主题 (如方法A vs 方法B)
- multi_hop: 需要跨多源追踪信息链 (如 A 如何导致 B)
- exploratory: 开放探索、梳理脉络 (需迭代检索)
- follow_up: 对上一轮回答的追问或补充 (如 "那显存呢？" "能再详细说说吗？")

领域: AI/ML 研究论文

请生成 {n} 条样本，每个分类约 {n//5} 条。
格式: JSON 数组 [{{"question": "...", "label": "factual|comparative|multi_hop|exploratory|follow_up", "reason": "..."}}]

要求:
- 问题要具体，涉及真实的 AI/ML 概念和论文
- follow_up 样本需带上一轮对话上下文 (可放在 question 中，如 "上一轮: ... 当前: 那显存呢？")
- 避免歧义: 每条的分类应该明确
"""

async def generate_router_training_data(n: int = 800) -> list[dict]:
    """用 GPT-4o 合成 Router 训练数据"""
    all_samples = []
    batch_size = 50

    for i in range(0, n, batch_size):
        response = await gpt4o.ainvoke(
            ROUTER_DATA_GEN_PROMPT.format(n=batch_size)
        )
        samples = json.loads(response.content)
        all_samples.extend(samples)

    # 质量过滤: 去重 + 随机 human review sample
    all_samples = deduplicate(all_samples)
    return all_samples
```

**最终训练数据格式 (Alpaca 格式，标签为 5 类意图):**

```json
{
    "instruction": "将以下研究问题分类为 factual/comparative/multi_hop/exploratory/follow_up 之一。\n\n问题: LoRA 论文中建议的默认 rank 值是多少？",
    "output": "factual"
}
{
    "instruction": "将以下研究问题分类为 factual/comparative/multi_hop/exploratory/follow_up 之一。\n\n问题: 对比 DPO 和 RLHF 的训练流程差异。",
    "output": "comparative"
}
{
    "instruction": "将以下研究问题分类为 factual/comparative/multi_hop/exploratory/follow_up 之一。\n\n上一轮: 用户问 LoRA 的 rank 是什么，助手回答了。\n当前: 那和 full fine-tuning 比显存差多少？",
    "output": "follow_up"
}
```

#### Slot Filling：无需训练，仅 Prompt 设计

槽位填充由 Cloud LLM 完成，只需在 `prompts/` 下维护 **Slot Filling Prompt** 与 `SlotFillingOutput` 的 Pydantic 定义。根据 `intent_type` 在 prompt 中强调：comparative 必填 dimensions；follow_up 时传入 conversation_context 并要求将代词/省略补全为 reformulated_query。

#### DPO 偏好对数据

```python
# training/data/generate_dpo_pairs.py

DPO_DATA_GEN_PROMPT = """
你是一个训练数据生成专家。请为"回答质量评估器"生成 DPO 偏好对。

场景: 给定一个研究问题和检索到的上下文，评估器需要判断回答质量。

请生成一组偏好对，每对包含:
- prompt: 包含 (问题 + 上下文 + 待评估回答) 的评估请求
- chosen: 准确、详细的质量评估 (正确识别优点和缺陷)
- rejected: 不准确或过于宽泛的质量评估 (遗漏关键问题或错误评价)

具体要求:
- chosen 应该精确指出幻觉、遗漏、引用缺失等具体问题
- rejected 应该犯典型错误: 过度宽容、忽略幻觉、评价模糊
- 涵盖不同质量水平的回答 (好回答 + 差回答 都需要评估)
"""

# 生成 ~500 偏好对
# 最终格式符合 TRL DPOTrainer 要求:
# {"prompt": "...", "chosen": "...", "rejected": "..."}
```

### 9.2 LoRA SFT 训练 (Router)

```python
# training/sft_router.py

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig   # trl >= 0.28.0: 使用 SFTConfig 替代 TrainingArguments
from datasets import load_dataset

def train_router():
    # ── 1. 加载基座模型 (4-bit 加载以节省显存 — QLoRA 模式) ──
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,     # 双重量化，进一步压缩
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    # ── 2. LoRA 配置 ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,                          # 秩: 16 (分类任务不需要太大)
        lora_alpha=32,                 # alpha/r = 2 (标准缩放)
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # 只改 attention 的 Q 和 V
        # 面试点: 为什么选 q_proj, v_proj?
        # → LoRA 原论文实验表明 Wq, Wv 效果最好，且参数最少
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 预期输出: trainable params: ~1.2M (0.08% of total)

    # ── 3. 数据加载 ──
    dataset = load_dataset("json", data_files="data/router_train.json")

    # ── 4. 训练 ──
    # trl >= 0.28.0 推荐使用 SFTConfig (继承自 TrainingArguments, 但集成了 SFT 专属参数)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        processing_class=tokenizer,    # ⚠️ trl>=0.27: tokenizer 参数已更名为 processing_class
        peft_config=lora_config,       # 也可以在这里传 LoRA config (SFTTrainer 内部处理)
        args=SFTConfig(
            output_dir="models/router-lora",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            max_seq_length=256,        # SFTConfig 集成了 max_seq_length
        ),
    )
    trainer.train()

    # ── 5. 保存 LoRA adapter ──
    model.save_pretrained("models/router-lora")
    # 保存的只是 adapter 权重 (~5MB)，不是完整模型
```

**关键面试点:**
- `r=16`: 为什么不用 4 或 64？→ 分类任务复杂度低，16 足够；rank 过高会过拟合
- `target_modules=["q_proj", "v_proj"]`: 为什么不加 `k_proj`？→ 原论文实验 Wq+Wv 效果最佳
- QLoRA 训练 (4-bit 基座 + LoRA): 显存需求 ~4GB，消费级 GPU 即可
- 可训练参数仅 ~1.2M (0.08%)，但分类准确率可达 95%+

### 9.3 DPO 训练 (Critic)

```python
# training/dpo_critic.py

from trl import DPOTrainer, DPOConfig    # trl >= 0.28.0
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from datasets import load_dataset

def train_critic_dpo():
    # ── 1. 加载模型 (先用 SFT 版本做基座，或直接用原始 Qwen) ──
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

    # ── 2. 可选: 加 LoRA (减少显存) ──
    # DPO 也可以用 LoRA，TRL 原生支持
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    # ── 3. 加载偏好对数据 ──
    dataset = load_dataset("json", data_files="data/dpo_pairs.json")
    # 格式: {"prompt": "...", "chosen": "...", "rejected": "..."}

    # ── 4. DPO 训练 ──
    dpo_config = DPOConfig(
        output_dir="models/critic-dpo",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,            # DPO 学习率通常比 SFT 低
        beta=0.1,                      # DPO 温度参数
        # 面试点: beta 控制偏好模型与参考模型的 KL 散度惩罚
        # beta 过大 → 太保守，几乎不学习
        # beta 过小 → 过度优化，可能 reward hacking
        warmup_ratio=0.1,
        logging_steps=10,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                # 使用 implicit reference (PEFT 模式)
        args=dpo_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer,    # ⚠️ trl>=0.27: tokenizer 参数已更名为 processing_class
        peft_config=peft_config,
    )
    trainer.train()

    # ── 5. 保存 ──
    trainer.save_model("models/critic-dpo")
```

**DPO 核心原理 (面试必讲):**

```
DPO Loss = -log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x))
                    - β · (log π_θ(y_l|x) - log π_ref(y_l|x)))

其中:
- π_θ: 训练中的模型
- π_ref: 参考模型 (SFT 后冻结)
- y_w: chosen (偏好回答)
- y_l: rejected (不偏好回答)
- β: 温度参数 (控制 KL 约束强度)

直觉: 训练模型增大 chosen 的概率，减小 rejected 的概率，
      同时用 KL 惩罚防止偏离参考模型太远。
      
相比 RLHF 的优势: 无需单独训练 reward model，将 reward 隐式编码在策略中。
```

### 9.4 量化部署

```python
# src/models/loader.py

class LocalModelManager:
    """管理本地微调模型的加载和推理"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.router_model = None
        self.critic_model = None

    async def load_router(self):
        """加载 Router 模型 (LoRA merged + 4-bit quantized)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # 方式1: 加载 merged 模型 (LoRA weights 已合并到基座)
        self.router_model = AutoModelForCausalLM.from_pretrained(
            self.config.router_model_path,    # "models/router-lora-merged"
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.router_tokenizer = AutoTokenizer.from_pretrained(
            self.config.router_model_path
        )

        # 方式2: 动态加载 LoRA adapter (更灵活，显存共享)
        # base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", ...)
        # self.router_model = PeftModel.from_pretrained(base, "models/router-lora")

    async def classify_question(self, question: str, recent_turn: str = "") -> tuple[IntentType, float]:
        """Router 推理：输出意图类型 + 置信度 (供 router_node 构造部分 Intent)。"""
        prompt = (
            "将以下研究问题分类为 factual/comparative/multi_hop/exploratory/follow_up 之一。\n\n"
            + (f"最近一轮: {recent_turn}\n\n" if recent_turn else "")
            + f"问题: {question}"
        )
        inputs = self.router_tokenizer(prompt, return_tensors="pt").to(self.router_model.device)
        with torch.no_grad():
            outputs = self.router_model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
            )
        result = self.router_tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = parse_classification(result)  # 解析出 factual | comparative | multi_hop | exploratory | follow_up
        return label, 0.95  # 可用 logits 计算真实置信度
```

**量化核心知识 (面试点):**

```
NF4 (NormalFloat 4-bit):
- 假设权重服从正态分布 N(0, σ²)
- 将正态分布的 CDF 均匀分成 16 个区间 (4-bit = 2^4)
- 每个区间的中位数作为量化值
- 信息论最优: 在正态分布假设下最大化信息保留

Double Quantization:
- 不仅量化权重，还量化量化常数本身
- 量化常数 (FP32) → 量化为 FP8
- 进一步节省 ~0.4 bit/param

显存计算:
- Qwen2.5-1.5B 原始: 1.5B × 2 bytes (FP16) = ~3GB
- 4-bit 量化: 1.5B × 0.5 bytes = ~0.75GB
- 加上 KV cache 和 overhead: ~1-1.5GB
- Router + Critic 两个模型: ~2-3GB，一张消费级 GPU 足够
```

---

## 10. 可观测性设计

### 10.1 Trace 数据结构

```python
class AgentTrace(BaseModel):
    """完整的 Agent 决策链路追踪"""
    trace_id: str
    question: str
    timestamp: float
    duration_ms: float

    intent: Intent
    strategy_executed: str
    reasoning_steps: list[ReasoningStep]

    retrieval_queries: list[str]
    contexts_retrieved: int
    tokens_used: dict[str, int]        # {"input": ..., "output": ...}

    critic_verdict: CriticVerdict
    retry_count: int
    final_answer: str

    # 性能指标
    router_latency_ms: float
    retrieval_latency_ms: float
    llm_latency_ms: float
    critic_latency_ms: float
```

### 10.2 流式输出 (Terminal UI)

```python
# 使用 Rich 库实现美观的终端输出

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

async def stream_agent_execution(graph, question: str):
    """流式展示 Agent 推理过程"""
    console = Console()

    console.print(Panel(f"[bold]{question}[/bold]", title="Question"))

    async for event in graph.astream_events(
        {"question": question},
        version="v2",
    ):
        kind = event["event"]
        name = event.get("name", "")

        if kind == "on_chain_start" and "node" in name:
            console.print(f"\n[blue]▶ Entering: {name}[/blue]")

        elif kind == "on_chat_model_stream":
            # 流式打印 LLM 输出
            chunk = event["data"]["chunk"]
            if chunk.content:
                console.print(chunk.content, end="")

        elif kind == "on_chain_end" and "critic" in name:
            verdict = event["data"]["output"].get("critic_verdict")
            if verdict:
                color = "green" if verdict.passed else "red"
                console.print(f"\n[{color}]✧ Critic: {verdict.score}/10 — {verdict.feedback}[/{color}]")
```

---

## 11. 项目结构

```
paper-pilot/
├── pyproject.toml                      # 项目配置 + 依赖
├── README.md                           # 项目介绍
├── config/
│   └── settings.yaml                   # Agent 配置 (LLM, 模型路径, 策略参数)
│
├── src/
│   ├── __init__.py
│   │
│   ├── agent/                          # === Agent 核心 (LangGraph) ===
│   │   ├── __init__.py
│   │   ├── graph.py                    # 主工作流 build_main_graph()
│   │   ├── state.py                    # AgentState + 子类型定义
│   │   ├── nodes/                      # 图节点
│   │   │   ├── __init__.py
│   │   │   ├── router.py              # Router 节点 (LoRA 意图分类)
│   │   │   ├── slot_filling.py    # Slot Filling 节点 (LLM 槽位填充)
│   │   │   ├── critic.py              # Critic 节点
│   │   │   ├── memory_nodes.py        # load_memory / save_memory
│   │   │   └── format_output.py       # 格式化最终输出
│   │   ├── strategies/                 # 四种执行策略
│   │   │   ├── __init__.py
│   │   │   ├── simple.py              # 直接检索 + 生成
│   │   │   ├── multi_hop.py           # 分解 + 顺序检索 + 合成
│   │   │   ├── comparative.py         # 并行检索 + 结构化对比
│   │   │   └── exploratory.py         # ReAct 循环子图
│   │   └── edges.py                   # 条件路由函数
│   │
│   ├── tools/                          # === 工具层 ===
│   │   ├── __init__.py
│   │   ├── mcp_client.py             # MCP Client 封装
│   │   └── tool_wrapper.py           # RAGToolWrapper
│   │
│   ├── models/                         # === 本地模型管理 ===
│   │   ├── __init__.py
│   │   ├── loader.py                  # LocalModelManager (量化加载)
│   │   └── inference.py               # Router/Critic 推理封装
│   │
│   ├── memory/                         # === 记忆系统 ===
│   │   ├── __init__.py
│   │   ├── short_term.py             # LangGraph Checkpointer 封装
│   │   └── long_term.py              # LongTermMemory (事实累积)
│   │
│   ├── llm/                            # === LLM 客户端 ===
│   │   ├── __init__.py
│   │   └── client.py                 # Cloud LLM 封装 (OpenAI/DeepSeek)
│   │
│   ├── prompts/                        # === Prompt 模板 ===
│   │   ├── __init__.py
│   │   ├── router.py                  # Router 分类 prompt (Fallback)
│   │   ├── slot_filling.py            # Slot Filling prompt + SlotFillingOutput
│   │   ├── strategies.py             # 策略相关 prompts (Decompose, Compare, ReAct)
│   │   ├── critic.py                  # Critic prompts
│   │   └── memory.py                 # 事实提取 prompts
│   │
│   └── tracing/                        # === 可观测性 ===
│       ├── __init__.py
│       └── tracer.py                  # AgentTrace + JSONL 记录
│
├── training/                           # === 微调 Pipeline ===
│   ├── data/
│   │   ├── generate_router_data.py    # 合成 Router 训练数据
│   │   └── generate_dpo_pairs.py      # 合成 DPO 偏好对
│   ├── sft_router.py                  # LoRA SFT 训练
│   ├── dpo_critic.py                  # DPO 训练
│   ├── merge_lora.py                  # 合并 LoRA 到基座
│   └── eval_models.py                 # 微调模型评估
│
├── notebooks/                          # === 实验 Notebook ===
│   ├── 01_data_generation.ipynb       # 数据合成实验
│   ├── 02_lora_sft.ipynb              # LoRA 训练实验
│   ├── 03_dpo_training.ipynb          # DPO 训练实验
│   └── 04_agent_demo.ipynb            # Agent 完整 Demo
│
├── data/
│   ├── training/                      # 训练数据
│   └── memory/                        # 长期记忆存储
│
├── models/                             # 微调后的模型 (gitignore)
│   ├── router-lora/
│   └── critic-dpo/
│
├── main.py                             # CLI 入口
└── tests/
    ├── test_router.py
    ├── test_strategies.py
    ├── test_critic.py
    └── test_graph.py
```

**文件数: ~35 个源文件**
**代码量估算: ~3000-4000 行 (Agent) + ~500 行 (Training)**

---

## 12. 实施路线图

### Phase 1: Agent 骨架 (3-4 天)

```
Day 1: 项目初始化 + State 定义 + LLM Client
  - [ ] pyproject.toml + settings.yaml
  - [ ] AgentState + 所有 Pydantic 子模型
  - [ ] Cloud LLM client 封装 (OpenAI/DeepSeek)

Day 2: MCP Client + Simple Strategy
  - [ ] MCP Client 连接 RAG Server
  - [ ] RAGToolWrapper 封装
  - [ ] Simple Strategy 实现 (端到端验证)

Day 3: LangGraph 主工作流 + 意图理解 (Cloud LLM 版)
  - [ ] build_main_graph() 骨架：route → slot_fill → 条件边
  - [ ] Router Node 输出部分 Intent (type + confidence)，Fallback 用 Cloud LLM
  - [ ] Slot Filling Node：SlotFillingOutput + Prompt，填充 entities/dimensions/constraints/reformulated_query
  - [ ] Critic Node (先用 Cloud LLM)
  - [ ] route_by_intent + 重试逻辑

Day 4: 验证 + 修 Bug
  - [ ] Simple 策略端到端测试 (使用 intent.reformulated_query)
  - [ ] 流程打通: 问题 → 意图分类 → 槽位填充 → 检索 → 生成 → 评估 → 输出
```

### Phase 2: 四种策略 (3-4 天)

```
Day 5: Multi-hop Strategy (Plan-and-Execute 子图)
  - [ ] build_plan_and_execute_subgraph()
  - [ ] plan_node: 分解子问题 + replan 逻辑
  - [ ] execute_step_node + replan_node + pe_synthesize_node
  - [ ] pe_should_continue 条件函数

Day 6: Comparative Strategy (ReWOO 模式)
  - [ ] 优先使用 intent.entities / intent.dimensions，缺失时 LLM 抽取
  - [ ] 并行检索 (asyncio.gather) — 无中间 LLM 调用
  - [ ] 统一结构化对比合成

Day 7: Exploratory Strategy (ReAct)
  - [ ] ReAct 子图 (think → act → observe 循环)
  - [ ] 循环终止条件
  - [ ] 知识累积

Day 8: 记忆系统 + Streaming
  - [ ] 长期记忆 (LongTermMemory)
  - [ ] load_memory / save_memory 节点
  - [ ] Rich terminal 流式输出
```

### Phase 3: 微调 Pipeline (4-5 天)

```
Day 9: 合成训练数据
  - [ ] Router 训练数据生成 (~800 条，5 类: factual/comparative/multi_hop/exploratory/follow_up)
  - [ ] Slot Filling Prompt 定稿 (按 intent_type 分支)
  - [ ] DPO 偏好对生成 (~500 对)
  - [ ] 数据清洗 + 质量检查

Day 10-11: LoRA SFT (Router)
  - [ ] 训练脚本
  - [ ] 训练 + 评估
  - [ ] LoRA merge + 量化导出

Day 12-13: DPO (Critic)
  - [ ] DPO 训练脚本
  - [ ] 训练 + 评估
  - [ ] 量化导出
```

### Phase 4: 集成 + 打磨 (2-3 天)

```
Day 14: 本地模型集成
  - [ ] LocalModelManager 实现
  - [ ] 替换 Router/Critic 为本地模型 (保留 Cloud LLM fallback)
  - [ ] 性能对比: 本地 vs Cloud

Day 15: 可观测性 + CLI
  - [ ] AgentTrace 完整记录
  - [ ] CLI 交互界面
  - [ ] Demo 场景准备

Day 16: 测试 + 文档
  - [ ] 核心路径测试
  - [ ] README 完善
  - [ ] 面试 Demo 脚本准备
```

**总工期: ~17-18 天 (含意图识别约 +1-2 天)**

---

## 13. 面试叙事指南

### 13.1 30 秒 Elevator Pitch

> "我做了一个叫 PaperPilot 的 AI 研究 Agent。核心理念是**不同复杂度的问题用不同推理模式**——事实查询直接执行，多跳推理用 Plan-and-Execute（先规划再执行、可中途修订），对比分析用 ReWOO（一次性并行检索），开放探索用 ReAct（边想边做）。意图识别层用 LoRA 微调小模型做 5 类分类 + Cloud LLM 做槽位填充，质量保障用 DPO 对齐的 Critic + Reflexion 反馈重试。"

### 13.2 技术深挖应答

| 面试官问 | 你的回答方向 |
|----------|-------------|
| "为什么不直接用 GPT-4o 做路由？" | 成本/延迟分析 → Router 是高频调用 → 本地小模型 <50ms vs API ~800ms → 生产环境的成本意识 |
| "意图识别为什么拆成分类 + 槽位填充？" | 分类是封闭集，小模型足够且省成本；实体/改写是开放集，交给 LLM 更稳。各取所长 (方法2) |
| "LoRA 的原理是什么？" | 低秩分解 ΔW = AB → 只训练 A, B → 参数从 d×d 降到 d×r + r×d → 1.5B 模型只训练 1.2M 参数 (0.08%) |
| "DPO 比 RLHF 好在哪？" | 无需单独训练 Reward Model → 将 reward 隐式编码在策略中 → Bradley-Terry 模型推导 → β 控制 KL 约束 |
| "量化会损失精度吗？" | NF4 假设正态分布 → 信息论最优量化 → 实测分类准确率下降 <1% → 权衡显存 vs 精度 |
| "ReAct 和 Plan-and-Execute 有什么区别？" | ReAct 无全局计划，每步临时决定；Plan-and-Execute 先有计划再执行，偏差时修订。我在 multi_hop 用 P&E (步骤可预估)，在 exploratory 用 ReAct (无法预估方向和深度) |
| "为什么 Comparative 用 ReWOO？" | 对比查询的实体和维度在意图识别时已经确定，不需要中间 LLM 推理，直接并行查完统一合成 → 省 LLM 调用 → ReWOO 模式 (Reasoning WithOut Observation) |
| "什么是 Reflexion？你怎么实现的？" | Critic + retry = Reflexion: 执行后自我评估 → 不合格时带具体反馈重试 → 下一轮执行能看到上一轮的失败原因。和简单重试的区别是**反馈信息**会传递给下一轮 |
| "为什么不用 LATS？" | LATS 用树搜索探索多条路径，适合有明确 pass/fail 的任务 (如写代码)。论文问答没有这种二元判定信号，且搜索空间太大，成本不划算 |
| "Critic 怎么判断幻觉？" | 将回答中的事实 claim 与检索 context 做对比 → 找不到支撑的 claim 标记为幻觉 → DPO 让模型学会识别这种 pattern |
| "为什么用 MCP 而不是直接调 API？" | MCP 是标准化工具协议 → 解耦 Agent 和 RAG → RAG 可以独立升级 → 符合微服务思想 |
| "记忆系统怎么设计的？" | 双层: 短期 (LangGraph Checkpointer 自动管理对话) + 长期 (LLM 提取关键事实 → JSONL 持久化 → 关键词+embedding 检索) |

### 13.3 现场 Demo 脚本

准备 4 个经典问题，覆盖 4 种推理模式 + Reflexion：

```
Demo 1 (Direct): "LoRA 论文中 rank 的默认推荐值是多少？"
→ 展示: factual → Direct Execution → 单次 RAG → 精确回答

Demo 2 (Plan-and-Execute): "QLoRA 是如何在 LoRA 的基础上引入量化的？"
→ 展示: multi_hop → plan 分解 3 个子问题 → 逐步执行 → 中途 replan 跳过冗余步骤 → 合成

Demo 3 (ReWOO): "对比 LoRA、QLoRA 和 Full Fine-tuning 的显存需求和效果"
→ 展示: comparative → intent.entities/dimensions 已填充 → 并行 3 路检索 → 结构化表格

Demo 4 (ReAct + Reflexion): "梳理从 RLHF 到 DPO 的技术演进脉络"
→ 展示: exploratory → ReAct 循环 (3-4 轮 think/act/observe) → Critic 评估 → 可能触发 Reflexion 重试
```

---

## 附录: 论文知识库推荐内容

以下论文 PDF 灌入 RAG，构成 PaperPilot 的知识库：

| 论文 | 与项目技术的关联 |
|------|-----------------|
| LoRA (Hu et al., 2021) | Router 微调技术 |
| QLoRA (Dettmers et al., 2023) | 量化 + LoRA 结合 |
| DPO (Rafailov et al., 2023) | Critic 对齐训练 |
| InstructGPT (Ouyang et al., 2022) | RLHF 对比基线 |
| RAG (Lewis et al., 2020) | 检索增强生成基础 |
| ReAct (Yao et al., 2022) | Exploratory 策略的推理范式 |
| Plan-and-Execute (Wang et al., 2023) | Multi-hop 策略的推理范式 |
| ReWOO (Xu et al., 2023) | Comparative 策略的推理范式 |
| Reflexion (Shinn et al., 2023) | Critic + retry 的自我反思机制 |
| Toolformer (Schick et al., 2023) | 工具使用 |
| Attention Is All You Need (Vaswani et al., 2017) | Transformer 基础 |
| Chain-of-Thought (Wei et al., 2022) | 推理链技术 |

这些论文构成一个自洽的知识网络，且每篇都与项目使用的技术直接相关。
