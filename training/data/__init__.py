"""训练数据生成：全 LLM 生成。

- generate_all_with_llm: 一键生成 Router / Critic SFT / DPO 的 JSONL（推荐）
- generate_router_data_with_llm: Router 问句（按意图）
- generate_critic_sft_with_llm: Critic SFT 评判数据
- generate_dpo_pairs_with_llm: DPO 偏好对
- llm_client: OpenAI 兼容调用与 chat_json / chat_text
"""
