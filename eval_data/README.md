# Evaluation datasets

这些 JSONL 是 **DEMO / synthetic fixture，not human-validated gold**。`metadata.demo=true` 和
`human_validated=false` 标记每题。示例标签用于演示评测流程，不能用于宣称模型达到某个能力水平。
`example.org` 下的 component passages 是合成文本，不是抓取的来源。

每行必须有唯一 `id` 和非空 `question`。`task` 默认 `end_to_end`，也可选
`planner / critic / verifier / retrieval`。可选字段包括 category、difficulty
(`easy / medium / hard`)、expected_aspects、reference_answer、notes、metadata。
未知字段、无效 JSON、重复 ID、普通任务的重复问题都会报出文件和行号。
Critic/Verifier 允许同一问题配不同 evidence，但拒绝完全重复的 fixture。

| Dataset | 额外输入 | 标签 / 输出解释 |
|---|---|---|
| `end_to_end/demo.jsonl` | 8 个中英文问题 | 可选 relevant_urls / relevant_chunk_ids；expected_aspects 只给人工审阅或可选 Judge 使用 |
| `planner/demo.jsonl` | 原始问题 | 检查组件最终返回的 queries，不反推被 parser 修复前的输出 |
| `critic/demo.jsonl` | evidence、executed_queries | expected_sufficient；expected_missing_aspects 保存供人工审阅，不做关键词伪匹配 |
| `verifier/demo.jsonl` | claim、evidence | label 为 supported/unsupported，正类为 supported；E1 按 evidence 列表顺序分配 |
| `retrieval/demo.jsonl` | 固定 candidates（现有 DocumentChunk 格式） | relevant_urls / relevant_chunk_ids；chunk ID 必须存在于候选语料且唯一 |

Evidence 字段直接使用现有类型：text、url、title、chunk_index、embedding_score、rerank_score。
固定 fixture 的分数可填 0 / null，组件判断依据仍是正文。

URL 标签应来自人工审核，**不能拿当前搜索输出当 gold**。URL-level 排名按项目现有 URLNormalizer
标准化并保留首见顺序去重。Offline retrieval 用候选的原始 chunk ID；E2E 的 Evidence 没有原始 ID，
因此使用已有公开转换方法的 `URL#chunk-N` 身份（未经 URL 标准化的原始 URL）。二者不能混填。
没有可信 relevant_urls 时省略字段，Recall/Precision/MRR 为 N/A。

人工扩充建议：

- 30–50 条 end-to-end：问题范围、预期方面、可靠来源、是否应明确证据不足。
- 50–100 条 planner：包括中文、英文、比较、窄定义和多方面问题。
- 50–100 条 critic：固定证据，人工判断 sufficient，并说明确切缺口。
- 100+ verifier claim/evidence pairs：覆盖直接支持、数字臆造、过度概括、局部支持、无效引用。
- Retrieval：固定候选文本并人工标注 relevant URL/chunk；保留困难负样本。

审核后创建新的 `gold_v1.jsonl`，记录审核者、日期和来源说明，再将 `human_validated` 标为 true。
保持 gold 文件版本固定；修改标签也会改变 fingerprint，须重新运行 baseline。
JSONL 随 Git 管理；`eval_runs/` 是忽略的实验产物，请自行备份重要 baseline。

```bash
python -m src.eval.dataset --validate eval_data/end_to_end/demo.jsonl
python -m src.eval.dataset --validate eval_data/verifier/demo.jsonl
```
