# V1.0-B Implementation Summary

## 1. Repository Assessment

开发前项目已有 Plan / Action / Tools / Memory / Agent、V0.6 Grounding 和 V1.0-A Runtime。
本版复用 `main.execute_runtime → AgentRuntime.run`、ResearchResult、RuntimeTrace、ResearchTrace、
每轮 RetrievalTrace、GroundingTrace、Evidence、DocumentChunk、SearchPlan、AnswerClaim。
没有修改研究策略、prompt、检索决策、Runtime 预算实现或 Grounding verifier。

## 2. Files Added

```text
src/eval/__init__.py
src/eval/models.py
src/eval/dataset.py
src/eval/runner.py
src/eval/extractor.py
src/eval/components.py
src/eval/report.py
src/eval/compare.py
src/eval/metrics/__init__.py
src/eval/metrics/base.py
src/eval/metrics/planner.py
src/eval/metrics/retrieval.py
src/eval/metrics/agent.py
src/eval/metrics/grounding.py
src/eval/metrics/runtime.py
src/eval/judges/__init__.py
src/eval/judges/llm_judge.py
tracker/eval.py
eval_data/README.md
eval_data/end_to_end/demo.jsonl
eval_data/planner/demo.jsonl
eval_data/critic/demo.jsonl
eval_data/verifier/demo.jsonl
eval_data/retrieval/demo.jsonl
tests/eval/__init__.py
tests/eval/conftest.py
tests/eval/test_dataset.py
tests/eval/test_metrics.py
tests/eval/test_runner.py
tests/eval/test_compare_report.py
tests/eval/test_components.py
EVALUATION.md
V1_0B_IMPLEMENTATION_SUMMARY.md
```

## 3. Files Modified

- `.gitignore`：忽略 `eval_runs/`，保留 eval_data 的版本控制。
- `README.md`：增加 V1.0-B 入口、命令和文档导航。
- `ARCHITECTURE.md`：增加 Evaluation observer 的依赖方向。
- `log_dev.md`：记录实现、自动化验证与真实评测结果。

## 4. Evaluation Architecture

```text
JSONL → DatasetLoader → EvaluationRunner → existing Runtime → Result / Trace
                                                               ↓
                                   JSONL → Metric functions → Report → Compare
```

Runner 顺序执行，逐题保存。Runtime failure/timeout/budget 不中断下一题；用户取消中断整批并保存 partial report。
Metrics 函数没有模型或联网调用。LLM Judge 必须显式开启，结果与确定性指标分开。

## 5. Dataset Schema

五个小型 case 模型共享 id/question/category/difficulty 等通用字段。
E2E 可选 URL/chunk 标签；Critic 使用 evidence + expected_sufficient；Verifier 使用 AnswerClaim + Evidence + label；
Retrieval 使用固定 DocumentChunk 候选和相关性标签。全部 demo 明确 `human_validated=false`。
每次 run 保存所选 dataset 快照和 SHA256；改变 limit、标签或题目顺序后不能直接比较。

## 6–7. Metrics Implemented / Definitions

| 类别 | 指标 |
|---|---|
| Planner | success、query count valid、empty、exact / normalized duplicate rate |
| Critic | component success、accuracy、TP/TN/FP/FN、FPR/FNR |
| Verifier | component success、accuracy、precision、recall、F1（supported 为正类） |
| Retrieval | unique URL / chunk Recall@10、Precision@10、MRR@10、funnel |
| Agent | rounds、initial / follow-up query count、useful follow-up、evidence gain、stop reasons |
| Grounding | draft/verified/unsupported/rewritten/rewrite passed/dropped/source counts、支持率、映射覆盖率、coverage |
| Runtime | success/failure/timeout/budget/cancellation、retry rate |
| Resource | calls、retries/timeouts、average/median/P95 latency、stage timings |

关键公式：Recall=命中/相关总数；Precision=命中/实际返回数；MRR=平均首个命中排名倒数；
Useful follow-up=新增证据的补搜轮/补搜轮；Grounding pass=verified/draft；Runtime success=SUCCEEDED/cases；
P95=排序后的 ceil(0.95×n) 项。零分母 N/A。详细逐指标定义与观察分母见 [EVALUATION.md](EVALUATION.md)。

## 8. Trace Reuse

- RuntimeTrace：status、latency、calls、retries、timeouts、stage durations、budget limits。
- ResearchTrace：rounds、queries、new evidence、stop reason、decision traces。
- RetrievalTrace：逐轮 funnel 和阶段耗时；最终 Evidence 提供 URL/chunk 身份及正文。
- GroundingTrace：claim counts、final mapping、coverage；不重新调用 verifier。
- Rewrite recovery 分母使用现有 trace 的非 null 改写数；不冒充全部尝试次数。

## 9. Tests

全量自动化测试 272 passed：已有 214 + 新增 eval 58。全部使用固定数据或 Fake 组件。
覆盖数据校验、指纹、隔离失败、取消传播、Runtime 接入、所有指标公式、报告、类别分解、
比较方向、dataset mismatch、pass/warn/fail、N/A 消失、NaN 拒绝和默认禁用 Judge。
compileall、pip check、git diff --check 通过；项目未配置 ruff/mypy，未引入新依赖。

## 10. Demo Evaluation

2026-09-06，本机 Ollama `qwen3:8b`，沿用当前配置，真实联网 smoke：
`eval_runs/v1.0b-smoke/report.md`。5/5 Runtime succeeded，共 1227.62 秒（约 20.46 分钟）。
平均每题 245.52 秒，median 240.55 秒，P95 313.59 秒。
发生过 DDGS timeout、HTTP 403 和模型 timeout，已有 Runtime 重试/来源隔离后完成。
这是开发 smoke，不是人工 gold 能力结论。

第一次完整基线 `v1.0b-qwen3-base` 在保存 2 题后遭外部执行中断，`completed=false`。
保留该目录，不将 partial run 当作 baseline；完整重跑使用 `v1.0b-qwen3-base-rerun`。
超时注入 `v1.0b-timeout-candidate` 已完成 8 题，8/8 `timed_out`，平均 0.062 秒；
确认 timeout 不阻止下一题。将 5 题 smoke 与该 8 题 run 比较，正确以 fingerprint mismatch 拒绝（exit 1）。

额外真实组件与故障评测：

| Run | 结果 |
|---|---|
| `v1.0b-budget` | 2/2 budget_exceeded；均在 llm_calls=1 停止，原因 max_llm_calls |
| `v1.0b-planner` | 2/2 返回有效 SearchPlan，query count valid=100%，无空/重复 query |
| `v1.0b-critic` | TP=0、TN=1、FP=0、FN=1；accuracy=50% |
| `v1.0b-verifier` | TP=1、TN=1、FP=0、FN=0；F1=1 |
| `v1.0b-retrieval` | 1 个固定语料 case；URL/chunk Recall@10=1、Precision@10=0.5、MRR@10=1 |

以上分类/检索参照只是明确标记的 demo 标签，不是人工 gold，样本极少。
Critic FN 案例是“What is caching?”：它认为简短定义仍缺 purpose/components，因此要求补搜。
这提示了 evaluator 可能过严，不能凭两道题直接认定模型整体性能。

完整基线：[v1.0b-qwen3-base-rerun report](eval_runs/v1.0b-qwen3-base-rerun/report.md)。
UTC 2026-09-07 09:26:55–09:54:35，8/8 cases 全部执行；6 succeeded、2 budget_exceeded。
累计 Runtime 耗时 1658.56 秒（约 27.64 分钟）。全部沿用现有配置，没有调整行为来提高分数。

| Case | Status | Runtime 秒 | 关键结果 |
|---|---|---:|---|
| rag_001 | succeeded | 270.87 | 1 round；11 verified claims |
| retrieval_001 | succeeded | 224.35 | 1 round；16 verified claims |
| moon_001 | succeeded | 165.79 | 1 round；9 verified claims |
| cache_zh_001 | succeeded | 289.10 | 2 rounds；structured planner 失败，legacy fallback，grounding_verified=false |
| lora_001 | succeeded | 167.16 | 1 round；4 verified claims |
| robotics_001 | succeeded | 240.63 | 3 rounds；no_new_evidence；7 verified claims |
| vla_001 | budget_exceeded | 179.56 | max_crawl_requests=30 |
| cost_001 | budget_exceeded | 121.10 | max_crawl_requests=30；没有产出最终答案 |

## 11. Scorecard

| Smoke 指标 | 实际值 |
|---|---:|
| Runtime success | 5/5 |
| 平均 research rounds | 1.2 |
| Useful follow-up | 1/1（样本很少） |
| Draft / verified / dropped claims | 28 / 27 / 1 |
| Grounding pass | 96.43% |
| Citation mapping coverage | 100%（不是人工正确率） |
| 平均 LLM calls | 5.8 |
| Retry rate | 80% |
| URL Recall@10 / Verifier F1 | N/A（E2E 未提供这些标签） |

原始记录抽查：smoke 的 `moon_001` 保留 “The Moon orbits Earth.”，映射到 E1/E3/E6；
E1 明确解释 geocentric orbit，E3 包含 orbit around Earth。另一个 claim “Earth is a planet.”
因 verifier 认为所引 E6 未明确支持而被删除，但 E6 存在间接指称，值得人工复核是否过严。
因此，检测到 drop 只能证明该行为被记录，不能自动宣称它是一次正确纠错。

完整 baseline scorecard（同时看观察数量）：

| 指标 | 实际值 |
|---|---:|
| Runtime success / budget exceeded | 75% / 25%（8 题） |
| Avg / median / P95 Runtime latency | 207.32 / 201.96 / 289.10 秒 |
| Agent trace count / avg rounds | 6 / 1.5 |
| Useful follow-up / evidence gain | 2/3 / 平均新增 3.67 |
| Stop reasons（有 ResearchTrace 的 6 题） | sufficient=5；no_new_evidence=1 |
| Grounding trace / fallback | 5 / 1 |
| Draft / verified / dropped | 47 / 47 / 0 |
| Grounding pass / citation mapping coverage | 100% / 100%（仅这 5 条有效 trace，不是人工正确率） |
| Coverage adequate / avg missing aspects | 100% / 0.2（系统自评） |
| Avg LLM calls / search requests / crawl requests | 5.25 / 11.125 / 19（8 题） |
| Retry rate | 100% |
| URL Recall@10 / Verifier F1 / LLM Judge | N/A / N/A / disabled |

值得人工复核：`robotics_001` 的 CoverageResult 同时给出 `adequate=true` 和 `missing_aspects=["limitations"]`。
Harness 如实记录，没有擅自修改现有 evaluator 的判断，所以“Coverage adequate=100%”不能当完整性真值。
两个困难题因 budget 未回答，中文题未通过 grounding；不能从其余 47/47 verified 推断整体答案可靠。

## 12–13. Comparison / Regression Example

[实际比较报告](eval_runs/v1.0b-fault-comparison.md)：baseline 与 8-case timeout candidate 指纹一致，
结果 **FAIL，CLI exit 2**。Runtime success 从 0.75 降至 0（下降 75 个百分点），超过 0.05 gate；
citation coverage 从 1 变 N/A，标记 lost_observation 并 FAIL，不把它当改进。
平均耗时从 207.32 秒降至 0.062 秒只是立即超时的副作用，不能叫性能优化。
两边 Recall@10 都缺 gold，gate 不可评估并明确显示 unavailable。

此比较验证测量与回归检测链路，不代表模型优化收益。单元测试另外覆盖 PASS/WARN/FAIL、
阈值方向与可配置 K。中断的 partial run 不参与比较。

## 14. How To Run

完整命令见 [EVALUATION.md](EVALUATION.md)，包含 validation、5-case smoke、完整 baseline、
component evaluation、可选 Judge、故障实验与 compare。

## 15. Files I Should Read / What I Should Learn

推荐的 12 个代码位置和 25 个学习概念见 [EVALUATION.md](EVALUATION.md)。
先读 dataset → runner → extractor → retrieval metrics → report → compare，理解实验输入、执行和结果如何关联。

## 为 Base vs LoRA / QLoRA 预留的部分

- 实际选题 snapshot + fingerprint：保证同一考卷。
- commit + dirty + source fingerprint + Settings + 模型 digest：标识代码、配置和模型。
- model_variant / adapter_name / prompt_version：记录实验标签，不负责加载 adapter。
- Component fixtures：分别检查 Planner/Critic/Verifier，定位微调收益和副作用。
- per-category scorecard + regression gates：避免局部收益掩盖其他能力退化。
- raw answers / evidence / traces：支持人工错误分析；不使用 Judge opinion 代替 gold。

## 你需要亲自完成什么

人工审核并冻结 gold；抽查答案与引用证据；在目标机器上跑并备份正式 baseline；
审核本次修改后提交代码；后续一次只改变一个实验变量，再运行同一 gold 的 candidate。
本版已经提供这些操作的工具和命令，未进行微调或自动制造人工标签。
