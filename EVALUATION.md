# V1.0-B — Evaluation Harness

Evaluation Harness 是一套固定 Dataset、统一 Runner、指标计算、报告与版本比较工具。
少量手工 query 能发现问题，却难以确认一次 prompt 修改是否让其他类别退化。
本版把每题结果和失败都保存下来，再用多维 scorecard 比较相同测试集。

## 运行命令

```bash
cd /home/yanwq/tracker
source .venv/bin/activate

# 检查数据；不加载模型、不联网
python -m src.eval.dataset --validate eval_data/end_to_end/demo.jsonl

# 真实联网 + 当前 .env 配置 + 现有 AgentRuntime
python -m tracker.eval --dataset eval_data/end_to_end/demo.jsonl \
  --run-name smoke-5 --limit 5

# 完整 demo baseline（8 题；run-name 必须尚不存在）
python -m tracker.eval --dataset eval_data/end_to_end/demo.jsonl \
  --run-name base-01 --model-variant base

# 保持数据集固定，修改显式模型/配置后另存 candidate
python -m tracker.eval --dataset eval_data/end_to_end/demo.jsonl \
  --run-name candidate-01 --model-variant candidate --prompt-version experiment-01

# 本地回归门禁；FAIL 退出 2，输入错误退出 1，PASS/WARN 退出 0
python -m src.eval.compare eval_runs/base-01/metrics.json \
  eval_runs/candidate-01/metrics.json --output eval_runs/comparison-01.md

# 可选 LLM Judge；会额外花推理时间，默认不启用
python -m tracker.eval --dataset eval_data/end_to_end/demo.jsonl \
  --run-name judged-smoke --limit 1 --judge

# Component evaluation：固定输入，直接测现有组件
python -m tracker.eval --dataset eval_data/planner/demo.jsonl --run-name planner-demo
python -m tracker.eval --dataset eval_data/critic/demo.jsonl --run-name critic-demo
python -m tracker.eval --dataset eval_data/verifier/demo.jsonl --run-name verifier-demo
python -m tracker.eval --dataset eval_data/retrieval/demo.jsonl --run-name retrieval-demo

# 人为故障实验，不改 .env；不要当默认模型质量 baseline
MAX_RUN_SECONDS=0.05 python -m tracker.eval \
  --dataset eval_data/end_to_end/demo.jsonl --run-name timeout-demo --limit 2
MAX_RUNTIME_LLM_CALLS=1 python -m tracker.eval \
  --dataset eval_data/end_to_end/demo.jsonl --run-name budget-demo --limit 2

# 所有自动化测试都是离线 Fake；真实评测不会被 pytest 自动启动
pytest -q
```

旧入口 `python -m tracker.cli "问题"` 继续使用。评测沿用 `.env` 的 GPU、模型、预算、prompt 和检索设置；
没有用于提高分数的隐藏 eval mode。`--model-variant / --adapter-name / --prompt-version` 仅记录实验标签，
**不会加载 adapter 或切换模型**；实际模型仍由 `OLLAMA_MODEL` 等已有配置选择。

## 数据流与结果文件

```text
JSONL → DatasetLoader → EvaluationRunner → main.execute_runtime → AgentRuntime
                                                   ↓
                                      RuntimeResult + existing traces
                                                   ↓
                                      Extractor → Metrics → Report → Compare
```

Component runner 直接调用 Planner、Critic、Verifier，或固定语料上的 Retriever/Reranker。
E2E runner 必须经过 Runtime。业务代码不导入 `src.eval`。

每次运行创建独立目录；已存在的 run-name 会报错，避免覆盖 baseline：

```text
eval_runs/{run_name}/
├── config.json       # 配置、实际选题 fingerprint、完整 dataset fingerprint、Git、模型元数据
├── dataset.jsonl     # 本次选题的规范化快照
├── cases.jsonl       # 每题状态、答案、来源、final evidence、原有 traces、标签、可选 Judge
├── metrics.json      # Scorecard、样本数、category/difficulty 分布、按类别指标
└── report.md         # 人可读报告、失败表与测量限制
```

Fingerprint 是按样本顺序序列化规范化 JSON 后的 SHA256。空行、JSON key 顺序和无意义空格不影响 hash；
题目、标签、样本顺序或 `--limit` 变化会影响实际评测指纹。5 题 smoke 不能直接和 8 题 full 比分数。
Git commit 加 dirty 标记及代码内容 SHA256 用于识别未提交实验；本地模型 digest 尽力从 Ollama tags 读取，无法读取则 null。
配置使用白名单，忽略未来可能出现的密钥字段，
不导出整个环境变量或代理凭证。记录 Python 和主要依赖版本，不声称这些能消除模型/网络非确定性。

每题完成后立即 flush/fsync。一个 Runtime failure 不阻止下一题；用户 Ctrl+C / cancellation 会终止整批，
保存已完成的行与标记为 incomplete 的报告。比较器拒绝 incomplete runs。
机器掉电、SIGKILL 或磁盘写满无法保证最终报告生成，但已经保存的逐题数据仍可用于分析。

## 指标定义与分母

缺标签和零分母统一返回 JSON null，Markdown 显示 **N/A**。N/A 不表示 0，也不表示通过。
汇总时必须一起看 `case_count / trace_count / labelled_count / coverage_count` 等观察数量。

| 指标 | 规则 |
|---|---|
| Planner success | 成功返回 SearchPlan 的 component cases / planner cases，包含一次已有 parser repair 的效果 |
| Query count valid | 1 ≤ queries 数量 ≤ 配置上限的成功输出占比 |
| Empty query rate | 空白 queries / 全部 queries，先逐题计算，再对可观测输出平均 |
| Exact duplicate rate | (query 数量 − unique 原始 query 数量) / query 数量 |
| Normalized duplicate rate | 同上，但先 casefold + 合并空白；不是语义多样性 |
| Critic accuracy | (TP+TN) / 有输出且有标签的 cases；正类 sufficient |
| Critic FPR / FNR | FP/(FP+TN)；FN/(FN+TP) |
| Verifier precision | TP/(TP+FP)，正类 supported |
| Verifier recall | TP/(TP+FN) |
| Verifier F1 | 2TP/(2TP+FP+FN)；有错误但 TP=0 时为 0，完全无正类分母时 N/A |
| Retrieval Recall@K | top-K unique URL 命中数 / 全部 relevant URL 数 |
| Retrieval Precision@K | top-K 命中数 / 实际返回的 unique URL 数；返回不足 K 使用实际数量 |
| Retrieval MRR@K | 每题第一个相关项在 top-K 内排名 r，RR=1/r；无命中 0；最后按题平均 |
| Chunk retrieval | 相同公式，identity 换成 chunk ID |
| Avg research rounds | 每题完整 ResearchTrace 的 round 数平均；失败 run 的已启动轮数另见 Runtime counter |
| Initial query count | 初始 SearchPlan 的 queries 数平均 |
| Follow-up query count | 已执行的 follow-up rounds 中 queries 数平均 |
| Useful follow-up rate | 新增 unique Evidence >0 的补搜轮数 / 全部补搜轮数，排除第一轮 |
| Avg evidence gain | 所有补搜轮新增 unique Evidence 总数 / 补搜轮数 |
| Stop reasons | ResearchTrace.stop_reason 的计数分布 |
| Grounding pass | 所有有效 grounding traces 的 verified 总数 / draft 总数 |
| Unsupported claim rate | 初次不支持总数 / draft 总数 |
| Claim drop rate | dropped 总数 / draft 总数 |
| Rewrite recovery | rewrite_passed / rewritten；现有 trace 的 rewritten 只数非 null 改写，**不是全部尝试** |
| Citation coverage | 最终 claim 有非空、真实存在的 Evidence IDs 且 supported / final factual claims |
| Verifier-passed citation rate | 当前 verified-only 输出路径与上述 mapping 覆盖率相同；不是人工正确率 |
| Coverage adequate | 有 CoverageResult 的 verified 回答中 adequate=true 的占比 |
| Avg missing aspects | 上述 CoverageResult.missing_aspects 数量平均 |
| Runtime success/failure/timeout/budget/cancellation | 对应最终 status 的 E2E cases / 全部已记录 E2E cases |
| Retry rate | 有至少一次 retry 的 Runtime traces / 可用 Runtime traces |
| Resource proxies | Runtime counters 的 search/crawl/LLM calls、retries、timeouts、started rounds 的平均 |
| Avg / median latency | 已记录 E2E cases 的 Runtime total_duration；setup exception 使用实际计时并标识原因 |
| P95 latency | 排序后的第 ceil(0.95×n) 个值（1-based nearest rank） |
| Stage latency | 平均已有 stage_durations 的实际观察值；缺失 stage 不假装为 0 |

Retriever 的 funnel 保存每轮原始计数，并报告跨轮之和的平均。跨轮 unique_urls 之和不等于整个任务的独立来源数。
final evidence 计数来自最终全局重排结果。默认 K=10；真实最终 evidence 可能少于 10。
没有 relevant URLs/chunks 的 demo E2E 题不会产生 Recall 分数；没有 component 标签也不会伪造 Verifier F1。

失败 run 可能没有 ResearchResult，Agent/Grounding/Funnel 指标仅计算存在的 trace，报告其样本数。
Runtime failure 一直留在全体可靠性分母。Legacy generation fallback 的零计数占位 trace 不进入 verified-grounding 分母。
没有完整 claim mapping 时 citation coverage 为 N/A。现有 Coverage Checker 的判断是系统内部评估，非人工 completeness truth。

并发 Search/Reading 的已有时长可能是请求累加，和墙钟时间不同；不同 stage 也可能重叠，不能相加后称总耗时。
本地模型不虚构美元成本，以调用数与时间作为资源代理。Judge 的耗时独立保存在 llm_judged，不混入 Runtime LLM calls。

## Comparison 与回归门禁

比较前检查 schema、实际选题 fingerprint、case IDs/order、K 和 completed 状态。输出每项 baseline、candidate、
delta、方向、变化判断与 gate。轮数、证据数等计数没有统一的“越多越好”，标为 neutral。

默认 gate：

- Runtime success、citation coverage、Recall@10 下降 **超过 0.05（5 个百分点）** → FAIL。
- 平均 latency 上升 **超过 baseline 的 20%** → WARN。
- 缺标签的 gate 不进行质量判定；baseline 有观察而 candidate 丢失时不能当改进，会警告或按 gate 严重级别失败。
- 全部绝对阈值使用原始比率差，不把 0.05 当 5% 相对变化。Judge 分数不参与自动 gate。

可选 `--thresholds file.json` 替换默认规则；未知指标、无方向指标、负值/NaN 阈值会拒绝：

```json
{
  "runtime.runtime_success_rate": {"tolerance": 0.02, "severity": "fail"},
  "resource.avg_latency_seconds": {"tolerance": 0.30, "severity": "warn", "relative": true}
}
```

PASS 只表示这次可观测 gate 未超阈值。要判断能力变化，还需人工标签、按类别分析、重复在线运行。
当前没有统计显著性检验，也没有把多维指标压成一个 Tracker 总分。

## What I Should Learn From V1.0-B

1. **Unit Test vs Evaluation**：前者验证公式/逻辑按定义执行；后者测当前模型在真实问题上表现如何。测试全绿不代表答案可靠。
2. **Benchmark**：固定题目、执行条件和评分规则，提供可比较的“考卷”。本版只是 demo 考卷。
3. **Golden Dataset**：经过人工审核的题目、证据和标签，不是把 LLM 自动输出换个名字。
4. **Ground Truth**：某次评测可信的参照，如人工确认哪些 URL 真正相关；无参照就不能算召回率。
5. **Component Evaluation**：只测 Planner/Critic/Verifier，能定位是哪一环造成退化。
6. **End-to-End Evaluation**：题目经过整个 AgentRuntime，包含网络、检索、决策、生成与故障。
7. **Offline vs Online**：固定文本更可重复；在线网页更贴近使用场景，但会随时间变化。固定文本上的模型推理也不保证逐字确定。
8. **Recall@K**：人工知道 4 个相关网页，top-10 找到 3 个，Recall=3/4。它问“该找到的漏了多少”。
9. **Precision@K**：上例若实际返回 6 个网页，Precision=3/6。它问“拿回来的有多少有用”。
10. **MRR**：第一个相关网页排第 4，则 RR=1/4；对多题求平均，衡量有用结果是否靠前。
11. **Accuracy / Precision / Recall / F1**：Verifier accuracy 看总判对率；precision 看判 supported 的有多少确实支持；recall 看真正支持的找回多少；F1 平衡后两者。
12. **FP / FN**：Critic 的 FP 是证据不足却判够，会提前停止；FN 是证据已经够却继续搜，增加延迟。Verifier 的 FP 会放行不支持的事实。
13. **Agent Metrics**：两个相同答案可能分别花 1 轮和 3 轮得到。需要观察行动、证据增长和停止原因才能解释差异。
14. **Useful Follow-up Rate**：两次补搜仅一次找到新 Evidence，指标=1/2。低值提示补搜低效，但新增证据不等于确实回答了缺口。
15. **Evidence Gain**：第 2 轮平均新增 3 条、第 3 轮新增 0 条，可以帮助选择轮数预算；这是资源决策依据而非自动断言最后一轮无价值。
16. **Grounding Metrics**：观察支持、改写、删除和引用映射，发现模型是否常写出超出证据的内容。
17. **Citation Coverage vs Correctness**：每句有 [1] 只能说明有编号。证据是否支持具体数字或结论需要支持判断及人工审核。
18. **Runtime Metrics**：超时、预算耗尽、retry 和完成率衡量系统是否能稳定运行。
19. **Reliability vs Quality**：成功返回“错误答案”的 run 仍可能是 Runtime success；代码测量必须保留两个维度。
20. **Latency / Quality Tradeoff**：Recall 提升 5 个百分点但耗时增加 80%，是否值得取决于用途，不能只挑最好看的分数。
21. **Regression Testing**：新版本在已有能力上明显下降。本版用固定阈值告警，未来可接 CI。
22. **Ablation Study**：明确配置分别比较 Search only、Multi Query、Embedding、Reranker、Critic 的贡献。当前只使用已有开关；没有实现的历史模式不能假装可切换。
23. **LLM-as-a-Judge**：能辅助评阅结构、相关性和覆盖，但受模型偏好和错误影响；同一个模型评自己的答案尤其不能视为 gold。
24. **Reproducibility**：保存 dataset、配置、模型名/variant、commit 和源代码 hash，才知道分数来自哪次实验。在线复现仍有网络波动。
25. **Evaluation before Fine-tuning**：先跑 Base baseline，再训练、跑 Candidate、比较。没有固定基线和可信标签，无法证明 LoRA/QLoRA 改善了目标能力。

## 需要你亲自完成的工作

1. 按 [eval_data/README.md](eval_data/README.md) 审核、扩充并冻结人工 gold，优先检查 Critic 和 Verifier 的边界案例。
2. 阅读 cases.jsonl 中的答案和 final evidence，人工抽查 claim→citation→passage，检查数字、日期、过度推断及明确拒答。
3. 在你的目标 GPU、代理和网络环境下运行完整 gold baseline，保留 config/cases/metrics/report 并备份忽略的 eval_runs。
4. 确认本次未提交开发代码后自行 commit，让后续正式实验有稳定 commit；Harness 已记录 dirty 和内容 hash，开发过程不会代你提交。
5. 后续每次只改变一个 prompt/模型/配置变量；用同一 gold 运行 candidate 和 compare，结合分类指标与人工抽查看收益。
6. 对 Base vs LoRA/QLoRA 填写 variant、adapter、prompt_version；实际模型部署与微调属于后续版本，本版不训练模型。

## 推荐阅读位置

| 文件 | 学习点 |
|---|---|
| `src/eval/models.py` | 小型任务 schema 与已有 Evidence/Claim 复用 |
| `src/eval/dataset.py` | validation 与实验身份 fingerprint |
| `src/eval/runner.py` | 逐题故障隔离、Runtime 边界、取消和持久化 |
| `src/eval/extractor.py` | Observer 消费已有 trace |
| `src/eval/metrics/base.py` | 分母、缺失值、分类混淆矩阵、P95 |
| `src/eval/metrics/retrieval.py` | 排名去重、Recall/Precision/MRR |
| `src/eval/metrics/agent.py` | 从行为轨迹衡量补搜价值 |
| `src/eval/metrics/grounding.py` | 真实 mapping、fallback 与分母选择 |
| `src/eval/report.py` | 多维聚合、类别分解与测量限制 |
| `src/eval/compare.py` | 方向、阈值与实验可比性 |
| `src/eval/judges/llm_judge.py` | 结构化但非 gold 的可选评阅 |
| `tracker/eval.py` | 复用现有执行入口、记录可复现配置 |
