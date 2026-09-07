# Tracker V1.0-B — Evaluation Harness

V1.0-B 增加 JSONL 数据集校验、批量评测、指标报告与 baseline/candidate 回归比较。
完整命令、每项指标公式、学习说明和人工验收步骤见 [EVALUATION.md](EVALUATION.md)。
示例数据是 demo，尚未经过人工 gold 审核；无可信标签的指标显示 N/A。

```bash
python -m src.eval.dataset --validate eval_data/end_to_end/demo.jsonl
python -m tracker.eval --dataset eval_data/end_to_end/demo.jsonl --run-name smoke --limit 5
python -m tracker.eval --dataset eval_data/end_to_end/demo.jsonl --run-name base --model-variant base
python -m src.eval.compare eval_runs/base/metrics.json eval_runs/candidate/metrics.json
```

Evaluation 通过现有 `execute_runtime → AgentRuntime` 运行 E2E；component evaluation 使用固定输入调用单个组件。
指标只读取 RuntimeTrace、ResearchTrace、RetrievalTrace、GroundingTrace，核心业务不依赖 eval。
运行产物保存在 Git 忽略的 `eval_runs/`，人工数据保存在可版本管理的 `eval_data/`。
Runtime success 衡量完成率，不能替代 answer correctness。LLM Judge 默认关闭，开启后的主观评分单独呈现。

Unit tests 检查代码逻辑；evaluation 衡量实际能力。先保留 Base baseline，后续才能用同一数据集判断
LoRA/QLoRA 或 prompt 修改的收益及成本。人工 gold 的构建方法见 [eval_data/README.md](eval_data/README.md)。

Tracker 是一个从底层学习 Search Agent / Deep Research Agent 的本地项目。它使用 Ollama/Qwen3 进行搜索规划、证据评估和 grounded generation，使用 DDGS + Wikipedia 搜索公开网页，并以 BGE-M3 + bge-reranker-v2-m3 在本地完成段落检索。项目不调用云端 LLM，也不依赖 LangChain、LlamaIndex、LangGraph 或 Agent framework。

V0.1–V0.6 依次实现“能搜、能读、搜得广、搜得准、搜得深、答得稳”。V1.0-A 不改变这些研究决策，而是在 `ResearchAgent` 外增加可靠的单机执行 Runtime：每次任务都有生命周期、预算、超时、有限重试、取消传播、结构化结果、可观测 trace，以及按研究轮恢复的轻量 checkpoint。

## 标准 Agent 项目结构

当前核心代码按 Agent 的五种职责组织。目录表示职责边界，不表示五个独立 Agent：

```text
src/
├── agent/                 # Orchestrator + Evaluator
│   ├── research_agent.py  # Observe → Evaluate → Decide → Act 闭环
│   ├── critic.py          # 证据充足性评估
│   └── models.py          # Agent 输出与 trace
├── plan/                  # Plan
│   ├── models.py          # SearchPlan
│   └── search_planner.py  # 原问题 → 初始查询计划
├── action/                # Action
│   ├── models.py          # ResearchAction / StopReason
│   └── policy.py          # Critic 结果 → Search 或 Finish
├── tools/                 # Tools
│   └── research.py        # Search → Read → Retrieve 的一次工具调用
├── memory/                # Memory
│   ├── evidence_pool.py   # 跨轮 Evidence 工作记忆
│   └── research_state.py  # 查询历史与可恢复状态
├── runtime/               # 执行治理：预算、超时、重试、checkpoint
├── grounding/             # claim 验证、改写、coverage、渲染
└── search/crawling/...    # ResearchTool 使用的底层能力实现
```

运行时的数据流为：

```text
Question
  → Plan: SearchPlanner 生成初始 queries
  → Action: ResearchActionPolicy 选择 Search
  → Tool: ResearchTool 执行 Search → Read → Retrieve
  → Memory: EvidencePool 累积证据，ResearchState 记录历史
  → Agent: EvidenceCritic 评估，Policy 选择 Search Again 或 Finish
  → Grounding: claims → verify → rewrite/drop → render
```

依赖规则：`ResearchAgent` 可以组合 Plan、Action、Tool、Memory；Tool 不读取 Agent Memory，也不决定下一步；Memory 不执行网络或模型调用；Action Policy 是确定性规则，不执行副作用；Runtime 管理执行可靠性，不替 Agent 做研究决策。

`src/planner`、`src/models/search_plan.py`、`src/agent/evidence_pool.py` 和 `src/agent/research_round.py` 暂时保留为轻量兼容路径，只做 re-export；真实实现各自只有一份。新代码应使用 `src.plan`、`src.action`、`src.tools`、`src.memory`。

## V1.0-A 的核心变化

```text
RunRequest
  → AgentRuntime
       ├── run_id + lifecycle
       ├── overall / operation timeout
       ├── retry classification + backoff
       ├── rounds / search / crawl / LLM budgets
       ├── cancellation + child cleanup
       ├── atomic checkpoint / stage-level resume
       └── RuntimeTrace + structured RuntimeResult
  → existing ResearchAgent (V0.5)
  → existing Grounding Layer (V0.6)
```

`ResearchAgent` 仍负责“搜什么、证据是否够、是否补搜、如何生成 grounded answer”；`AgentRuntime` 只负责“任务怎样可靠执行”。Runtime 没有复制 Search、Crawler、Retrieval、Critic 或 Grounding 逻辑。

## Agent 与 Agent Runtime

- Agent Logic 决定任务行为：Search → Read → Evaluate → Search Again → Grounded Answer。
- Execution Runtime 管理行为的边界：任务状态、deadline、工具调用配额、暂时故障重试、取消、持久化和诊断。

两者分离后，研究策略可以独立演进；CLI 或未来 API 也能稳定获得统一的 `RuntimeResult`，而不是只能依赖顶层异常。

## 生命周期与结构化结果

每次新任务使用 UUID 生成唯一 `run_id`，状态只使用 `RunStatus`：

```text
PENDING → RUNNING → SUCCEEDED
                  ├→ FAILED
                  ├→ CANCELLED
                  ├→ TIMED_OUT
                  └→ BUDGET_EXCEEDED
```

成功结果包含 `research_result` 与 answer；失败、超时、取消或预算终止也会返回 `RuntimeResult`，其中有明确 `status`、`termination_reason`、`RuntimeErrorInfo` 和 `RuntimeTrace`。内部仍使用异常传播错误，但异常不是 Runtime 对外的唯一协议。

Agent State 与 Runtime State 不同：

- Agent State 表示“研究到了哪里”：EvidencePool、executed queries、已完成 round、下一轮 queries。
- Runtime State 表示“任务怎么运行”：status、开始/结束时间、当前 stage、调用计数、retry 和 timeout。

## Error Taxonomy 与有限重试

Runtime 区分 `RetryableError` 与 `NonRetryableError`，并提供 Search、Crawler、Model operation timeout 和 Budget/Checkpoint 错误。以下暂时故障可以重试：transport error、operation timeout、HTTP 429、502、503、504。配置错误、schema/structured-output 错误、模型缺失和程序错误不会盲目重试。

Retry 使用有上限的指数退避和小幅 jitter：

```text
attempt 1 fails → base delay
attempt 2 fails → 2 × base delay
...             → capped max delay
```

Network 与 LLM 使用独立最大尝试次数。Retry 只是重复同一个失败操作；Agent Iteration 是 Critic 根据新 Evidence 生成新的行动，两者不是一个概念。

## Operation Timeout 与 Overall Timeout

- `SEARCH_TIMEOUT` / `HTTP_TIMEOUT` / `LLM_TIMEOUT` 限制单次外部 I/O。
- `MAX_RUN_SECONDS` 限制整次 research task。

Runtime 使用 `asyncio.timeout` 管理整体异步 deadline，并将剩余 deadline 下传给同步 Ollama HTTP transport，因此很短的 overall timeout 不会等完整模型推理结束。Operation timeout 可以按策略重试；overall timeout 是任务终止条件，不启动新的 retry。

## Runtime Budget

`RuntimeBudget` 在真正调用工具之前原子地消费以下计数：

```text
max_research_rounds
max_search_requests
max_crawl_requests
max_llm_calls
```

Semaphore 只限制“同时有多少请求”；Budget 限制“整个 run 最多发出多少请求”。预算耗尽时不会再启动超额调用，任务返回 `BUDGET_EXCEEDED` 和精确原因，例如 `max_crawl_requests`。

## Cancellation 与 Fault Isolation

`asyncio.CancelledError` 不会被普通 `except Exception` 当成 provider/页面失败吞掉。ResearchTool 为 crawl 建立显式 child tasks；父任务取消或某个终止错误发生时，会取消并等待所有 sibling tasks 清理。外部库已经启动的不可取消同步线程只能等待其自身 transport timeout，但不会再被 Runtime 接受为有效结果。

一个搜索 provider 暂时失败时，`SourceManager` 仍保留其他 provider 的成功结果；单页抓取失败也只丢弃该页。只有所有候选路径都无法形成 Evidence，或遇到全局 timeout/budget/cancellation，失败才升级到 Runtime。

## Idempotency、Checkpoint 与 Resume

默认 checkpoint 路径：

```text
.runtime/runs/{run_id}/state.json
.runtime/runs/{run_id}/trace.json
```

Checkpoint 只保存小状态：问题、SearchPlan、已完成 round、executed queries、累计 Evidence 文本及来源、下一轮 queries、Runtime counters 和 lifecycle；不保存模型权重、embedding tensor 或 HTTP client。写入流程是同目录临时文件 → flush/fsync → `os.replace`，避免半截 JSON。

恢复粒度是 stage-level，不尝试恢复函数执行到一半：

```text
completed Round 1 checkpoint
  → process stops during Round 2
  → --resume RUN_ID
  → restore Round 1 Evidence + executed queries
  → restart from Round 2
```

EvidencePool 的 `(URL, chunk_index)` 去重、executed-query 去重以及固定 checkpoint 路径共同保证幂等。损坏或缺失的 checkpoint 会返回清晰的 structured failure，不会猜测状态。

## RuntimeTrace 与结构化日志

`RuntimeTrace` 位于已有 `ResearchTrace`、`GroundingTrace` 之上，记录：run_id、问题、状态历史、开始/结束时间、总耗时、预算上限、round/search/crawl/LLM 计数、retries、timeouts、终止原因、各阶段耗时和 checkpoint path。

CLI 日志统一带 `run_id` 与标准 stage：`planning`、`search`、`reading`、`retrieval`、`evaluation`、`grounding`、`rendering`、`runtime`。`--debug-runtime` 才额外展开完整 RuntimeTrace；普通模式仍主要显示 Answer + Sources。

## V1.0-A 配置

```dotenv
LLM_TIMEOUT=120
MAX_RUN_SECONDS=900
MAX_RUNTIME_SEARCH_REQUESTS=30
MAX_RUNTIME_CRAWL_REQUESTS=30
MAX_RUNTIME_LLM_CALLS=40
NETWORK_RETRY_MAX_ATTEMPTS=2
LLM_RETRY_MAX_ATTEMPTS=2
RETRY_BASE_DELAY_SECONDS=0.5
RETRY_MAX_DELAY_SECONDS=4
RETRY_JITTER_SECONDS=0.1
CHECKPOINT_ENABLED=true
CHECKPOINT_DIR=.runtime/runs
```

默认 round budget 继续复用 `MAX_RESEARCH_ROUNDS`，避免两套含义相同的配置。

## V1.0-A 使用

普通入口保持兼容：

```bash
python -m tracker.cli "What is retrieval-augmented generation?"
```

查看完整 Runtime、Agent、Retrieval 和 Grounding trace：

```bash
python -m tracker.cli --debug-runtime --debug-agent \
  --debug-retrieval --debug-grounding "your question"
```

临时覆盖 deadline 与 budget：

```bash
python -m tracker.cli --run-timeout 300 \
  --max-research-rounds 2 --max-search-requests 12 \
  --max-crawl-requests 10 --max-llm-calls 20 "your question"
```

从最近完成的 research-round checkpoint 恢复，问题从 checkpoint 读取：

```bash
python -m tracker.cli --debug-runtime --resume run_0123456789abcdef
```

`--resume` 不能同时提供新问题。`.runtime/` 默认被 Git 忽略。

## V0.6 — Grounded Answer Generation / 答得稳

## V0.6 的核心变化

V0.5 的 Generation Layer 是一步生成：

```text
Final Evidence → ContextBuilder → Qwen3 → Free-form Answer
```

V0.6 使用结构化中间表示：

```text
Final Evidence
  → EvidenceRegistry (E1, E2, ...)
  → GroundedAnswerPlanner
  → Atomic Claims + Evidence IDs
  → CitationVerifier
       ├── supported → keep
       └── unsupported → rewrite once → verify again → keep or drop
  → AnswerCoverageChecker
  → Deterministic Renderer
  → Grounded Answer + Used Sources
```

ResearchAgent 的 Search → Critic → Search Again 闭环保持不变；V0.6 没有复制 Search、Crawler、Embedding、Reranker 或 EvidencePool。

## Grounding 是什么

Grounding 表示最终事实可以明确追溯到检索证据：

```text
Final sentence
  → AnswerClaim C3
  → Evidence E7
  → DocumentChunk #4
  → Document
  → URL
```

`E1`、`E2` 只是在当前回答内稳定的 Evidence ID，不是长期数据库 ID。最终 Renderer 再把实际使用的 URL 映射成 `[1]`、`[2]`；同一 URL 的多个 chunk 共用一个 citation number，未被 verified claims 使用的网页不会进入 Sources。

## Claim 与结构化答案

`AnswerClaim` 是一个可独立判断真假的原子事实，例如：

```text
BGE-M3 supports multilingual retrieval.
```

“模型支持多语言、比所有 reranker 更快、并且在 2026 年最流行”包含三个需要不同证据的事实，必须拆成多个 claim。`GroundedAnswerDraft` 按 section 保存 claims，每个 claim 明确携带 `evidence_ids`。组织句不需要伪装成事实 claim。

把 `Evidence → Free-form prose` 拆成 `Evidence → Structured Claims → Verify → Render`，可以在最终文字出现前检查事实与引用，也能单独删除一个失败 claim，而不丢掉其他已验证内容。

## Citation Presence 不等于 Citation Correctness

```text
Claim: Model A improves accuracy by 30%. [1]
Evidence [1]: This paper studies Model A.
```

这里虽然“有引用”，但 Evidence 没有给出 30%，所以 citation 不正确。V0.6 的 `CitationVerifier` 检查的是具体 claim 是否被引用段落直接陈述或合理蕴含，而不是只检查是否存在 `[1]`。

## Relevance 与 Entailment

- Retriever / Reranker 回答“这段内容和问题相关吗？”
- CitationVerifier 回答“只根据这段内容，能支持这个具体事实吗？”

例如 Evidence 说“论文研究四足机器人强化学习”，Claim 却说“方法将能耗降低 18%”。二者主题高度相关，但 Evidence 不能推出 18%，因此 relevant 但不 entail。搜索质量、检索质量和 grounding 质量分别对应“找到正确网页”“选出正确段落”“最终 claim 真被段落支持”，不能混为一个指标。

## Unsupported Claim：Rewrite 或 Drop

Planner 仍可能过度推断。Verifier 判定 unsupported 后，`ClaimRewriter` 最多尝试一次，把强表述弱化为 Evidence 真正支持的事实；重写后必须再次验证。没有有意义的重写，或二次验证仍失败，就删除 claim。

```text
Evidence insufficient → say less
Evidence insufficient ≠ guess more
```

Rewrite 适合“核心意思有依据但措辞过强”的情况；Drop 适合数字、因果、最高级或比较关系完全没有依据的情况。有限重写预算避免 generate → verify 无限循环。

## Faithfulness 与 Completeness

- Faithfulness：已经说出的事实是否忠实于 Evidence。CitationVerifier 负责它。
- Completeness：回答是否覆盖用户要求的重要方面。AnswerCoverageChecker 负责它。

所有留下的 claims 都正确，不代表回答完整。例如问题要求 methods + limitations + deployment cost，verified claims 只有 methods，答案仍缺两个方面。Coverage Checker 只比较 Original Question 与 Verified Claims，不触发搜索；缺失部分会被 Renderer 明确写成“证据局限”，不会补写未经支持的内容。

V0.5 EvidenceCritic 检查“Evidence 是否足够”，V0.6 Coverage Checker 检查“最终 claims 是否覆盖问题”。前者位于搜索闭环，后者位于答案生成末端。

## 为什么 Renderer 是确定性的

通过 verification 的 claim 不会再交给 LLM 自由润色，因为自由改写可能重新加入未经支持的数字或因果。`GroundedAnswerRenderer` 只做 section 排版、citation number 映射、coverage limitation 和 used-source 去重。

## Fault Tolerance 与 Legacy Mode

- Planner 或 Verifier 的 structured output 连续两次解析失败：ResearchAgent 回退到 V0.5 `ContextBuilder → AnswerGenerator`，并设置 `grounding_verified=false` 与 `fallback_reason`，不会假装答案已验证。
- 单个 claim 不支持：只重写/删除该 claim，其他 claim 继续使用。
- Rewriter 失败：删除 unsupported claims。
- Coverage Checker 失败：仍渲染已验证 claims，只是不附 coverage limitation。
- `ENABLE_GROUNDED_GENERATION=false`：显式运行 V0.5 legacy generation，便于对照实验。

## V0.6 分层架构

```text
Planning Layer       SearchPlanner
       ↓
Search Layer         SourceManager / SearchProvider
       ↓
Reading Layer        Crawler / Extractor
       ↓
Retrieval Layer      Chunk / BGE-M3 / Reranker
       ↓
Evaluation Layer     EvidencePool / EvidenceCritic
       ↓
Grounding Layer      AnswerPlanner / Verifier / Rewriter / Coverage
       ↓
Rendering Layer      deterministic citations + used Sources
```

这里 `SearchProvider` 是 Tool，执行一次搜索；`ResearchAgent` 是 Orchestrator，维护状态、预算和执行顺序；`EvidenceCritic` 是搜索阶段 Evaluator；`CitationVerifier` 是 claim 阶段 Evaluator；Renderer 是不引入新事实的输出组件。

## V0.6 配置

```dotenv
ENABLE_GROUNDED_GENERATION=true
MAX_CLAIM_REWRITE_ATTEMPTS=1
MAX_CLAIMS=30
MAX_EVIDENCE_PER_CLAIM=3
VERIFICATION_BATCH_SIZE=8
ENABLE_COVERAGE_CHECK=true
```

`MAX_CLAIMS`、每 claim 最大 Evidence 数和 verification batch size 控制本地 Qwen 的延迟与上下文规模。Planner、Verifier、Rewriter、Coverage 共用一个 structured-output helper：容忍 `<think>`、code fence、JSON 前后文字，并只做一次修复。

## V0.6 使用

普通模式只显示 Grounded Answer 和真正使用的 Sources：

```bash
cd /home/yanwq/tracker
source .venv/bin/activate
python -m tracker.cli "What is retrieval augmented generation?"
```

查看 claim、verification、rewrite、coverage 和 citation mapping：

```bash
python -m tracker.cli --debug-grounding \
  "Compare dense embedding retrieval and cross-encoder reranking in terms of architecture, speed, and typical role in a search system."
```

查看全部 Agent、Retrieval 和 Grounding trace：

```bash
python -m tracker.cli --debug-agent --debug-retrieval --debug-grounding \
  "What are the major recent approaches to reinforcement learning for robotic manipulation, what problems do they solve, and what limitations remain?"
```

Legacy 对照：

```bash
ENABLE_GROUNDED_GENERATION=false python -m tracker.cli "your question"
```

## 为什么 V0.6 是“答得稳”

Evidence 找对只说明模型获得了正确材料，不保证它在写作时不会混入记忆、夸大比较或编造数字。V0.6 把“生成”改为“生成候选事实 → 验证支持关系 → 保守处理失败事实 → 检查覆盖 → 确定性输出”。它不能保证绝对正确，但让“每个事实到底有什么证据”成为代码中的显式约束和可调试 trace。

## V0.5 — Iterative Deep Search / 搜得深

## V0.5 的核心变化

V0.4 是一次性 Linear Pipeline：

```text
Question → Search → Read → Retrieve → Answer
```

V0.5 是 Evidence-Driven Adaptive Search Loop：

```text
Question
  → Initial Search Plan
  → Full Research Round
  → Evidence Pool
  → Evidence Critic
       ├── sufficient → Final Global Rerank → Answer
       └── insufficient
             → Detect Gaps
             → Targeted Follow-up Queries
             → Full Research Round
             → Merge into Evidence Pool
             → Critic Again
```

区别不只是“多调用一次 Qwen”。Linear Pipeline 的下一步预先固定；Agent Loop 会观察当前结果、评估状态、做出继续或停止决策，并让下一步搜索依赖上一轮证据。

## 分层架构

```text
┌──────────────────────────────────┐
│ Planning Layer                   │
│ SearchPlanner: broad queries     │
└────────────────┬─────────────────┘
                 ↓
┌──────────────────────────────────┐
│ Search Layer                     │
│ SourceManager: DDGS + Wikipedia  │
└────────────────┬─────────────────┘
                 ↓
┌──────────────────────────────────┐
│ Reading Layer                    │
│ Crawl → Extract → Content Dedup  │
└────────────────┬─────────────────┘
                 ↓
┌──────────────────────────────────┐
│ Retrieval Layer                  │
│ Chunk → BGE-M3 → Cross-Encoder   │
└────────────────┬─────────────────┘
                 ↓
┌──────────────────────────────────┐
│ Evaluation Layer                 │◄──────────────┐
│ EvidencePool + EvidenceCritic    │               │
└────────────────┬─────────────────┘               │
                 │ insufficient                    │
                 └→ gaps → follow-up queries ──────┘
                 │ sufficient / budget stop
                 ↓
┌──────────────────────────────────┐
│ Generation Layer                 │
│ Global Rerank → Context → Qwen3  │
└──────────────────────────────────┘
```

单次 `ResearchTool` 调用仍完整复用 V0.4：

```text
Queries
  → Multi-Source Search
  → Domain Filter + URL Normalization/Deduplication
  → Crawl + Extract
  → Content Deduplication
  → Chunk
  → BGE-M3 Retrieval against the original question
  → bge-reranker-v2-m3
  → New Evidence
```

补搜结果不会把搜索摘要直接塞进 EvidencePool，也不会绕过抓取和检索。

## Evidence Pool 与 Agent State

`EvidencePool` 是单次 research task 的累计证据状态，而不是 `list[Evidence]` 的别名。它按首次出现顺序保存 Evidence，以 `(URL, chunk_index)` 作为稳定键去重，并提供 evidence count、unique source count 和 source URLs。

```text
Round 1: A, B
Round 2: B, C
Pool:    A, B, C
```

`ResearchState` 保存原始问题、当前轮次、EvidencePool 和 `executed_queries`。只要系统存在 Loop，就必须记得过去：否则 Critic 无法基于累计证据判断覆盖，也可能反复执行同一个 query。

State 只存在于本次任务内；V0.5 没有 persistent memory 或长期向量数据库。

## Evidence Critic / Judge

项目只有一个 evaluator：`EvidenceCritic`。它不写最终答案，只判断当前 Evidence 是否足以准确、较完整地回答原问题，输出严格的 `CriticResult`：

```json
{
  "sufficient": false,
  "confidence": 0.7,
  "missing_aspects": ["real-world deployment limitations"],
  "follow_up_queries": ["safety limitations RL physical robot manipulation"],
  "reason": "Methods are covered, but deployment limitations are weakly supported."
}
```

Critic 比较的是：

```text
Question Requirements − Available Evidence = Evidence Gaps
```

“足够”表示主要子问题已有可支持回答的相关证据，而不是网络上已不存在更多材料。Prompt 明确阻止 Critic 为追求绝对完美而无止境补搜。Critic 只看到最多 `MAX_CRITIC_EVIDENCE` 条、每条被截断的证据，不接收 embedding/rerank 分数，也不会把其内部判断交给最终 AnswerGenerator。

结构化解析与 SearchPlanner 共用同一个 JSON object extractor，可容忍 `<think>`、Markdown code fence 和 JSON 前后多余文本。首次校验失败时只修复一次；仍失败则抛出 `CriticError`，由 ResearchAgent 降级处理。

## Gap Detection 与 Adaptive Search

V0.3 Multi-Query 的所有 query 都在证据出现前生成，目标是用不同术语扩大初始召回：

```text
Question → query A + query B + query C → Search
```

V0.5 的补搜 query 在读过 Evidence 之后才生成：

```text
Evidence lacks deployment safety
  → "safety limitations RL physical robotic systems"
```

因此第二轮不是重复第一轮，而是用已观察到的 Evidence Gap 调整行动，这就是 Adaptive Search。补搜 query 会 strip、轮内精确去重、删除 `executed_queries` 中已有项，并受每轮最大数量限制。

必须区分两个目标：

- Follow-up Query 是为寻找某项缺失证据而生成的工具指令。
- Original Question 是研究任务的根目标，所有轮次的 Retriever、最终 Global Reranker 和 AnswerGenerator 都仍针对它工作。

## Search Budget 与停止条件

任何自主 Loop 都必须有 cost、latency 和 iteration budget。默认预算：

```dotenv
MAX_RESEARCH_ROUNDS=3
MAX_FOLLOWUP_QUERIES_PER_ROUND=3
MIN_NEW_EVIDENCE_TO_CONTINUE=1
```

`MAX_RESEARCH_ROUNDS=3` 包括初始搜索一轮和最多两轮补搜。ResearchAgent 实现以下明确停止原因，并记录到 `ResearchTrace.stop_reason`：

1. `sufficient`：Critic 判断当前证据足够，立即停止。
2. `max_rounds`：达到最大研究轮数。
3. `no_follow_up_queries`：Critic 判断不足但没有给出 query。
4. `duplicate_queries`：所有补搜 query 都已执行过。
5. `no_new_evidence`：补搜轮没有让 EvidencePool 达到最小增长。
6. `critic_failure`：Critic 调用或 structured parsing 失败。

“知道何时停止”和“知道下一步做什么”同样重要；缺少停止规则会导致无限循环、延迟爆炸和资源失控。

## Graceful Degradation

Critic 是控制优化组件，不是已获得 Evidence 的唯一消费者。如果搜索和检索成功，但 Critic 随后失败，系统仍有可用于回答的证据。因此 ResearchAgent 会记录 warning 和 `critic_failure`，停止补搜，继续 Final Rerank、ContextBuilder 和 AnswerGenerator。只有没有任何可用 Evidence 时，任务才明确失败。

## Final Global Reranking

每轮 Evidence 已在当轮排序，但不同轮次的局部分数和候选集合不可简单拼接后直接交给 LLM。最终阶段会把整个 EvidencePool 重新转换为候选，并以 Original Question 运行一次全局 Cross-Encoder Rerank：

```text
Round 1 Evidence + Round 2 Evidence + Round 3 Evidence
  → Final Global Rerank against Original Question
  → FINAL_EVIDENCE_TOP_K (default 8)
  → ContextBuilder
  → AnswerGenerator
```

每 URL 的 `MAX_CHUNKS_PER_DOCUMENT` 限制仍生效，避免单篇长文垄断上下文。AnswerGenerator 只看到最终筛选后的 Evidence，不看到 Critic reason、missing aspects 或中间搜索指令。

## Tool、Orchestrator、Evaluator 与 Generator

- `SearchProvider` 是底层 Tool Adapter：执行一个明确的外部搜索动作，不决定是否继续研究。
- `ResearchTool` 是 Agent 直接调用的能力链：把 queries 变成经过完整读取与检索的新 Evidence，不管理跨轮状态。
- `ResearchAgent` 是 Orchestrator：组合 Plan、Action、Tool、Memory、Critic 与最终重排。
- `ResearchActionPolicy` 是 Action Selector：把 Critic 输出确定性地变成 Search 或 Finish。
- `EvidenceCritic` 是 Evaluator：判断证据覆盖并提出 gap-driven queries，不生成用户答案。
- `AnswerGenerator` 是 Generator：只根据 Final Evidence 回答 Original Question。

组件保持单一职责，因此没有把 Critic 逻辑塞入 SearchProvider、Retriever 或 Reranker。

## Agent Trace 与性能

`ResearchRoundTrace` 记录每轮 queries、搜索结果数、新增/累计 Evidence、Critic 判断、缺失方面、后续 queries 和分阶段耗时。`ResearchTrace` 保存所有轮次、stop reason、Critic failure 和总性能：

- planning
- search（所有轮之和）
- crawl/extract（所有轮之和）
- retrieval（chunk + embedding + rerank）
- critic
- final rerank
- context
- generation
- total

普通模式只显示 Answer + Sources。`--debug-agent` 才显示内部决策 trace；`--debug-retrieval` 显示每轮漏斗、最终排序与性能。

## 安装

```bash
cd /home/yanwq/tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

确认 Ollama 与本地模型：

```bash
ollama list
ollama run qwen3:8b "请只回答：LOCAL_LLM_OK"
```

首次检索会从 Hugging Face 下载 `BAAI/bge-m3` 与 `BAAI/bge-reranker-v2-m3` 到用户 cache。设备默认 `auto`；PyTorch 检测到 CUDA 时使用 GPU，否则使用 CPU。8 GB GPU 下默认 `OLLAMA_KEEP_ALIVE=0`，Embedding 和 Reranker 在阶段结束后移到 CPU 并保留权重，避免三个模型同时常驻显存。

## 配置

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
OLLAMA_KEEP_ALIVE=0
LLM_TIMEOUT=120

MAX_SEARCH_QUERIES=3
RESULTS_PER_QUERY_PER_PROVIDER=5
MAX_COMBINED_SEARCH_RESULTS=20
MAX_PAGES_TO_READ=10
MAX_SEARCH_CONCURRENCY=5
SEARCH_TIMEOUT=10
ALLOWED_DOMAINS=
BLOCKED_DOMAINS=

HTTP_TIMEOUT=10
MAX_PAGE_BYTES=2000000
MIN_CONTENT_LENGTH=200

EMBEDDING_MODEL_NAME=BAAI/bge-m3
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RETRIEVAL_DEVICE=auto
EMBEDDING_BATCH_SIZE=16
RERANK_BATCH_SIZE=8
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
MIN_CHUNK_LENGTH=100
EMBEDDING_TOP_K=20
RERANK_TOP_K=6
MAX_CHUNKS_PER_DOCUMENT=2
ENABLE_RERANKER=true

MAX_RESEARCH_ROUNDS=3
MAX_FOLLOWUP_QUERIES_PER_ROUND=3
MIN_NEW_EVIDENCE_TO_CONTINUE=1
MAX_CRITIC_EVIDENCE=10
MAX_CHARS_PER_CRITIC_EVIDENCE=800
MAX_TOTAL_CRITIC_CONTEXT_CHARS=10000
FINAL_EVIDENCE_TOP_K=8

MAX_CHARS_PER_EVIDENCE=1200
MAX_TOTAL_CONTEXT_CHARS=15000

ENABLE_GROUNDED_GENERATION=true
MAX_CLAIM_REWRITE_ATTEMPTS=1
MAX_CLAIMS=30
MAX_EVIDENCE_PER_CLAIM=3
VERIFICATION_BATCH_SIZE=8
ENABLE_COVERAGE_CHECK=true

MAX_RUN_SECONDS=900
MAX_RUNTIME_SEARCH_REQUESTS=30
MAX_RUNTIME_CRAWL_REQUESTS=30
MAX_RUNTIME_LLM_CALLS=40
NETWORK_RETRY_MAX_ATTEMPTS=2
LLM_RETRY_MAX_ATTEMPTS=2
RETRY_BASE_DELAY_SECONDS=0.5
RETRY_MAX_DELAY_SECONDS=4
RETRY_JITTER_SECONDS=0.1
CHECKPOINT_ENABLED=true
CHECKPOINT_DIR=.runtime/runs
```

## 运行

普通模式：

```bash
source .venv/bin/activate
python -m tracker.cli \
  "What are the major recent approaches to reinforcement learning for robotic manipulation, what problems do they solve, and what are their limitations?"
```

显示 Agent 的每轮决策：

```bash
python -m tracker.cli --debug-agent \
  "What are the recent improvements in neural dynamics based control for quadrotor UAVs, and what limitations remain?"
```

同时显示 Agent Trace、每轮 Retrieval Funnel、全局重排和性能：

```bash
python -m tracker.cli --debug-runtime --debug-agent \
  --debug-retrieval --debug-grounding \
  "Compare major approaches for vision-language-action robotic manipulation and explain their strengths, weaknesses, and deployment challenges."
```

旧入口仍可用：

```bash
python main.py --debug-agent "What is retrieval augmented generation?"
```

还可使用 `--allow-domain`、`--ban-domain`、`--disable-reranker`、`--embedding-top-k` 和 `--rerank-top-k`。环境变量可临时限制集成测试轮数：

```bash
MAX_RESEARCH_ROUNDS=2 python -m tracker.cli --debug-agent "your question"
```

## 测试

自动化测试不联网、不调用 Qwen/BGE/Reranker，全部使用 Fake 组件：

```bash
source .venv/bin/activate
pytest -q
python -m compileall -q main.py src tracker tests scripts
pip check
git diff --check
```

测试覆盖 V0.5 全部 Search Agent 行为、V0.6 Grounding，以及 V1.0-A lifecycle、run ID、错误分类、retry limit/backoff、operation/overall timeout、四类预算、取消与 child cleanup、atomic checkpoint、损坏检测、stage-level resume、query/evidence 幂等、RuntimeTrace、fallback 和旧入口兼容。所有测试均使用 Fake，不联网、不加载模型。

## 项目结构

```text
tracker/
├── main.py
├── tracker/cli.py
├── src/
│   ├── agent/
│   │   ├── models.py
│   │   ├── critic_context.py
│   │   ├── critic.py
│   │   └── research_agent.py
│   ├── plan/
│   │   ├── models.py
│   │   └── search_planner.py
│   ├── action/
│   │   ├── models.py
│   │   └── policy.py
│   ├── tools/
│   │   └── research.py
│   ├── memory/
│   │   ├── evidence_pool.py
│   │   └── research_state.py
│   ├── grounding/
│   │   ├── models.py
│   │   ├── registry.py
│   │   ├── planner.py
│   │   ├── verifier.py
│   │   ├── rewriter.py
│   │   ├── service.py
│   │   ├── coverage.py
│   │   ├── renderer.py
│   │   └── generator.py
│   ├── runtime/
│   │   ├── models.py
│   │   ├── errors.py
│   │   ├── budgets.py
│   │   ├── retry.py
│   │   ├── context.py
│   │   ├── checkpoint.py
│   │   └── runtime.py
│   ├── search/
│   ├── crawling/
│   ├── extraction/
│   ├── retrieval/
│   ├── context/
│   ├── answer/
│   └── models/
└── tests/
    └── runtime/
```

## 为什么 V0.5 开始接近 Deep Research Agent

V0.1–V0.4 主要是在固定数据流中改进 retrieval：更多来源、更可靠的正文、更高召回和更精确的证据选择。V0.5 第一次加入：

```text
Observe  = 运行 ResearchTool，读取新 Evidence
Evaluate = Critic 判断 Evidence Sufficiency
Decide   = ResearchActionPolicy 选择 Search 或 Finish
Act      = 执行针对 Gap 的 Follow-up Search
Observe  = 新 Evidence 合并到 EvidencePool
```

这仍是一个简单的单 Agent 闭环，不是多 Agent 或复杂 planning tree；但它已经拥有 iterative reasoning、stateful decisions 和 adaptive action，因此项目从 Advanced RAG / Search Pipeline 开始转向 Autonomous Search Agent。
