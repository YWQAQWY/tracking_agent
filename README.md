# Tracker V0.6 — Grounded Answer Generation / 答得稳

Tracker 是一个从底层学习 Search Agent / Deep Research Agent 的本地项目。它使用 Ollama/Qwen3 进行搜索规划、证据评估和 grounded generation，使用 DDGS + Wikipedia 搜索公开网页，并以 BGE-M3 + bge-reranker-v2-m3 在本地完成段落检索。项目不调用云端 LLM，也不依赖 LangChain、LlamaIndex、LangGraph 或 Agent framework。

V0.1–V0.5 依次实现“能搜、能读、搜得广、搜得准、搜得深”。V0.6 在 V0.5 Final Evidence 之后增加 claim-level Grounding Layer：最终事实必须先映射到 Evidence、通过支持关系验证，才能进入确定性 Renderer。

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

单轮 `ResearchRound` 仍完整复用 V0.4：

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

- `SearchProvider` 是 Tool：执行一个明确的外部搜索动作，不决定是否继续研究。
- `ResearchRound` 是可重复的能力链：把 queries 变成经过完整读取与检索的新 Evidence，不管理跨轮状态。
- `ResearchAgent` 是 Orchestrator：维护 State 和预算，安排 ResearchRound、Critic、停止与最终重排。
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
python -m tracker.cli --debug-agent --debug-retrieval --debug-grounding \
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

测试覆盖 V0.5 全部 Search Agent 行为，以及 EvidenceRegistry、claim schema、structured repair、unknown Evidence ID、batch verification、rewrite success/failure/limit、coverage、确定性 citation mapping、同 URL 合并、unused source 排除、planner/verifier fallback、GroundingTrace 和 legacy mode。

## 项目结构

```text
tracker/
├── main.py
├── tracker/cli.py
├── src/
│   ├── agent/
│   │   ├── evidence_pool.py
│   │   ├── models.py
│   │   ├── critic_context.py
│   │   ├── critic.py
│   │   ├── research_round.py
│   │   └── research_agent.py
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
│   ├── planner/
│   ├── search/
│   ├── crawling/
│   ├── extraction/
│   ├── retrieval/
│   ├── context/
│   ├── answer/
│   └── models/
└── tests/
```

## 为什么 V0.5 开始接近 Deep Research Agent

V0.1–V0.4 主要是在固定数据流中改进 retrieval：更多来源、更可靠的正文、更高召回和更精确的证据选择。V0.5 第一次加入：

```text
Observe  = 运行 ResearchRound，读取新 Evidence
Evaluate = Critic 判断 Evidence Sufficiency
Decide   = 停止，或选择具体 Evidence Gap
Act      = 执行针对 Gap 的 Follow-up Search
Observe  = 新 Evidence 合并到 EvidencePool
```

这仍是一个简单的单 Agent 闭环，不是多 Agent 或复杂 planning tree；但它已经拥有 iterative reasoning、stateful decisions 和 adaptive action，因此项目从 Advanced RAG / Search Pipeline 开始转向 Autonomous Search Agent。
