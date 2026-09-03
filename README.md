# Tracker V0.5 — Iterative Deep Search / 搜得深

Tracker 是一个从底层学习 Search Agent / Deep Research Agent 的本地项目。它使用 Ollama/Qwen3 进行搜索规划、证据评估和答案生成，使用 DDGS + Wikipedia 搜索公开网页，并以 BGE-M3 + bge-reranker-v2-m3 在本地完成段落检索。项目不调用云端 LLM，也不依赖 LangChain、LlamaIndex 或 Agent framework。

V0.1–V0.4 依次实现“能搜、能读、搜得广、搜得准”。V0.5 在完整 V0.4 Retrieval Pipeline 外层加入一个有状态、受预算约束的闭环：Agent 会评估已获得的证据，发现具体缺口，并据此进行定向补搜。

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
python -m tracker.cli --debug-agent --debug-retrieval \
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

测试覆盖 EvidencePool add/dedup/source count、Critic structured output/repair、query 去重与限额、两轮闭环、max rounds、全部提前停止条件、Critic failure、跨轮累积和去重、针对 Original Question 的 Final Rerank、Context 只接收 Final Evidence、Trace stop reason，以及完整 V0.4 retrieval funnel 回归。

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
