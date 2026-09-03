# Tracker V0.3 — 搜得广

Tracker 是一个使用本地 Ollama/Qwen3 的联网检索 Agent。V0.3 会把一个自然语言问题扩展成多个互补查询，同时访问 DDGS 和 Wikipedia 两个独立搜索来源，再将候选网页交给 V0.2 的抓取、正文提取和回答链路。

所有 LLM 推理均在本机完成，不调用云端 LLM。互联网只用于搜索和读取公开网页。

## V0.3 架构

```text
User Question
    ↓
SearchPlanner / Local Qwen3
    ↓
Query 1 + Query 2 + Query 3
    ↓
SourceManager
    ↓ queries × providers
DDGS + Wikipedia
    ↓ asyncio concurrent search
SearchBatch + SearchCoverage
    ↓ round-robin + exact URL dedup + result cap
DomainFilter
    ↓
Selected URLs
    ↓
WebCrawler → ContentExtractor → Document
    ↓
ContextBuilder → Local Qwen3
    ↓
Answer + Actual Sources
```

V0.2 是：

```text
Question → Single Query → Single Provider → Read → Answer
```

V0.3 是：

```text
Question → Multi-Query → Multi-Source → Concurrent Search
         → Domain Filter → Read → Answer
```

## 为什么需要 Multi-Query

自然语言问题不等于唯一搜索关键词。单个 query 很容易受到词汇选择影响，例如同一个机器人问题可能使用 `robotic manipulation`、`dexterous manipulation`、`robot learning policy` 或 `object grasping` 等不同术语。

Multi-Query 让本地模型生成少量互补查询，从不同术语和角度发现候选网页，目标是提高搜索召回率（Recall）。系统只做 strip、空值移除、数量限制和 exact query duplicate removal，不做语义去重。

## 为什么需要 Multi-Source

不同搜索来源有不同的索引、排名机制和内容覆盖。依赖单一 Provider 会放大它的覆盖盲区和临时故障。

V0.3 默认启用：

- `ddgs`：广泛的 Web 搜索结果。
- `wikipedia`：通过 MediaWiki API 使用独立的百科搜索索引。

两个 Provider 都实现同一个异步 `SearchProvider` 接口，互不依赖。Wikipedia 不需要 API Key，因此默认安装即可真正运行双来源搜索，也没有任何密钥被写进代码。

## SourceManager 是什么

SourceManager 不是搜索引擎，而是搜索编排器（orchestrator）。它负责：

```text
queries × providers
    ↓
创建异步搜索任务
    ↓
Semaphore 限制并发
    ↓
单任务错误隔离
    ↓
Round-robin 合并
    ↓
Exact URL 去重和总数限制
```

例如默认配置最多形成：

```text
3 queries × 2 providers = 6 concurrent search tasks
3 × 2 × 5 results = 30 raw results
```

候选结果随后被限制为 20 条，并且最多读取 5 个页面。

## 为什么使用 asyncio

搜索和网页请求属于 IO-bound 工作。程序等待网络响应时，CPU 大部分时间没有工作。`asyncio` 可以让一个请求等待时继续推进其他请求，避免按顺序等待六次网络往返。

`asyncio.gather(..., return_exceptions=True)` 同时等待多个搜索任务，并把单个 Provider 的失败作为结果收集，不让它取消其他成功任务。`asyncio.Semaphore` 则限制同时运行的请求数，避免产生不受控制的网络突发。

DDGS 提供的是阻塞接口，因此 Provider 使用 `asyncio.to_thread` 把阻塞调用移出事件循环。Wikipedia 使用原生异步 `httpx.AsyncClient`。

## 搜索结果与来源追踪

每个 `SearchResult` 包含：

```text
title
url
snippet
provider
query
```

因此日志和 CLI 可以显示每条候选结果来自哪个 query 和哪个 Provider。搜索摘要只用于发现候选网页，不会作为最终回答证据。

`SearchCoverage` 会保存每个 query/provider 组合的成功状态、结果数量或错误信息。一个搜索任务失败时，其他任务仍然继续。

## Domain Allowlist / Banlist

DomainFilter 在 crawler 之前执行，被拒绝的网站不会浪费抓取请求，也不会进入模型上下文。

匹配规则：

- `example.com` 同时匹配 `example.com`、`www.example.com` 和 `docs.example.com`。
- allowlist 非空时，首先只保留允许的域名。
- banlist 随后执行，可以进一步排除 allowlist 中的子域名。
- 不执行 canonical URL、UTM 清理、内容哈希或语义去重。

## 配置

复制示例文件：

```bash
cp .env.example .env
```

默认配置：

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
MAX_SEARCH_QUERIES=3
RESULTS_PER_QUERY_PER_PROVIDER=5
MAX_COMBINED_SEARCH_RESULTS=20
MAX_PAGES_TO_READ=5
MAX_SEARCH_CONCURRENCY=5
SEARCH_TIMEOUT=10
ALLOWED_DOMAINS=
BLOCKED_DOMAINS=
HTTP_TIMEOUT=10
MAX_PAGE_BYTES=2000000
MIN_CONTENT_LENGTH=200
MAX_CHARS_PER_DOCUMENT=6000
MAX_TOTAL_CONTEXT_CHARS=15000
```

多个域名使用逗号分隔：

```dotenv
ALLOWED_DOMAINS=arxiv.org,github.com
BLOCKED_DOMAINS=pinterest.com,facebook.com
```

V0.2 的 `SEARCH_MAX_RESULTS` 和 `MAX_PAGES` 仍可作为兼容回退，但新配置应使用 `RESULTS_PER_QUERY_PER_PROVIDER` 和 `MAX_PAGES_TO_READ`。

## 安装

```bash
cd /home/yanwq/tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

确认本地模型：

```bash
ollama list
ollama run qwen3:8b "你好"
```

## 运行

当前仓库保留已有 `main.py` argparse 入口，同时支持模块入口。

交互模式：

```bash
source .venv/bin/activate
python main.py
```

普通查询：

```bash
python main.py "What are the major recent research directions in embodied AI?"
```

等价的模块命令：

```bash
python -m tracker.cli "What are the major recent research directions in embodied AI?"
```

只允许特定域名，可重复传入参数：

```bash
python -m tracker.cli \
  --allow-domain arxiv.org \
  --allow-domain github.com \
  "What are recent methods for robotic manipulation?"
```

排除特定域名：

```bash
python -m tracker.cli \
  --ban-domain pinterest.com \
  --ban-domain facebook.com \
  "latest robotics research"
```

CLI 会显示：

- LLM 生成的全部查询；
- 启用的搜索 Provider；
- 每个 query/provider 任务的成功状态和结果数；
- 原始、去重/截断、域名过滤后的结果数；
- 实际读取成功的网页；
- 最终回答与真实来源。

## 测试

```bash
source .venv/bin/activate
pytest -q
python -m compileall -q main.py src tests
pip check
```

测试覆盖：

- Multi-Query JSON 解析、数量限制和精确去重；
- DDGS 与 Wikipedia 结果归一化和来源元数据；
- `queries × providers` 任务数量；
- 异步并发与 Semaphore 上限；
- 单 Provider 超时后的故障隔离；
- Round-robin 合并和 exact URL 去重；
- allowlist、subdomain、banlist 和 allow+ban 优先级；
- 域名过滤发生在 crawler 之前；
- V0.2 的抓取、提取、上下文和回答链路。

## 项目结构

```text
tracker/
├── main.py
├── tracker/cli.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── network.py
│   ├── pipeline.py
│   ├── planner/search_planner.py
│   ├── search/
│   │   ├── base.py
│   │   ├── ddgs_provider.py
│   │   ├── wikipedia_provider.py
│   │   ├── source_manager.py
│   │   └── domain_filter.py
│   ├── crawling/
│   ├── extraction/
│   ├── context/
│   ├── answer/
│   └── models/
└── tests/
```

## 当前边界

V0.3 的目标是 Recall / Coverage，即尽量发现更多可能有用的信息。它没有实现 relevance ranking、embedding、向量数据库、Reranker、语义去重、Critic Agent、搜索循环、MCP、LangChain 或 LlamaIndex。

候选结果的简单顺序并不保证最相关网页一定进入前五个，这是 V0.4「搜得准」需要解决的问题。
