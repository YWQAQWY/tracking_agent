# Tracker V0.2 — 能读

Tracker 是一个基于本地大模型的联网检索 Agent。它先规划搜索词，通过 DuckDuckGo 发现网页，再抓取和抽取网页正文，把真正读到的内容交给本地 Ollama/Qwen3 生成带来源的回答。

联网仅发生在两个位置：DuckDuckGo 搜索和网页抓取。问题规划、上下文构建与答案生成均在本地完成。

## V0.2 数据流

```text
User Question
    ↓
SearchPlanner
    ↓
SearchProvider → SearchResult（候选网页）
    ↓
WebCrawler（httpx，并发抓取前 N 个 URL）
    ↓ HTML
ContentExtractor（Trafilatura，失败时回退 BeautifulSoup）
    ↓
Document（实际读取到的正文）
    ↓
ContextBuilder（来源标记与字符预算）
    ↓
Local Qwen3 via Ollama
    ↓
Answer + Sources
```

## V0.2 能力

- 并发抓取搜索结果中的前 N 个网页。
- 处理跳转、超时、HTTP 错误、不支持的内容类型和过大页面。
- 优先使用 Trafilatura 提取正文，失败时使用 BeautifulSoup 清理页面噪声。
- 拒绝正文过短、质量不足的页面。
- 使用 `Document` 明确区分“搜索发现的候选网页”和“实际读到的网页正文”。
- 为单篇文档和总上下文分别设置字符上限，避免提示词无限增长。
- 单个网页失败不会中断整次任务；只要仍有可读网页就继续回答。
- CLI 显示查询词、搜索数量、实际阅读页面、答案和来源。

本版本不包含 embedding、向量数据库、Reranker、LangChain、LlamaIndex 或浏览器渲染。JavaScript 重度网页可能无法提取，后续版本可按需加入 Playwright。

## 核心对象

- `SearchResult`：搜索引擎返回的候选结果，包含标题、URL 和摘要。摘要仅用于发现，不作为最终阅读正文。
- `WebCrawler`：负责网络 I/O，获取 HTML，并执行超时、类型和页面大小保护。
- `ContentExtractor`：负责从原始 HTML 中移除导航、脚本等噪声，提取标题和正文。
- `Document`：表示系统确实抓取并读取成功的网页内容。
- `ContextBuilder`：把多个 `Document` 编排为带 `[Source n]` 标记的有限长度上下文。V0.2 使用透明的字符截断，不使用语义切片或向量检索。

## 环境要求

- Python 3.12+
- 已安装并运行 Ollama
- 已拉取本地模型，例如 `qwen3:8b`
- 可访问 DuckDuckGo 和目标网页的网络环境

## 安装

```bash
cd /home/yanwq/tracker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

确认 Ollama 与模型可用：

```bash
ollama list
ollama run qwen3:8b "你好"
```

## 配置

在 `.env` 中配置：

```dotenv
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
SEARCH_MAX_RESULTS=5
MAX_PAGES=3
HTTP_TIMEOUT=10
MAX_PAGE_BYTES=2000000
MIN_CONTENT_LENGTH=200
MAX_CHARS_PER_DOCUMENT=6000
MAX_TOTAL_CONTEXT_CHARS=15000
```

`MAX_PAGES` 控制读取的候选页面数。`MAX_PAGE_BYTES` 限制下载体积；`MAX_CHARS_PER_DOCUMENT` 和 `MAX_TOTAL_CONTEXT_CHARS` 分别控制单篇正文和总提示词上下文。

Crawler 只采用 HTTP/HTTPS 代理并忽略 `ALL_PROXY` 中的 SOCKS 地址，从而避免未安装 SOCKS 支持时 `httpx` 在启动阶段报错。Ollama 客户端直接连接配置的本地地址。

## 运行

交互模式：

```bash
source .venv/bin/activate
python main.py
```

单次查询：

```bash
python main.py "What is retrieval augmented generation?"
```

正常运行时可以看到搜索、抓取、正文抽取、上下文构建和本地模型调用日志。最终只列出实际读取成功的来源。

## 测试

```bash
source .venv/bin/activate
pytest -q
python -m compileall -q main.py src tests
pip check
```

测试覆盖网页抓取、HTTP 错误、超时、内容类型与体积限制、正文抽取及回退、短正文拒绝、上下文预算、并发抓取、单页故障容忍，以及从搜索到回答的完整 Pipeline。

## 项目结构

```text
tracker/
├── main.py
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── answer/
│   │   └── answer_generator.py
│   ├── context/
│   │   └── context_builder.py
│   ├── crawling/
│   │   ├── crawler.py
│   │   └── models.py
│   ├── extraction/
│   │   └── content_extractor.py
│   ├── llm/
│   │   └── client.py
│   ├── models/
│   │   ├── document.py
│   │   └── search_result.py
│   ├── planner/
│   │   └── search_planner.py
│   └── search/
│       ├── base.py
│       └── ddgs_provider.py
└── tests/
    ├── test_answer_generator.py
    ├── test_content_extractor.py
    ├── test_context_builder.py
    ├── test_crawler.py
    ├── test_pipeline.py
    └── ...
```

## 故障处理

- 某个页面抓取或提取失败：记录警告，继续处理其他页面。
- 所有页面均不可读：明确报错，不让模型基于搜索摘要编造答案。
- Ollama 不可用或模型缺失：检查 `ollama serve`、`ollama list` 与 `.env`。
- 页面主要由 JavaScript 动态生成：当前版本可能读不到正文，这是 V0.2 的已知边界。

下一阶段可在保持现有边界清晰的前提下增加分块、embedding、向量检索和证据级引用。
