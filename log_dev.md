# Tracker 开发日志

## V0.1「能搜」

- 开始时间：2026-09-01 15:32 CST
- 当前系统：Ubuntu 24.04.2 LTS，Python 3.12.3
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU 8 GB；Driver 535.309.01；驱动 CUDA 能力 12.2

### 已完成

- 创建 `.venv`，安装 `ollama`、`pydantic`、`httpx`、`python-dotenv`、`rich`、`ddgs`、`pytest`。
- 实现本地地址限定的 `LLMClient` 及 Ollama/模型健康检查。
- 实现 Pydantic `SearchPlan`、`SearchResult`。
- 实现单次查询 `SearchPlanner`，包括 thinking/code fence 清理、JSON 提取、校验和一次有限重试。
- 实现 `SearchProvider` 接口和唯一的 `DDGSSearchProvider`。
- 实现仅使用搜索摘要的 `AnswerGenerator`。
- 实现 Rich 交互 CLI、来源列表与用户可理解的异常信息。
- 添加离线单元测试、配置样例、Git 忽略规则和 README。
- 安装并启用 Ollama 0.33.2，下载本地 `qwen3:8b`（5.2 GB）。
- 完成 CLI、Python SDK、项目 LLMClient 和完整联网问答闭环的实际验证。

### 实际安装版本

- ollama Python client 0.6.2
- pydantic 2.13.5
- httpx 0.28.1
- python-dotenv 1.2.3
- rich 15.0.0
- ddgs 9.16.0
- pytest 9.1.1

### 验证结果

- `.venv/bin/python -m pytest -q`：31 passed（包含默认本地模型、本地地址限制、代理导入隔离、健康检查、模型缺失、Ollama 连接失败和响应解析测试）。
- `compileall`：passed。
- `pip check`：No broken requirements found。
- 真实 DDGS 搜索：passed，返回 3 条已归一化结果。
- GPU 宿主环境检查：passed。沙箱内无 `/dev/nvidia*`，沙箱外 `nvidia-smi` 正常。
- Ollama 0.33.2：installed；systemd 服务 enabled + active。
- `http://localhost:11434/api/tags`：HTTP 200。
- `ollama list`：存在 `qwen3:8b`，ID `500a1f067a9f`，大小 5.2 GB。
- `ollama run qwen3:8b "请只回答：LOCAL_LLM_OK"`：passed。
- Python Ollama SDK：返回 `PYTHON_LOCAL_LLM_OK`。
- 项目 `LLMClient.check_health()`：passed；`chat()` 返回 `PROJECT_LLMCLIENT_OK`。
- `ollama ps`：`qwen3:8b` 100% GPU，运行大小 5.6 GB；`nvidia-smi` 观察到 `llama-server` 约占用 5.3 GB 显存。
- 完整 CLI：问题 → SearchPlan（`Qwen3 最新重要进展`）→ DDGS 5 条结果 → 本地摘要回答 → Sources，退出码 0。

### 遇到的问题与处理

- Ollama 原先未安装。官方 `curl -fsSL https://ollama.com/install.sh | sh` 已执行到系统安装步骤，但交互式 `sudo` 需要用户密码，自动化环境无法继续。
- 用户完成 sudo 安装后，继续自动完成了服务验证、模型下载和全部推理验收。
- 当前 shell 设置了指向 `127.0.0.1:7890` 的 HTTP(S) 代理，但没有 `NO_PROXY`。`LLMClient` 使用 `trust_env=False`，保证本地 Ollama 请求不经过代理；README 的 curl 命令使用 `--noproxy '*'`。
- Ollama 日志提示 NVIDIA 535 驱动低于其 CUDA 后端要求的 550，因此自动选择 Vulkan 后端。实测仍为 100% GPU 推理，V0.1 可用；未安装 PyTorch、CUDA Toolkit 或 cuDNN。
- 修复 `ALL_PROXY=socks://127.0.0.1:7890` 导致 Ollama Python 包在导入阶段崩溃的问题：导入期间临时隔离代理，随后恢复给 DDGS 使用。相同代理环境下 CLI 健康检查和完整 Search → Answer 流程均实测通过。

### 当前验收

- V0.1：完成。
- Local LLM Deployment：完成。
- Cloud LLM used：NO。
- 完成时间：2026-09-02 09:54 CST。

### 后续 TODO

- V0.4：候选网页相关性排序、精度提升与证据质量控制。

## V0.2「能读」

- 开始时间：2026-09-02
- 目标：把 V0.1 的 Search → Answer 升级为 Search → Fetch → Extract → Read → Answer，让模型基于网页正文而非搜索摘要回答。

### 已完成

- 新增 `Document`，明确区分搜索候选结果和实际读取成功的网页正文。
- 新增异步 `WebCrawler`，支持并发抓取、User-Agent、跳转、超时、HTTP 错误、内容类型检查和 2 MB 页面体积限制。
- 新增 `ContentExtractor`，优先使用 Trafilatura，失败时回退 BeautifulSoup，并过滤过短正文。
- 新增 `ContextBuilder`，为来源添加稳定编号，并实施单文档与总上下文字符预算。
- 新增 `TrackerPipeline`，串联规划、搜索、并发抓取、正文抽取、上下文构建和本地回答，单页失败时可继续执行。
- 升级 `AnswerGenerator`，只接收实际网页正文上下文，并要求使用 `[Source n]` 引用且在证据不足时明确说明。
- 升级 CLI，支持交互和命令行参数两种模式，显示查询、搜索数量、实际阅读页面、答案与真实来源。
- 补充 README、环境变量样例以及 crawler、extractor、context、pipeline 的离线测试。

### 新增依赖

- trafilatura 2.2.0
- beautifulsoup4 4.15.0
- lxml 6.1.2

### 验证结果

- `.venv/bin/python -m pytest -q`：48 passed。
- 离线测试覆盖：抓取成功、404、超时、非 HTML、页面过大、正文提取、BeautifulSoup 回退、短正文拒绝、上下文截断、并发读取和单页故障容忍。
- `compileall`：passed。
- `pip check`：No broken requirements found。
- 真实集成查询 `What is retrieval augmented generation?`：DDGS 返回 5 条结果，并发读取前三条。
- Wikipedia、IBM 和 NVIDIA 页面均返回 HTTP 200；分别抽取约 14.5k、13.2k 和 10.2k 字符正文。
- 本地 `qwen3:8b` 基于构建后的网页正文生成带 `[Source n]` 引用的回答；CLI 仅展示实际读取成功的来源，退出码 0。

### 实现备注

- Pipeline 使用 `asyncio.gather` 并发网页 I/O。规划、搜索和本地模型调用保留同步边界，避免没有收益的线程切换。
- Crawler 显式选择合法的 HTTP/HTTPS 代理并关闭 `httpx` 的环境代理自动解析，因此无效的 SOCKS 环境变量不会再次导致启动失败。
- V0.2 未引入 embedding、向量数据库、Reranker、LangChain、LlamaIndex 或 Playwright。

### 当前验收

- V0.2：完成。
- Local LLM used：YES（Ollama `qwen3:8b`）。
- Cloud LLM used：NO。
- 完成时间：2026-09-02 13:24 CST。

## V0.3「搜得广」

- 开始时间：2026-09-03
- 目标：将 Single Query / Single Provider 升级为 Multi-Query / Multi-Source 并发搜索，提升候选信息覆盖率，同时保持 V0.2 的网页阅读与本地回答链路。

### 已完成

- 将 `SearchPlan.query` 升级为经过清洗、精确去重和数量限制的 `queries[]`。
- 更新 SearchPlanner prompt，让本地 Qwen3 从不同术语和角度生成最多 3 个互补查询。
- 将 `SearchProvider.search()` 升级为统一异步接口，并为 Provider 增加稳定名称。
- 保留 DDGS Provider，通过 `asyncio.to_thread` 隔离其阻塞 API。
- 新增无需 API Key 的 Wikipedia MediaWiki Search Provider，默认即可真实运行双来源搜索。
- 新增 SourceManager，执行 `queries × providers` 搜索任务，使用 `asyncio.gather` 并发、Semaphore 限流和逐任务失败隔离。
- 使用 round-robin 混合各任务结果，并只做 exact URL duplicate removal 和总结果数量限制。
- 为 `SearchResult` 增加 `provider` 和 `query` 元数据。
- 新增 DomainFilter，支持 allowlist、banlist 和子域名匹配，过滤发生在 crawler 之前。
- 新增 `SearchBatch` 与 `SearchCoverage`，记录 Provider、任务状态、原始结果数和重复数量。
- 将 V0.3 搜索链路接回 WebCrawler、ContentExtractor、Document、ContextBuilder 和 AnswerGenerator。
- CLI 新增多查询、Provider、覆盖情况和数量变化展示，以及 `--allow-domain`、`--ban-domain` 参数。
- 保留 `python main.py`，并新增等价的 `python -m tracker.cli` 模块入口。
- 集中增加搜索查询数、每任务结果数、合并上限、读取页数、并发数、超时和域名策略配置。

### 自动化验证

- `.venv/bin/python -m pytest -q`：64 passed。
- 单元测试覆盖 Multi-Query、query 数量限制、query 精确去重、两个 Provider、3×2 搜索任务、Provider 失败、来源元数据、域名策略、exact URL 去重、并发上限和完整 Pipeline。
- `compileall`：passed。
- `pip check`：No broken requirements found。
- `git diff --check`：passed。

### 真实联网与本地模型验证

- 通用问题 `What is retrieval augmented generation?`：生成 3 个 query；DDGS 与 Wikipedia 的 6 个任务均成功；30 条 raw results 合并为 18 条候选；读取 5 页并生成带引用回答，退出码 0。
- 技术问题首次运行：1 个 DDGS 任务超时，其余 5 个任务仍返回 25 条结果，证明搜索失败隔离有效；随后代理对所有候选抓取请求返回连接错误，系统明确拒绝在无正文时回答。
- 使用 `--allow-domain wikipedia.org` 重试技术问题：27 raw → 20 unique/capped → 8 allowed；crawler 只访问 Wikipedia，读取 4 页并生成证据不足的诚实回答，退出码 0。
- Embodied AI 问题：生成 3 个不同 query；两个 Provider 共返回 30 条 raw results；抓取阶段容忍 HTTP 403 和超大页面，读取 3 页并完成本地回答，退出码 0。
- Local LLM used：YES（Ollama `qwen3:8b`）。
- Cloud LLM used：NO。

### 当前边界

- V0.3 优化 Recall，不实现 embedding、向量数据库、Reranker、高级 URL normalization、内容/语义去重或多轮搜索循环。
- Wikipedia 是无需密钥的默认第二来源；当前版本没有需要配置或可能泄露的 Provider API Key。
- 简单 round-robin 只改善来源覆盖，不判断相关性；候选精排属于 V0.4。

### 当前验收

- V0.3：完成。
- 完成时间：2026-09-03 10:44 CST。
