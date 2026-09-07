# Tracker 开发日志

## Agent 标准分层重构（2026-09-04）

- 将混放在 `src/agent` 和 `src/models` 的职责整理为 `agent / tools / memory / action / plan` 五个核心目录。
- `ResearchAgent` 只负责编排；`ResearchTool` 封装一次完整 Search→Read→Retrieve；`EvidencePool` 与 `ResearchState` 归入任务 Memory；`SearchPlanner` 与 `SearchPlan` 归入 Plan。
- 新增确定性的 `ResearchActionPolicy` 和 `ResearchAction`，把 Critic 之后的 Search/Finish 决策显式化，并集中处理查询清洗、重复查询、最大轮数和无后续查询等停止条件。
- 旧导入路径保留为 re-export compatibility shim，真实实现只有一份，没有复制 Search、Crawler、Retriever、Planner 或 EvidencePool。
- V0.5 搜索闭环、V0.6 grounding、V1.0-A Runtime 的行为和 CLI 均保持兼容。

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

## V0.6 — Claim-Level Grounded Generation（2026-09-04）

### 目标与架构

- 保留 V0.5 Search → Critic → Follow-up Search、EvidencePool 和 Final Global Rerank，不复制检索组件。
- 把 `Final Evidence → free-form Answer` 改为 `EvidenceRegistry → Structured Claims → Verify → Rewrite/Drop → Coverage → deterministic render`。
- 新增 `src/grounding/`，按 Planner、Verifier、Rewriter、Coverage、Renderer 单一职责拆分；共同复用 `src/llm/structured.py` 的一次修复式 structured-output helper。
- ResearchAgent 只增加一个可选 `GroundedAnswerGenerator` 入口；关闭 `ENABLE_GROUNDED_GENERATION` 或 Planner/Verifier 失败时，回退 V0.5 AnswerGenerator 并明确记录未验证状态。

### 关键行为

- Final Evidence 在当前回答中稳定映射为 E1/E2/...；unknown Evidence ID 本地直接判 unsupported，不调用 LLM，也不崩溃。
- CitationVerifier 批量判断 claim 是否被引用 Evidence entail；topic relevance 本身不算支持。
- Unsupported claim 最多重写一次，重写后必须再验证；失败或无有意义重写则删除。
- Coverage Checker 只检查 Original Question 与 Verified Claims，不启动新搜索；缺口变成 answer limitation。
- Renderer 不调用 LLM，只排版 verified claim、稳定映射 URL citation number、合并同 URL chunks，并只列实际引用来源。
- GroundingTrace 记录 draft、initially supported、unsupported、rewritten、rewrite passed、dropped、final claims、sources、coverage、claim traces 和阶段耗时。

### Structured Output 集成问题

- 比较型真实测试中，Qwen 返回了合法的单个 `{heading, claims}` section，却省略外层 `{sections: [...]}`；首次实现严格拒绝并正确回退 V0.5。
- `GroundedAnswerDraft` 增加一个窄范围归一化：只把这种常见单 section 形状包装进 `sections`，claim/evidence 字段仍严格校验。
- 同时允许空 `sections`，因为“没有任何 Evidence 支持的 claim”是保守生成的合法结果；Coverage/Renderer 会输出证据不足，而不是强迫 Planner 编造一个 claim。

### 自动化验证

- `.venv/bin/pytest -q`：157 passed；涵盖 V0.1–V0.5 回归和 V0.6 Grounding 行为。
- `python -m compileall -q main.py src tracker tests scripts`：passed。
- `.venv/bin/pip check`：No broken requirements found。
- `git diff --check`：passed。
- 项目未安装或配置 ruff/mypy，未为本阶段额外引入工具链。

### 真实端到端集成

- 问题：`What are the major recent approaches to reinforcement learning for robotic manipulation, what problems do they solve, and what limitations remain?`
- 环境：RTX 4060 Laptop GPU 8 GB、CUDA、真实 DDGS/Wikipedia、网页抓取、BGE-M3、bge-reranker-v2-m3、Ollama Qwen3-8B。
- Round 1：3 queries；30 raw → 20 capped → 5 documents → 307 chunks → 20 candidates → 4 new Evidence / 2 sources；Critic 判 insufficient。
- Missing：近期 manipulation-specific approaches、解决的问题、场景限制；生成 3 条针对性补搜 query。
- Round 2：27 raw → 20 capped → 9 documents / 8 unique → 374 chunks → 20 candidates → 6 round evidence；跨轮去重后新增 5，Pool 共 9 / 5 sources。
- Round 2 Critic：sufficient；Final Global Rerank 9 → 8 Evidence。
- Grounding：12 draft → 12 verified immediately → 0 unsupported → 0 rewritten → 0 dropped；最终使用 4 个唯一 URL。
- Coverage：adequate，覆盖 major approaches / problems solved / limitations。
- 性能：planning 12.634s；search 14.003s；crawl/extract 5.975s；retrieval 47.374s；critic 57.651s；final rerank 0.404s；grounded generation 201.909s；total 342.677s。

### Citation 人工审计与证据不足测试

- 重建真实网页的相同 DocumentChunk，抽查超过 5 个 claim：DRL/IL manipulation survey、role-model sampling efficiency、offline RL reality gap、real-world data collection/distribution shift、sample inefficiency、battery/sensor hardware limits均可在对应 chunk 中直接找到支持文本。
- 对抗问题要求“每个方法的精确提升百分比和部署成本”，只提供明确缺少这些数据的 Evidence。真实 Qwen Grounding 生成 3 个“证据未报告”claims，Coverage 标记 `exact percentage improvement` 与 `deployment costs` 缺失，最终没有编造数字或成本。
- 比较型联网测试第一次运行由 Critic 发现 speed comparison 缺口并以 max_rounds 停止，同时验证 Planner structured failure 会安全回退，而不是让已完成 Research 失败。
- 修复 common single-section wrapper 后再次运行完整比较测试：4 Final Evidence → 4 draft claims → 4 verified → 0 dropped，`grounding_verified=true`。最终只使用 1 个 URL；Coverage 虽然看到 Critic 判 sufficient，仍独立发现 claims 缺少 speed 维度，并确定性输出 `Evidence limitations: speed`，没有编造速度结论。

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

## V0.4「搜得准」

- 开始/完成日期：2026-09-03
- 目标：在 V0.3 的 Multi-Query / Multi-Source 高召回候选之后，引入 URL/正文去重、Chunk、Dense Retrieval 和 Cross-Encoder Reranking，只把高相关 Evidence 交给本地 Qwen。

### 已完成

- 新增 URLNormalizer 和 URLDeduplicator：统一 scheme/host、默认端口、fragment、尾斜杠、tracking 参数及 query 参数顺序，抓取前去重。
- 新增 ContentDeduplicator：正文抽取后对大小写/whitespace 归一化文本做 SHA-256 精确去重。
- 新增 DocumentChunk、ScoredChunk、Evidence 和 RetrievalTrace 数据模型。
- 新增 1200/200 字符滑窗 Chunker，过滤不足 100 字符的短尾块并保留 URL/title/index。
- 新增 BGEEmbedder abstraction 与 `BAAI/bge-m3` SentenceTransformer 实现；支持 batch、normalized embedding、CUDA 自动检测和 lazy loading。
- 新增 SemanticRetriever：以原始用户问题为 query，在内存中计算 cosine similarity 并选 Top 20。
- 新增 Reranker abstraction、`BAAI/bge-reranker-v2-m3` CrossEncoder 和 EmbeddingOnly fallback；默认 Top 6、每 URL 最多 2 段。
- ContextBuilder 改为只接收 Evidence；AnswerGenerator 明确禁止使用所选证据之外的信息。
- Pipeline 加入完整 retrieval funnel、计数、分阶段 `perf_counter` 耗时和清晰空文档/空 Chunk/模型错误。
- CLI 新增 `--debug-retrieval`、`--disable-reranker`、`--embedding-top-k`、`--rerank-top-k`。
- 针对 RTX 4060 8 GB 增加阶段间显存管理：Ollama 默认 `keep_alive=0`，BGE 模型在阶段结束后移到 CPU 并保留已加载权重。
- 阻止 Transformers 5 为只有 `.bin` 的 BGE-M3 后台重复下载 2.27 GB safetensors 自动转换副本。

### 新增依赖与实际版本

- numpy 2.5.2
- sentence-transformers 6.0.1
- torch 2.14.0+cu130
- transformers 5.16.1（sentence-transformers 传递依赖）
- PyTorch CUDA runtime 13.0
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，Driver 595.84，显存 8188 MiB

### 自动化验证

- `.venv/bin/python -m pytest -q`：95 passed；V0.3 既有测试和 V0.4 Fake 模型测试全部通过。
- 覆盖 URL/tracking/query 排序去重、正文 hash 去重、Chunk size/overlap/metadata/短块、余弦排序/Top-K/batch、Reranker 排名变化/Top-K/每 URL 上限、无 Reranker、模型 offload/reuse、Evidence Context、空 Documents/Chunks 和完整 Pipeline。
- `python -m compileall -q main.py src tracker tests scripts`：passed。
- `pip check`：No broken requirements found。
- `git diff --check`：passed。

### 真实模型 smoke test

- Hugging Face cache：`bge-m3` 约 2.2 GB；`bge-reranker-v2-m3` 约 2.2 GB；均位于用户 cache，不进入 Git。
- BGE-M3：CUDA 成功；机器人操作相关段 `0.718966` > 天气段 `0.283877`。
- Reranker：CUDA 成功；相关段 `0.994636` > 天气段 `0.000016`。
- 缓存权重 + 显存卸载版本：`REAL_RETRIEVAL_SMOKE_OK`；28.96s；max RSS 5,617,256 KiB；结束后无 Python GPU 进程。

### 完整联网集成验证

1. `What are recent approaches to reinforcement learning for robotic manipulation?`
   - 3 queries；DDGS 1/3 成功，Wikipedia 3/3 成功；单 Provider 失败被隔离。
   - 20 raw → 16 combined/unique URLs → 9 Documents → 9 unique Documents → 353 Chunks → 20 embedding Candidates → 6 Evidence / 4 URLs。
   - Reranker 将 embedding 第 9 名提升到 Evidence 第 1；最终含 chunk 31，验证长文非开头内容可进入上下文。
   - search 8.591s；crawl 2.061s；embedding 13.965s；rerank 5.037s；LLM 29.427s；total 75.769s；退出码 0。

2. `What are recent improvements in neural dynamics based control for quadrotor UAVs?`
   - 23 raw → 20 URLs → 5 Documents → 248 Chunks → 20 Candidates → 6 Evidence / 3 URLs。
   - PDF、HTTP 403 和 timeout 均被逐页隔离；Reranker 将 embedding 第 9 名提升到 Evidence 第 1；选中 81k 字符长文的 chunk 72。
   - search 6.653s；crawl 10.556s；embedding 10.441s；rerank 2.868s；LLM 22.185s；total 80.126s；退出码 0。

3. Wikipedia allowlist 长文测试：`How do temporal-difference learning and eligibility traces work in reinforcement learning?`
   - 27 raw → 20 capped → 9 allowed URLs → 8 Documents → 495 Chunks → 20 Candidates → 6 Evidence / 3 URLs。
   - Evidence 来自 chunk 0、4、14、15、39、58；每 URL 最多 2 段，证明长文中后段检索和 diversity cap 生效。
   - search 3.514s；crawl 3.344s；embedding 14.519s；rerank 2.647s；LLM 31.464s；total 70.955s；退出码 0。

### 当前验收

- V0.4：核心 20 项验收全部完成。
- Local Embedding / Reranker / LLM used：YES。
- Cloud LLM used：NO。

## V0.5「搜得深」

- 开始/完成日期：2026-09-03
- 目标：在 V0.4 Retrieval Pipeline 外层加入 Evidence Pool、Evidence Critic 和受预算约束的 Search → Critic → 补搜闭环。

### 已完成

- 新增按 `(URL, chunk_index)` 去重的 EvidencePool，提供累计数量、唯一来源数量和 source URLs。
- 新增 ResearchState、CriticResult、ResearchRoundTrace、ResearchTrace 和 ResearchResult。
- 新增独立 CriticContextBuilder，只向 Critic 提供有数量和字符预算的 Evidence 内容。
- 新增 EvidenceCritic：严格 structured output、共享 JSON extractor、一次 repair、typed failure。
- 把 V0.4 Search → Crawl → Extract → Chunk → Embed → Rerank 抽为可重复 ResearchRound；原 TrackerPipeline 改为复用它，V0.4 回归保持通过。
- 新增 ResearchAgent：首轮 SearchPlanner、后续 gap-driven queries、executed query 去重、跨轮 Evidence 累积和最多三轮预算。
- 实现 sufficient、max_rounds、no_follow_up_queries、duplicate_queries、no_new_evidence、critic_failure 六类停止原因。
- Critic 失败时 graceful degradation：保留已取得 Evidence，继续全局重排和回答。
- 最终把整个 EvidencePool 针对 Original Question 再次 rerank，默认只向 ContextBuilder 交付 Top 8。
- CLI 升级为 V0.5，普通模式只显示 Answer/Sources，新增 `--debug-agent`，`--debug-retrieval` 支持逐轮漏斗。
- 更新配置样例、README、模块化 `python -m tracker.cli` 入口和完整 Fake 测试。

### 自动化验证

- `.venv/bin/pytest -q`：121 passed；V0.4 基线 95 项继续通过，新增测试覆盖 Agent Loop 的状态、决策、停止、降级与最终重排。
- `python -m compileall -q src tracker main.py`：passed。
- 项目没有安装或配置 ruff/mypy，按“如果项目已有”约束未额外引入。

### 真实联网集成验证

- Query：`What are the major recent approaches to reinforcement learning for robotic manipulation, what problems do they solve, and what are their limitations?`
- 使用 `MAX_RESEARCH_ROUNDS=2`、真实 DDGS/Wikipedia、网页读取、BGE-M3、bge-reranker-v2-m3 和 Qwen3-8B。
- Round 1：3 queries；25 raw → 19 URLs → 8 Documents → 713 Chunks → 20 candidates → 5 new Evidence / 3 sources。
- Round 1 Critic：insufficient；发现“近期具体 PPO/SAC 类算法”和“真实部署的安全/能耗限制”缺口；生成 2 条定向补搜 query。
- Round 2：2 follow-up queries；10 raw → 9 URLs → 7 Documents → 526 Chunks → 20 candidates → 6 new Evidence。
- EvidencePool：5 → 11 chunks / 6 sources；证明补搜与跨轮累积实际生效。
- Round 2 Critic 仍判不足，达到测试预算后以 `max_rounds` 停止。
- Final Global Rerank：11 pooled → 8 final Evidence，明确使用 Original Question；最终 Qwen 生成带来源引用的回答。
- 性能：planning 12.031s；search 16.920s；crawl/extract 9.593s；retrieval 55.140s；critic 55.259s；final rerank 0.458s；generation 36.068s；total 188.002s。
- DDGS 个别任务失败被 SourceManager 隔离；Wikipedia 仍保证两轮继续运行。
- 真实 sufficient 控制测试：`What is reinforcement learning?` 在 Round 1 得到 6 Evidence / 4 sources 后，Critic 返回 `sufficient=true`、空 gaps 和空 follow-up queries；Agent 立即以 `sufficient` 停止，没有执行 Round 2。
- 控制测试曾暴露本地 Qwen 对窄问题擅自要求比较、应用和数学细节的 scope creep；Critic prompt 已增加“缺口必须映射到原问题”和 `What is X?` scope example，修复后真实端到端复测通过。

### 当前验收

- V0.5：Evidence-Driven Adaptive Search Loop 完成。
- Real follow-up search triggered：YES。
- Local Embedding / Reranker / Critic / Answer LLM used：YES。
- Cloud LLM used：NO。

### 运行问题：非法 ALL_PROXY 导致本地模型加载失败

- 记录日期：2026-09-04
- 现象：搜索、网页抓取和 Chunking 均成功，但在 `Loading embedding model BAAI/bge-m3` 后退出。即使设置 `HF_HUB_OFFLINE=1`，并分别使用 CUDA 与 CPU，仍报告模型加载失败。
- 原通用错误信息误导为 Hugging Face 网络或模型下载问题；增加底层异常显示后，实际错误为：

```text
ValueError: Unknown scheme for proxy URL URL('socks\://127.0.0.1:7890/')
```

- 根因：shell 中的 `ALL_PROXY` / `all_proxy` 使用了非法地址 `socks\://127.0.0.1:7890`。其中反斜杠不属于 URL，Hugging Face/httpx 在初始化本地模型相关客户端时解析失败。模型权重本身已完整缓存；CUDA、磁盘和系统内存均正常。
- 为什么 CPU 也失败：异常发生在代理 URL 解析和模型初始化阶段，早于实际 CPU/GPU 推理，因此更换计算设备不能解决。

#### 代码修复

- 在 `src/network.py` 新增 `hide_unsupported_proxy_environment()` context manager。
- BGE-M3 和 Reranker 初始化期间，只临时隐藏 scheme 不是 `http`/`https` 的代理变量，加载结束后原样恢复。
- 合法的 HTTP(S) 代理继续保留；模型加载发生在搜索任务结束后，因此不会影响 DDGS 联网搜索。
- `BGEEmbedder` 的加载异常现在保留底层异常类型和消息，不再把所有问题统一误报为 Hugging Face 网络失败。
- 同一保护同时用于 `SentenceTransformer` 和 `CrossEncoder`，避免 Final Rerank 再次受到非法代理影响。

#### 用户环境处理

如果已经配置合法的 `HTTP_PROXY` / `HTTPS_PROXY`，可删除无效的 ALL_PROXY：

```bash
unset ALL_PROXY all_proxy
```

并从 `~/.bashrc` 或 `~/.profile` 删除 `socks\://127.0.0.1:7890`。合法 SOCKS URI 应写成 `socks5://127.0.0.1:7890`，不能包含反斜杠；当前项目加载本地模型时仍会隔离 SOCKS-only proxy。

#### GPU 使用说明

- 推荐 `RETRIEVAL_DEVICE=cuda`，或使用默认 `auto` 自动选择 CUDA。
- `INFO Offloaded reranker model to CPU` 表示 GPU 推理结束后把模型权重暂存到系统内存，为 Qwen Critic/Answer Generator 释放 8 GB 显存；不表示 Reranker 使用 CPU 完成了该次推理。
- Final Global Rerank 时，Reranker 会自动从 CPU 移回 CUDA，完成计算后再次 offload。
- `OLLAMA_KEEP_ALIVE=0` 让 Qwen 每次调用后释放显存，使 Qwen、BGE-M3 和 Reranker 按阶段轮流使用 GPU。

#### 修复验证

- 在真实非法环境 `ALL_PROXY=socks\://127.0.0.1:7890`、`HF_HUB_OFFLINE=1` 下，本地 BGE-M3 成功加载并完成编码：`INVALID_PROXY_LOCAL_MODEL_OK (1, 1024)`。
- `.venv/bin/pytest -q`：124 passed。
- `compileall`：passed。
- `git diff --check`：passed。

## V1.0-A「Agent Runtime Hardening」

- 开发日期：2026-09-04
- 目标：不改变 V0.5 ResearchAgent 决策和 V0.6 Grounding 的前提下，增加可靠的单机执行生命周期、预算、超时、有限重试、取消传播、checkpoint/resume 与 RuntimeTrace。

### 实现

- 新增 `src/runtime/`：RunRequest/RunStatus/RuntimeState/RuntimeResult/RuntimeTrace、错误 taxonomy、BudgetTracker、RetryPolicy、RuntimeExecutionContext、CheckpointStore 和 AgentRuntime。
- 每个新 run 使用唯一 UUID；外部始终获得 structured terminal status，而不是只能捕获顶层异常。
- Search/Crawl/LLM 在调用前消费独立 budget；超过上限时不启动超额请求，返回 `BUDGET_EXCEEDED` 与精确 termination reason。
- Search/Crawl 使用 operation timeout；Ollama 使用 `LLM_TIMEOUT`，同时把 overall deadline 的剩余时间下传给 HTTP transport，避免同步本地模型调用阻塞极短 overall timeout。
- 仅对 transport timeout/error、429、502/503/504 和显式 RetryableError 做有限指数退避；配置、schema 和程序错误不重试。
- `asyncio.CancelledError` 独立传播到 Runtime 的 `CANCELLED` 状态；ResearchRound 会取消并等待 crawl sibling tasks。
- 每轮完成后保存 SearchPlan、executed queries、EvidencePool、下一轮 queries 和 counters；resume 从最后完成 round 继续，不恢复函数中间状态。
- checkpoint 使用 `.runtime/runs/{run_id}/state.json` 和 `trace.json`，通过临时文件、fsync、`os.replace` 原子覆盖；损坏/缺失 checkpoint 返回清晰错误。
- CLI 新增 `--debug-runtime`、`--resume`、`--run-timeout` 和 rounds/search/crawl/LLM budget overrides；原 `python -m tracker.cli "question"` 与 `execute_pipeline()` 默认行为保留。
- 所有 CLI 日志通过 Runtime context 注入统一 `run_id` 与 stage；RuntimeTrace 关联已有 ResearchTrace/GroundingTrace，并记录 lifecycle、limits、counters、retry、timeout、timings 和 checkpoint path。

### Fault Injection 与自动化验证

- Retryable failure：第三次成功、retry 次数和 0.5→1.0 秒 backoff 正确。
- Non-retryable failure：只调用一次。
- Operation timeout：按策略尝试两次后成为结构化失败；timeout/retry/search counters 正确。
- Overall timeout：Fake slow task 返回 `TIMED_OUT`；真实 `--run-timeout 0.1` 在约 0.113 秒终止，round/search/crawl 均未启动。
- Budget：四类 budget 均在调用前阻止超额工作。真实 crawl limit=8 时在第 9 页前返回 `BUDGET_EXCEEDED`，counter 保持 8。
- Cancellation：父 task 取消后 child `finally` 执行，结果为 `CANCELLED`。
- Checkpoint：固定路径可重复覆盖、atomic replace、missing/corrupt/unsafe run ID 均有测试。
- Resume：自动化测试从 Round 1 checkpoint 只执行 query b，复用 Evidence A，并对跨轮重复 Evidence 去重。
- `.venv/bin/python -m pytest -q`：210 passed，其中 `tests/runtime/` 38 个专项测试。
- `.venv/bin/python -m compileall -q src main.py tracker tests`：passed。
- `git diff --check`：passed。
- 项目未安装或配置 ruff/mypy，因此未额外引入工具链。

### 真实集成验证

- 正常问题：`What is retrieval-augmented generation?`
- 结果：`SUCCEEDED / evidence_sufficient`；1 research round；7 search calls（含 1 retry）；4 crawl calls；5 LLM calls；1 retry；2 timed-out attempts；总耗时 230.354s。
- Provider fault isolation：一个 DDGS operation 连续 timeout，Wikipedia 与其余 DDGS tasks 保留，最终仍形成 4 Final Evidence。
- Grounding：5 draft claims → 5 verified → 0 dropped；3 unique cited sources，citation coverage 100%。
- Checkpoint：`.runtime/runs/run_0afcad79f153457ea756eeef8e791e28/state.json` 与 `trace.json`。
- 真实 resume：同一 run ID 从 `research_complete=true` 恢复，research rounds/search/crawl counters 分别保持 1/7/4，没有重做搜索或抓取；只重新执行 final rerank 与 grounded generation。
- 真实 timeout：`run_d6d49a60064a4f5d90a415331bfb3e49` 在 0.113s 返回 `TIMED_OUT / overall_timeout`。
- 真实 budget：`run_29f3d596b6ce4a21ac833a0bdb29e8c5` 返回 `BUDGET_EXCEEDED / max_crawl_requests`。

### 当前边界

- Resume 是 research-round stage-level，不恢复单个 HTTP request、embedding batch 或 LLM token generation 的中间位置。
- 对 DDGS 等第三方库内部已启动的同步 worker thread，Python cancellation 不能强制杀线程；Runtime 会取消 asyncio child、忽略其迟到结果，并依靠 transport timeout 收敛。
- Checkpoint 是本机单进程 JSON，不提供分布式锁、任务队列、远程 worker 或跨机器恢复。

## V1.0-B「Evaluation Harness」

- 开发日期：2026-09-06 至 2026-09-07。
- 范围：只增加评测消费者，不修改 Agent、Runtime、Grounding、检索策略或 prompt。
- 新增严格 JSONL loader、五类 case schema、顺序 EvalRunner、公开 trace 提取、纯函数指标、
  多维报告、同题集 compare/regression gates，以及默认关闭的结构化 LLM Judge。
- E2E 复用 `main.execute_runtime → AgentRuntime.run`；组件评测使用固定输入调用已有组件。
- 每题立即保存；单题故障继续、用户取消终止并保存 partial report；拒绝覆盖已有 run 目录。
- 保存数据快照/指纹、Git commit/dirty、代码指纹、实际配置、模型 digest 和实验标签。
- `eval_runs/` 已忽略，不提交模型、运行产物或自动制造的 gold；demo 全部标记未人工审核。

### 自动化验证

- 全量 `pytest -q`：272 passed（原 214 + 新增 58）。
- `compileall`、`pip check`、`git diff --check` 通过；未配置 ruff/mypy，不引入额外依赖。
- 覆盖数据校验/重复/指纹、Runtime 接入、失败隔离、超时/预算/取消、指标分母、N/A、
  默认 Judge 禁用、报告分类统计、同数据集限制及 pass/warn/fail 门禁。

### 真实评测

- 5-case smoke：`eval_runs/v1.0b-smoke/report.md`，5/5 succeeded。
- 总耗时 1227.62 秒；平均 245.52 秒，median 240.55 秒，P95 313.59 秒。
- 平均 1.2 research rounds；一次补搜新增 6 条 Evidence；28 draft → 27 verified → 1 dropped。
- 重试发生在 4/5 题；DDGS/网页/LLM 超时等故障由既有 Runtime 隔离或重试。
- 从保存的 `cases.jsonl` 重新计算，与 `metrics.json` 完全一致。
- Citation mapping coverage=100% 不是人工正确率；E2E 无相关性 gold，所以 Recall@10 为 N/A。
- 首次完整基线外部中断，仅留下 2 题和 completed=false；保留原目录，以 `v1.0b-qwen3-base-rerun` 完整重跑。
- 完整 baseline：8 题全部执行，6 succeeded / 2 budget_exceeded，Runtime success=75%；累计 1658.56 秒。
- 两个困难题 `vla_001 / cost_001` 达到 max_crawl_requests=30；下一题继续执行，未隐藏失败。
- 中文题 structured planner 失败后 legacy fallback，grounding_verified=false；不进入 verified-grounding 分母。
- 其余 5 个有效 GroundingTrace：47 draft / 47 verified；mapping coverage=100%，不是人工正确率。
- 机器人题 3 轮后 no_new_evidence 停止；CoverageResult 同时报 adequate=true 和 missing limitations，保留该自评矛盾供人工复核。
- timeout candidate：8/8 timed_out；budget 注入：2/2 budget_exceeded，LLM counter 均保持 1。
- Planner demo：2/2 输出有效；Critic demo：TP=0/TN=1/FP=0/FN=1；Verifier demo：TP=1/TN=1/FP=0/FN=0。
- 固定语料 Retrieval demo：URL/chunk Recall@10=1、Precision@10=0.5、MRR@10=1；均非人工 gold。
- 相同 8-case 指纹比较正确 FAIL（exit 2），Runtime success 0.75→0；丢失 Grounding observation 也被标出。
- 不同 5/8-case 比较正确拒绝（exit 1）。报告：`eval_runs/v1.0b-fault-comparison.md`。

### 人工工作与边界

- 用户需要审核/扩充 gold、人工抽查 citation support、冻结代码和题集，再做正式 Base/Candidate 比较。
- LLM Judge 只提供辅助相关性/覆盖/清晰度意见，不能成为事实真值或唯一总分。
- 详细定义、命令、25 个学习概念见 `EVALUATION.md`；交付与真实结果见 `V1_0B_IMPLEMENTATION_SUMMARY.md`。
