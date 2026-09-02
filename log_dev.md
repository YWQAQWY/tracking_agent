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

- V0.2：Crawler + 网页正文解析。
