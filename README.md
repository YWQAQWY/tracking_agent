# Tracker V0.1

Tracker 是一个“本地大模型 + 在线搜索”的最小联网问答系统。V0.1 使用本机 Ollama 中的 `qwen3:8b` 规划一个搜索查询，通过 DDGS 获取网页标题、URL 和摘要，再由同一个本地模型根据这些摘要生成带来源编号的回答。

所有 LLM 推理都在本机完成。项目不调用任何云端 LLM，也没有云模型 fallback；只有 DDGS SearchProvider 会主动访问互联网。

## V0.1 的能力与边界

当前版本能够：

- 将用户问题转换为经过 Pydantic 校验的 `SearchPlan`。
- 通过唯一的 DDGS Provider 联网搜索。
- 将 DDGS 原始字段归一化为 `SearchResult`。
- 使用本地 `qwen3:8b` 根据搜索摘要回答，并显示来源。
- 对 Ollama 不可用、模型缺失、非法 LLM JSON、网络失败和空结果给出清晰错误。

当前版本不能：

- 抓取或解析网页正文。
- 使用 Embedding、Reranker、向量数据库或 RAG 框架。
- 执行多查询、多搜索源、搜索循环或多 Agent 协作。

V0.1 只使用搜索引擎返回的标题、URL 和摘要，因此回答质量受搜索摘要的完整性与准确性影响。

## 环境要求

- Ubuntu 24.04
- Python 3.12
- NVIDIA GPU 与可用驱动（建议先运行 `nvidia-smi`）
- Ollama
- 足够存放 `qwen3:8b` 的磁盘空间

Ollama 自带所需运行能力。不要为本项目额外安装 PyTorch、CUDA Toolkit 或 cuDNN；`nvidia-smi` 显示的 CUDA Version 是驱动支持能力，不等于需要安装 CUDA Toolkit。

## 安装

安装官方 Ollama：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

确认并启动服务：

```bash
sudo systemctl enable --now ollama
systemctl status ollama --no-pager
curl --noproxy '*' http://localhost:11434/api/tags
```

下载并实际运行本地模型：

```bash
ollama pull qwen3:8b
ollama run qwen3:8b "请只回答：LOCAL_LLM_OK"
```

创建项目虚拟环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

项目不会修改 Ubuntu 系统 Python，也不需要 `sudo pip`。

## 配置

复制示例配置：

```bash
cp .env.example .env
```

默认配置如下：

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
SEARCH_MAX_RESULTS=5
```

`SEARCH_MAX_RESULTS` 允许 1 到 10。为保证不使用云端 LLM，程序只接受 `localhost`、`127.0.0.1` 或 `::1` 形式的 `OLLAMA_HOST`。

如果系统设置了 HTTP 或 SOCKS 代理，命令行验证本地 API 时使用 `curl --noproxy '*'`。项目会在导入 Ollama Python 包时隔离代理，并在导入后恢复原环境；`LLMClient` 的本地请求也不会读取代理变量，因此 `socks://127.0.0.1:7890` 不会影响 Ollama，同时 DDGS 仍可继续使用系统代理。

## 运行

确保 Ollama 正在运行且 `qwen3:8b` 已下载，然后执行：

```bash
source .venv/bin/activate
python main.py
```

输入问题后，CLI 会依次显示搜索关键词、结果数量、最终回答和来源。

## 测试

默认测试完全离线，不依赖真实 DDGS 或 Ollama：

```bash
.venv/bin/python -m pytest
```

手动检查本地 Python → Ollama 链路：

```bash
.venv/bin/python - <<'PY'
from ollama import Client

client = Client(host="http://localhost:11434", trust_env=False)
response = client.chat(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "请只回答 PYTHON_LOCAL_LLM_OK"}],
)
print(response.message.content)
PY
```

## 项目架构

```text
tracker/
├── .env.example
├── .gitignore
├── README.md
├── log_dev.md
├── main.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── llm/client.py
│   ├── models/
│   │   ├── search_plan.py
│   │   └── search_result.py
│   ├── planner/search_planner.py
│   ├── search/
│   │   ├── base.py
│   │   └── ddgs_provider.py
│   └── answer/answer_generator.py
└── tests/
    ├── test_answer_generator.py
    ├── test_config.py
    ├── test_llm_client.py
    ├── test_search_plan.py
    └── test_search_provider.py
```

核心数据流：

```text
Question
→ LLMClient → localhost Ollama/qwen3:8b
→ SearchPlanner → SearchPlan
→ DDGSSearchProvider → list[SearchResult]
→ AnswerGenerator
→ LLMClient → localhost Ollama/qwen3:8b
→ Answer + Sources
```

`main.py` 只负责组织流程和 CLI 展示；其他模块不直接调用 DDGS，Planner 和 AnswerGenerator 也不直接调用 Ollama SDK。

## 常见错误

无法连接本地 Ollama：

```bash
systemctl status ollama --no-pager
curl --noproxy '*' http://localhost:11434/api/tags
```

本地没有模型：

```bash
ollama list
ollama pull qwen3:8b
```

DDGS 搜索失败时，请检查网络、DNS 和代理。程序不会因搜索失败切换到任何云端 LLM。

## 下一阶段

V0.2 计划加入 Crawler 与网页正文解析，提升检索上下文质量；本版本不实现这些能力。
# tracking_agent
