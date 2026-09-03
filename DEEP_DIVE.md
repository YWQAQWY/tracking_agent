# 值得深挖的代码：原理详解

> 配合 [ARCHITECTURE.md](ARCHITECTURE.md) 阅读。本文把其中最值得学的三段代码逐层讲透：**结构化输出与自我修复**（planner）、**工具抽象与适配器**（search）、**LLM 网关**（llm/client），最后串起来对照真实 Agent 框架。

## 1. search_planner.py：让概率性的 LLM 输出变得可靠

### 1.1 根本矛盾：模型是「概率生成器」，你需要的是「确定性数据」

LLM 的工作原理是**自回归生成**：逐 token 地从概率分布中采样。

```
P(token_t | 之前的全部 token) → 采样 → token_t → 拼上去 → 生成下一个
```

由此推出两个工程推论：

1. **任何输出都只是「概率上合理」，不是「被保证正确」**。即使 prompt 里写「只输出 JSON」，模型也没有任何机制被强制这么做——它只是被训练成「倾向于」这么做。temperature > 0 时，同一个 prompt 每次采样结果都可能不同。
2. **模型被 RLHF 训练成「乐于助人」**——它天然倾向于解释、打招呼、排版、给思考过程。所以让它输出 JSON 时，它经常带 `<think>`（Qwen3 的思维链模式）、带 ``` 代码围栏、带「好的，这是结果：」这类废话。

所以核心工程原则：**把 LLM 当作一个不可靠的远程 API——输出必须校验，失败必须反馈，绝不直接信任。** 这就是 [search_planner.py](src/planner/search_planner.py) 全部代码存在的理由。

### 1.2 结构化输出的三种方案（本项目用的是第一种）

| 方案 | 机制 | 可靠性 | 成本 |
|------|------|--------|------|
| **A. 提示词约束 + 解析清洗**（本项目） | 在 prompt 里描述 schema，模型「自觉」遵守，代码兜底 | 中（解析层能救回大部分） | 低，任何模型都行 |
| **B. Function calling / 服务端结构化输出** | 把 JSON Schema 作为 API 参数传入，服务端约束解码（OpenAI structured outputs、Ollama 的 `format=` 参数、vLLM） | 高（语法层面保证合法 JSON） | 需要服务端支持 |
| **C. 服务端 grammar 约束解码**（llama.cpp GBNF / Outlines / XGrammar） | 解码时每一步只允许采样「符合文法」的 token，在概率分布上做掩码 | 最高（硬保证） | 实现复杂；B 方案的底层就是它 |

关键认知：**B/C 只能保证「语法对」，不能保证「语义对」**。就算 grammar 强制输出合法 JSON，`{"query": "...", "max_results": 99}` 照样违反你的业务规则。所以无论选哪种方案，**Pydantic 校验这一层永远不能省**——本项目的 `SearchPlan.model_validate()` 就是这道永远在场的闸门（[search_planner.py:72-74](src/planner/search_planner.py#L72-L74)）。

（可做实验：给 `LLMClient.chat` 加一个 Ollama 的 `format=` 参数传 JSON Schema，观察失败率变化——这是理解 B 方案最快的方式。注意 pydantic 生成的 schema 里的 min/max 会被 grammar 利用，但 `_parse` 清洗和重试仍要保留作第二道防线。）

### 1.3 解析管道逐行拆解：为什么是这三步

[search_planner.py:64-74](src/planner/search_planner.py#L64-L74)：

```python
cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
```

- `re.DOTALL`：让 `.` 匹配换行。没有它，跨行的 `<think>\n...\n</think>` 就删不掉——思维链输出几乎必然跨行。
- 非贪婪 `.*?`：匹配到第一个 `</think>` 就停，避免吞掉后面的正文。
- 围栏正则用 `^` `$` 锚定：只剥**最外层**围栏，不动正文里出现的反引号（JSON 字符串里可能有）。
- 清洗只做「已知的常见污染」，真正决定成败的是下一步——**解析器永远比正则强**：

```python
try:
    payload = json.loads(cleaned)
except JSONDecodeError:
    payload = cls._extract_json_object(cleaned)
```

为什么 `json.loads` 优先？因为它是真正的递归下降解析器：嵌套对象、字符串里的 `{`、转义引号全部正确处理。任何用正则「匹配花括号」的尝试都会在这些情况翻车（比如 query 里含带 `{` 的 URL）。

[search_planner.py:76-87](src/planner/search_planner.py#L76-L87) 的 `_extract_json_object` 是兜底扫描：

```python
for index, character in enumerate(text):
    if character != "{":
        continue
    try:
        value, _ = decoder.raw_decode(text[index:])
    except JSONDecodeError:
        continue
    return value
```

原理：`json.JSONDecoder.raw_decode` 是标准库的**增量解析器**——从某个位置开始尝试解析「一个完整 JSON 值」，返回值和结束位置。所以这个循环的含义是：「从每个 `{` 开始试，第一个能构成完整 JSON 的位置就是答案」。它天然跳过开头的废话（「好的，结果如下：」）、天然跳过假的 `{`（「参考{图1}」解析会失败）。这是「**用解析器找边界，而不是猜边界**」——比正则稳健一个量级。

注意这条管道的**层次性**：regex 清洗处理「已知污染」→ `json.loads` 处理「整体合法」→ `raw_decode` 扫描处理「局部合法」→ Pydantic 处理「语法对但语义错」。四层各管一段，互不依赖，缺一层都会漏。

### 1.4 自我修复循环：为什么要把上次输出塞回去

[search_planner.py:45-60](src/planner/search_planner.py#L45-L60)：

```python
except (JSONDecodeError, ValidationError, TypeError, ValueError) as first_error:
    repair_prompt = (
        "上一次输出无法通过 JSON 校验。请重新输出且只输出 JSON 对象。\n"
        f"用户问题：{clean_question}\n"
        f"max_results：{self.default_max_results}\n"
        f"上一次输出：{raw}"
    )
```

三个原理层面的要点：

1. **每次 `chat()` 都是失忆的。** LLMClient 无状态，修复请求是一个**全新的请求**——模型完全不知道刚才发生过什么。所以修复 prompt 必须**自包含**：重复问题、重复约束、附上失败输出。这是新手最容易踩的坑：写「请修正」却发现模型根本不知道要修正什么。
2. **外部反馈优于自我反思。** 研究（Huang et al., *Large Language Models Cannot Self-Correct Reasoning Yet*, ICLR 2024）表明：没有外部信号时，模型「自我纠错」常常把对的改成错的——因为它自己判断不出错。但**给它外部反馈（解析器报错 + 原文）时，修复成功率显著提高**。本项目的重试正是后者：反馈来自解析器，不是模型「自觉」。相关方法还有 Self-Refine（Madaan et al. 2023）和 Reflexion（Shinn et al. 2023）。
3. **有限重试是成本纪律。** 每次重试 = 一次完整推理（本地 8B 也要数秒），无限重试还可能死循环。一次失败通常是采样波动 + 反馈不足，重抽一次已能覆盖绝大多数情况；两次都失败说明问题不在采样，抛错比硬刚更诚实。

再看异常分类的精确性（[search_planner.py:55-62](src/planner/search_planner.py#L55-L62)）：`LLMError`（服务挂了）直接上抛**不重试**——模型不可达时重试没有意义。只有「输出不好」才值得再试。**区分「环境故障」和「输出故障」是 agent 重试逻辑的基本功。**

### 1.5 补充知识

- **temperature 与解析失败率**：本项目的 `chat()` 没设 temperature（用 Ollama 默认值）。采样越随机，解析失败率越高；需要严格 JSON 时可以调低（Ollama 支持 `options={"temperature": 0}`）。但低温也可能让重试「两次输出一模一样」——反正有反馈修复兜底。
- **`<think>` 的来历**：Qwen3 支持思考模式，开启时在正文前用 `<think>...</think>` 输出思维链（本项目实测出现过）。Ollama 侧可用 `think: false` 参数关闭（对支持该参数的模型）。本项目选择在**解析层**清洗而不是依赖服务端配置——防御性编程：不依赖外部配置，只依赖自己的代码。
- **Pydantic `extra="forbid"`**（[search_plan.py:9](src/models/search_plan.py#L9)）：模型多吐一个字段（`{"query": ..., "max_results": ..., "reason": "..."}`）就整体拒绝。对 LLM 输出宁严勿宽——多出来的字段说明模型没真正理解指令，重试比放行更安全。

## 2. search/：工具抽象与适配器

### 2.1 Agent 里的「工具」到底是什么

Agent 框架里一个 tool 的标准四要素：

```
name + description   → 写进 system prompt，让 LLM 知道「有这个能力、何时该用」
args_schema          → 参数约束（JSON Schema）
execute 函数         → 真正干活，返回结果字符串
```

对照本项目：`SearchProvider` 接口只有 execute 函数（[base.py:12-16](src/search/base.py#L12-L16)）；name/description 以 SYSTEM_PROMPT 文本的形式写在 planner 里；参数以 `SearchPlan` 的形式定义。**同一个思想，不同的实现精度。** 真框架把四要素结构化，这样 LLM 才能「挑选」工具；V0.1 只有一个工具，用 prompt 文本描述就够了。

### 2.2 适配器模式与规范模型

[ddgs_provider.py:44-64](src/search/ddgs_provider.py#L44-L64)。DDGS 返回的字段叫 `href`/`body`（DuckDuckGo 的遗产命名），你的系统里叫 `url`/`snippet`。如果直接把 DDGS 的 dict 传给 AnswerGenerator，你的核心代码就和第三方库的字段名**焊死**了：DDGS 改版、换搜索引擎、换搜索 API，全都要动核心代码。

适配器模式（GoF）的解法：**在边界上做一次翻译，让脏的外部世界进门前变成你的规范模型**。

```
DDGS 世界（href/body/脏行/缺字段）→ _normalize → 你的世界（SearchResult：字段非空、URL 合法）
```

`SearchResult` 就是领域里的「中间表示」（canonical model）。这是所有接外部 API 的系统的标准动作，agent 领域尤其常见：**每个工具的返回都应该被归一化成你的内部类型，而不是让 LLM 直接吃原始 JSON。**

### 2.3 外部数据的防御式处理

- **坏行跳过而不是整体失败**：`_normalize` 每行独立 try/except，空标题、坏 URL、空摘要的行静默丢弃。外部数据脏是常态，5 条里坏 2 条，给 3 条比整体报错好。
- **宽容接受 + 严格验证**：字段值先 `str()` 强转（防 None / 非字符串），再交给 `SearchResult` 校验——两层各司其职。
- **提前断流**：凑够 `max_results` 就 break，不浪费网络和内存。

### 2.4 测试接缝（seam）

[ddgs_provider.py:20-21](src/search/ddgs_provider.py#L20-L21)：

```python
def __init__(self, search_func: RawSearch | None = None):
    self._search_func = search_func
```

Michael Feathers《修改代码的艺术》的核心概念：**seam（接缝）是「不改代码就能改变行为的位置」**。构造函数注入就是最经典的接缝——测试传一个 lambda，生产传 None 走真实 DDGS。没有这个接缝，测试就必须真的联网（慢、不稳定、不可重复）。

同样的思路贯穿全项目：`LLMClient._client` 可替换（[test_llm_client.py:31-34](tests/test_llm_client.py#L31-L34) 直接换成 FakeOllama）、planner 接收任意带 `chat` 方法的对象（StubLLM，鸭子类型）。**为测试设计的接缝，同时就是「未来换实现」的接缝——接缝是两用的。**

## 3. llm/client.py：LLM 网关

### 3.1 为什么全项目只能有一个文件碰 LLM

- **替换点唯一**：未来换 OpenAI/Anthropic/vLLM，只改这一个文件，其余模块不动。
- **横切关注点集中**：代理处理、异常翻译、响应格式兼容、健康检查，全在一扇门后面。
- **可观测**：加日志、加 token 计数、加统一重试策略，都只有一个位置。

这是「网关」（gateway）模式。LLM 调用是变化最快的外部依赖，必须隔离。

### 3.2 import 时代理隔离：一个真实的踩坑故事

[client.py:10-29](src/llm/client.py#L10-L29)。事故链条（开发日志里真实发生过）：

1. 用户 shell 里有 `ALL_PROXY=socks://127.0.0.1:7890`（代理软件）。
2. `ollama` Python 包在 **import 时**创建模块级默认 client，这个 client 读取代理环境变量。
3. HTTPX 处理 socks 代理需要 `httpx[socks]` 附加依赖；ollama 包没装它 → **import 直接崩溃**。
4. 于是整个项目连 import 都失败——即使你根本不想让本地请求走代理。

Python 原理：**import 有副作用是脆弱的**。模块级代码在 import 时执行一次，如果它依赖环境变量、网络、文件系统，那么「能不能 import」就取决于环境——这正是本项目踩的坑。

修复原理：**只在 import 那一瞬间给环境消毒，之后恢复原状**。

```python
_saved_proxy_environment = {key: os.environ.pop(key) for ... if key in os.environ}
try:
    from ollama import Client
finally:
    os.environ.update(_saved_proxy_environment)
```

- `finally` 保证恢复一定执行（import 成败都不污染环境）。
- 消毒范围精确到 import 语句本身：DDGS 后续还要用代理，所以不能全局删。
- 加上构造时的 `trust_env=False`（[client.py:50](src/llm/client.py#L50)）构成双保险：import 期消毒 + 运行期绕行。`trust_env=False` 是 HTTPX 的 transport 级开关——**根本不读任何代理/SSL 环境变量**，比 `NO_PROXY` 更硬（NO_PROXY 是「部分豁免」，trust_env=False 是「根本不看」）。

教训推广到 agent 开发：**本地回环流量和出网流量是两个世界，必须显式分开**。凡是你自己的 agent 项目，LLM 调用（本地或云端）与工具的网络请求（搜索、爬虫）要各自明确代理策略。

### 3.3 健康检查 = 飞行前检查

[client.py:52-66](src/llm/client.py#L52-L66)：进主流程之前先验证「服务可达？模型已下载？」，任何一环失败立刻退出，**绝不带着坏状态跑长流程**。这是 agent 里的铁律：一次 agent run 可能调用 LLM 几十次，第一分钟就知道模型坏了，比跑到一半才发现强得多。

细节：`_model_is_available` 把两边的 `:latest` 后缀都剥掉再比较（[client.py:121-123](src/llm/client.py#L121-L123)）——Ollama 的 tag 默认带 `:latest`，`qwen3:8b` 和 `qwen3:8b:latest` 是同一个模型。

### 3.4 异常链：`raise LLMError(...) from exc`

`from exc` 保留原始异常的 `__cause__`——traceback 会显示「直接原因是 ConnectionError，间接原因是 LLMError」。用户看到中文指引，开发者调试能看到原始堆栈，两层都不丢。**边界翻译、内部保真**：每层用自己的异常类型说话（模块间契约清晰），只在最外层翻译成人话。

### 3.5 纯函数封装的意义

[client.py:68-87](src/llm/client.py#L68-L87) 的 `chat(user_prompt, system_prompt) -> str`：

- **无状态**：上层不关心 messages 数组怎么拼。当前无状态是 V0.1 的刻意简化（多轮记忆的缺口见 ARCHITECTURE.md §8.3）。
- **协议无关**：上层不知道底层是 HTTP 还是 SDK、响应是对象还是 dict（`_extract_content` 两种都兼容）。换协议不动上层。
- **类型极简**：字符串进字符串出。「结构化输出」的职责不在网关，而在调用方（planner）——网关只负责「说话」，**解析是调用方的责任**。这个职责划分值得记住。

## 4. answer_generator.py：RAG 的「G」

### 4.1 这其实就是 RAG

RAG = Retrieve → Augment → Generate。本项目的 Retrieve 是 DDGS 搜索（语料是整个互联网），Augment 是 `_format_context` 把结果拼进 prompt（[answer_generator.py:35-45](src/answer/answer_generator.py#L35-L45)），Generate 是最后一次 LLM 调用。**任何「检索 + 注入上下文 + 生成」都是 RAG**，不需要向量数据库——向量库只是 Retrieve 的一种实现。

### 4.2 反幻觉提示词的结构

[answer_generator.py:11-15](src/answer/answer_generator.py#L11-L15)，四句话各干一件事：

1. 「优先且只能依据提供的搜索结果中的具体事实回答」→ **约束信息来源**（grounding）
2. 「如果结果不足以支持结论，明确说明信息不足，不要虚构」→ **显式允许说「不知道」**。幻觉的根源之一是模型被训练成「总要给答案」；给它说不知道的许可，显著降低编造率
3. 「用 [1]、[2] 等标注事实来源」→ **引用机制**，让用户能验证，也让模型「说话有凭据」
4. 「使用与用户问题相同的语言」→ 用户体验约束

实用技巧：**把约束写成一条条独立的、动词明确的句子，而不是一段散文**——模型对「规则列表」的遵守率高于对「语气要求」的遵守率。

### 4.3 编号上下文的双重作用

`[1] Title/URL/Snippet` 的编号块既是「给模型看的定位符」（引用时写 [1] 比抄 URL 省 token 且稳定），也是「给用户看的对照表」（CLI 按相同编号打印 Sources，[main.py:61-64](main.py#L61-L64)）。**同一个编号体系贯穿模型侧和人侧**，这个设计很干净。

## 5. main.py：组合根（composition root）

[main.py:38-40](main.py#L38-L40) 是**手动依赖注入**：对象在哪里创建、怎么连线，全部集中在一个地方。好处是看 main.py 就知道系统由哪些零件组成——没有魔法、没有全局单例、没有框架。

两个 try/except 边界（[main.py:24-31](main.py#L24-L31) 和 [main.py:42-58](main.py#L42-L58)）把「启动故障」和「流程故障」分开处理，但都转换成**用户能行动**的中文信息——错误信息里带命令（`ollama pull qwen3:8b`）而不是只描述问题。以及 `console.status` 的存在：LLM 推理要数秒，没有进度反馈用户会以为程序死了。**agent 系统的每个慢步骤都需要可见性。**

## 6. 串起来：对照真实 Agent 框架

| Tracker 组件 | 真实框架对应物 |
|--------------|---------------|
| `LLMClient` | LangChain `BaseChatModel` 封装 / OpenAI SDK 的 client |
| `SYSTEM_PROMPT`（planner） | 工具的 name+description 注入 + PromptTemplate |
| `SearchPlan` | 工具的 args_schema / function calling 的 parameters |
| `_parse` + 重试 | `JsonOutputParser` + `OutputFixingParser`（LangChain 里有同名组件，做的事一模一样） |
| `SearchProvider` | `BaseTool`（`_run` 方法 + schema） |
| `AnswerGenerator._format_context` | tool result 序列化 / RAG 的 augmentation |
| `main.run()` | 固定管道的 AgentExecutor / 手工展开一次的 agent loop |

真实 agent loop（ReAct 风格）长这样：

```python
messages = [system_prompt_with_tools]
while True:
    response = llm.chat(messages)            # 模型输出：要么调用工具，要么最终回答
    if response.is_final:
        return response
    tool_result = execute(response.tool, response.args)  # 框架执行工具
    messages.append(tool_result)             # 结果回填，进入下一轮
```

对照 V0.1：`plan → search → answer` 就是这个循环**手工展开一次**，且「调用哪个工具」不是模型决定的，是代码写死的。**V0.2 的最小升级就是把 main.py 的固定顺序换成这个 while 循环**——planner 的「输出 JSON 动作 + 校验 + 修复」直接复用为「动作解析器」，provider 直接复用为「工具执行器」，AnswerGenerator 的格式化复用为「结果回填」。三块积木一块都不用重写，只换编排方式。

## 总结

**V0.1 的价值不在功能（联网问答谁都会写），而在它把 agent 的关键骨架用最少的代码暴露出来了**：LLM 网关、工具抽象、结构化输出、自我修复、上下文回填、组合根。把这份代码吃透，再看 LangChain、Claude Code 或任何 agent 框架，你看到的都将是同一批模式的不同实现。
