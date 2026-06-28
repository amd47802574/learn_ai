# CLI LLM Demo (LangChain)

一个基于 **LangChain** 的 Python CLI 工具，用于调用 OpenAI 兼容的聊天补全 API，支持 Agent 工具调用与 RAG 医学文献检索。

## 技术栈

- **LangChain** — LLM 应用框架
- **LangGraph** — ReAct Agent 构建
- **langchain-openai** — OpenAI 兼容 API 接入（DeepSeek 等）
- **Chroma** — 向量数据库（文献检索）
- **pypdf / unstructured** — 多格式文档加载（PDF / Markdown / TXT）

## 安装

1. 创建虚拟环境：

```powershell
python -m venv .venv
```

2. 激活虚拟环境：

PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

CMD：

```cmd
.venv\Scripts\activate.bat
```

3. 安装依赖：

```powershell
pip install -r requirements.txt
```

## 设置

复制 `.env.example` 为 `.env` 文件：

```powershell
cp .env.example .env
```

然后编辑 `.env` 文件，填入你的 API 密钥：

```
OPENAI_API_KEY=your_api_key_here
```

也可通过环境变量临时设置（PowerShell）：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

可选配置：

```powershell
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

默认使用 DeepSeek API（`https://api.deepseek.com/v1`，模型 `deepseek-v4-flash`）。

## RAG 医学文献检索

将医学文献放入 `medical_docs/` 目录（支持 `.txt`、`.md`、`.pdf`），即可构建向量索引并进行检索。

### 环境变量

RAG 模块使用独立的 Embedding 服务配置（优先读取 `EMBEDDING_*`，回退到 `OPENAI_*`）：

```
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
EMBEDDING_MODEL=embedding-3
```

默认使用智谱 Embedding API。

### 构建索引

```powershell
python .\rag.py --build
```

向量数据库持久化在 `chroma_db/` 目录，后续运行会自动加载已有索引。

### 检索

```powershell
python .\rag.py --query "高血压的用药建议" --top-k 3
```

## 运行

单次提问（带工具支持）：

```powershell
python .\llm_cli.py "列出当前目录的文件"
```

交互式对话（带工具支持）：

```powershell
python .\llm_cli.py
```

禁用工具（纯文本模式）：

```powershell
python .\llm_cli.py --no-tools "用三句话解释什么是大模型"
```

对话模式命令：

- `/clear` 清空对话上下文
- `/exit` 退出

## 可用工具

Agent 可自动调用以下工具：

| 工具               | 说明           |
| ------------------ | -------------- |
| `read_file`        | 读取文件内容   |
| `list_directory`   | 列出目录内容   |
| `execute_command`  | 执行系统命令   |
| `get_current_time` | 获取当前时间   |
| `calculate`        | 计算数学表达式 |

工具定义在 `tools.py` 中，使用 LangChain 的 `@tool` 装饰器。

## 项目结构

```
learn_ai/
├── llm_cli.py          # 主程序：LangChain Agent + CLI
├── tools.py            # 工具定义（@tool 装饰器）
├── rag.py              # RAG 模块（文献索引与检索）
├── test_tools.py       # 测试脚本
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量示例文件
├── medical_docs/       # 医学文献目录（.txt/.md/.pdf）
├── chroma_db/          # Chroma 向量数据库（自动生成）
└── README.md
```

## 命令行参数

```
python .\llm_cli.py [prompt] [options]

参数:
  prompt                提示文本，省略则进入交互模式

选项:
  --api-key API_KEY     API 密钥（或设置 OPENAI_API_KEY 环境变量）
  --base-url BASE_URL   API 基础 URL（默认: https://api.deepseek.com/v1）
  --model MODEL         模型名称（默认: deepseek-v4-flash）
  --system SYSTEM       系统提示词
  --temperature TEMP    温度参数（默认: 0.7）
  --no-stream           禁用流式输出
  --no-tools            禁用工具调用
```

## 架构说明

本项目使用 LangChain + LangGraph 的 ReAct Agent 架构，并集成 RAG 医学文献检索：

### Agent 架构

1. **ChatOpenAI** — 通过 OpenAI 兼容接口调用 DeepSeek 等 LLM
2. **@tool 装饰器** — 声明式工具定义，自动生成 JSON Schema
3. **create_react_agent** — LangGraph 预构建的 ReAct Agent，自动处理工具调用循环
4. **StreamingHandler** — 回调处理器，实现流式输出和工具调用日志

### RAG 架构

1. **DocumentLoader** — 从 `medical_docs/` 加载 `.txt`/`.md`/`.pdf` 文献
2. **RecursiveCharacterTextSplitter** — 按中文字符分块
3. **Chroma** — 向量数据库，使用智谱 Embedding API 构建索引
4. **similarity_search** — 基于语义相似度检索相关文献片段
