# CLI LLM Demo (LangChain)

一个基于 **LangChain** 的 Python CLI 工具，用于调用 OpenAI 兼容的聊天补全 API，支持 Agent 工具调用。

## 技术栈

- **LangChain** — LLM 应用框架
- **LangGraph** — ReAct Agent 构建
- **langchain-openai** — OpenAI 兼容 API 接入（DeepSeek 等）

## 安装

```powershell
pip install -r requirements.txt
```

## 设置

PowerShell：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

可选配置：

```powershell
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

默认使用 DeepSeek API（`https://api.deepseek.com/v1`，模型 `deepseek-v4-flash`）。

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

| 工具 | 说明 |
|------|------|
| `read_file` | 读取文件内容 |
| `list_directory` | 列出目录内容 |
| `execute_command` | 执行系统命令 |
| `get_current_time` | 获取当前时间 |
| `calculate` | 计算数学表达式 |

工具定义在 `tools.py` 中，使用 LangChain 的 `@tool` 装饰器。

## 项目结构

```
learn_ai/
├── llm_cli.py          # 主程序：LangChain Agent + CLI
├── tools.py            # 工具定义（@tool 装饰器）
├── test_tools.py       # 测试脚本
├── requirements.txt    # Python 依赖
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

本项目使用 LangChain + LangGraph 的 ReAct Agent 架构：

1. **ChatOpenAI** — 通过 OpenAI 兼容接口调用 DeepSeek 等 LLM
2. **@tool 装饰器** — 声明式工具定义，自动生成 JSON Schema
3. **create_react_agent** — LangGraph 预构建的 ReAct Agent，自动处理工具调用循环
4. **StreamingHandler** — 回调处理器，实现流式输出和工具调用日志
