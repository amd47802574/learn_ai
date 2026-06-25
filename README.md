# CLI LLM Demo

一个零依赖的 Python CLI 工具，用于调用 OpenAI 兼容的聊天补全 API，支持 MCP 工具集成。

## 设置

PowerShell（PowerShell 命令）：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

可选配置：

```powershell
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

对于其他 OpenAI 兼容的提供商，请修改 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。

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

- `/clear` 清空对话上下文。
- `/exit` 退出。

## MCP 工具

项目内置了 MCP (Model Context Protocol) 工具服务器，LLM 可以自动调用以下工具：

- `read_file` - 读取文件内容
- `list_directory` - 列出目录内容
- `execute_command` - 执行系统命令
- `get_current_time` - 获取当前时间
- `calculate` - 计算数学表达式

工具会在 LLM 认为需要时自动调用，无需手动干预。

## 示例对话

```
You> 当前目录有哪些文件？

[调用工具: list_directory({'path': '.'})]
[工具结果: 📁 __pycache__/
📄 .gitignore (123 bytes)
📄 README.md (1.2 KB)
📄 llm_cli.py (8.5 KB)
📄 mcp_tools.py (6.2 KB)]

AI> 当前目录包含以下文件：
- __pycache__/ (Python 缓存目录)
- .gitignore
- README.md
- llm_cli.py (主程序)
- mcp_tools.py (MCP 工具服务器)
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
  --no-tools            禁用 MCP 工具
  --mcp-server PATH     MCP 服务器脚本路径（默认: mcp_tools.py）
```
