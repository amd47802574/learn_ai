#!/usr/bin/env python3
"""简单的 MCP (Model Context Protocol) 工具服务器。

提供基本的工具功能，可以被 LLM 调用来扩展其能力。
"""

import json
import sys
import os
import subprocess
from typing import Any, Callable


# 工具注册表
TOOLS: dict[str, dict[str, Any]] = {}
TOOL_HANDLERS: dict[str, Callable] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Callable
) -> None:
    """注册一个 MCP 工具。

    Args:
        name: 工具名称。
        description: 工具描述。
        parameters: 工具参数的 JSON Schema。
        handler: 工具处理函数。
    """
    TOOLS[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
    }
    TOOL_HANDLERS[name] = handler


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    """处理 MCP 请求并返回响应。

    Args:
        request: MCP 请求字典。

    Returns:
        MCP 响应字典。
    """
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")

    # 处理初始化请求
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "simple-mcp-tools",
                    "version": "1.0.0"
                }
            }
        }

    # 处理工具列表请求
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": list(TOOLS.values())
            }
        }

    # 处理工具调用请求
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in TOOL_HANDLERS:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }

        try:
            result = TOOL_HANDLERS[tool_name](**arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Tool execution error: {str(e)}"
                }
            }

    # 处理未知方法
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


# ========== 工具实现 ==========

def read_file(path: str) -> str:
    """读取文件内容。

    Args:
        path: 文件路径。

    Returns:
        文件内容字符串。
    """
    if not os.path.exists(path):
        return f"错误：文件不存在 - {path}"

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"错误：读取文件失败 - {str(e)}"


def list_directory(path: str = ".") -> str:
    """列出目录内容。

    Args:
        path: 目录路径，默认为当前目录。

    Returns:
        目录内容列表。
    """
    if not os.path.exists(path):
        return f"错误：目录不存在 - {path}"

    try:
        items = os.listdir(path)
        result = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"📄 {item} ({size} bytes)")
        return "\n".join(result) if result else "空目录"
    except Exception as e:
        return f"错误：列出目录失败 - {str(e)}"


def execute_command(command: str) -> str:
    """执行系统命令。

    Args:
        command: 要执行的命令。

    Returns:
        命令输出。
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output if output else "命令执行完成，无输出"
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（30秒）"
    except Exception as e:
        return f"错误：命令执行失败 - {str(e)}"


def get_current_time() -> str:
    """获取当前时间。

    Returns:
        当前时间的字符串表示。
    """
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 数学表达式。

    Returns:
        计算结果。
    """
    try:
        # 安全检查：只允许基本数学运算
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"错误：计算失败 - {str(e)}"


# ========== 注册工具 ==========

register_tool(
    name="read_file",
    description="读取指定路径的文件内容",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件路径"
            }
        },
        "required": ["path"]
    },
    handler=read_file
)

register_tool(
    name="list_directory",
    description="列出指定目录的内容",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录路径，默认为当前目录",
                "default": "."
            }
        }
    },
    handler=list_directory
)

register_tool(
    name="execute_command",
    description="执行系统命令并返回输出",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的命令"
            }
        },
        "required": ["command"]
    },
    handler=execute_command
)

register_tool(
    name="get_current_time",
    description="获取当前时间",
    parameters={
        "type": "object",
        "properties": {}
    },
    handler=get_current_time
)

register_tool(
    name="calculate",
    description="计算数学表达式",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式"
            }
        },
        "required": ["expression"]
    },
    handler=calculate
)


# ========== 主程序 ==========

def main() -> int:
    """MCP 工具服务器主程序。

    从 stdin 读取 JSON-RPC 请求，处理后将响应写入 stdout。
    """
    print("MCP 工具服务器已启动，等待请求...", file=sys.stderr)
    print("可用工具：", file=sys.stderr)
    for name, info in TOOLS.items():
        print(f"  - {name}: {info['description']}", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response), flush=True)
        except json.JSONDecodeError:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": "Parse error"
                }
            }
            print(json.dumps(error_response), flush=True)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            print(json.dumps(error_response), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
