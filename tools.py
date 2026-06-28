#!/usr/bin/env python3
"""LangChain 工具定义。

使用 @tool 装饰器定义可被 LLM Agent 调用的工具。
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """读取指定路径的文件内容。

    Args:
        path: 要读取的文件路径。
    """
    if not os.path.exists(path):
        return f"错误：文件不存在 - {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"错误：读取文件失败 - {e}"


@tool
def list_directory(path: str = ".") -> str:
    """列出指定目录的内容。

    Args:
        path: 要列出的目录路径，默认为当前目录。
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
        return f"错误：列出目录失败 - {e}"


@tool
def execute_command(command: str) -> str:
    """执行系统命令并返回输出。

    Args:
        command: 要执行的命令。
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output if output else "命令执行完成，无输出"
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（30秒）"
    except Exception as e:
        return f"错误：命令执行失败 - {e}"


@tool
def get_current_time() -> str:
    """获取当前时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 要计算的数学表达式，只支持基本运算（+-*/）。
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"错误：计算失败 - {e}"


# 所有工具的列表，供 Agent 使用
ALL_TOOLS = [read_file, list_directory, execute_command, get_current_time, calculate]
