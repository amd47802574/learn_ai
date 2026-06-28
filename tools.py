#!/usr/bin/env python3
"""LangChain 工具定义。

使用 @tool 装饰器定义可被 LLM Agent 调用的工具。
包含基础系统工具和医学文献 RAG 检索工具。
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime

from langchain_core.tools import tool

from rag import search as rag_search, format_search_results, build_vector_store


@tool
def read_file(path: str) -> str:
    """Read file content at the given path.

    Args:
        path: File path to read.
    """
    if not os.path.exists(path):
        return f"Error: file not found - {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: failed to read file - {e}"


@tool
def list_directory(path: str = ".") -> str:
    """List contents of a directory.

    Args:
        path: Directory path, defaults to current directory.
    """
    if not os.path.exists(path):
        return f"Error: directory not found - {path}"
    try:
        items = os.listdir(path)
        result = []
        for item in items:
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                result.append(f"[DIR] {item}/")
            else:
                size = os.path.getsize(full_path)
                result.append(f"[FILE] {item} ({size} bytes)")
        return "\n".join(result) if result else "Empty directory"
    except Exception as e:
        return f"Error: failed to list directory - {e}"


@tool
def execute_command(command: str) -> str:
    """Execute a system command and return output.

    Args:
        command: Command to execute.
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
        return output if output else "Command completed, no output"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"
    except Exception as e:
        return f"Error: command failed - {e}"


@tool
def get_current_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression (basic arithmetic only: +-*/).

    Args:
        expression: Math expression to evaluate.
    """
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains disallowed characters"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: calculation failed - {e}"


# ---------- Medical Literature RAG Tools ----------

@tool
def medical_search(query: str, top_k: int = 5) -> str:
    """搜索医学文献知识库获取相关段落。

    当用户询问医学问题时使用此工具，如疾病诊断标准、
    治疗方案、用药指南或临床指南等。
    从本地医学文献库中检索最相关的内容。

    Args:
        query: 医学查询文本，例如"糖尿病诊断标准"。
        top_k: 返回的结果数量，默认为5。
    """
    try:
        results = rag_search(query, top_k=top_k)
        return format_search_results(results)
    except FileNotFoundError as e:
        return f"Error: literature library not found - {e}. Please put medical documents in medical_docs/ directory."
    except Exception as e:
        return f"Error: literature search failed - {e}"


@tool
def medical_index_rebuild(docs_dir: str = "") -> str:
    """重建医学文献向量索引。

    在添加或修改 medical_docs/ 目录中的文档后调用此工具，
    以使新文献可被搜索。

    Args:
        docs_dir: 文献目录路径。留空则使用默认的 medical_docs/。
    """
    try:
        target_dir = docs_dir if docs_dir else None
        kwargs = {"force_rebuild": True}
        if target_dir:
            kwargs["docs_dir"] = target_dir
        db = build_vector_store(**kwargs)
        count = db._collection.count()
        return f"Index rebuilt successfully! Indexed {count} document chunks."
    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: index rebuild failed - {e}"


# Agent 工具列表
ALL_TOOLS = [
    # 基础系统工具
    read_file,
    list_directory,
    execute_command,
    get_current_time,
    calculate,
    # 医学文献 RAG 工具
    medical_search,
    medical_index_rebuild,
]
