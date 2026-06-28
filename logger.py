#!/usr/bin/env python3
"""统一美观的日志输出工具。"""

import sys
import os
from typing import Any


class Color:
    """ANSI 颜色代码"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


class Icon:
    """图标符号"""
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    LOADING = "⏳"
    ARROW = "→"
    STAR = "★"
    DOT = "•"


def supports_color() -> bool:
    """检查终端是否支持颜色"""
    plat = sys.platform
    supported_platform = plat != 'Pocket PC' and (plat != 'win32' or 'ANSICON' in os.environ)
    is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    return supported_platform and is_a_tty


USE_COLOR = supports_color()


def colored(text: str, color: str) -> str:
    """给文本添加颜色"""
    if USE_COLOR:
        return f"{color}{text}{Color.RESET}"
    return text


def print_section(title: str, icon: str = Icon.INFO) -> None:
    """打印章节标题"""
    print()
    print(colored(f"{icon} {title}", Color.BOLD + Color.BLUE))
    print(colored("─" * 50, Color.DIM))


def print_success(message: str, icon: str = Icon.SUCCESS) -> None:
    """打印成功消息"""
    print(f"{icon} {message}")


def print_error(message: str, icon: str = Icon.ERROR) -> None:
    """打印错误消息"""
    print(colored(f"{icon} {message}", Color.RED))


def print_warning(message: str, icon: str = Icon.WARNING) -> None:
    """打印警告消息"""
    print(colored(f"{icon} {message}", Color.YELLOW))


def print_info(message: str, icon: str = Icon.INFO) -> None:
    """打印信息消息"""
    print(colored(f"{icon} {message}", Color.CYAN))


def print_item(key: str, value: Any, indent: int = 2) -> None:
    """打印键值对项"""
    indent_str = " " * indent
    print(f"{indent_str}{colored(key + ':', Color.DIM)} {value}")


def print_divider(char: str = "─", width: int = 50) -> None:
    """打印分隔线"""
    print(colored(char * width, Color.DIM))


def tag(tag_name: str, color: str = Color.BLUE) -> str:
    """创建标签"""
    return colored(f"[{tag_name}]", color)


def rag_tag() -> str:
    """RAG 模块标签"""
    return tag("RAG", Color.MAGENTA)


def llm_tag() -> str:
    """LLM 模块标签"""
    return tag("LLM", Color.CYAN)


def tool_tag() -> str:
    """工具模块标签"""
    return tag("TOOL", Color.YELLOW)

