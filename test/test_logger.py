#!/usr/bin/env python3
"""测试美观的日志输出。"""

from logger import (
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    print_item,
    print_divider,
    rag_tag,
    llm_tag,
    tool_tag,
    colored,
    Color,
    Icon
)


def main():
    print()
    print(colored("=" * 60, Color.CYAN))
    print(colored("    美观日志系统演示", Color.BOLD + Color.CYAN))
    print(colored("=" * 60, Color.CYAN))

    print_section("基本功能测试", Icon.INFO)
    print_success("这是一条成功消息")
    print_error("这是一条错误消息")
    print_warning("这是一条警告消息")
    print_info("这是一条信息消息")

    print_section("键值对显示", Icon.ARROW)
    print_item("配置项1", "值1")
    print_item("配置项2", "值2")
    print_item("配置项3", "这是一个较长的值内容")

    print_section("模块标签", Icon.DOT)
    print(f"{rag_tag()} 这是 RAG 模块的日志")
    print(f"{llm_tag()} 这是 LLM 模块的日志")
    print(f"{tool_tag()} 这是工具模块的日志")

    print_divider()
    print(colored("演示完成！", Color.GREEN + Color.BOLD))
    print()


if __name__ == "__main__":
    main()

