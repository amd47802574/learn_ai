#!/usr/bin/env python3
"""测试 LangChain 工具定义。"""

import sys


def test_tools_import():
    """测试工具模块是否能正常导入。"""
    print("测试工具模块导入...")
    try:
        from tools import ALL_TOOLS, read_file, list_directory, get_current_time, calculate, execute_command
        print(f"✓ 成功导入 {len(ALL_TOOLS)} 个工具")
        for tool in ALL_TOOLS:
            desc = tool.description.split('\n')[0]
            print(f"  - {tool.name}: {desc}")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("  请先安装依赖: pip install -r requirements.txt")
        return False


def test_tool_execution():
    """测试工具是否能正常执行。"""
    print("\n测试工具执行...")
    from tools import get_current_time, calculate, list_directory, read_file

    # 测试获取时间
    result = get_current_time.invoke({})
    print(f"✓ get_current_time: {result}")

    # 测试计算
    result = calculate.invoke({"expression": "2 + 3 * 4"})
    print(f"✓ calculate: {result}")

    # 测试列出目录
    result = list_directory.invoke({"path": "."})
    lines = result.split("\n")
    print(f"✓ list_directory: {len(lines)} 个条目")

    # 测试读取文件
    result = read_file.invoke({"path": "README.md"})
    print(f"✓ read_file: {len(result)} 字符")

    return True


def test_cli_import():
    """测试主程序模块是否能正常导入。"""
    print("\n测试主程序模块导入...")
    try:
        from llm_cli import build_agent, extract_reply, parse_args
        print("✓ 主程序模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_cli_help():
    """测试 CLI 帮助信息。"""
    print("\n测试 CLI 帮助信息...")
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, "llm_cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print("✓ CLI 帮助信息正常")
            if "--no-tools" in result.stdout:
                print("✓ 包含 --no-tools 选项")
            if "--no-stream" in result.stdout:
                print("✓ 包含 --no-stream 选项")
            return True
        else:
            print(f"✗ CLI 帮助信息异常: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("LangChain 工具集成测试")
    print("=" * 50)

    success = True
    success &= test_tools_import()
    success &= test_tool_execution()
    success &= test_cli_import()
    success &= test_cli_help()

    print("\n" + "=" * 50)
    if success:
        print("所有测试通过！LangChain 工具集成成功。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 50)

    sys.exit(0 if success else 1)
