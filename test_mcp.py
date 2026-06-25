#!/usr/bin/env python3
"""测试 MCP 工具集成。"""

import json
import subprocess
import sys


def test_mcp_tools():
    """测试 MCP 工具服务器是否正常工作。"""
    print("测试 MCP 工具服务器...")

    # 启动 MCP 服务器
    proc = subprocess.Popen(
        [sys.executable, "mcp_tools.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )

    try:
        # 等待服务器启动
        import time
        time.sleep(0.5)

        # 测试初始化
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }) + "\n"
        proc.stdin.write(request.encode())
        proc.stdin.flush()

        response = proc.stdout.readline().decode().strip()
        data = json.loads(response)
        print(f"✓ 初始化成功: {data.get('result', {}).get('serverInfo', {}).get('name', 'unknown')}")

        # 测试工具列表
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }) + "\n"
        proc.stdin.write(request.encode())
        proc.stdin.flush()

        response = proc.stdout.readline().decode().strip()
        data = json.loads(response)
        tools = data.get("result", {}).get("tools", [])
        print(f"✓ 工具列表获取成功: {len(tools)} 个工具")
        for tool in tools:
            print(f"  - {tool['name']}")

        # 测试调用工具
        request = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_current_time",
                "arguments": {}
            }
        }) + "\n"
        proc.stdin.write(request.encode())
        proc.stdin.flush()

        response = proc.stdout.readline().decode().strip()
        data = json.loads(response)
        result = data.get("result", {}).get("content", [{}])[0].get("text", "")
        print(f"✓ 工具调用成功: {result}")

        print("\n所有测试通过！")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    finally:
        proc.terminate()
        proc.wait()


def test_llm_cli_help():
    """测试 LLM CLI 帮助信息。"""
    print("\n测试 LLM CLI 帮助信息...")
    try:
        result = subprocess.run(
            [sys.executable, "llm_cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✓ LLM CLI 帮助信息正常")
            # 检查是否包含工具相关选项
            if "--no-tools" in result.stdout:
                print("✓ 包含 --no-tools 选项")
            if "--mcp-server" in result.stdout:
                print("✓ 包含 --mcp-server 选项")
            return True
        else:
            print(f"✗ LLM CLI 帮助信息异常: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("MCP 工具集成测试")
    print("=" * 50)

    success = True
    success &= test_mcp_tools()
    success &= test_llm_cli_help()

    print("\n" + "=" * 50)
    if success:
        print("所有测试通过！MCP 工具集成成功。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 50)

    sys.exit(0 if success else 1)
