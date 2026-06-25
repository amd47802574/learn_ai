#!/usr/bin/env python3
"""MCP 工具演示脚本。

展示如何与 MCP 工具服务器交互。
"""

import json
import subprocess
import sys
from typing import Any


def send_request(proc: subprocess.Popen, request: dict[str, Any]) -> dict[str, Any]:
    """向 MCP 服务器发送请求并获取响应。

    Args:
        proc: MCP 服务器进程。
        request: 请求字典。

    Returns:
        响应字典。
    """
    # 发送请求
    request_json = json.dumps(request) + "\n"
    proc.stdin.write(request_json.encode())
    proc.stdin.flush()

    # 读取响应
    response_line = proc.stdout.readline().decode().strip()
    return json.loads(response_line)


def main() -> int:
    """演示 MCP 工具的使用。"""
    print("启动 MCP 工具服务器...")
    print("-" * 50)

    # 启动 MCP 服务器
    proc = subprocess.Popen(
        [sys.executable, "mcp_tools.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )

    try:
        # 1. 初始化请求
        print("\n1. 发送初始化请求...")
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        response = send_request(proc, init_request)
        print(f"   响应: {json.dumps(response, indent=2, ensure_ascii=False)}")

        # 2. 获取工具列表
        print("\n2. 获取工具列表...")
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        response = send_request(proc, list_request)
        tools = response.get("result", {}).get("tools", [])
        print(f"   可用工具: {len(tools)} 个")
        for tool in tools:
            print(f"   - {tool['name']}: {tool['description']}")

        # 3. 调用工具示例
        print("\n3. 调用工具示例...")

        # 3.1 获取当前时间
        print("\n   3.1 获取当前时间:")
        time_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_current_time",
                "arguments": {}
            }
        }
        response = send_request(proc, time_request)
        print(f"   结果: {response.get('result', {}).get('content', [{}])[0].get('text', '')}")

        # 3.2 列出目录内容
        print("\n   3.2 列出当前目录:")
        list_dir_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "list_directory",
                "arguments": {"path": "."}
            }
        }
        response = send_request(proc, list_dir_request)
        print(f"   结果:\n{response.get('result', {}).get('content', [{}])[0].get('text', '')}")

        # 3.3 计算数学表达式
        print("\n   3.3 计算数学表达式:")
        calc_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "calculate",
                "arguments": {"expression": "2 + 3 * 4"}
            }
        }
        response = send_request(proc, calc_request)
        print(f"   结果: {response.get('result', {}).get('content', [{}])[0].get('text', '')}")

        # 3.4 读取文件
        print("\n   3.4 读取 README.md:")
        read_request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "read_file",
                "arguments": {"path": "README.md"}
            }
        }
        response = send_request(proc, read_request)
        content = response.get('result', {}).get('content', [{}])[0].get('text', '')
        # 只显示前 200 个字符
        print(f"   结果 (前 200 字符): {content[:200]}...")

        print("\n" + "-" * 50)
        print("演示完成！")

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        proc.wait()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
