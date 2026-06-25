#!/usr/bin/env python3
"""带 MCP 工具支持的 OpenAI 兼容 LLM 命令行客户端。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant with access to tools. Use them when needed."
# 默认的系统提示，用于在对话开始时设定助手的行为与角色


class MCPClient:
    """MCP 工具客户端，用于与 MCP 工具服务器通信。"""

    def __init__(self, server_script: str = "mcp_tools.py"):
        """初始化 MCP 客户端。

        Args:
            server_script: MCP 服务器脚本路径。
        """
        self.server_script = server_script
        self.process: subprocess.Popen | None = None
        self.request_id = 0
        self.tools: list[dict[str, Any]] = []

    def start(self) -> bool:
        """启动 MCP 服务器。

        Returns:
            是否成功启动。
        """
        try:
            self.process = subprocess.Popen(
                [sys.executable, self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False
            )
            # 等待服务器启动
            import time
            time.sleep(0.5)

            # 初始化服务器
            response = self._send_request({
                "method": "initialize",
                "params": {}
            })

            if "error" in response:
                print(f"MCP 服务器初始化失败: {response['error']}", file=sys.stderr)
                return False

            # 获取工具列表
            response = self._send_request({
                "method": "tools/list",
                "params": {}
            })

            if "result" in response:
                self.tools = response["result"].get("tools", [])
                print(f"已连接 MCP 服务器，可用工具: {len(self.tools)} 个", file=sys.stderr)
                for tool in self.tools:
                    print(f"  - {tool['name']}: {tool['description']}", file=sys.stderr)

            return True
        except Exception as e:
            print(f"启动 MCP 服务器失败: {e}", file=sys.stderr)
            return False

    def stop(self):
        """停止 MCP 服务器。"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def _send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """发送请求到 MCP 服务器。

        Args:
            request: 请求字典（不含 jsonrpc 和 id）。

        Returns:
            响应字典。
        """
        self.request_id += 1
        full_request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            **request
        }

        # 发送请求
        request_json = json.dumps(full_request) + "\n"
        self.process.stdin.write(request_json.encode())
        self.process.stdin.flush()

        # 读取响应
        response_line = self.process.stdout.readline().decode().strip()
        return json.loads(response_line)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具。

        Args:
            name: 工具名称。
            arguments: 工具参数。

        Returns:
            工具执行结果。
        """
        response = self._send_request({
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        })

        if "error" in response:
            return f"工具调用错误: {response['error']['message']}"

        # 提取文本内容
        content = response.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            return content[0].get("text", "")

        return str(response.get("result", {}))

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """获取工具的 OpenAI 格式 schema。

        Returns:
            工具 schema 列表。
        """
        tools = []
        for tool in self.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("parameters", {})
                }
            })
        return tools


def env(name: str, default: str | None = None) -> str | None:
    """从环境变量读取值，如果未设置则返回默认值。

    Args:
        name: 环境变量名称。
        default: 环境变量未设置时的默认值。

    Returns:
        环境变量的字符串值，或默认值（可能为 None）。
    """
    value = os.environ.get(name)
    return value if value else default


def build_url(base_url: str) -> str:
    """构造聊天完成的 API 端点 URL。

    会移除末尾多余的斜杠并追加固定路径。
    """
    return base_url.rstrip("/") + "/chat/completions"


def read_error(error: urllib.error.HTTPError) -> str:
    """从 HTTPError 中解析出友好的错误信息。

    优先尝试从响应体解析 JSON 中的 error.message 字段，解析失败则返回原始响应体或错误字符串。
    """
    body = error.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
        message = data.get("error", {}).get("message")
        if message:
            return message
    except json.JSONDecodeError:
        pass
    return body or str(error)


def call_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    stream: bool,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """调用 LLM API 并返回生成的文本或工具调用。

    参数说明与网络细节：
    - 使用 POST 将消息和参数发送到 `base_url` 的 `/chat/completions` 路径。
    - 支持 `stream` 模式：若为 True，逐片段打印流式响应并返回合并后的字符串。
    - 支持工具调用：若提供 tools 参数，LLM 可能返回工具调用请求。

    返回值：包含 'content' 和/或 'tool_calls' 的字典。
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    # 添加工具定义
    if tools:
        payload["tools"] = tools

    request = urllib.request.Request(
        build_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if stream:
                # 流式输出会边打印边累积片段
                return read_stream(response)
            data = json.loads(response.read().decode("utf-8"))
            message = data["choices"][0]["message"]
            return message
    except urllib.error.HTTPError as error:
        raise RuntimeError(read_error(error)) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error: {error.reason}") from error
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unexpected API response: {error}") from error


def read_stream(response) -> dict[str, Any]:
    """读取并处理流式响应，打印输出并返回完整消息。

    该函数期望每行以 `data:` 开头并包含 JSON 事件，遇到 `[DONE]` 则结束。
    """
    content_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue

        event = line.removeprefix("data:").strip()
        if event == "[DONE]":
            break

        try:
            data = json.loads(event)
            delta = data["choices"][0].get("delta", {})

            # 处理文本内容
            content = delta.get("content", "")
            if content:
                print(content, end="", flush=True)
                content_chunks.append(content)

            # 处理工具调用（流式）
            if "tool_calls" in delta:
                for tc_delta in delta["tool_calls"]:
                    tc_index = tc_delta.get("index", 0)
                    # 确保 tool_calls 列表足够长
                    while len(tool_calls) <= tc_index:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""}
                        })
                    # 累积工具调用信息
                    if "id" in tc_delta:
                        tool_calls[tc_index]["id"] = tc_delta["id"]
                    if "function" in tc_delta:
                        if "name" in tc_delta["function"]:
                            tool_calls[tc_index]["function"]["name"] += tc_delta["function"]["name"]
                        if "arguments" in tc_delta["function"]:
                            tool_calls[tc_index]["function"]["arguments"] += tc_delta["function"]["arguments"]

        except (KeyError, IndexError, json.JSONDecodeError):
            continue

    print()
    return {
        "content": "".join(content_chunks) if content_chunks else None,
        "tool_calls": tool_calls if tool_calls else None
    }


def ask_once(args: argparse.Namespace, user_text: str, mcp_client: MCPClient | None = None) -> str:
    """发送一次性提示并返回模型回复（非交互式）。

    会构造 system + user 两条消息并调用 `call_llm`。
    """
    messages = [{"role": "system", "content": args.system}]
    messages.append({"role": "user", "content": user_text})

    # 获取工具定义
    tools = mcp_client.get_tools_schema() if mcp_client else None

    # 调用 LLM
    result = call_llm(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        stream=args.stream,
        tools=tools,
    )

    # 处理工具调用
    if mcp_client and result.get("tool_calls"):
        return handle_tool_calls(args, messages, result, mcp_client)

    return result.get("content", "")


def handle_tool_calls(
    args: argparse.Namespace,
    messages: list[dict[str, str]],
    result: dict[str, Any],
    mcp_client: MCPClient
) -> str:
    """处理工具调用循环。

    Args:
        args: 命令行参数。
        messages: 消息历史。
        result: LLM 返回的结果。
        mcp_client: MCP 客户端实例。

    Returns:
        最终的文本回复。
    """
    tools = mcp_client.get_tools_schema()
    final_content = ""

    while result.get("tool_calls"):
        # 添加助手消息（包含工具调用）
        assistant_message = {"role": "assistant", "content": result.get("content")}
        if result.get("tool_calls"):
            assistant_message["tool_calls"] = result["tool_calls"]
        messages.append(assistant_message)

        # 执行每个工具调用
        for tool_call in result["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            try:
                arguments = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                arguments = {}

            print(f"\n[调用工具: {tool_name}({arguments})]", file=sys.stderr)

            # 调用 MCP 工具
            tool_result = mcp_client.call_tool(tool_name, arguments)
            print(f"[工具结果: {tool_result[:200]}...]" if len(tool_result) > 200 else f"[工具结果: {tool_result}]", file=sys.stderr)

            # 添加工具结果消息
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })

        # 再次调用 LLM 获取回复
        print("\nAI> ", end="", flush=True)
        result = call_llm(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            stream=args.stream,
            tools=tools,
        )

        if result.get("content"):
            final_content = result["content"]

    return final_content


def repl(args: argparse.Namespace, mcp_client: MCPClient | None = None) -> None:
    """交互式 REPL：持续接收用户输入并与模型对话，支持简单命令。

    支持命令：`/exit` 或 `/quit` 退出，`/clear` 清除上下文（只保留 system）。
    """
    messages = [{"role": "system", "content": args.system}]
    tools = mcp_client.get_tools_schema() if mcp_client else None

    print("LLM CLI is ready. Type /exit to quit, /clear to reset context.")
    if mcp_client:
        print(f"MCP 工具已启用，可用工具: {len(mcp_client.tools)} 个")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        if user_text == "/clear":
            # 重置对话上下文，只保留 system 提示
            messages = [{"role": "system", "content": args.system}]
            print("Context cleared.")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            print("\nAI> ", end="", flush=True)
            result = call_llm(
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                stream=args.stream,
                tools=tools,
            )

            # 处理工具调用
            if mcp_client and result.get("tool_calls"):
                handle_tool_calls(args, messages, result, mcp_client)
            else:
                if not args.stream:
                    print(result.get("content", ""))
                # 添加助手消息到历史
                messages.append({
                    "role": "assistant",
                    "content": result.get("content", "")
                })

        except RuntimeError as error:
            # 当调用失败时，移除刚加入的用户消息以保持上下文一致
            messages.pop()
            print(f"\nError: {error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """解析命令行参数并进行简单校验。

    返回解析后的 `argparse.Namespace`，并在缺少必需的 API key 时报错。
    """
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible LLM from CLI with MCP tools support.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. Omit it to start chat mode.")
    parser.add_argument("--api-key", default=env("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=env("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=env("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--system", default=env("LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--temperature", type=float, default=float(env("LLM_TEMPERATURE", "0.7")))
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming output.")
    parser.add_argument("--no-tools", dest="use_tools", action="store_false", help="Disable MCP tools.")
    parser.add_argument("--mcp-server", default="mcp_tools.py", help="MCP server script path.")
    parser.set_defaults(stream=True, use_tools=True)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
    return args


def main() -> int:
    """程序入口：根据是否提供 prompt 运行单次请求或进入 REPL。"""
    args = parse_args()
    prompt = " ".join(args.prompt).strip()

    # 启动 MCP 客户端（如果启用）
    mcp_client = None
    if args.use_tools:
        mcp_client = MCPClient(args.mcp_server)
        if not mcp_client.start():
            print("警告: MCP 工具不可用，继续运行...", file=sys.stderr)
            mcp_client = None

    try:
        if prompt:
            # 单次调用模式
            answer = ask_once(args, prompt, mcp_client)
            if not args.stream:
                print(answer)
        else:
            # 交互式模式
            repl(args, mcp_client)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        # 停止 MCP 服务器
        if mcp_client:
            mcp_client.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
