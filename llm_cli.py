#!/usr/bin/env python3
"""基于 LangChain 的 LLM 命令行客户端，支持工具调用。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import ALL_TOOLS

# 加载 .env 文件中的环境变量
load_dotenv()

class StreamingHandler(BaseCallbackHandler):
    """流式输出回调，逐 token 打印 LLM 生成内容。"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        print(f"\n[调用工具: {tool_name}({input_str})]", file=sys.stderr)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        preview = output[:200] + "..." if len(output) > 200 else output
        print(f"[工具结果: {preview}]", file=sys.stderr)


def env(name: str, default: str | None = None) -> str | None:
    """从环境变量读取值。"""
    value = os.environ.get(name)
    return value if value else default


def build_agent(args: argparse.Namespace) -> Any:
    """构建 LangChain ReAct Agent。

    Args:
        args: 命令行参数。

    Returns:
        可调用的 Agent 实例。
    """
    callbacks = [StreamingHandler()] if args.stream else []

    llm = ChatOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        streaming=args.stream,
        callbacks=callbacks,
    )

    tools = ALL_TOOLS if args.use_tools else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=args.system,
    )

    return agent


def extract_reply(result: dict[str, Any]) -> str:
    """从 Agent 返回结果中提取最终文本回复。

    Args:
        result: Agent invoke 返回的字典。

    Returns:
        最终回复文本。
    """
    messages = result.get("messages", [])
    # 从后往前找到最后一条 AI 消息
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def ask_once(agent: Any, system_prompt: str, user_text: str) -> str:
    """发送一次性提示并返回模型回复。

    Args:
        agent: LangChain Agent 实例。
        system_prompt: 系统提示。
        user_text: 用户输入。

    Returns:
        模型回复文本。
    """
    result = agent.invoke({
        "messages": [HumanMessage(content=user_text)],
    })
    return extract_reply(result)


def repl(agent: Any, system_prompt: str) -> None:
    """交互式 REPL：持续接收用户输入并与 Agent 对话。

    支持命令：/exit 或 /quit 退出，/clear 清除上下文。

    Args:
        agent: LangChain Agent 实例。
        system_prompt: 系统提示。
    """
    messages: list = []

    print("LLM CLI (LangChain) is ready. Type /exit to quit, /clear to reset context.")

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
            messages = []
            print("Context cleared.")
            continue

        messages.append(HumanMessage(content=user_text))

        try:
            print("\nAI> ", end="", flush=True)
            result = agent.invoke({"messages": messages})
            # 更新消息历史为 agent 返回的完整消息列表
            messages = result.get("messages", messages)
            reply = extract_reply(result)
            # 非流式模式下手动打印回复（流式模式已通过回调打印）
            if reply:
                print()  # 换行
        except Exception as error:
            messages.pop()  # 移除失败的用户消息
            print(f"\nError: {error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基于 LangChain 的 LLM CLI，支持工具调用。"
    )
    parser.add_argument("prompt", nargs="*", help="提示文本，省略则进入交互模式。")
    parser.add_argument("--api-key", default=env("OPENAI_API_KEY"))
    parser.add_argument(
        "--base-url", default=env("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    )
    parser.add_argument("--model", default=env("OPENAI_MODEL", "deepseek-v4-flash"))
    parser.add_argument(
        "--system", default=env("LLM_SYSTEM_PROMPT", "You are a helpful assistant with access to tools. Use them when needed.")
    )
    parser.add_argument(
        "--temperature", type=float, default=float(env("LLM_TEMPERATURE", "0.7"))
    )
    parser.add_argument(
        "--no-stream", dest="stream", action="store_false", help="禁用流式输出。"
    )
    parser.add_argument(
        "--no-tools", dest="use_tools", action="store_false", help="禁用工具。"
    )
    parser.set_defaults(stream=True, use_tools=True)
    args = parser.parse_args()

    if not args.api_key:
        parser.error("缺少 API key。请设置 OPENAI_API_KEY 环境变量或传入 --api-key。")
    return args


def main() -> int:
    """程序入口。"""
    args = parse_args()
    prompt = " ".join(args.prompt).strip()

    try:
        agent = build_agent(args)
    except Exception as e:
        print(f"Error: 创建 Agent 失败 - {e}", file=sys.stderr)
        return 1

    try:
        if prompt:
            reply = ask_once(agent, args.system, prompt)
            if not args.stream and reply:
                print(reply)
        else:
            repl(agent, args.system)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
