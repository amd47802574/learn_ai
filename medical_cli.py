#!/usr/bin/env python3
"""医学诊断辅助 Agent - 命令行入口。

用法：
    python medical_cli.py                    # 进入交互式诊断会话
    python medical_cli.py "我最近总是头晕"     # 单次诊断咨询
    python medical_cli.py --no-stream        # 禁用流式输出
    python medical_cli.py --model gpt-4o     # 指定模型
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from medical_agent import create_medical_agent, run_diagnosis_session, extract_reply
from langchain_core.messages import HumanMessage
from logger import print_error, print_info, colored, Color


def _env(name: str, default: str | None = None) -> str | None:
    """读取环境变量。"""
    value = os.environ.get(name)
    return value if value else default


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="医学诊断助手 - AI 驱动的症状分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          进入交互式诊断会话
  %(prog)s "胸口疼了两天"             单次症状分析
  %(prog)s --temperature 0.1        使用较低的温度以获得更精确的回答
        """,
    )
    parser.add_argument(
        "symptoms",
        nargs="*",
        help="用于单次分析的症状描述。省略则进入交互模式。",
    )
    parser.add_argument("--api-key", default=_env("OPENAI_API_KEY"))
    parser.add_argument(
        "--base-url", default=_env("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    )
    parser.add_argument(
        "--model", default=_env("OPENAI_MODEL", "deepseek-v4-flash")
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="温度 (默认: 0.3, 越低越精确)。",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="禁用流式输出。",
    )
    parser.set_defaults(stream=True)
    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "缺少 API 密钥。请设置 OPENAI_API_KEY 环境变量或通过 --api-key 参数传递。"
        )
    return args


def main() -> int:
    """程序入口点。"""
    args = parse_args()
    symptoms_text = " ".join(args.symptoms).strip()

    try:
        agent = create_medical_agent(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            streaming=args.stream,
        )
    except Exception as e:
        print_error(f"创建 Agent 失败: {e}")
        return 1

    try:
        if symptoms_text:
            # 单次诊断模式
            print(colored("\n[医学助手] ", Color.CYAN), end="", flush=True)
            result = agent.invoke(
                {"messages": [HumanMessage(content=symptoms_text)]}
            )
            reply = extract_reply(result)
            if reply and not args.stream:
                print(reply)
            elif reply:
                print()
        else:
            # 交互式模式
            run_diagnosis_session(agent, streaming=args.stream)
    except KeyboardInterrupt:
        print()
        return 0
    except RuntimeError as error:
        print_error(f"运行时错误: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
