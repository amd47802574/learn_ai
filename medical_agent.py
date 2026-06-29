#!/usr/bin/env python3
"""医学诊断辅助 Agent。

基于 LangChain ReAct Agent + RAG 医学文献检索，提供：
- 根据症状描述检索文献并给出可能的诊断方向
- 追问用户更多细节（症状持续时间、既往病史等）
- 提供相关检查建议和注意事项
"""

from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from tools import medical_search, medical_index_rebuild, get_current_time
from logger import (
    llm_tag,
    tool_tag,
    colored,
    Color,
    Icon,
    print_info,
    print_success,
    print_error,
    print_item,
    print_divider,
)


# ---------- 系统提示词 ----------

MEDICAL_SYSTEM_PROMPT = """你是一位专业的医学诊断辅助 AI 助手。你的职责是根据用户描述的症状，
结合医学文献知识库中的权威信息，提供专业的医学参考建议。

## 核心工作流程

1. **症状收集**：仔细倾听用户描述的症状，必要时主动追问以下关键信息：
   - 症状的具体表现和部位
   - 发病时间和持续时长
   - 症状的严重程度和变化趋势
   - 是否有诱因或加重/缓解因素
   - 既往病史、家族史
   - 当前用药情况
   - 年龄、性别等基本信息

2. **文献检索**：使用 medical_search 工具检索相关医学文献，获取权威的诊断标准和治疗方案。
   你应该针对用户的症状，构造精准的检索查询词。例如：
   - 用户说"最近总是口渴、尿多" -> 检索"多饮多尿 糖尿病诊断"
   - 用户说"头晕、头痛" -> 检索"头晕头痛 高血压诊断标准"
   你可以多次检索不同关键词，以获取更全面的信息。

3. **诊断分析**：基于收集到的症状和文献信息，提供结构化的分析：
   - 可能的诊断方向（按可能性排序）
   - 每个诊断方向的依据
   - 需要排除的其他疾病

4. **检查建议**：推荐相关的检查项目，说明每项检查的目的。

5. **注意事项**：提供日常注意事项和就医建议。

## 重要原则

- **安全第一**：始终提醒用户本系统仅供参考，不能替代专业医生的面诊和诊断。
- **追问细节**：信息不充分时，主动追问关键细节，不要在信息不足时草率给出诊断方向。
- **循证医学**：回答必须基于检索到的医学文献，不凭空编造医学信息。
- **通俗易懂**：用患者能理解的语言解释医学概念，必要时附上专业术语。
- **紧急识别**：如果症状提示可能的急症（如胸痛、突发剧烈头痛、呼吸困难等），
  立即提醒用户尽快就医，不要延误。

## 回复格式建议

在充分收集信息并检索文献后，使用以下格式组织回复：

---
**可能的诊断方向：**
1. [最可能的诊断] - 依据说明
2. [次可能的诊断] - 依据说明

**建议检查项目：**
- [检查项目1]：目的说明
- [检查项目2]：目的说明

**日常注意事项：**
- 具体建议

**就医建议：**
- 建议就诊科室和时机

> 免责声明：以上分析仅基于您描述的症状和医学文献，仅供参考。
> 具体诊断和治疗请务必咨询专业医生。
---
"""


class MedicalStreamingHandler(BaseCallbackHandler):
    """医学 Agent 专用的流式输出回调。"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        tool_name = serialized.get("name", "unknown")
        # 截断过长的输入显示
        display_input = input_str[:80] + "..." if len(input_str) > 80 else input_str
        print(
            colored(
                f"\n{tool_tag()} {Icon.LOADING} 正在检索: {tool_name}({display_input})",
                Color.YELLOW,
            ),
            file=sys.stderr,
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        # 只显示简要预览
        lines = output.strip().split("\n")
        preview = lines[0][:100] if lines else "(empty)"
        print(
            colored(f"{tool_tag()} {Icon.SUCCESS} 检索完成: {preview}", Color.GREEN),
            file=sys.stderr,
        )


def _env(name: str, default: str | None = None) -> str | None:
    """从环境变量读取值。"""
    value = os.environ.get(name)
    return value if value else default


def create_medical_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    streaming: bool = True,
) -> Any:
    """创建医学诊断辅助 Agent。

    Args:
        api_key: LLM API Key，默认从环境变量读取。
        base_url: LLM API Base URL。
        model: 模型名称。
        temperature: 温度参数，医学场景建议较低值。
        streaming: 是否启用流式输出。

    Returns:
        可调用的 Agent 实例。
    """
    api_key = api_key or _env("OPENAI_API_KEY")
    base_url = base_url or _env("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    model = model or _env("OPENAI_MODEL", "deepseek-v4-flash")

    if not api_key:
        raise ValueError("缺少 API Key，请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数。")

    callbacks = [MedicalStreamingHandler()] if streaming else []

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        streaming=streaming,
        callbacks=callbacks,
    )

    # 医学 Agent 只需要医学相关工具
    medical_tools = [medical_search, medical_index_rebuild, get_current_time]

    agent = create_react_agent(
        model=llm,
        tools=medical_tools,
        prompt=MEDICAL_SYSTEM_PROMPT,
    )

    return agent


def extract_reply(result: dict[str, Any]) -> str:
    """从 Agent 返回结果中提取最终文本回复。"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def run_diagnosis_session(
    agent: Any,
    streaming: bool = True,
) -> None:
    """运行交互式诊断会话。

    Args:
        agent: 医学诊断 Agent 实例。
        streaming: 是否启用流式输出。
    """
    messages: list = []

    # 打印欢迎界面
    print()
    print(colored("+" + "-" * 58 + "+", Color.CYAN))
    print(colored("|" + " " * 10 + "Medical Diagnosis Assistant" + " " * 21 + "|", Color.CYAN))
    print(colored("+" + "-" * 58 + "+", Color.CYAN))
    print()
    print(colored("  " + Icon.INFO + " ", Color.CYAN) + "我是医学诊断辅助助手，可以根据您的症状描述，")
    print("    结合医学文献为您提供初步的诊断方向和建议。")
    print()
    print(colored("  " + Icon.WARNING + " ", Color.YELLOW) + "本系统仅供参考，不能替代专业医生的诊断。")
    print()
    print(colored("  命令：/exit 退出  /clear 清除对话  /help 帮助", Color.DIM))
    print_divider()

    while True:
        try:
            user_text = input(colored("\n[患者] ", Color.GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _print_goodbye()
            return

        if not user_text:
            continue

        # 处理命令
        if user_text in {"/exit", "/quit"}:
            _print_goodbye()
            return
        if user_text == "/clear":
            messages = []
            print_success("对话已清除，可以开始新的咨询。")
            continue
        if user_text == "/help":
            _print_help()
            continue

        messages.append(HumanMessage(content=user_text))

        try:
            print(colored("\n[医学助手] ", Color.CYAN), end="", flush=True)
            result = agent.invoke({"messages": messages})
            messages = result.get("messages", messages)
            reply = extract_reply(result)
            if reply and not streaming:
                print(reply)
            elif reply:
                print()  # 流式输出后换行
        except Exception as error:
            messages.pop()
            print_error(f"分析出错: {error}")
            print_info("请重试或换一种方式描述您的症状。")


def _print_help() -> None:
    """打印帮助信息。"""
    print()
    print_divider()
    print(colored("  使用说明：", Color.BOLD))
    print()
    print("  1. 描述您的症状，例如：")
    print(colored("     \"最近一周经常头晕，早上起床时特别明显\"", Color.DIM))
    print(colored("     \"我爸今年60岁，最近总说口渴，尿也变多了\"", Color.DIM))
    print()
    print("  2. 助手会根据需要追问更多细节")
    print("  3. 助手检索医学文献后给出分析建议")
    print()
    print("  命令：")
    print(f"    /exit   - 退出程序")
    print(f"    /clear  - 清除对话历史，开始新咨询")
    print(f"    /help   - 显示此帮助")
    print_divider()


def _print_goodbye() -> None:
    """打印告别信息。"""
    print()
    print_divider()
    print(colored(f"  {Icon.SUCCESS} 感谢使用医学诊断辅助系统。祝您健康！", Color.CYAN))
    print_divider()
    print()
