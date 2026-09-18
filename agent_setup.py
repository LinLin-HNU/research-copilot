"""
Agent 设置模块
创建 Research Copilot Agent。

架构说明（token 治理）：
- 论文证据由后端 Python 预检索（app.build_evidence），通过 LangGraph runtime
  context（RequestContext）传入，再由 @dynamic_prompt 中间件拼进「系统提示」。
- 证据只在当轮发给模型，**不写进 messages，因此不会进 SqliteSaver 历史**，
  从根上避免大段检索原文跨轮重发（token 线性膨胀）以及换论文后旧证据残留。
- Agent 自身不注册任何检索工具：检索是确定性的，后端做完即可，模型只负责
  基于当轮证据综合与表达。
"""
import sqlite3
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langgraph.checkpoint.sqlite import SqliteSaver

from config import model
from prompts import get_system_prompt


@dataclass
class RequestContext:
    """单次请求级数据：仅本轮使用，永不写入 checkpoint 消息历史。"""
    evidence: str = ""
    guidance: str = ""


@dynamic_prompt
def request_prompt(request) -> str:
    """每轮动态系统提示 = 基础人设/铁律 + 本轮要求 + 本轮论文证据。

    证据放这里而不是用户消息里，因此不会被持久化、不会跨轮重发。
    """
    ctx = getattr(request.runtime, "context", None)

    def _field(name: str) -> str:
        if ctx is None:
            return ""
        if isinstance(ctx, dict):
            return ctx.get(name, "") or ""
        return getattr(ctx, name, "") or ""

    prompt = get_system_prompt()
    guidance = _field("guidance")
    evidence = _field("evidence")
    if guidance:
        prompt += f"\n\n【本轮回答要求】\n{guidance}"
    if evidence:
        prompt += (
            "\n\n【本轮论文证据】以下证据仅供本轮使用，不会保留到后续对话，"
            "请只基于这些证据陈述论文事实，并按证据中标注的来源章节/页码引用：\n\n"
            f"{evidence}"
        )
    return prompt


def create_research_agent():
    # 相对路径：需在 Research_Copilot 目录下运行
    connection = sqlite3.connect("resources/research_copilot.db", check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return create_agent(
        model,
        tools=[],
        middleware=[request_prompt],
        context_schema=RequestContext,
        checkpointer=checkpointer,
    )
