"""
Agent 设置模块
创建 Research Copilot Agent，包含 RAG 检索工具
"""
import sqlite3
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from config import model
from prompts import get_system_prompt
from rag_store import get_rag

# 当前会话的 thread_id（工具通过此变量获取上下文）
_current_thread_id = [""]


@tool
def retrieve_paper(query: str) -> str:
    """
    从当前已上传的论文中检索与问题相关的内容。
    当用户提问涉及论文具体内容时，必须调用此工具。
    """
    tid = _current_thread_id[0]
    if not tid:
        return "当前没有活跃的论文会话。"
    rag = get_rag()
    results = rag.search(tid, query)
    if not results:
        return "论文库中未找到相关内容。"
    parts = []
    for i, r in enumerate(results, 1):
        section = r["section"].title()
        content = r["content"][:600]
        page = f" | 第 {r.get('page_start', 0)}-{r.get('page_end', 0)} 页" if r.get("page_end", 0) else ""
        parts.append(f"【证据 #{i} | 来源: {section}{page}】\n{content}")
    return "\n\n---\n\n".join(parts)


def set_current_thread(tid: str):
    _current_thread_id[0] = tid


def create_research_agent():
    connection = sqlite3.connect("resources/research_copilot.db", check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return create_agent(
        model,
        tools=[retrieve_paper],
        system_prompt=get_system_prompt(),
        checkpointer=checkpointer,
    )
