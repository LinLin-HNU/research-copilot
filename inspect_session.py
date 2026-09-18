"""
只读「黑盒成果观察器」：查看你自己刚上传/对话的真实会话，不写入、不删除任何数据。

能看三件事：
1. 每个会话对应向量集合的 chunk 数（覆盖上传后核对数量/关键词）。
2. 会话「最新状态」里的消息构成：Human / AI / ToolMessage 各几条、各多少字
   （验证 ToolMessage 是否还在；以及证据是否藏在巨型 HumanMessage 里跨轮保留）。
3. 集合正文里是否还残留某个关键词（验证旧论文是否被新论文物理覆盖）。

用法（在 Research_Copilot 目录下）：
  python inspect_session.py                       # 总览：所有会话
  python inspect_session.py 3c2d6b6d              # 看某个会话详情（thread_id 前缀即可）
  python inspect_session.py 3c2d6b6d --grep zebra # 该会话向量库里是否含关键词（覆盖验证）

注意：本脚本只调用 list/get/count 等只读接口，绝不 store/delete/update_state。
"""
import io
import sys
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import chromadb
from chromadb.config import Settings

from database import list_sessions
from rag_store import get_rag
from agent_setup import create_research_agent
from langchain.messages import HumanMessage, AIMessage, ToolMessage

# 复用真实 agent，仅用于 get_state 只读历史；不会触发任何模型调用
agent = create_research_agent()
rag = get_rag()
chroma_client = chromadb.PersistentClient(
    path="resources/chroma_db", settings=Settings(anonymized_telemetry=False)
)


def text_len(msg) -> int:
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return len(c)
    try:
        return len(str(c))
    except Exception:
        return -1


def latest_state(thread_id: str):
    st = agent.get_state({"configurable": {"thread_id": thread_id}})
    if not st.values:
        return []
    return st.values.get("messages", [])


def count_kinds(msgs):
    kinds = Counter(type(m).__name__ for m in msgs)
    tool_calls_ai = sum(
        isinstance(m, AIMessage) and bool(getattr(m, "tool_calls", None)) for m in msgs
    )
    return kinds, tool_calls_ai


def collection_of(thread_id: str):
    name = f"paper_{thread_id}"
    for c in chroma_client.list_collections():
        if c.name == name:
            return c
    return None


def show_overview():
    sessions = list_sessions()
    print(f"共 {len(sessions)} 个会话\n")
    print(f"{'thread_id 前缀':<10}{'chunks':>7}  {'最新state消息(H/A/T)':<22}标题")
    print("-" * 90)
    for s in sessions:
        tid = s["thread_id"]
        col = collection_of(tid)
        n_chunks = col.count() if col else 0
        msgs = latest_state(tid)
        kinds, tc_ai = count_kinds(msgs)
        h, a, t = kinds.get("HumanMessage", 0), kinds.get("AIMessage", 0), kinds.get("ToolMessage", 0)
        flag = ""
        if t:
            flag = "  ⚠ 最新state仍含ToolMessage"
        print(f"{tid[:8]:<10}{n_chunks:>7}  H{h}/A{a}/T{t}（带tool_calls的AI {tc_ai}）{'':<4}{s['title'][:34]}{flag}")
    print("\n提示：T=ToolMessage。新架构会话正常应为 0；8 月老会话是旧架构样本。")


def show_detail(thread_id: str, grep: str | None):
    sessions = list_sessions()
    match = [s for s in sessions if s["thread_id"].startswith(thread_id)]
    if not match:
        print(f"没找到以 {thread_id} 开头的会话")
        return
    if len(match) > 1:
        print("前缀匹配到多个会话，请写长一点：")
        for s in match:
            print(" ", s["thread_id"], s["title"][:30])
        return
    tid = match[0]["thread_id"]
    print(f"会话 {tid}\n标题：{match[0]['title']}")

    print("\n── 向量库（chroma_db / paper_<thread_id>）──")
    col = collection_of(tid)
    if col is None:
        print("  该会话没有论文集合（可能已删除或从未上传成功）")
    else:
        data = col.get(include=["documents", "metadatas"])
        docs, metas = data["documents"], data["metadatas"]
        print(f"  chunk 总数 = {len(docs)}")
        secs = Counter((m.get("section", "?") for m in metas))
        print(f"  章节分布 = {dict(secs)}")
        title = next((m.get("title") for m in metas if m.get("title")), "")
        print(f"  集合元数据 title = {title}")
        if grep:
            kw = grep.lower()
            hit = [d[:60].replace(chr(10), " ") for d in docs if kw in d.lower()]
            print(f"\n  关键词「{grep}」命中 {len(hit)} 条：")
            if hit:
                for h in hit:
                    print("   残留 →", h)
                print("  → 若这是【旧论文】的词却仍命中，说明覆盖未生效。")
            else:
                print("  → 无残留，旧论文内容已不在集合中（覆盖成功）。")

    print("\n── 最新 state 消息构成（多轮对话看这里）──")
    msgs = latest_state(tid)
    if not msgs:
        print("  （无消息历史）")
    for i, m in enumerate(msgs, 1):
        kind = type(m).__name__
        tc = " tool_calls!" if isinstance(m, AIMessage) and bool(getattr(m, "tool_calls", None)) else ""
        preview = ""
        c = getattr(m, "content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        preview = str(c).replace("\n", " ")[:46]
        print(f"  {i:>2}. {kind:<14}{len(str(c)):>6} 字{tc:<14} {preview}")
    kinds, tc_ai = count_kinds(msgs)
    print(
        f"\n  合计：Human {kinds.get('HumanMessage',0)} / "
        f"AI {kinds.get('AIMessage',0)}（其中带tool_calls {tc_ai}）/ "
        f"ToolMessage {kinds.get('ToolMessage',0)}"
    )
    print("  读法：ToolMessage=0 表示最新状态没有大段检索原文；")
    print("        若某些 HumanMessage 长达上千字，那是注入的证据，注意它会跨轮保留。")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    grep = None
    if "--grep" in sys.argv:
        i = sys.argv.index("--grep")
        if i + 1 < len(sys.argv):
            grep = sys.argv[i + 1]
    if args:
        show_detail(args[0], grep)
    else:
        show_overview()
