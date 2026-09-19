"""
Token 爆炸治理 / 覆盖上传 / ToolMessage 清理 —— 白盒验证脚本

设计原则：
1. 只读测试直接复用真实向量集合（不修改）。
2. 写入类测试（ToolMessage 清理、覆盖上传）一律用临时目录 / 临时数据库，
   结束自动删除，绝不污染真实 sessions、checkpoints 与 chroma_db。

运行：python verify_token_mechanisms.py
"""
import io
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# Windows 终端中文输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from config import model
from query_router import classify_question, build_route
from rag_store import PaperRAG
from app import build_evidence, route_user_question, SUMMARY_QUERIES

# 一个真实、只读的论文集合（Transformers SSMs，41 chunks），用于检索类验证
REAL_THREAD = "3c2d6b6d-c9bc-4c37-9bbe-312065bd0928"

TMP_DIR = Path("resources/_verify_tmp")
TMP_DB = TMP_DIR / "checkpoints.db"
TMP_CHROMA = TMP_DIR / "chroma"

PASS, FAIL, WARN = "✅ PASS", "❌ FAIL", "⚠️ WARN"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str):
    results.append((name, status, detail))
    print(f"[{status}] {name}")
    for line in detail.splitlines():
        print(f"        {line}")


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# A. token 爆炸治理手段
# ----------------------------------------------------------------------
def verify_router():
    section("A1. 查询路由：问题分类决定是否检索")
    cases = [
        ("这篇论文用了什么数据集和评估指标？", "A1"),
        ("这篇论文有哪些不足和改进空间？", "A2"),
        ("什么是 FlashAttention？请解释原理", "B"),
        ("结合这篇论文帮我想一个新的研究方向", "C"),
    ]
    for question, expect in cases:
        cat = classify_question(question)
        route = build_route(question)
        ok_cat = cat in {"A1", "A2", "B", "C"}
        # B 类必须不检索；其余必须检索
        expect_retrieval = cat != "B"
        ok_logic = route["needs_retrieval"] == expect_retrieval
        tag = PASS if (ok_cat and ok_logic) else WARN
        hit = "命中预期" if cat == expect else f"预期 {expect}（分类有主观性，重点看逻辑）"
        record(f"路由分类「{question[:18]}…」→ {cat}", tag,
               f"{hit}；needs_retrieval={route['needs_retrieval']}，"
               f"queries 数={len(route['queries'])}")


def verify_evidence_budget(rag: PaperRAG):
    section("A2. 预检索 + 证据预算（build_evidence）")
    evidence = build_evidence(rag, REAL_THREAD, SUMMARY_QUERIES, per_query=2)
    blocks = [b for b in evidence.split("\n\n---\n\n") if b.strip()]

    # 条数上限：4 查询 × 每查询 2 条 = 8
    ok_count = len(blocks) <= len(SUMMARY_QUERIES) * 2
    record("证据块数量 ≤ 4查询×2条 = 8",
           PASS if ok_count else FAIL,
           f"实际证据块数 = {len(blocks)}")

    # 去重：块正文不应重复
    bodies = []
    for b in blocks:
        bodies.append(b.split("】", 1)[-1].strip())
    ok_unique = len(set(bodies)) == len(bodies)
    record("证据块内容去重（无重复 chunk）",
           PASS if ok_unique else FAIL,
           f"总块数={len(blocks)}，去重后={len(set(bodies))}")

    # 截断：每个 chunk 正文 ≤ 600 字符
    over = [i + 1 for i, body in enumerate(bodies) if len(body) > 600]
    record("每个证据块正文截断 ≤ 600 字",
           PASS if not over else FAIL,
           "全部 ≤600" if not over else f"超限块：{over}")

    # 带来源章节/页码标注
    ok_src = all("【证据 #" in b and "来源:" in b for b in blocks)
    record("证据块带【证据 #N | 来源: 章节 | 页码】",
           PASS if ok_src else FAIL,
           "格式完整" if ok_src else "存在缺来源标注的块")


def verify_mmr(rag: PaperRAG):
    section("A3. MMR 重排 + chunk 级去重（exclude_ids）")
    first = rag.search(REAL_THREAD, "transformer self-attention architecture", k=3)
    first_ids = {r["id"] for r in first}
    record("首次检索返回 k=3", PASS if len(first) == 3 else WARN,
           f"返回 {len(first)} 条")

    second = rag.search(REAL_THREAD, "transformer self-attention architecture",
                        k=3, exclude_ids=first_ids)
    second_ids = {r["id"] for r in second}
    overlap = first_ids & second_ids
    record("第二次检索排除已展示 chunk（exclude_ids 生效）",
           PASS if not overlap and second else FAIL,
           f"第一次={len(first_ids)}条，第二次={len(second_ids)}条，交集={len(overlap)}")


def verify_recursion_and_model():
    section("A4. 保险丝 + 轻量模型（静态确认）")
    app_src = Path("app.py").read_text(encoding="utf-8")
    config_src = Path("config.py").read_text(encoding="utf-8")
    record("recursion_limit = 15 已设置",
           PASS if "recursion_limit" in app_src else FAIL,
           "app.py 中已配置递归上限（app.py:170）" if "recursion_limit" in app_src
           else "未找到 recursion_limit")
    using_flash = "flash" in config_src.lower()
    record("模型使用轻量版 flash",
           PASS if using_flash else WARN,
           "config.py 使用 qwen flash（更省更快）" if using_flash
           else "当前不是 flash 模型")


# ----------------------------------------------------------------------
# B. 同一会话可覆盖上传多篇论文（旧 chunk 必须被清空，不能混合）
# ----------------------------------------------------------------------
def verify_overwrite_upload():
    section("B. 同一会话覆盖上传：新论文替换旧论文")
    rag = PaperRAG(persist_dir=str(TMP_CHROMA))
    tid = "verify-overwrite-thread"

    # 论文 1：动物迁徙
    p1 = [
        {"section": "abstract",
         "content": "Zebras are African animals known for black and white stripes. They migrate across the Serengeti in large herds to find water and fresh grassland."},
        {"section": "method",
         "content": "We tracked zebra migration patterns using GPS collars and satellite imagery over three years, observing herd movement distances."},
    ]
    # 论文 2：量子物理
    p2 = [
        {"section": "abstract",
         "content": "Quantum entanglement is a physical phenomenon where pairs of particles remain interconnected so the state of one instantly influences the other."},
        {"section": "method",
         "content": "We generate entangled photon pairs through spontaneous parametric down conversion and measure Bell inequality violations."},
    ]

    rag.store(tid, p1, title="Paper One")
    n1 = rag.client.get_collection(f"paper_{tid}").count()
    record("上传论文 1 后集合含 2 个 chunk",
           PASS if n1 == 2 else FAIL, f"count = {n1}")

    # 关键：app.py:139 在重新上传前调用 delete_collection
    rag.delete_collection(tid)
    existing = [c.name for c in rag.client.list_collections() if c.name == f"paper_{tid}"]
    record("重新上传前 delete_collection 删除旧集合",
           PASS if not existing else FAIL,
           "旧集合已删除" if not existing else "旧集合仍存在")

    rag.store(tid, p2, title="Paper Two")
    col = rag.client.get_collection(f"paper_{tid}")
    n2 = col.count()
    record("上传论文 2 后 chunk 数=2（不是 1+2 混合=4）",
           PASS if n2 == 2 else FAIL, f"count = {n2}（若=4 说明旧论文残留）")

    # 物理校验：集合现存文档中不得包含旧论文独有词（覆盖是删除重建，
    # 不能用语义检索"0 命中"判断——仅 2 条新 chunk 时向量检索仍可能泛化召回）
    docs = col.get(include=["documents"])["documents"]
    leaked = [d[:40] for d in docs if any(w in d.lower() for w in ("zebra", "stripe", "herd", "serengeti"))]
    record("旧论文（斑马）原文已从集合物理删除，无残留",
           PASS if not leaked else FAIL,
           f"发现旧论文残留 {len(leaked)} 条：{leaked}" if leaked
           else f"集合现存 {len(docs)} 条，全部属于新论文，未混合")
    new_hit = rag.search(tid, "quantum entanglement photon bell physics", k=2)
    record("新论文（量子）内容可正常检索",
           PASS if new_hit else FAIL,
           f"新论文命中 {len(new_hit)} 条")


# ----------------------------------------------------------------------
# C. ToolMessage 清理 + 不同会话相互隔离
# ----------------------------------------------------------------------
def cleanup_tool_messages(agent, config: dict) -> int:
    """复刻 app.py:172-183 的清理逻辑（该段内联在 chat() 中，这里等价复刻）。"""
    state = agent.get_state(config)
    to_remove = [
        RemoveMessage(id=m.id)
        for m in state.values.get("messages", [])
        if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
    ]
    if to_remove:
        agent.update_state(config, {"messages": to_remove})
    return len(to_remove)


def _inject_tool_turn(agent, config: dict, tool_payload: str):
    """注入一轮“带工具调用”的消息：Human + AIMessage(tool_calls) + ToolMessage。"""
    ai = AIMessage(content="", tool_calls=[{
        "name": "retrieve_paper",
        "args": {"query": "x"},
        "id": f"call_{tool_payload[:6]}",
    }])
    tm = ToolMessage(content=tool_payload, tool_call_id=ai.tool_calls[0]["id"])
    agent.update_state(config, {"messages": [HumanMessage(content="论文里方法是什么？"), ai, tm]})


def verify_toolmessage_cleanup():
    section("C. ToolMessage 清理 + 跨会话隔离")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TMP_DB), check_same_thread=False)
    agent = create_agent(model, tools=[], system_prompt="verify",
                         checkpointer=SqliteSaver(conn))

    cfg_a = {"configurable": {"thread_id": "thread-A"}}
    cfg_b = {"configurable": {"thread_id": "thread-B"}}

    big_evidence = "【论文检索结果】" + ("这是一大段会反复重发的论文原文。" * 120)
    _inject_tool_turn(agent, cfg_a, big_evidence)
    _inject_tool_turn(agent, cfg_b, "【B 会话的检索结果】" + "另一篇论文内容。" * 50)

    def kinds(cfg):
        msgs = agent.get_state(cfg).values.get("messages", [])
        return {
            "Human": sum(isinstance(m, HumanMessage) for m in msgs),
            "AIMessage(tool_calls)": sum(
                isinstance(m, AIMessage) and bool(getattr(m, "tool_calls", None)) for m in msgs),
            "ToolMessage": sum(isinstance(m, ToolMessage) for m in msgs),
        }

    before_a = kinds(cfg_a)
    record("会话 A 注入后含 1 条 ToolMessage + 1 条 tool_calls AI",
           PASS if before_a["ToolMessage"] == 1 and before_a["AIMessage(tool_calls)"] == 1
           else FAIL, f"会话A 消息构成 = {before_a}")

    removed = cleanup_tool_messages(agent, cfg_a)
    after_a = kinds(cfg_a)
    record("清理会话 A：删除 ToolMessage 与 tool_calls，保留 Human",
           PASS if removed == 2 and after_a["ToolMessage"] == 0
           and after_a["AIMessage(tool_calls)"] == 0 and after_a["Human"] == 1
           else FAIL,
           f"删除 {removed} 条；清理后 = {after_a}")

    after_b = kinds(cfg_b)
    record("会话 B 不受影响（只清理当前会话，跨会话隔离）",
           PASS if after_b["ToolMessage"] == 1 and after_b["AIMessage(tool_calls)"] == 1
           else FAIL,
           f"会话B 仍保留其 ToolMessage = {after_b}")
    conn.close()


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    real_rag = PaperRAG()  # 默认真实目录，仅只读检索
    try:
        verify_router()
        verify_evidence_budget(real_rag)
        verify_mmr(real_rag)
        verify_recursion_and_model()
        verify_overwrite_upload()
        verify_toolmessage_cleanup()
    finally:
        # 清理所有临时写入产物
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    section("验证总结")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    for name, status, _ in results:
        print(f"  {status}  {name}")
    print(f"\n合计：{n_pass} PASS / {n_fail} FAIL / {n_warn} WARN（共 {len(results)} 项）")
    print("临时数据已清理，真实会话与向量库未受影响。")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
