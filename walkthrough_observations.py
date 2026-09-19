"""
逐步拆解观察脚本（理解向，非测试）
把每个 token 优化机制的"中间产物"和"省了多少"直接打印出来。
只读真实向量集合，不写入、不删除任何数据。

运行：python walkthrough_observations.py
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from query_router import build_route
from rag_store import PaperRAG
from app import build_evidence, SUMMARY_QUERIES

REAL_THREAD = "3c2d6b6d-c9bc-4c37-9bbe-312065bd0928"  # 真实 Transformers SSMs 集合，只读

# 粗估 token：论文中英混排，约 1 token ≈ 2 个字符（仅用于直观对比，非精确计费值）
def approx_tokens(chars: int) -> int:
    return chars // 2


def title(t: str):
    print("\n" + "─" * 68)
    print(f"  {t}")
    print("─" * 68)


rag = PaperRAG()

# 1) 查询路由 --------------------------------------------------------------
title("① 查询路由：先花极少 token 判断「要不要检索」")
questions = [
    "这篇论文用了什么数据集？",      # A1 要检索
    "这篇论文有哪些不足？",          # A2 要检索
    "什么是 FlashAttention？",       # B  不检索 ← 省 token 关键
    "结合这篇论文想个新研究方向",     # C  要检索
]
for q in questions:
    r = build_route(q)
    flag = "🔍 检索论文" if r["needs_retrieval"] else "🚫 不检索（用模型自身知识，0 论文 token）"
    print(f"  「{q}」→ {r['category']:<3} {flag}")
print("  说明：B 类问题在旧版本照样把论文片段塞进上下文；现在直接跳过检索。")

# 2) 证据包预算 ------------------------------------------------------------
title("② 预检索 + 证据预算：一轮问答实际注入多少内容")
evidence = build_evidence(rag, REAL_THREAD, SUMMARY_QUERIES, per_query=2)
blocks = [b for b in evidence.split("\n\n---\n\n") if b.strip()]
lens = [len(b.split("】", 1)[-1].strip()) for b in blocks]
total_chars = sum(len(b) for b in blocks)
print(f"  证据块数量   : {len(blocks)} 块（上限 4查询 × 2条 = 8）")
print(f"  各块正文字符 : {lens}（每块硬截断 ≤600）")
print(f"  证据包总字符 : {total_chars}  ≈ {approx_tokens(total_chars)} tokens（粗估）")
print("  预览第一块：")
print("    " + blocks[0].replace("\n", "\n    ")[:400])

# 3) MMR 去重 --------------------------------------------------------------
title("③ chunk 级去重：同一片段不发第二遍")
q = "transformer self-attention architecture"
first = rag.search(REAL_THREAD, q, k=3)
ids1 = {r["id"] for r in first}
second = rag.search(REAL_THREAD, q, k=3, exclude_ids=ids1)
ids2 = {r["id"] for r in second}
suffix = lambda ids: sorted(int(i.split("_")[-1]) for i in ids)
print(f"  第一次命中 chunk 序号 : {suffix(ids1)}")
print(f"  第二次(排除已展示)    : {suffix(ids2)}，交集 = {len(ids1 & ids2)} 条（0 表示没有重复发送）")

# 4) 多轮 token 增长对比（核心） -------------------------------------------
title("④ 多轮对话：上下文随轮数怎么变（省 token 的主战场）")
per_turn = total_chars  # 用上面真实证据包大小代表"每轮检索注入量"
print(f"  以真实证据包 ≈{approx_tokens(per_turn)} tokens/轮 为基准，模拟连续 6 轮问答：")
print(f"  {'轮次':<4}{'旧:证据常驻历史(累积)':<26}{'新:每轮清空工具消息(恒定)':<26}")
for n in range(1, 7):
    old = n * approx_tokens(per_turn)
    new = approx_tokens(per_turn)
    print(f"  第{n}轮   {old:<26}{new:<26}")
print(f"  到第 6 轮：旧≈{6*approx_tokens(per_turn)} vs 新≈{approx_tokens(per_turn)} tokens（仅证据部分，未计提示词与问答）")
print("  机制：app.py 每轮请求前用 RemoveMessage 删除历史 ToolMessage / tool_calls，")
print("       证据不常驻，需要时重新检索，所以每轮发送量基本恒定。")

print("\n观察完成。以上检索均为只读，未改动你的任何会话或向量数据。")
