"""
RAG 存储模块
使用 ChromaDB + DashScope Embedding API
"""
import math
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

load_dotenv(Path(__file__).parent / '.env')

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = (os.getenv("DASHSCOPE_BASE_URL") or "").rstrip("/")


def _embed(texts: list[str]) -> list[list[float]]:      #下划线前缀_embeg，这表示是一个内部函数，仅供本模块内的类使用，不对外暴露
    """调用 DashScope embedding API。接口单批最多 20 条，超出自动分批。"""
    batch_size = 20                                     #DashScope 兼容接口的批次上限
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = requests.post(                           #向阿里云的embeddings接口发送post请求
            f"{DASHSCOPE_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": "qwen3.7-text-embedding", "input": batch},
        )
        if resp.status_code != 200:                     #异常处理：把响应体一起抛出，便于定位 4xx 根因
            raise RuntimeError(f"embedding API {resp.status_code}: {resp.text}")
        data = sorted(resp.json()["data"], key=lambda d: d["index"])   #按 index 排序，保证与输入顺序一致
        results.extend(d["embedding"] for d in data)
    return results


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return v
    return [x / norm for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _mmr_select(query_vec: list[float], cand_vecs: list[list[float]], k: int, lam: float = 0.7) -> list[int]:
    """MMR 最大边际相关：同时考虑"与问题的相关度"和"与已选内容的重复度"，
    返回选中的候选下标。lam 越大越偏向相关度，越小越偏向多样性。"""
    selected: list[int] = []
    remaining = list(range(len(cand_vecs)))
    while len(selected) < k and remaining:
        best_i, best_score = -1, float("-inf")
        for i in remaining:
            relevance = _cos(query_vec, cand_vecs[i])
            redundancy = max((_cos(cand_vecs[i], cand_vecs[j]) for j in selected), default=0.0)
            score = lam * relevance - (1 - lam) * redundancy
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


class PaperRAG:
    """基于 ChromaDB 的论文向量存储与检索"""

    def __init__(self, persist_dir: str = "resources/chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    def store(self, thread_id: str, chunks: list[dict], title: str = "") -> None:
        """存储论文章节到向量库"""
        collection = self.client.get_or_create_collection(f"paper_{thread_id}")
        texts = [c["content"] for c in chunks]
        embeddings = _embed(texts)
        collection.add(
            ids=[f"{thread_id}_{i}" for i in range(len(chunks))],
            documents=texts,
            embeddings=embeddings,
            metadatas=[{
                "section": c["section"],
                "title": title,
                "page_start": c.get("page_start", 0),
                "page_end": c.get("page_end", 0),
            } for c in chunks],
        )

    def search(self, thread_id: str, query: str, k: int = 5,
               exclude_ids: set | None = None, pool: int = 20, mmr_lambda: float = 0.7) -> list[dict]:
        """检索与 query 最相关的论文内容。

        流程：先召回 pool 个候选（扩大召回面）→ 排除已展示过的 exclude_ids →
        用 MMR 在"相关度"与"多样性"间折中，挑出 k 个，避免同一 chunk 被反复返回重发。
        目标：只把最有用的、不重复的少量文本喂给大模型
        """
        collection = self.client.get_or_create_collection(f"paper_{thread_id}")
        total = collection.count()
        if total == 0:
            return []
        emb = _embed([query])
        query_vec = _normalize(emb[0])
        results = collection.query(
            query_embeddings=[emb[0]],
            n_results=min(pool, total),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        if results["ids"] and results["ids"][0]:
            excluded = exclude_ids or set()
            for i in range(len(results["ids"][0])):
                if results["ids"][0][i] in excluded:
                    continue
                candidates.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "section": results["metadatas"][0][i].get("section", "unknown"),
                    "page_start": results["metadatas"][0][i].get("page_start", 0),
                    "page_end": results["metadatas"][0][i].get("page_end", 0),
                    "score": results["distances"][0][i] if results.get("distances") else 0,
                })

        if not candidates:
            return []
        if len(candidates) <= k:
            return candidates

        emb_rows = collection.get(ids=[c["id"] for c in candidates], include=["embeddings"])
        emb_map = dict(zip(emb_rows["ids"], emb_rows["embeddings"]))
        cand_vecs = [_normalize(emb_map[c["id"]]) for c in candidates]
        chosen = _mmr_select(query_vec, cand_vecs, k, mmr_lambda)
        return [candidates[i] for i in chosen]

    def delete_collection(self, thread_id: str) -> None:
        """删除某个会话的论文向量"""
        try:
            self.client.delete_collection(f"paper_{thread_id}")
        except (ValueError, NotFoundError):
            pass


# 全局单例
_rag = None


def get_rag() -> PaperRAG:
    global _rag
    if _rag is None:
        _rag = PaperRAG()
    return _rag
