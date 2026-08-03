"""
RAG 存储模块
使用 ChromaDB + DashScope Embedding API
"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import chromadb
from chromadb.config import Settings

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

    def search(self, thread_id: str, query: str, k: int = 5) -> list[dict]:
        """检索与 query 最相关的论文内容"""
        collection = self.client.get_or_create_collection(f"paper_{thread_id}")
        emb = _embed([query])
        results = collection.query(query_embeddings=[emb[0]], n_results=k)  #把已向量化的用户问题传入chromeDB中，查询与问题最相似的前k个内容

        items = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                items.append({
                    "content": results["documents"][0][i],
                    "section": results["metadatas"][0][i].get("section", "unknown"),
                    "page_start": results["metadatas"][0][i].get("page_start", 0),
                    "page_end": results["metadatas"][0][i].get("page_end", 0),
                    "score": results["distances"][0][i] if results.get("distances") else 0,
                })
        return items

    def delete_collection(self, thread_id: str) -> None:
        """删除某个会话的论文向量"""
        try:
            self.client.delete_collection(f"paper_{thread_id}")
        except ValueError:
            pass


# 全局单例
_rag = None


def get_rag() -> PaperRAG:
    global _rag
    if _rag is None:
        _rag = PaperRAG()
    return _rag
