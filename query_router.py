"""
Query Router：在生成前把用户问题路由到 A1/A2/B/C，决定是否需要预检索论文证据。
避免所有问题都走 Agent 工具循环（单轮内重复检索重发 + B 类问题白烧 token）。
"""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from config import model

ROUTER_PROMPT = """你是问题路由器。判断下面的用户问题是否依赖"已上传论文"的内容。只输出 JSON，不要输出任何其他内容。

- {"category": "A1"} —— 论文事实：问论文里明确写的内容（方法/实验/数据/结论/贡献等），必须检索论文
- {"category": "A2"} —— 论文不足：问论文的不足/局限/改进空间，需要检索论文的 limitation、future work、方法细节
- {"category": "B"} —— 领域知识：问通用的领域知识、与这篇具体论文无关（如"什么是FlashAttention"），不需要检索论文
- {"category": "C"} —— 创新推理：要求基于论文提出新想法/结合论文做推理，需要检索论文的局限和方法细节，再结合领域知识

输出格式：{"category": "A1"}"""

VALID = {"A1", "A2", "B", "C"}

CATEGORY_QUERIES = {
    "A1": None,  # 用问题原文检索
    "A2": ["limitations and future work of this paper", "method design weaknesses and assumptions"],
    "C": ["limitations and future work", "method architecture design details and assumptions"],
    "B": [],
}


def _parse_category(text: str) -> str:
    """从模型输出稳健提取 category，失败回退 A1（保守走检索）。"""
    try:
        content = text.strip()
        if content.startswith("```"):
            content = content.split("```")[1].strip()
        start, end = content.find("{"), content.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(content[start : end + 1])
            if data.get("category") in VALID:
                return data["category"]
    except Exception:
        pass
    return "A1"


def classify_question(text: str) -> str:
    """一次廉价 LLM 调用，返回 A1/A2/B/C。任何异常回退 A1。"""
    try:
        # qwen3 flash 默认开启思考模式，空 reasoning 会占满 max_tokens 导致分类 JSON
        # 还没输出就被截断（返回空串 → 全部兜底成 A1，路由形同虚设）。
        # 必须 enable_thinking=False，让这次调用直接产出分类 JSON。
        resp = model.bind(
            temperature=0,
            max_tokens=100,
            extra_body={"enable_thinking": False},
        ).invoke(
            [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=f"用户问题：{text}")]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _parse_category(content)
    except Exception:
        return "A1"


def build_route(text: str) -> dict:
    """分类 + 生成检索策略。queries 为空表示不需要检索论文。"""
    category = classify_question(text)
    if category not in CATEGORY_QUERIES:
        category = "A1"
    if category == "A1":
        queries = [text]
    else:
        queries = CATEGORY_QUERIES[category]
    return {
        "category": category,
        "queries": queries,
        "needs_retrieval": bool(queries),
    }
