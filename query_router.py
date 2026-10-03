"""
Query Router：在生成前把用户问题路由到 A1/A2/B/C，决定是否需要预检索论文证据。
避免所有问题都走 Agent 工具循环（单轮内重复检索重发 + B 类问题白烧 token）。
"""
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from config import model

ROUTER_PROMPT = """你是问题路由器。当前对话中已经有一篇上传的论文。判断下面的问题是否需要依赖这篇论文。只输出 JSON，不要输出任何其他内容。

- {"category": "A1"} —— 论文事实：问论文里明确写的内容（方法/实验/数据/结论/贡献等），必须检索论文
- {"category": "A2"} —— 论文不足：问论文的不足/局限/改进空间，需要检索论文的 limitation、future work、方法细节
- {"category": "B"} —— 领域知识：问通用的领域知识、与这篇具体论文无关（如"什么是FlashAttention"），不需要检索论文
- {"category": "C"} —— 创新推理：要求基于论文提出新想法/结合论文做推理，需要检索论文的局限和方法细节，再结合领域知识

判定规则：
1. B 只能用于“即使没有上传论文，答案也完全相同”的通用知识问题。
2. 只要问题指向该论文、作者、论文中的具体模型/假设/实验/结论，就不得判为 B。
3. 询问局限、不足、风险、失效条件、为什么难以做到，优先判 A2；不能因为问题没写“本文”就判 B。
4. 询问论文中的具体模型或专有概念如何实现/为何有效，判 A1。
5. 要求设计比较、验证、消融或改进方案，判 C。

例子：
- “什么是 FlashAttention？” -> B
- “PatchTST 如何处理多变量时间序列？” -> A1
- “传统 self-attention 用于长序列预测时有哪些限制？” -> A2
- “如何设计消融实验验证该论文的通道独立假设？” -> C

输出格式：{"category": "A1"}"""

VALID = {"A1", "A2", "B", "C"}

LIMITATION_CUES = (
    "局限", "不足", "限制", "缺陷", "弱点", "瓶颈", "风险", "失效",
    "难以", "无法", "不适用", "为什么可能", "limitations", "weakness",
    "failure mode", "risk",
)
INNOVATION_CUES = (
    "如何比较", "如何验证", "如何设计", "如何改进", "如何结合",
    "怎样比较", "怎样验证", "消融实验", "ablation", "how to compare",
    "how to validate", "how to improve",
)
PAPER_REFERENCE_CUES = (
    "这篇论文", "该论文", "本文", "文中", "作者", "该方法", "该模型",
    "该研究", "论文中", "this paper", "the paper", "the authors",
)
GENERIC_DEFINITION_PREFIXES = (
    "什么是", "请介绍", "介绍一下", "请解释", "解释一下", "what is ",
)


def _has_paper_specific_identifier(text: str) -> bool:
    """识别专有模型名或连字符技术词，例如 PatchTST、Time-LLM、channel-independence。"""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", text)
    return any(
        "-" in token
        or any(ch.isdigit() for ch in token)
        or sum(ch.isupper() for ch in token) >= 2
        for token in tokens
    )


def apply_retrieval_guard(text: str, predicted: str) -> str:
    """B 类是高风险决策：它会完全跳过论文检索。

    模型仍负责四分类；这个确定性保护层只处理“模型预测 B，但问题明显
    需要论文证据”的情况。优先避免漏检，代价是少量通用局限问题可能多做一次检索。
    """
    if predicted != "B":
        return predicted if predicted in VALID else "A1"
    normalized = " ".join(text.strip().lower().split())
    explicit_reference = any(cue in normalized for cue in PAPER_REFERENCE_CUES)
    generic_definition = any(normalized.startswith(prefix) for prefix in GENERIC_DEFINITION_PREFIXES)
    # 显式的通用概念释义仍保留 B，防止保护层把所有问题都推向检索。
    if generic_definition and not explicit_reference:
        return "B"
    if any(cue in normalized for cue in LIMITATION_CUES):
        return "A2"
    if any(cue in normalized for cue in INNOVATION_CUES):
        return "C"
    if explicit_reference:
        return "A1"
    if _has_paper_specific_identifier(text) and not generic_definition:
        return "A1"
    return "B"

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
            max_tokens=50,
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
    raw_category = classify_question(text)
    category = apply_retrieval_guard(text, raw_category)
    if category == "B":
        queries = []
    elif category == "A1":
        queries = [text]
    else:
        # 保留用户原问，否则 A2/C 只用通用英文模板检索，会丢掉
        # PatchTST、quantization 等关键实体。模板查局限/方法章，原问查具体实体。
        queries = [text, *CATEGORY_QUERIES[category]]
    return {
        "category": category,
        "raw_category": raw_category,
        "guard_applied": category != raw_category,
        "queries": queries,
        "needs_retrieval": bool(queries),
    }
