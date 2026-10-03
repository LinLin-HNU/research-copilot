"""Create review-required benchmark question drafts from local PaperRAG chunks.

This script is deliberately offline and templated: it never calls an LLM and never
changes benchmark/questions.json. A human must verify every expected term, section,
reference answer and PDF citation before a draft becomes a benchmark case.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_store import get_rag

SKIP_TERMS = {
    "The", "This", "That", "These", "With", "From", "For", "And", "Our", "We", "In", "To",
    "Figure", "Table", "Section", "Abstract", "Introduction", "Method", "Methods", "Results",
    "Conclusion", "Appendix", "References", "IEEE", "ICLR", "NeurIPS", "Transformer",
}
TERM_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z]{2,}[A-Za-z0-9-]*)\b")


def normal_section(name: str) -> str:
    value = (name or "unknown").strip().lower()
    aliases = {
        "related works": "related work", "experiments": "experiment", "experimental results": "results",
        "methodology": "method", "methods": "method", "approaches": "approach", "conclusions": "conclusion",
        "limitations": "limitation", "future work": "future work",
    }
    return aliases.get(value, value)


def preferred(chunks: list[dict], groups: list[set[str]]) -> list[dict]:
    for group in groups:
        rows = [c for c in chunks if normal_section(c.get("section")) in group and len(c.get("content", "")) >= 120]
        if rows:
            return rows
    return [c for c in chunks if len(c.get("content", "")) >= 120] or chunks


def terms(rows: list[dict], count: int = 3) -> list[str]:
    found = []
    for row in rows:
        found.extend(t for t in TERM_RE.findall(row.get("content", "")) if t not in SKIP_TERMS)
    frequency = Counter(found)
    ranked = sorted(frequency, key=lambda term: (-frequency[term], -len(term), term))
    return ranked[:count] or ["time series"]


def pick(rows: list[dict], fallback: str = "time series") -> tuple[str, str, int, int, str]:
    row = max(rows, key=lambda r: len(r.get("content", "")))
    term = terms([row], 1)[0] if terms([row], 1) else fallback
    return normal_section(row.get("section")), term, row.get("page_start", 0), row.get("page_end", 0), row.get("content", "")


def make_cases(thread_id: str, prefix: str, chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    if not chunks:
        raise ValueError(f"No chunks found for {thread_id}")
    abstract = preferred(chunks, [{"abstract", "preamble"}, {"introduction", "background"}])
    method = preferred(chunks, [{"method", "approach", "architecture", "model", "preliminaries"}, {"introduction"}])
    experiment = preferred(chunks, [{"experiment", "evaluation", "results"}, {"method", "approach"}])
    conclusion = preferred(chunks, [{"limitation", "discussion", "conclusion", "future work"}, {"conclusion"}, {"introduction"}])
    a_section, a_term, a_start, a_end, a_text = pick(abstract)
    m_section, m_term, m_start, m_end, m_text = pick(method, a_term)
    e_section, e_term, e_start, e_end, e_text = pick(experiment, m_term)
    c_section, c_term, c_start, c_end, c_text = pick(conclusion, m_term)
    candidates = [
        ("A1", f"论文提出的 {a_term} 主要要解决什么问题？作者概括的核心贡献是什么？", a_section, a_term, a_start, a_end, a_text),
        ("A1", f"论文中 {m_term} 的核心结构或工作机制是什么？", m_section, m_term, m_start, m_end, m_text),
        ("A1", f"作者为何采用 {m_term}，它相对传统设计解决了什么困难？", m_section, m_term, m_start, m_end, m_text),
        ("A1", f"论文围绕 {e_term} 做了哪些实验设置或任务验证？主要结论是什么？", e_section, e_term, e_start, e_end, e_text),
        ("A1", f"从论文结果看，{e_term} 在哪些条件、数据集或评价指标下表现如何？", e_section, e_term, e_start, e_end, e_text),
        ("A2", f"论文对 {c_term} 相关方法提到哪些局限、挑战或未来工作？", c_section, c_term, c_start, c_end, c_text),
        ("B", "什么是时间序列预测中的训练集、验证集和测试集？它们分别用于什么？", "", "", 0, 0, ""),
        ("B", "MAE 和 MSE 分别衡量什么？在比较预测模型时应如何理解它们？", "", "", 0, 0, ""),
        ("C", f"基于论文中 {m_term} 的设计，若要提升对更长历史窗口的适应性，你会优先改进哪个环节？请区分论文事实与自己的推断。", m_section, m_term, m_start, m_end, m_text),
        ("C", f"结合论文对 {c_term} 的讨论，提出一个可验证的后续研究方向，并说明需要增加什么实验。", c_section, c_term, c_start, c_end, c_text),
    ]
    cases, review = [], []
    for index, (category, question, section, term, start, end, excerpt) in enumerate(candidates, 1):
        case_id = f"{prefix}-{category.lower()}-{index:02d}"
        cases.append({"id": case_id, "question": question, "expected_category": category,
                      "expected_sections": [section] if section else [], "expected_terms": [term] if term else [],
                      "thread_id": thread_id, "split": "dev"})
        review.append({"id": case_id, "expected_category": category, "proposed_section": section,
                       "proposed_term": term, "pdf_pages": f"{start}-{end}" if start else "",
                       "candidate_excerpt": excerpt[:1000], "review_status": "needs_human_verification"})
    return cases, review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, required=True, help="JSON array: {prefix, thread_id, paper_name}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sessions = json.loads(args.sessions.read_text(encoding="utf-8"))
    if not sessions or len({row["prefix"] for row in sessions}) != len(sessions):
        raise ValueError("Sessions must be non-empty with unique prefixes.")
    args.output.mkdir(parents=True, exist_ok=False)
    rag = get_rag()
    all_cases, manifest = [], []
    for session in sessions:
        thread_id = session["thread_id"]
        chunks = rag.list_chunks(thread_id)
        cases, review = make_cases(thread_id, session["prefix"], chunks)
        (args.output / f"{session['prefix']}.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        (args.output / f"{session['prefix']}_review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        all_cases.extend(cases)
        manifest.append({**session, "chunk_count": len(chunks), "case_count": len(cases)})
    (args.output / "all_drafts.json").write_text(json.dumps(all_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"papers": len(sessions), "draft_cases": len(all_cases), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
