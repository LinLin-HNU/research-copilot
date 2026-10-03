"""Run routing and retrieval checks for a manually curated paper benchmark.
自动测路由准确率 + 检索命中率（不生成回答，省钱）
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import build_question_evidence
from query_router import build_route
from rag_store import get_rag


def evidence_sections(evidence: str) -> set[str]:
    sections = set()
    for part in evidence.split("\n\n---\n\n"):
        if "来源:" not in part:
            continue
        source = part.split("来源:", 1)[1].split("|", 1)[0].split("】", 1)[0]
        sections.add(source.strip().lower())
    return sections


def evaluate_case(rag, case: dict, default_thread_id: str) -> dict:
    thread_id = case.get("thread_id") or default_thread_id
    if not thread_id:
        raise ValueError(f"Case {case['id']} has no thread_id.")

    started = perf_counter()
    route = build_route(case["question"])
    route_ms = round((perf_counter() - started) * 1000, 2)
    expected_category = case["expected_category"]
    result = {
        "id": case["id"],
        "expected_category": expected_category,
        "actual_category": route["category"],
        "route_correct": route["category"] == expected_category,
        "route_ms": route_ms,
        "retrieval_ms": 0.0,
        "evidence_blocks": 0,
        "evidence_chars": 0,
        "expected_sections": case.get("expected_sections", []),
        "expected_terms": case.get("expected_terms", []),
        "retrieval_hit": None,
    }

    if route["needs_retrieval"]:
        retrieval_started = perf_counter()
        evidence = build_question_evidence(rag, thread_id, route)
        result["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 2)
        result["evidence_blocks"] = len([p for p in evidence.split("\n\n---\n\n") if p.strip()])
        result["evidence_chars"] = len(evidence)
        actual_sections = evidence_sections(evidence)
        expected_sections = {s.lower() for s in case.get("expected_sections", [])}
        expected_terms = [term.lower() for term in case.get("expected_terms", [])]
        section_hit = not expected_sections or bool(expected_sections & actual_sections)
        term_hit = not expected_terms or all(term in evidence.lower() for term in expected_terms)
        result["retrieval_hit"] = section_hit and term_hit
        result["actual_sections"] = sorted(actual_sections)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--thread-id", default="")
    parser.add_argument("--output", type=Path, default=Path("benchmark/reports/latest.json"))
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark cases must be a non-empty JSON array.")
    rag = get_rag()
    results = [evaluate_case(rag, case, args.thread_id) for case in cases]
    routed = [row for row in results if row["route_correct"]]
    retrievable = [row for row in results if row["retrieval_hit"] is not None]
    hits = [row for row in retrievable if row["retrieval_hit"]]
    report = {
        "generated_at": datetime.now().isoformat(),
        "case_count": len(results),
        "routing_accuracy": len(routed) / len(results),
        "retrieval_hit_rate": len(hits) / len(retrievable) if retrievable else None,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
