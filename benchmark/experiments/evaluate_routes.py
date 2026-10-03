"""Evaluate the router only; no retrieval, embedding, or answer generation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.experiments.common import digest, read_jsonl


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["expected_category"]].append(row)
    return {
        "n": len(rows),
        "correct": sum(row["route_correct"] for row in rows),
        "accuracy": sum(row["route_correct"] for row in rows) / len(rows) if rows else None,
        "by_expected_category": {
            category: {
                "n": len(group),
                "correct": sum(row["route_correct"] for row in group),
                "recall": sum(row["route_correct"] for row in group) / len(group),
            }
            for category, group in sorted(groups.items())
        },
        "confusion": {
            f"{expected}->{actual}": count
            for (expected, actual), count in sorted(Counter(
                (row["expected_category"], row["predicted_category"]) for row in rows
            ).items())
        },
        "predicted_b_for_paper_question": sum(
            row["expected_category"] != "B" and row["predicted_category"] == "B"
            for row in rows
        ),
    }


def baseline_rows(path, selected_ids):
    rows = []
    for record in read_jsonl(path):
        if record.get("variant") != "v1_current" or record.get("id") not in selected_ids:
            continue
        route = record.get("route") or {}
        predicted = route.get("category")
        if predicted is None:
            continue
        rows.append({
            "id": record["id"],
            "expected_category": record["expected_category"],
            "predicted_category": predicted,
            "route_correct": predicted == record["expected_category"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--baseline-answers", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if any("split" not in case for case in cases):
        parser.error("every case must have an explicit split")
    cases = [case for case in cases if case["split"] == args.split]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    elif not args.confirm_full:
        parser.error("use --limit for a smoke test or --confirm-full for the complete router run")
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("selected cases must be nonempty with unique ids")
    if args.output.exists():
        parser.error("--output already exists; do not overwrite evaluation evidence")

    import query_router

    current = []
    for case in cases:
        route = query_router.build_route(case["question"])
        row = {
            "id": case["id"],
            "expected_category": case["expected_category"],
            "raw_category": route.get("raw_category", route["category"]),
            "predicted_category": route["category"],
            "guard_applied": route.get("guard_applied", False),
            "route_correct": route["category"] == case["expected_category"],
        }
        current.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "protocol": {
            "cases_sha256": digest(args.cases),
            "split": args.split,
            "answer_generation_called": False,
            "retrieval_called": False,
            "router_api_called": True,
            "labels": "AI-reviewed development labels; not human expert ground truth",
        },
        "after": summarize(current),
        "rows": current,
    }
    if args.baseline_answers:
        before = baseline_rows(args.baseline_answers, {case["id"] for case in cases})
        report["before"] = summarize(before)
        report["comparison"] = {
            "before_n": len(before),
            "after_n": len(current),
            "accuracy_change_percentage_points": (
                100 * (report["after"]["accuracy"] - report["before"]["accuracy"])
                if before else None
            ),
            "a2_recall_change_percentage_points": (
                100 * (
                    report["after"]["by_expected_category"]["A2"]["recall"]
                    - report["before"]["by_expected_category"]["A2"]["recall"]
                )
                if "A2" in report["before"]["by_expected_category"] else None
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("before", "after", "comparison") if key in report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
