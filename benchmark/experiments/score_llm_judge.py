"""Reveal blinded LLM judgements and summarize them separately from human review."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.experiments.common import agreement, proportion, read_jsonl

METRICS = ("citation_correct", "grounded", "hallucination", "completeness")


def consensus(values):
    """Strict consensus: missing values or disagreement remain unscored (NA)."""
    if not values or any(value is None for value in values):
        return None
    return values[0] if all(value == values[0] for value in values) else None


def metric_summary(records):
    return {metric: proportion([record.get(metric) for record in records]) for metric in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; do not overwrite evaluation evidence")

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    judgements = read_jsonl(args.judgements)
    grouped = defaultdict(list)
    seen_passes = set()
    for row in judgements:
        review_id = row.get("review_id")
        key = (review_id, row.get("pass"))
        if review_id not in mapping:
            parser.error(f"unknown review_id: {review_id}")
        if key in seen_passes:
            parser.error(f"duplicate review_id/pass: {key}")
        seen_passes.add(key)
        grouped[review_id].append(row)

    revealed = []
    for review_id, passes in grouped.items():
        passes.sort(key=lambda row: row["pass"])
        source = mapping[review_id]
        revealed.append({
            "review_id": review_id,
            "id": source["id"],
            "variant": source["variant"],
            "expected_category": source.get("expected_category"),
            "pass_count": len(passes),
            **{
                metric: consensus([row.get(metric) for row in passes])
                for metric in METRICS
            },
            "any_error": any(row.get("error") for row in passes),
            "any_insufficient_evidence": any(
                row.get("insufficient_evidence", False) for row in passes
            ),
        })

    variants = sorted({row["variant"] for row in revealed})
    report = {
        "kind": "llm_as_a_judge_not_human_review",
        "judge_rows": len(judgements),
        "review_items": len(revealed),
        "ai_consensus_scored_cells": sum(
            row[metric] is not None for row in revealed for metric in METRICS
        ),
        "overall": {
            variant: metric_summary([row for row in revealed if row["variant"] == variant])
            for variant in variants
        },
        "inter_pass": {},
        "paired_deltas": [],
        "insufficient_evidence_items": sum(
            row["any_insufficient_evidence"] for row in revealed
        ),
        "limitations": [
            "These are model judgements, not human or expert scores.",
            "Strict consensus converts pass disagreement or missing evidence to NA.",
            "The development reference points were AI-reviewed rather than human gold.",
        ],
        "records": revealed,
    }
    pass_numbers = sorted({row["pass"] for row in judgements})
    if len(pass_numbers) >= 2:
        first, second = pass_numbers[:2]
        for metric in METRICS:
            pairs = []
            for rows in grouped.values():
                by_pass = {row["pass"]: row for row in rows}
                if first in by_pass and second in by_pass:
                    a, b = by_pass[first].get(metric), by_pass[second].get(metric)
                    if a is not None and b is not None:
                        pairs.append((a, b))
            report["inter_pass"][metric] = agreement(pairs)

    if len(variants) >= 2:
        baseline_name = variants[0]
        baseline = {row["id"]: row for row in revealed if row["variant"] == baseline_name}
        for variant in variants[1:]:
            current = {row["id"]: row for row in revealed if row["variant"] == variant}
            for metric in METRICS:
                pairs = [
                    (baseline[case_id].get(metric), current[case_id].get(metric))
                    for case_id in baseline.keys() & current.keys()
                ]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                report["paired_deltas"].append({
                    "baseline": baseline_name,
                    "variant": variant,
                    "metric": metric,
                    "paired_n": len(pairs),
                    "difference_percentage_points": (
                        100 * sum(b - a for a, b in pairs) / len(pairs) if pairs else None
                    ),
                    "direction": "lower_is_better" if metric == "hallucination" else "higher_is_better",
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "kind": report["kind"],
        "review_items": report["review_items"],
        "ai_consensus_scored_cells": report["ai_consensus_scored_cells"],
        "insufficient_evidence_items": report["insufficient_evidence_items"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
