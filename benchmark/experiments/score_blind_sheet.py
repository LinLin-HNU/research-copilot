"""Binary human scores, Wilson intervals, machine strata and paired deltas."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from common import proportion, agreement

SCORES = ("citation_correct", "grounded", "hallucination", "completeness")


def read_scores(path, mapping):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = row["review_id"]
            if rid not in mapping or rid in result:
                raise ValueError("Unknown or duplicate review_id")
            if row["question"] != mapping[rid]["question"] or row["answer"] != mapping[rid]["answer"]:
                raise ValueError("Question/answer edited; only fill scoring columns")
            scores = {}
            for key in SCORES:
                value = row[key].strip()
                if value not in ("", "0", "1", "NA"):
                    raise ValueError(f"{key}: use 0, 1, NA or blank")
                scores[key] = int(value) if value in ("0", "1") else None
            result[rid] = {**scores, "notes": row.get("notes", "")}
    return result


def summarize(records):
    metrics = (*SCORES, "route_correct", "retrieval_hit")
    return {"rows": len(records), **{key: proportion([r.get(key) for r in records]) for key in metrics}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--second-review", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    scores = read_scores(args.review, mapping)
    rows = [{**r, **scores.get(rid, {}), "review_id": rid} for rid, r in mapping.items()]
    variants = sorted({r["variant"] for r in rows})
    report = {"scope": "complete successful pairs in mapping; see run audit for all failures",
              "human_scored_cells": sum(s[k] is not None for s in scores.values() for k in SCORES),
              "overall": {}, "by_category": {}, "by_paper": {}, "paired_deltas": [], "errors": []}
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        report["overall"][variant] = summarize(subset)
        for field, target in (("expected_category", "by_category"), ("thread_id", "by_paper")):
            groups = defaultdict(list)
            for r in subset:
                groups[r.get(field, "unknown")].append(r)
            report[target][variant] = {key: summarize(group) for key, group in groups.items()}
    for variant in variants[1:]:
        base = {r["id"]: r for r in rows if r["variant"] == variants[0]}
        other = {r["id"]: r for r in rows if r["variant"] == variant}
        for metric in (*SCORES, "route_correct", "retrieval_hit"):
            pairs = [(base[cid].get(metric), other[cid].get(metric)) for cid in base.keys() & other.keys()]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            report["paired_deltas"].append({"baseline": variants[0], "variant": variant, "metric": metric, "paired_n": len(pairs),
                "difference_percentage_points": 100*sum(b-a for a,b in pairs)/len(pairs) if pairs else None})
    for r in rows:
        labels = []
        if r.get("route_correct") is False:
            labels.append("routing_mismatch")
        if r.get("section_hit") is False:
            labels.append("section_miss")
        if r.get("term_hit") is False:
            labels.append("term_absent_from_evidence")
        if r.get("hallucination") == 1:
            labels.append("human_reported_hallucination")
        if r.get("citation_correct") == 0:
            labels.append("human_reported_citation_error")
        if labels or r.get("notes"):
            report["errors"].append({"id": r["id"], "variant": r["variant"], "labels": labels, "notes": r.get("notes", "")})
    if args.second_review:
        second = read_scores(args.second_review, mapping)
        report["inter_rater"] = {key: agreement([(scores[rid][key], second[rid][key]) for rid in scores.keys() & second.keys() if scores[rid][key] is not None and second[rid][key] is not None]) for key in SCORES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({"variants": variants, "human_scored_cells": report["human_scored_cells"]}))


if __name__ == "__main__":
    main()
