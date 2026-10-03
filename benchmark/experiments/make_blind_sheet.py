"""Create an unscored, shuffled paired review sheet; mapping stays private."""
import argparse
import csv
import json
from pathlib import Path
import secrets
from common import read_jsonl

FIELDS = ["review_id", "question", "answer", "reference_key_points", "citation_correct", "grounded", "hallucination", "completeness", "notes"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("answers", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records = [r for path in args.answers for r in read_jsonl(path)]
    keys = [(r["variant"], r["id"]) for r in records]
    if len(keys) != len(set(keys)):
        p.error("Duplicate variant/case; do not merge reruns")
    variants = {r["variant"] for r in records}
    if len(variants) < 2:
        p.error("Paired review requires at least two variants")
    eligible = {r["id"] for r in records if all(any(s["id"] == r["id"] and s["variant"] == v and not s.get("error") and s.get("answer") for s in records) for v in variants)}
    selected = [r for r in records if r["id"] in eligible]
    if not selected:
        p.error("No complete successful pairs to review")
    for cid in eligible:
        paired = [r for r in selected if r["id"] == cid]
        if len({(r["question"], r.get("thread_id")) for r in paired}) != 1:
            p.error("Case id refers to different questions or corpora")
    secrets.SystemRandom().shuffle(selected)
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = {}
    with (args.output/"review.csv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in selected:
            rid = secrets.token_hex(8)
            mapping[rid] = r
            gold = r.get("reference_key_points", r.get("gold", ""))
            if not isinstance(gold, str):
                gold = json.dumps(gold, ensure_ascii=False)
            writer.writerow(dict(review_id=rid, question=r["question"], answer=r["answer"], reference_key_points=gold))
    (args.output/"mapping.private.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"review_rows": len(selected), "paired_cases": len(eligible), "excluded_rows": len(records)-len(selected)}))


if __name__ == "__main__":
    main()
