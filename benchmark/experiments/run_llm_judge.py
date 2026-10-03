"""Run a blinded, evidence-bounded LLM-as-a-Judge evaluation.

The model never sees variant names or mapping.private.json.  Results are AI
judgements, not human scores, and are written to a separate output directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.experiments.common import digest


METRICS = ("citation_correct", "grounded", "hallucination", "completeness")
SYSTEM_PROMPT = """You are a strict evaluator of answers about research papers.
You are blind to system identity. Evaluate only from the supplied question,
reference key points, and candidate source excerpt.

Important evidence rule: the candidate excerpt may be incomplete. Absence from
the excerpt is not proof that a claim is false or hallucinated. Use null (NA)
whenever the supplied evidence is insufficient. Do not use your memorized
knowledge of the paper.

Rubric:
- citation_correct=1: citations/source attributions in the answer are supported;
  0: a required citation is missing or a cited source contradicts/does not support
  the claim; null: no paper claim to cite or evidence cannot verify the citation.
- grounded=1: all assessable paper claims are supported; 0: at least one claim is
  clearly unsupported or contradicted; null: source packet is insufficient.
- hallucination=1: at least one clearly invented or contradicted factual claim;
  0: no hallucination is found within the supplied evidence; null: cannot tell.
  Lower is better for this metric.
- completeness=1: every supplied reference key point is covered; 0: at least one
  supplied key point is missing; null: no usable reference key points.

Return one JSON object only, with exactly these keys:
{"citation_correct":0|1|null,"grounded":0|1|null,
 "hallucination":0|1|null,"completeness":0|1|null,
 "confidence":"low"|"medium"|"high","insufficient_evidence":true|false,
 "reason":"concise evidence-based explanation"}
"""


def parse_judgement(text):
    content = text.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1].removeprefix("json").strip() if len(parts) > 1 else content
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge did not return a JSON object")
    data = json.loads(content[start:end + 1])
    for metric in METRICS:
        value = data.get(metric)
        if value not in (0, 1, None):
            raise ValueError(f"{metric} must be 0, 1 or null")
    if data.get("confidence") not in ("low", "medium", "high"):
        raise ValueError("confidence must be low, medium or high")
    if not isinstance(data.get("insufficient_evidence"), bool):
        raise ValueError("insufficient_evidence must be boolean")
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a nonempty string")
    return {metric: data.get(metric) for metric in METRICS} | {
        "confidence": data["confidence"],
        "insufficient_evidence": data["insufficient_evidence"],
        "reason": reason.strip()[:1000],
    }


def user_packet(row):
    return "\n\n".join([
        f"[review_id]\n{row['review_id']}",
        f"[question]\n{row['question']}",
        f"[answer]\n{row['answer']}",
        f"[reference_key_points]\n{row['reference_key_points'] or '(none)'}",
        f"[candidate_source_excerpt]\n{row['candidate_excerpt'] or '(none)'}",
        f"[source_metadata]\nexpected_section={row.get('expected_section','')}; "
        f"pdf_pages={row.get('pdf_pages','')}",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-packets", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.passes <= 0:
        parser.error("--passes must be positive")
    with args.review_packets.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"review_id", "question", "answer", "reference_key_points", "candidate_excerpt"}
    if not rows or not required.issubset(rows[0]):
        parser.error("review packet CSV is empty or missing required columns")
    if len({row["review_id"] for row in rows}) != len(rows):
        parser.error("review_id values must be unique")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        rows = rows[:args.limit]
    elif not args.confirm_full:
        parser.error("use --limit for a paid smoke test or --confirm-full for all rows")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    from config import model
    from langchain_core.messages import HumanMessage, SystemMessage

    judge = model.bind(
        temperature=0,
        max_tokens=700,
        extra_body={"enable_thinking": False},
    )
    manifest = {
        "kind": "blinded_llm_as_a_judge_not_human_review",
        "input_sha256": digest(args.review_packets),
        "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "rows": len(rows),
        "passes": args.passes,
        "mapping_opened": False,
        "limitations": [
            "Reference key points were AI-reviewed, not approved by a human expert.",
            "Candidate excerpts can be incomplete, so insufficient cases must be NA.",
            "Repeated passes of one model measure stability, not independent human agreement.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_path = args.output / "judgements.jsonl"
    with output_path.open("x", encoding="utf-8") as output:
        for pass_index in range(1, args.passes + 1):
            ordered = rows if pass_index % 2 else list(reversed(rows))
            for row in ordered:
                record = {"review_id": row["review_id"], "pass": pass_index, "error": None}
                try:
                    response = judge.invoke([
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=user_packet(row)),
                    ])
                    content = response.content if isinstance(response.content, str) else str(response.content)
                    record.update(parse_judgement(content))
                    usage = getattr(response, "usage_metadata", None)
                    record["usage"] = usage if isinstance(usage, dict) else None
                except Exception as exc:
                    record["error"] = {"type": type(exc).__name__}
                    record.update({metric: None for metric in METRICS})
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps({
                    "review_id": row["review_id"],
                    "pass": pass_index,
                    "error": record["error"],
                    "confidence": record.get("confidence"),
                }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
