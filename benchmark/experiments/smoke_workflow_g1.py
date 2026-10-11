"""Run one synthetic G1 request and write a reviewable trace, without network calls."""

import argparse
import json
from pathlib import Path

from deep_research import run_deep_research


class FakeRetriever:
    def has_paper(self, thread_id):
        return thread_id == "demo"

    def search(self, thread_id, query, *, k, exclude_ids, timeout_seconds):
        rows = [
            {"id": "demo_1", "content": "SYNTHETIC: this page describes the forecasting method. " * 8,
             "section": "method", "page_start": 3, "page_end": 3},
            {"id": "demo_2", "content": "SYNTHETIC: this page reports the experiment result. " * 8,
             "section": "results", "page_start": 7, "page_end": 7},
        ]
        return [row for row in rows if row["id"] not in exclude_ids][:1]


class FakeModel:
    def call(self, stage, payload, *, max_output_tokens, timeout_seconds):
        outputs = {
            "plan": {"subquestions": [{"id": "q1", "text": "What method and result were reported?",
                                           "query": "forecasting method"}]},
            "assess": {"subquestions": [{"subquestion_id": "q1", "assessment": "missing_specific_fact",
                                          "evidence_ids": ["demo_1"],
                                          "reason": "The first excerpt contains the method, but no result.",
                                          "missing_information": "experiment result"}],
                       "action": {"type": "search", "query": "experiment result"}},
            "final": {"items": [{"kind": "paper_fact", "text": "The synthetic result is on page 7.",
                                 "evidence_ids": ["demo_2"]}], "unresolved": []},
        }
        result = outputs[stage]
        return {"data": result, "raw": json.dumps(result),
                "usage": {"input_tokens": 12, "output_tokens": 8}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/experiments/private/workflow_g1_smoke"))
    args = parser.parse_args()
    result = run_deep_research(
        "Explain the synthetic method and experiment result.", "demo", FakeRetriever(),
        FakeModel(), trace_dir=args.output,
    )
    if result["status"] != "ok":
        raise SystemExit(f"Synthetic workflow failed: {result['stop_reason']}")
    print(json.dumps({"synthetic_only": True, "network_calls": 0,
                      "request_id": result["request_id"], "status": result["status"],
                      "actions": result["actions"], "model_calls": result["model_calls"],
                      "stop_reason": result["stop_reason"], "answer": result["answer"],
                      "trace_file": str(args.output / f"{result['request_id']}.json"),
                      "warning": "Synthetic control-flow check; not a paper-quality metric."},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
