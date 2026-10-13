"""Offline replay of a saved final response; original experiment is immutable."""
import argparse
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deep_research import _prepare_final_contract, _render_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, target = Path(args.experiment).resolve(), Path(args.output).resolve()
    private = Path(__file__).resolve().parent / "private"
    if not target.is_relative_to(private.resolve()):
        raise ValueError("Replay output must remain private")
    target.mkdir(parents=True, exist_ok=False)
    raw = json.loads((source / "tw-01_adaptive_call_1.json").read_text(encoding="utf-8"))
    original = json.loads((source / "tw-01_adaptive_result.json").read_text(encoding="utf-8"))
    trace = json.loads((source / "traces" / (original["request_id"] + ".json")).read_text(encoding="utf-8"))
    evidence = next(e["provided_segments"] for e in trace["events"] if e["step"] == "full_context")
    seen = {r["id"]: r for r in evidence}
    replay = {"events": []}
    final, coverage, caveats, structured = _prepare_final_contract(
        raw["data"], [{"id": "q1", "text": trace["question"]}], seen, True, replay)
    answer, claims, unresolved = _render_final(final, seen, True)
    report = {"offline_replay_only": True, "network_calls": 0, "model_calls": 0,
              "original_status": original["status"], "original_stop_reason": original["stop_reason"],
              "new_status": "partial" if replay.get("answer_degradation") else "rendered",
              "semantic_accuracy_proven": False, "claims": claims, "answer": answer,
              "answer_degradation": replay.get("answer_degradation")}
    (target / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "answer.html").write_text('<meta charset="utf-8"><h1>Informer 离线显示修复回放</h1>'
        '<p>不是新模型回答；原始实验保持不变。来源身份校验不等于事实正确。</p>'
        '<pre style="white-space:pre-wrap;line-height:1.8;max-width:1000px">' + html.escape(answer) + '</pre>', encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("offline_replay_only", "network_calls", "model_calls", "new_status")}, ensure_ascii=False))
    print(json.dumps(report["answer_degradation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
