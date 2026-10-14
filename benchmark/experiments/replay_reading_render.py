"""Restore display from archived pilot responses; no model calls or rewrites."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_research import build_full_paper_context
from deep_research import _render_final
from paper_structure import validate_structure
from benchmark.experiments.run_reading_pilot import report_html, save_json


def replay(source, output, case_id):
    private = (ROOT / "benchmark/experiments/private").resolve()
    if not source.is_relative_to(private) or not output.is_relative_to(private):
        raise ValueError("Input/output must stay private")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    case = next(row for row in manifest["cases"] if row["id"] == case_id)
    journal = json.loads((source / f"{case_id}_attempt.json").read_text(encoding="utf-8"))
    _, evidence = build_full_paper_context(case["rows"], max_chars=100000)
    seen = {row["id"]: row for row in evidence}
    results, checks = {}, {"replay_only": True, "network_calls": 0, "rendering": {}}
    for variant, stage in (("direct", "full_context"), ("structured", "structured_context")):
        # Select the returned writer from the archived journal, not fixed call numbers.
        candidates = [row for row in journal["calls"] if row["stage"] == stage and row["status"] == "returned"]
        if not candidates:
            raise ValueError(f"No completed {stage} response; this tool does not create one")
        raw = json.loads((source / f"{case_id}_call_{candidates[-1]['number']}_response.json").read_text(encoding="utf-8"))
        original = json.loads((source / f"{case_id}_{variant}.json").read_text(encoding="utf-8"))
        result = copy.deepcopy(original)
        result["answer"], result["claims"], result["unresolved"] = _render_final(raw["data"], seen, True)
        result["status"] = "historical_render_replay"
        checks["rendering"][variant] = {"original_displayed": len(original["claims"]),
                                        "replayed_displayed": len(result["claims"])}
        results[variant] = result
    structured = results["structured"]
    trace = json.loads((source / "traces" / f"{structured['request_id']}.json").read_text(encoding="utf-8"))
    extraction = next(row for row in journal["calls"] if row["stage"] == "source_structure")
    raw = json.loads((source / f"{case_id}_call_{extraction['number']}_response.json").read_text(encoding="utf-8"))
    _, checks["structure"] = validate_structure(raw["data"], evidence, trace["structure_source_aliases"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "checks.json", checks)
    for variant, result in results.items():
        save_json(output / f"{case_id}_{variant}_replayed.json", result)
    content = report_html(case, results)
    content = content.replace("href='traces/", f"href='{(source / 'traces').as_uri()}/")
    content = content.replace("<h1>", "<p>历史响应的离线展示修复：没有重新调用模型；原状态/用量是历史记录，不代表新流程验收。原资料表在新门禁下不合格。页面 Trace 链接请查看原试跑目录。</p><h1>", 1)
    (output / f"{case_id}_comparison_replayed.html").write_text(content, encoding="utf-8")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    result = replay(Path(args.input).resolve(), Path(args.output).resolve(), args.case)
    print(json.dumps({"replay_only": True, "network_calls": 0, "rendering": result["rendering"],
                      "structure_completeness": result["structure"].get("completeness")}, ensure_ascii=False))
