"""Freeze local paper snapshots, then run one user-triggered 3-call comparison.

Prepare is offline. No automatic retries; one explicit fresh recovery is allowed.
"""
import argparse
import hashlib
import html
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_research import build_full_paper_context, run_adaptive_research


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def save_json(path, value):
    """Atomic checkpoint; a failed write must prevent the next paid call."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def recovery_output(output, case_id):
    """One explicit fresh retry, never delete or overwrite the previous attempt."""
    if (output / "recovery.json").exists():
        raise ValueError("Nested recovery is forbidden; this pilot permits only one fresh retry")
    previous = output / f"{case_id}_attempt.json"
    if not previous.is_file():
        raise ValueError("No previous attempt to recover")
    record = json.loads(previous.read_text(encoding="utf-8"))
    if record.get("status") == "complete" or (output / f"{case_id}_summary.json").is_file():
        raise ValueError("Completed attempts cannot be retried here")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if digest(manifest["cases"]) != manifest["cases_sha256"]:
        raise ValueError("Frozen case manifest changed")
    retry = output / f"{case_id}_retry_1"
    # Exclusive creation also blocks concurrent recovery and a second retry.
    retry.mkdir()
    save_json(retry / "manifest.json", manifest)
    save_json(retry / "recovery.json", {
        "created_at": utc_now(), "previous_attempt": str(previous),
        "previous_record": record, "previous_billing": "unknown_unless_provider_confirmed",
        "new_max_calls": 3, "is_fresh_retry_not_resume": True,
    })
    return retry


def read_rows(db, thread_id):
    prefix = thread_id + "_"
    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        found = conn.execute("SELECT e.embedding_id,m.key,m.string_value,m.int_value FROM embeddings e "
                             "JOIN embedding_metadata m ON e.id=m.id WHERE substr(e.embedding_id,1,?)=?",
                             (len(prefix), prefix)).fetchall()
    finally:
        conn.close()
    records = {}
    for cid, key, text, integer in found:
        records.setdefault(cid, {})[key] = text if text is not None else integer
    rows = []
    for cid, meta in records.items():
        if "chroma:document" not in meta:
            continue
        rows.append({"id": cid, "content": meta["chroma:document"],
                     "section": meta.get("section", "unknown"), "page_start": meta.get("page_start", 0),
                     "page_end": meta.get("page_end", meta.get("page_start", 0)),
                     "_order": meta.get("_order", int(cid.rsplit("_", 1)[-1]))})
    if not rows:
        raise ValueError("Paper snapshot has no chunks")
    return sorted(rows, key=lambda row: (row["page_start"] or 0, row["_order"], row["id"]))


def prepare(cases_path, db, output):
    from dotenv import load_dotenv
    from research_model_settings import resolve_research_model
    load_dotenv(ROOT / ".env")
    settings = resolve_research_model(require_key=False)
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    if len(cases) != 2 or {row["id"] for row in cases} != {"sundial", "autoformer"}:
        raise ValueError("This pilot must have exactly the two approved cases")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ValueError("Manifest already exists; do not overwrite the frozen experiment")
    output.mkdir(parents=True, exist_ok=True)
    frozen = []
    for row in cases:
        rows = read_rows(db, row["thread_id"])
        context, _ = build_full_paper_context(rows, max_chars=100000)
        frozen.append({"id": row["id"], "thread_id": row["thread_id"], "question": row["question"],
                       "question_sha256": digest(row["question"]), "rows": rows, "source_sha256": digest(rows),
                       "context_chars": len(context)})
    manifest = {"cases": frozen, "cases_sha256": digest(frozen), "network_calls_prepare": 0,
                "history_mode": "fresh", "depth": "advanced", "strategies": ["direct", "structured"],
                "max_model_calls_total": 6, "max_model_calls_per_case": 3,
                "model": settings.name, "provider_host": settings.host,
                "retries": 0, "writer_output_limit": 3600}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "prepared", "network_calls": 0, "case_count": 2,
            "paper_chars": {row["id"]: row["context_chars"] for row in frozen}, "max_paid_model_calls": 6}


def report_html(case, results):
    parts = ["<!doctype html><meta charset='utf-8'><title>论文讲解小样本对照</title>",
             "<style>body{font-family:system-ui;max-width:1400px;margin:30px auto;padding:15px}"
             ".grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{border:1px solid #ddd;padding:20px}"
             "pre{white-space:pre-wrap;line-height:1.7}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>",
             f"<h1>{html.escape(case['id'])}：同源文、无历史、相同深度</h1>",
             f"<p>{html.escape(case['question'])}</p><p>没有自动评分。请比较核心方法、事实错误和新增退化。</p><div class='grid'>"]
    for variant, result in results.items():
        parts.append(f"<section class='card'><h2>{variant}</h2><p>状态 {html.escape(result['status'])}；模型调用 {result['model_calls']}；"
                     f"输入 {result['input_tokens']} / 输出 {result['output_tokens']} Token；{result['elapsed_ms']} ms</p>"
                     f"<pre>{html.escape(result['answer'])}</pre></section>")
        if variant == "structured":
            parts.append(f"<p>资料表实际使用：{bool(result.get('reading_brief_used'))}；"
                         f"降级原因：{html.escape(str(result.get('structure_fallback_reason') or '无'))}</p>")
    if "structured" in results:
        result = results["structured"]
        trace_path = Path("traces") / f"{result['request_id']}.json"
        parts.append(f"<p>资料表、检查过程和降级原因：<a href='{html.escape(trace_path.as_posix())}'>打开新版Trace</a></p>")
    parts.append("</div>")
    return "".join(parts)


def run(output, case_id, confirmed, *, model_factory=None):
    if not confirmed:
        raise ValueError("Run needs --confirm-paid; maximum 3 paid calls for this paper")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if digest(manifest["cases"]) != manifest["cases_sha256"]:
        raise ValueError("Frozen case manifest changed")
    case = next(row for row in manifest["cases"] if row["id"] == case_id)
    if digest(case["rows"]) != case["source_sha256"] or digest(case["question"]) != case["question_sha256"]:
        raise ValueError("Source/question fingerprint mismatch")
    attempt = output / f"{case_id}_attempt.json"
    if attempt.exists():
        raise ValueError("This case was already attempted; automatic repeats are forbidden")
    journal = {"case": case_id, "max_calls": 3, "status": "started",
               "started_at": utc_now(), "pid": os.getpid(), "attempted_calls": 0,
               "completed_variants": [], "calls": []}
    with attempt.open("x", encoding="utf-8") as handle:
        json.dump(journal, handle)

    def checkpoint():
        journal["updated_at"] = utc_now()
        save_json(attempt, journal)

    try:
        from deep_research_runtime import AliyunResearchModel
        model = (model_factory or AliyunResearchModel)()
        endpoint = urlparse(model._base).hostname
        if model_factory is None and (model._settings.name != manifest["model"] or
                manifest.get("provider_host") not in {None, endpoint}):
            raise ValueError("Frozen pilot model differs from current provider; prepare a new pilot before paid runs")
        print(json.dumps({"paper": case_id, "provider_host": endpoint, "model": manifest["model"],
                          "max_paid_model_calls": 3, "history": "fresh", "whole_paper_sent": True}), flush=True)
    except BaseException as exc:
        journal.update(status="failed", error_type=type(exc).__name__, failed_phase="model_initialization")
        checkpoint()
        raise

    class BudgetedModel:
        calls = 0
        def describe_call(self, stage, payload, **limits):
            describe = getattr(model, "describe_call", None) or AliyunResearchModel.describe_request
            return describe(stage, payload, **limits)
        def call(self, stage, payload, **limits):
            if self.calls >= 3:
                raise RuntimeError("pilot_call_budget")
            self.calls += 1
            call = {"number": self.calls, "stage": stage, "status": "dispatch_pending",
                    "started_at": utc_now(), "billing": "unknown"}
            journal["attempted_calls"] = self.calls
            journal["calls"].append(call)
            checkpoint()  # Persist BEFORE dispatch; interrupted calls may have been billed.
            print(json.dumps({"event": "call_started", "call": self.calls, "stage": stage}), flush=True)
            try:
                response = model.call(stage, payload, **limits)
                save_json(output / f"{case_id}_call_{self.calls}_response.json", response)
            except BaseException as exc:
                call.update(status="failed_or_interrupted", error_type=type(exc).__name__, ended_at=utc_now())
                call.update(AliyunResearchModel.failure_metadata(exc))
                checkpoint()
                raise
            call.update(status="returned", ended_at=utc_now(),
                        usage=response.get("usage") if isinstance(response, dict) else None,
                        finish_reason=response.get("finish_reason") if isinstance(response, dict) else None)
            checkpoint()
            print(json.dumps({"event": "call_returned", "call": self.calls, "stage": stage}), flush=True)
            return response

    bounded = BudgetedModel()
    results = {}
    try:
        for variant in ("direct", "structured"):
            journal["phase"] = variant
            checkpoint()
            result = run_adaptive_research(case["question"], case["thread_id"], case["rows"], None, bounded,
                                          strategy="deep", answer_depth="advanced", history_mode="fresh",
                                          explanation_pipeline=variant, enable_quality_review=False,
                                          trace_dir=output / "traces")
            results[variant] = result
            save_json(output / f"{case_id}_{variant}.json", result)
            journal["completed_variants"].append(variant)
            checkpoint()
            print(json.dumps({"case": case_id, "variant": variant, "status": result["status"],
                              "model_calls": result["model_calls"], "input_tokens": result["input_tokens"],
                              "output_tokens": result["output_tokens"], "elapsed_ms": result["elapsed_ms"],
                              "stop_reason": result.get("stop_reason"),
                              "reading_brief_used": result.get("reading_brief_used", False),
                              "reading_brief_completeness": result.get("reading_brief_completeness"),
                              "structure_fallback_reason": result.get("structure_fallback_reason")}), flush=True)
            if result["status"] in {"stopped", "error"}:
                journal["stop_reason"] = result.get("stop_reason")
                break  # Do not spend on the next variant after an execution failure.
        (output / f"{case_id}_comparison.html").write_text(report_html(case, results), encoding="utf-8")
        status = "complete" if len(results) == 2 and all(r["status"] not in {"stopped", "error"} for r in results.values()) else "incomplete"
        summary = {"status": status, "case": case_id, "attempted_calls": bounded.calls,
                   "quality_improvement": "not_yet_reviewed", "automatic_rewrites": 0,
                   "structured_effective": bool(results.get("structured", {}).get("reading_brief_used")),
                   "reading_brief_completeness": results.get("structured", {}).get("reading_brief_completeness"),
                   "structure_fallback_reason": results.get("structured", {}).get("structure_fallback_reason")}
        save_json(output / f"{case_id}_summary.json", summary)
        journal["status"] = status
        checkpoint()
        return summary
    except BaseException as exc:
        journal.update(status="failed", error_type=type(exc).__name__)
        checkpoint()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cases")
    parser.add_argument("--chroma-db", default="resources/chroma_db/chroma.sqlite3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", choices=["sundial", "autoformer"])
    parser.add_argument("--confirm-paid", action="store_true")
    parser.add_argument("--retry-interrupted", action="store_true",
                        help="Confirm the previous process has stopped; preserve it and allow ONE fresh retry (up to 3 new paid calls)")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    private_root = (ROOT / "benchmark/experiments/private").resolve()
    if not output.is_relative_to(private_root):
        parser.error("Output must stay under benchmark/experiments/private")
    if args.prepare:
        if args.retry_interrupted:
            parser.error("Recovery is only valid for run")
        if not args.cases:
            parser.error("Prepare needs --cases")
        result = prepare(args.cases, args.chroma_db, output)
    else:
        if not args.case:
            parser.error("Run needs --case")
        if args.retry_interrupted:
            if not args.confirm_paid:
                parser.error("Recovery needs --confirm-paid for up to 3 NEW calls; old billing may be unknown")
            output = recovery_output(output, args.case)
            print(json.dumps({"recovery_output": str(output), "old_billing": "unknown_unless_provider_confirmed",
                              "new_max_paid_calls": 3}), flush=True)
        result = run(output, args.case, args.confirm_paid)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
