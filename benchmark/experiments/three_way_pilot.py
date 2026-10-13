"""Prepare offline; human-approved batches only. No silent paid reruns."""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.experiments.three_way_cases import CASES, TITLE_CUES
from benchmark.experiments.run_reading_pilot import read_rows, digest, save_json
from reading_limits import FULL_PAPER_OUTPUT_LIMITS

VARIANTS = ("legacy_v1", "adaptive", "direct_full")
FROZEN_FILES = ("app.py", "rag_store.py", "query_router.py", "prompts.py", "agent_setup.py")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT / "benchmark/experiments/private").resolve()):
        raise ValueError("Output must stay in the private experiment directory")
    return path


def model_settings():
    from dotenv import load_dotenv
    from research_model_settings import resolve_research_model
    load_dotenv(ROOT / ".env")
    return resolve_research_model(require_key=False)


def prepare(source, output, frozen_ref, paper_map=None):
    from adaptive_research import build_full_paper_context, select_research_mode
    settings = model_settings()
    output = private(output)
    output.mkdir(parents=True, exist_ok=False)
    db = source / "chroma.sqlite3"
    # Backup SQLite transactionally; index files are copied privately. Pause
    # imports during prepare and verify row fingerprints again before dispatch.
    shutil.copytree(source, output / "chroma", ignore=shutil.ignore_patterns("*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm"))
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        target = sqlite3.connect(output / "chroma/chroma.sqlite3")
        try:
            connection.backup(target)
        finally:
            target.close()
        titles = connection.execute("SELECT e.embedding_id,m.string_value FROM embeddings e JOIN embedding_metadata m ON e.id=m.id WHERE m.key='title'").fetchall()
    finally:
        connection.close()
    mapping = {}
    for alias, cue in TITLE_CUES.items():
        if paper_map and alias in paper_map:
            read_rows(output / "chroma/chroma.sqlite3", paper_map[alias])
            mapping[alias] = paper_map[alias]
            continue
        candidates = {cid.rsplit("_", 1)[0] for cid, title in titles if cue in (title or "").lower()}
        if len(candidates) != 1:
            raise ValueError(f"Paper mapping missing or ambiguous: {alias}; do not guess")
        mapping[alias] = candidates.pop()
    cases = []
    for draft in CASES:
        rows = read_rows(output / "chroma/chroma.sqlite3", mapping[draft["paper"]])
        try:
            context, _ = build_full_paper_context(rows, max_chars=100000)
            eligible, reason = True, None
        except ValueError:
            context, eligible, reason = "", False, "full_context_source_limit"
        cases.append({**draft, "thread_id": mapping[draft["paper"]], "rows": rows,
                      "source_sha256": digest(rows), "full_eligible": eligible,
                      "full_context_chars": len(context), "full_ineligible_reason": reason,
                      "auto_mode_preflight": select_research_mode([rows], question=draft["question"], strategy="auto").mode})
    commit = subprocess.check_output(["git", "rev-parse", frozen_ref], cwd=ROOT, text=True).strip()
    frozen = {name: subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT).decode("utf-8") for name in FROZEN_FILES}
    save_json(output / "legacy_sources.json", frozen)
    code_files = ["adaptive_research.py", "deep_research.py", "deep_research_runtime.py", "research_model_settings.py",
                  "provider_chat_model.py", "partial_output.py", "reading_limits.py", "rag_store.py", "paper_structure.py",
                  "quality_review.py", "benchmark/experiments/three_way_pilot.py", "benchmark/experiments/three_way_legacy.py"]
    manifest = {"schema": 1, "cases": cases, "cases_sha256": digest(cases), "legacy_commit": commit,
                "legacy_sources_sha256": file_hash(output / "legacy_sources.json"),
                "code_hashes": {name: file_hash(ROOT / name) for name in code_files},
                "model": settings.name, "provider_host": settings.host,
                "base_url_sha256": digest(settings.base_url),
                "embedding_base_sha256": digest(os.getenv("DASHSCOPE_BASE_URL", "")),
                "embedding_model": "qwen3.7-text-embedding-flash",
                "writer_output_limit": FULL_PAPER_OUTPUT_LIMITS["advanced"], "history": "fresh",
                "review_enabled": False, "structured_enabled": False,
                "max_model_calls_per_case": 6, "max_embedding_requests_per_case": 8,
                "prepared_network_calls": 0, "is_expert_gold": False, "is_holdout": False}
    save_json(output / "manifest.json", manifest)
    rows = "\n".join(f"|{c['id']}|{c['batch']}|{c['paper']}|{c['question']}|{c['review_focus']}|{c['full_eligible']}|" for c in cases)
    (output / "questions.md").write_text("# 待确认八题\n\n不是专家金标准。full_eligible=false时全文组明确不适用，不偷偷截全文。\n\n|ID|批次|论文|问题|人工核查重点|全文可用|\n|---|---|---|---|---|---|\n" + rows, encoding="utf-8")
    return {"prepared": True, "network_calls": 0, "cases": len(cases), "first_batch_answers": 12,
            "first_batch_max_model_calls": 24, "first_batch_max_embedding_requests": 32,
            "full_ineligible": [c["id"] for c in cases if not c["full_eligible"]], "awaits_human_confirmation": True}


def checked_manifest(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if digest(manifest["cases"]) != manifest["cases_sha256"]:
        raise ValueError("Frozen questions/sources changed")
    for name, expected in manifest["code_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Code changed after prepare: {name}; prepare a new experiment")
    if file_hash(output / "legacy_sources.json") != manifest["legacy_sources_sha256"]:
        raise ValueError("Frozen V1 sources changed")
    if manifest["writer_output_limit"] != FULL_PAPER_OUTPUT_LIMITS["advanced"]:
        raise ValueError("Output budget drift")
    return manifest


def approve(output):
    checked_manifest(output)
    target = output / "approval.json"
    if target.exists():
        raise ValueError("Already approved")
    save_json(target, {"manifest_sha256": file_hash(output / "manifest.json"), "questions_and_workload_approved": True})
    return {"approved": True, "network_calls": 0}


class RecordingModel:
    def __init__(self, output, case_id, variant, max_calls):
        from deep_research_runtime import AliyunResearchModel
        self.base = AliyunResearchModel()
        self.output, self.case_id, self.variant, self.max_calls = output, case_id, variant, max_calls
        self.calls, self.failed = [], False
        self.options = {}

    def bind(self, **options):
        # Frozen router thinking flags are the ONLY provider adaptation; route
        # prompt and retrieval logic stay frozen.
        self.options = {k: v for k, v in options.items() if k != "extra_body"}
        return self

    def describe_call(self, stage, payload, **limits):
        return self.base.describe_call(stage, payload, **limits)

    def dispatch(self, stage, function):
        if self.failed or len(self.calls) >= self.max_calls:
            raise RuntimeError("Model call gate blocked")
        record = {"stage": stage, "state": "dispatch_pending", "billing": "unknown"}
        self.calls.append(record)
        path = self.output / f"{self.case_id}_{self.variant}_calls.json"
        save_json(path, self.calls)
        try:
            result = function()
            record.update(state="returned", usage=result.get("usage"), finish_reason=result.get("finish_reason"))
            save_json(self.output / f"{self.case_id}_{self.variant}_call_{len(self.calls)}.json", result)
            return result
        except Exception as exc:
            self.failed = True
            from deep_research_runtime import AliyunResearchModel
            record.update(state="failed", error_type=type(exc).__name__, **AliyunResearchModel.failure_metadata(exc))
            raise
        finally:
            save_json(path, self.calls)

    def call(self, stage, payload, **limits):
        return self.dispatch(stage, lambda: self.base.call(stage, payload, **limits))

    def invoke(self, messages):
        from research_model_settings import build_chat_model
        from deep_research_runtime import AliyunResearchModel
        cap = self.options.pop("max_tokens", 6400)
        self.options = {}
        saved = []
        def request():
            answer = build_chat_model(settings=self.base._settings, max_output_tokens=cap, timeout_seconds=60).invoke(messages)
            saved.append(answer)
            return {"raw": answer.content, "usage": AliyunResearchModel._extract_usage(answer.usage_metadata or {}, answer.response_metadata or {}),
                    "finish_reason": (answer.response_metadata or {}).get("finish_reason"), "output_cap": cap}
        self.dispatch("routing" if cap == 50 else "legacy_generation", request)
        return saved[0]


def run_batch(output, batch, confirmed):
    if not confirmed:
        raise ValueError("Needs --confirm-paid; up to 24 model calls and 32 embedding requests for this batch")
    manifest = checked_manifest(output)
    approved = json.loads((output / "approval.json").read_text(encoding="utf-8"))
    if approved.get("manifest_sha256") != file_hash(output / "manifest.json"):
        raise ValueError("Human approval does not match manifest")
    settings = model_settings()
    if settings.name != manifest["model"] or settings.host != manifest["provider_host"] or digest(settings.base_url) != manifest["base_url_sha256"]:
        raise ValueError("Model/provider changed")
    if digest(os.getenv("DASHSCOPE_BASE_URL", "")) != manifest["embedding_base_sha256"]:
        raise ValueError("Embedding provider changed")
    if not os.getenv("DASHSCOPE_API_KEY") or not os.getenv("DASHSCOPE_BASE_URL"):
        raise ValueError("Embedding configuration missing; no paid calls dispatched")
    marker = output / f"batch_{batch}.started"
    with marker.open("x", encoding="utf-8") as handle:
        handle.write("No automatic rerun; interrupted billing may be unknown")
    import requests
    import rag_store
    from unittest.mock import patch
    from deep_research import ChromaPaperRetriever
    from adaptive_research import run_adaptive_research
    from benchmark.experiments.three_way_legacy import legacy_answer, load
    from benchmark.experiments.run_reading_pilot import read_rows
    original_post = requests.post
    sources = json.loads((output / "legacy_sources.json").read_text(encoding="utf-8"))
    for case in [c for c in manifest["cases"] if c["batch"] == batch]:
        if digest(read_rows(output / "chroma/chroma.sqlite3", case["thread_id"])) != case["source_sha256"]:
            raise ValueError("Snapshot drift")
        embedding_calls = []
        def post(url, **kwargs):
            if len(embedding_calls) >= 8:
                raise RuntimeError("Per-case embedding request budget exhausted")
            if url.rstrip("/") != (os.getenv("DASHSCOPE_BASE_URL", "").rstrip("/") + "/embeddings"):
                raise ValueError("Unexpected embedding destination")
            entry = {"state": "dispatch_pending", "model": kwargs.get("json", {}).get("model"), "billing": "unknown"}
            embedding_calls.append(entry)
            save_json(output / f"{case['id']}_embeddings.json", embedding_calls)
            kwargs["timeout"] = 20
            try:
                response = original_post(url, **kwargs)
                entry.update(state="returned", http_status=response.status_code,
                             usage=response.json().get("usage") if response.status_code == 200 else None)
                return response
            finally:
                save_json(output / f"{case['id']}_embeddings.json", embedding_calls)
        # Rotate execution order; an embedding query cache is deliberately not shared.
        start = (int(case["id"][-2:]) - 1) % 3
        order = VARIANTS[start:] + VARIANTS[:start]
        for variant in order:
            model = RecordingModel(output, case["id"], variant, {"legacy_v1": 2, "adaptive": 3, "direct_full": 1}[variant])
            began = time.perf_counter()
            before = len(embedding_calls)
            try:
                with patch("requests.post", post):
                    if variant == "legacy_v1":
                        result = legacy_answer(sources, model, rag_store._embed, output / "chroma", case)
                    elif variant == "direct_full" and not case["full_eligible"]:
                        result = {"status": "not_applicable", "answer": "全文超过既定输入限制，未截断或发送。", "stop_reason": case["full_ineligible_reason"]}
                    else:
                        rag = rag_store.PaperRAG(str(output / "chroma"))
                        result = run_adaptive_research(case["question"], case["thread_id"], case["rows"], ChromaPaperRetriever(rag), model,
                            strategy="auto" if variant == "adaptive" else "deep", answer_depth="advanced", history_mode="fresh",
                            explanation_pipeline="direct", enable_quality_review=False, trace_dir=output / "traces")
                result.update(variant=variant, case_id=case["id"], model_calls=len(model.calls),
                              model_usage=[c.get("usage") for c in model.calls], embedding_requests=len(embedding_calls)-before,
                              wall_ms=round((time.perf_counter()-began)*1000, 2))
            except Exception as exc:
                result = {"variant": variant, "case_id": case["id"], "status": "error", "answer": "本次实验执行失败。",
                          "error_type": type(exc).__name__, "model_calls": len(model.calls),
                          "model_usage": [c.get("usage") for c in model.calls], "embedding_requests": len(embedding_calls)-before}
                if any(c.get("http_status") in {401, 403, 429} for c in model.calls):
                    result["stop_reason"] = "provider_request_rejected"
            save_json(output / f"{case['id']}_{variant}_result.json", result)
            print(json.dumps({k: result.get(k) for k in ("case_id", "variant", "status", "model_calls", "embedding_requests")}), flush=True)
            if result.get("stop_reason") in {"provider_quota_exhausted", "provider_request_rejected"}:
                raise RuntimeError("Provider rejection; stop batch without further spending")
    return render(output, batch)


def render(output, batch, preview=False):
    manifest = checked_manifest(output)
    review = output / f"{'preview' if preview else 'review'}_batch_{batch}"
    review.mkdir(exist_ok=False)
    rng, mapping, sheet, operations = random.Random(20261013 + batch), {}, [], []
    body = ["<!doctype html><meta charset='utf-8'><title>匿名答案核查</title><style>body{font-family:system-ui;max-width:1100px;margin:auto;padding:24px}pre{white-space:pre-wrap;line-height:1.8}article{border:1px solid #ddd;padding:18px;margin:16px 0}</style><h1>匿名答案核查</h1><p>先核对事实与要点，再看流程。缺失、失败和不适用样本不隐藏。每份最多核对两处核心事实，最多24处/批。匿名化不保证文本风格完全无法辨认。</p>"]
    for case in [c for c in manifest["cases"] if c["batch"] == batch]:
        variants = list(VARIANTS)
        rng.shuffle(variants)
        body.append(f"<h2>{case['id']}：{html.escape(case['question'])}</h2>")
        for index, variant in enumerate(variants):
            rid = f"{case['id']}-{chr(65+index)}"
            path = output / f"{case['id']}_{variant}_result.json"
            result = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "not_run", "answer": "尚未运行。"}
            mapping[rid] = variant
            sheet.append({"review_id": rid, "case_id": case["id"], "readable_answer": None, "supported_core_facts": None,
                          "unsupported_claims": None, "core_coverage": None, "uncertain": None, "notes": ""})
            operations.append({"case_id": case["id"], "variant": variant, "metrics": {k: result.get(k) for k in
                               ("status", "stop_reason", "model_calls", "model_usage", "embedding_requests", "wall_ms")}})
            body.append(f"<article><h3>{rid}</h3><pre>{html.escape(str(result.get('answer') or '没有可读输出；需查看执行记录。'))}</pre></article>")
    (review / "answers.html").write_text("\n".join(body), encoding="utf-8")
    save_json(review / "review.json", sheet)
    save_json(review / "mapping.private.json", mapping)
    save_json(review / "operations.private.json", operations)
    return {"review_rows": len(sheet), "batch": batch, "automatic_quality_score": False}


def smoke(output):
    """Execute all three branches with fake providers, frozen V1 definitions."""
    from adaptive_research import run_adaptive_research
    from deep_research import ChromaPaperRetriever
    from benchmark.experiments import three_way_legacy as legacy
    from langchain_core.messages import AIMessage
    from unittest.mock import patch
    manifest = checked_manifest(output)
    sources = json.loads((output / "legacy_sources.json").read_text(encoding="utf-8"))
    destination = output / "synthetic_smoke"
    destination.mkdir(exist_ok=False)
    shutil.copy2(output / "manifest.json", destination / "manifest.json")
    shutil.copy2(output / "legacy_sources.json", destination / "legacy_sources.json")
    real_load = legacy.load
    for case in [c for c in manifest["cases"] if c["batch"] == 1]:
        class FakeRAG:
            def __init__(self, *args):
                pass
            def list_chunks(self, thread):
                return case["rows"]
            def search(self, thread, query, k=8, exclude_ids=None, **kwargs):
                return [row for row in case["rows"] if row["id"] not in (exclude_ids or set())][:k]
        class FakeChat:
            failed = False
            calls = 0
            cap = 6400
            def bind(self, **kwargs):
                self.cap = kwargs.get("max_tokens", 6400)
                return self
            def invoke(self, messages):
                self.calls += 1
                value = '{"category":"A1"}' if self.cap == 50 else "合成答案，仅验证冻结V1流水线，不是模型论文回答。"
                self.cap = 6400
                return AIMessage(content=value)
        def fake_load(*args):
            namespaces = real_load(*args)
            namespaces[0]["PaperRAG"] = FakeRAG
            return namespaces
        chat = FakeChat()
        with patch.object(legacy, "load", side_effect=fake_load):
            old = legacy.legacy_answer(sources, chat, lambda texts: (_ for _ in ()).throw(AssertionError("No real embedding")), destination, case)
        old.update(model_calls=chat.calls)
        class FakeModel:
            calls = 0
            def call(self, stage, payload, **limits):
                self.calls += 1
                if stage == "plan":
                    data = {"subquestions": [{"id": "q1", "text": "合成子问题", "query": "synthetic"}]}
                elif stage == "assess":
                    data = {"subquestions": [{"subquestion_id": "q1", "assessment": "supported", "reason": "合成分支",
                              "missing_information": "", "evidence_ids": [payload["evidence"][0]["id"]]}], "action": {"type": "stop"}}
                else:
                    data = {"items": [{"kind": "background", "text": "合成答案，仅验证执行分支，不是项目质量测评。", "evidence_ids": []}], "unresolved": []}
                return {"data": data, "raw": json.dumps(data), "usage": {"input_tokens": 1, "output_tokens": 1, "cached_tokens": 0}}
        results = {"legacy_v1": old}
        for variant in ("adaptive", "direct_full"):
            model = FakeModel()
            result = run_adaptive_research(case["question"], case["thread_id"], case["rows"], ChromaPaperRetriever(FakeRAG()), model,
                           strategy="auto" if variant == "adaptive" else "deep", answer_depth="advanced", history_mode="fresh",
                           enable_quality_review=False, explanation_pipeline="direct")
            if result["status"] in {"stopped", "error"}:
                raise ValueError(f"Synthetic branch failed: {variant}/{result['stop_reason']}")
            results[variant] = result
        for variant, result in results.items():
            save_json(destination / f"{case['id']}_{variant}_result.json", {**result, "synthetic_only": True})
    report = render(destination, 1)
    report.update(synthetic_only=True, network_calls=0, frozen_v1_definitions_exercised=True)
    save_json(destination / "smoke_summary.json", report)
    return report


def check_index(output):
    """Private snapshot retrieval with a stored vector, not an embedding API."""
    from benchmark.experiments.three_way_legacy import load
    manifest = checked_manifest(output)
    sources = json.loads((output / "legacy_sources.json").read_text(encoding="utf-8"))
    seed = []
    def embed(texts):
        if len(texts) != 1 or not seed:
            raise AssertionError("Offline index probe must use one stored vector")
        return [seed[0]]
    namespace, _, _, _ = load(sources, None, embed)
    rag = namespace["PaperRAG"](str(output / "chroma"))
    counts = []
    for case in [c for c in manifest["cases"] if c["batch"] == 1]:
        collection = rag.client.get_collection(f"paper_{case['thread_id']}")
        vector = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
        seed[:] = [vector]
        if digest(rag.list_chunks(case["thread_id"])) != case["source_sha256"]:
            raise ValueError("Chroma text snapshot differs from frozen manifest")
        found = rag.search(case["thread_id"], "synthetic stored-vector query", k=8)
        if not found or any(not row["id"].startswith(case["thread_id"] + "_") for row in found):
            raise ValueError("Offline frozen retrieval failed")
        counts.append({"case_id": case["id"], "candidates_selected": len(found), "vector_dimension": len(vector)})
    result = {"network_calls": 0, "synthetic_vector_query": True, "cases": counts,
              "not_a_retrieval_quality_metric": True}
    save_json(output / "index_preflight.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "approve", "run", "render", "smoke", "preview", "check-index"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--chroma-source")
    parser.add_argument("--paper-map", help="Explicit private alias/thread mapping; never guess ambiguous titles")
    parser.add_argument("--legacy-ref", default="release/v1-freeze-20261002")
    parser.add_argument("--batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--confirm-paid", action="store_true")
    args = parser.parse_args()
    output = private(args.output)
    if args.mode == "prepare":
        if not args.chroma_source:
            parser.error("prepare needs --chroma-source")
        paper_map = json.loads(Path(args.paper_map).read_text(encoding="utf-8")) if args.paper_map else None
        result = prepare(Path(args.chroma_source).resolve(), output, args.legacy_ref, paper_map)
    elif args.mode == "approve":
        result = approve(output)
    elif args.mode == "run":
        result = run_batch(output, args.batch, args.confirm_paid)
    elif args.mode == "smoke":
        result = smoke(output)
    elif args.mode == "check-index":
        result = check_index(output)
    else:
        result = render(output, args.batch, preview=args.mode == "preview")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
