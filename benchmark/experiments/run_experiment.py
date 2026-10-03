"""Isolated single-turn experiment; production retrieval/prompt, no chat DB writes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from contextlib import nullcontext
from time import perf_counter
from unittest.mock import patch
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark.experiments.common import digest, windows, top_k, retrieval_metrics

VARIANTS = ("v0_naive", "v1_current", "v2_experimental")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--variant", choices=(*VARIANTS, "all"), required=True)
    p.add_argument("--split", choices=("dev", "test"))
    p.add_argument("--limit", type=int)
    p.add_argument("--confirm-full", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--test-manifest", type=Path)
    args = p.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.split:
        if any("split" not in c for c in cases):
            p.error("--split requires explicit split on every case")
        cases = [c for c in cases if c["split"] == args.split]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        p.error("Cases must be nonempty with unique ids")
    marker = None
    if any(c.get("split") == "test" for c in cases):
        if not args.test_manifest:
            p.error("Test runs require a human-approved frozen --test-manifest")
        manifest = json.loads(args.test_manifest.read_text(encoding="utf-8"))
        if manifest.get("cases_sha256") != digest(args.cases) or not manifest.get("human_gold_approved"):
            p.error("Test manifest hash/approval mismatch")
        marker = args.test_manifest.with_suffix(".started")
        if marker.exists():
            p.error("Frozen test already started; report failures, do not silently rerun")
    if args.limit is not None:
        if args.limit <= 0:
            p.error("--limit must be positive")
        cases = cases[:args.limit]
    elif not args.confirm_full:
        p.error("Use --limit for smoke tests or --confirm-full for a full paid run")
    variants = VARIANTS if args.variant == "all" else [args.variant]
    if args.top_k <= 0:
        p.error("--top-k must be positive")
    args.output = args.output.resolve()
    # Output directories cannot be reused: accidental reruns must not mix data.
    args.output.mkdir(parents=True, exist_ok=False)
    for variant in variants:
        print(f"将为变体 {variant} 生成 {len(cases)} 个答案（会调用付费 API）", flush=True)
    os.chdir(ROOT)
    import rag_store
    import query_router
    from app import build_question_evidence, GUIDANCE_B, GUIDANCE_NO_EVIDENCE, ROUTE_GUIDANCE
    from agent_setup import request_prompt, RequestContext
    from config import model
    from langchain.agents import create_agent
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.callbacks import BaseCallbackHandler

    class Usage(BaseCallbackHandler):
        def __init__(self):
            self.calls = []
            self.errors = 0

        def on_llm_end(self, response, **kwargs):
            for batch in response.generations:
                for generation in batch:
                    self.calls.append(getattr(getattr(generation, "message", None), "usage_metadata", None))

        def on_llm_error(self, error, **kwargs):
            self.errors += 1

        def totals(self):
            return {k: sum(c[k] for c in self.calls) if self.calls and all(c and c.get(k) is not None for c in self.calls) else None
                    for k in ("input_tokens", "output_tokens")}

    agent = create_agent(model, tools=[], middleware=[request_prompt], context_schema=RequestContext)
    source_files = ["app.py", "query_router.py", "rag_store.py", "agent_setup.py", "config.py", "prompts.py", "requirements.txt", "benchmark/experiments/run_experiment.py", "benchmark/experiments/common.py"]
    run_meta = {"created_at": datetime.now(timezone.utc).isoformat(), "cases_sha256": digest(args.cases),
                "sources": {f: digest(ROOT/f) for f in source_files},
                "protocol": "single_turn_no_checkpoint", "variants": list(variants),
                "experimental_alias": "v2_experimental equals v1_current; not a new product version",
                "limitations": ["Approximate full text from existing overlapping parsed chunks", "Not an ablation: chunking, routing and prompts differ", "No billing-price estimate; usage may exclude failed provider requests"]}
    (args.output/"manifest.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    original_post = rag_store.requests.post
    embedding_calls = []

    def tracked_post(*a, **kw):
        kw.setdefault("timeout", 60)
        response = original_post(*a, **kw)
        usage = response.json().get("usage") if response.ok else None
        embedding_calls.append(usage)
        return response

    # Snapshot first; never create/delete a collection in the user's live Chroma.
    with nullcontext(str(args.output/"corpus_snapshot")) as temp:
        snapshot = Path(temp)/"chroma"
        shutil.copytree(ROOT/"resources/chroma_db", snapshot)
        rag = rag_store.PaperRAG(str(snapshot))
        chunk_cache, index_cache = {}, {}
        for case in cases:
            thread = case["thread_id"]
            if thread not in chunk_cache:
                rag.client.get_collection(f"paper_{thread}")  # fail for absent corpus
                chunk_cache[thread] = rag.list_chunks(thread)
                if not chunk_cache[thread]:
                    raise ValueError("Empty paper corpus")
        corpus = {thread: {"chunks": len(chunks), "sha256": __import__("hashlib").sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()} for thread, chunks in chunk_cache.items()}
        (args.output/"corpus.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        if marker is not None:
            with marker.open("x", encoding="utf-8") as f:
                f.write(datetime.now(timezone.utc).isoformat())
        for variant in variants:
            with (args.output/f"answers_{variant}.jsonl").open("x", encoding="utf-8") as output:
                for case in cases:
                    row = {**case, "variant": variant, "answer": "", "evidence": "", "route": None,
                           "route_correct": None, "retrieval_hit": None, "error": None,
                           "input_tokens": None, "output_tokens": None, "timing_ms": {}}
                    started = perf_counter()
                    usage, router_usage = Usage(), Usage()
                    embedding_calls.clear()
                    stage = "setup"
                    try:
                        with patch.object(rag_store.requests, "post", tracked_post):
                            if variant == "v0_naive":
                                thread = case["thread_id"]
                                stage, before = "index", perf_counter()
                                if thread not in index_cache:
                                    pieces = windows(chunk_cache[thread])
                                    index_cache[thread] = (pieces, rag_store._embed([w["content"] for w in pieces]))
                                row["timing_ms"]["index"] = (perf_counter()-before)*1000
                                pieces, vectors = index_cache[thread]
                                stage, before = "retrieval", perf_counter()
                                selected = [pieces[i] for i in top_k(rag_store._embed([case["question"]])[0], vectors, args.top_k)]
                                evidence = "\n\n---\n\n".join(f"【来源: {'/'.join(w['sections'])} | 第 {w['page_start']}-{w['page_end']} 页】\n{w['content']}" for w in selected)
                                sections = {s for w in selected for s in w["sections"]}
                                row["retrieved_windows"] = selected
                                row["timing_ms"]["retrieval"] = (perf_counter()-before)*1000
                            else:
                                stage, before = "routing", perf_counter()
                                with patch.object(query_router, "model", model.with_config(callbacks=[router_usage])):
                                    route = query_router.build_route(case["question"])
                                row["route"] = route
                                row["route_correct"] = route["category"] == case["expected_category"]
                                row["router_errors"] = router_usage.errors
                                row["timing_ms"]["routing"] = (perf_counter()-before)*1000
                                stage, before = "retrieval", perf_counter()
                                evidence = build_question_evidence(rag, case["thread_id"], route) if route["needs_retrieval"] else ""
                                from benchmark.run_benchmark import evidence_sections
                                sections = evidence_sections(evidence)
                                row["timing_ms"]["retrieval"] = (perf_counter()-before)*1000
                            row["evidence"] = evidence
                            row["actual_sections"] = sorted(sections)
                            row.update(retrieval_metrics(case, evidence, sections))
                            stage, before = "generation", perf_counter()
                            if variant == "v0_naive":
                                answer = model.invoke([SystemMessage(content="根据以下上下文回答问题，上下文没有就说不知道。\n\n"+evidence), HumanMessage(content=case["question"])], config={"callbacks": [usage]})
                            else:
                                guidance = GUIDANCE_B if not route["needs_retrieval"] else (ROUTE_GUIDANCE[route["category"]] if evidence else GUIDANCE_NO_EVIDENCE)
                                result = agent.invoke({"messages": [HumanMessage(content=case["question"])]}, context=RequestContext(evidence=evidence, guidance=guidance), config={"callbacks": [usage]})
                                answer = result["messages"][-1]
                            row["answer"] = answer.content
                            if not row["answer"]:
                                raise ValueError("Empty model answer")
                            row["timing_ms"]["generation"] = (perf_counter()-before)*1000
                            row.update(usage.totals())
                    except Exception as exc:
                        # Do not persist exception bodies containing API URLs/credentials.
                        row["error"] = {"stage": stage, "type": type(exc).__name__}
                    row["routing_usage"] = router_usage.totals()
                    row["embedding_usage"] = list(embedding_calls)
                    row["timing_ms"]["total"] = (perf_counter()-started)*1000
                    output.write(json.dumps(row, ensure_ascii=False)+"\n")
                    output.flush()
                    print(json.dumps({"id": row["id"], "variant": variant, "error": row["error"], "retrieval_hit": row["retrieval_hit"], "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
