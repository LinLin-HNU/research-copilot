"""Run the R0 vector vs R1 MMR retrieval-only ablation.

Both variants share the original question embedding, the same Chroma candidate
pool, original chunk ids and exact evidence budget.  The script sends question
texts to the configured embedding endpoint, but never calls the router or an
answer-generation model.  It copies Chroma first and never opens the live store.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.experiments.ablation_protocol import (
    EvidenceBudget,
    STAGE1_VARIANTS,
    assert_shared_evidence_budget,
    build_manifest,
    chunk_fingerprint,
    validate_cases,
)
from benchmark.experiments.common import pack_chunks, proportion, retrieval_metrics


def _candidate_rows(collection, query_embedding, pool):
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(pool, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    if not result.get("ids") or not result["ids"][0]:
        return rows
    distances = result.get("distances") or [[]]
    for index, chunk_id in enumerate(result["ids"][0]):
        metadata = result["metadatas"][0][index]
        rows.append({
            "id": chunk_id,
            "content": result["documents"][0][index],
            "section": metadata.get("section", "unknown"),
            "page_start": metadata.get("page_start", 0),
            "page_end": metadata.get("page_end", 0),
            "distance": distances[0][index] if distances and distances[0] else None,
        })
    return rows


def _collection_rows(collection):
    result = collection.get(include=["documents", "metadatas"])
    rows = []
    for index, chunk_id in enumerate(result.get("ids") or []):
        metadata = result["metadatas"][index]
        rows.append({
            "id": chunk_id,
            "content": result["documents"][index],
            "section": metadata.get("section", "unknown"),
            "page_start": metadata.get("page_start", 0),
            "page_end": metadata.get("page_end", 0),
        })
    return rows


def _mmr_rows(collection, candidates, query_embedding, k, rag_store):
    if len(candidates) <= k:
        return list(candidates)
    stored = collection.get(ids=[row["id"] for row in candidates], include=["embeddings"])
    vector_by_id = dict(zip(stored["ids"], stored["embeddings"]))
    candidate_vectors = [rag_store._normalize(vector_by_id[row["id"]]) for row in candidates]
    chosen = rag_store._mmr_select(
        rag_store._normalize(query_embedding), candidate_vectors, k, 0.7
    )
    return [candidates[index] for index in chosen]


def _joined_text(chunks, *, emitted=False):
    """Join full chunk text or the exact truncated text placed in evidence."""
    def text_of(row):
        if emitted and "content_emitted" in row:
            return row.get("content_emitted") or ""
        return row.get("content") or ""

    return "\n".join(text_of(row) for row in chunks)


def _sections(chunks):
    return sorted({row.get("section") or "unknown" for row in chunks})


def _missing_terms(case, chunks, *, emitted=False):
    text = _joined_text(chunks, emitted=emitted).lower()
    return [term for term in case.get("expected_terms", []) if term.lower() not in text]


def _diagnose_term_stage(case, corpus, candidates, selected_full, emitted):
    if case.get("expected_category") == "B":
        return None
    if not case.get("expected_terms"):
        return "no_expected_terms"
    if _missing_terms(case, corpus):
        return "expected_terms_not_found_in_corpus"
    if _missing_terms(case, candidates):
        return "candidate_pool_miss"
    if _missing_terms(case, selected_full):
        return "selection_miss"
    if _missing_terms(case, emitted, emitted=True):
        return "packing_or_truncation_loss"
    return None


def _stage_metrics(case, chunks):
    return retrieval_metrics(case, _joined_text(chunks), _sections(chunks))


def _metric_summary(rows):
    return {
        metric: proportion([row[metric] for row in rows])
        for metric in ("retrieval_hit", "section_hit", "term_hit")
    }


def _grouped(rows, key):
    groups = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        groups[value] = {
            variant: {"cases": len([row for row in subset if row["variant"] == variant]),
                      **_metric_summary([row for row in subset if row["variant"] == variant])}
            for variant in STAGE1_VARIANTS
        }
    return groups


def _paired_changes(rows):
    by_key = {(row["variant"], row["id"]): row for row in rows}
    ids = sorted({row["id"] for row in rows})
    result = {}
    for metric in ("retrieval_hit", "section_hit", "term_hit"):
        miss_to_hit, hit_to_miss = [], []
        for case_id in ids:
            before = by_key[(STAGE1_VARIANTS[0], case_id)][metric]
            after = by_key[(STAGE1_VARIANTS[1], case_id)][metric]
            if before is False and after is True:
                miss_to_hit.append(case_id)
            elif before is True and after is False:
                hit_to_miss.append(case_id)
        result[metric] = {"R0_miss_to_R1_hit": miss_to_hit, "R0_hit_to_R1_miss": hit_to_miss}
    return result


def _summarize(rows):
    assert_shared_evidence_budget(rows, STAGE1_VARIANTS)
    summary = {"overall": {}, "paired_changes": _paired_changes(rows)}
    for variant in STAGE1_VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        summary["overall"][variant] = {
            "cases": len(subset),
            **_metric_summary(subset),
            "mean_evidence_chars": sum(row["evidence_chars"] for row in subset) / len(subset),
            "mean_emitted_chunks": sum(len(row["selected_chunk_ids"]) for row in subset) / len(subset),
            "term_failure_stage_counts": dict(sorted(Counter(
                row["term_failure_stage"] or "no_term_failure" for row in subset
            ).items())),
        }
    summary["by_category"] = _grouped(rows, "expected_category")
    summary["by_thread"] = _grouped(rows, "thread_id")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--chroma-source", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-chunk-chars", type=int, default=600)
    parser.add_argument("--char-budget", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path, label in ((args.cases, "cases"), (args.chroma_source, "chroma source"), (args.env_file, "env file")):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    budget = EvidenceBudget(
        candidate_pool=args.candidate_pool,
        max_chunks=args.max_chunks,
        max_chunk_chars=args.max_chunk_chars,
        char_budget=args.char_budget,
    )
    try:
        budget.validate()
    except ValueError as exc:
        parser.error(str(exc))

    raw_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = validate_cases(raw_cases, split=args.split)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    elif not args.confirm_full:
        parser.error("use --limit for a smoke run or --confirm-full for all selected cases")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "corpus_snapshot" / "chroma"
    shutil.copytree(args.chroma_source.resolve(), snapshot)

    from dotenv import load_dotenv
    load_dotenv(args.env_file.resolve(), override=False)
    if not os.getenv("DASHSCOPE_API_KEY") or not os.getenv("DASHSCOPE_BASE_URL"):
        raise RuntimeError("DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required")

    import rag_store

    rag = rag_store.PaperRAG(str(snapshot))
    eligible = [case for case in cases if case["expected_category"] != "B"]
    query_vectors = rag_store._embed([case["question"] for case in eligible]) if eligible else []
    vector_by_case = {case["id"]: vector for case, vector in zip(eligible, query_vectors)}

    collection_cache, corpus_cache, fingerprints = {}, {}, {}
    rows = []
    rows_path = output / "retrieval_rows.jsonl"
    with rows_path.open("x", encoding="utf-8") as sink:
        for case in cases:
            thread_id = case["thread_id"]
            if thread_id not in collection_cache:
                collection = rag.client.get_collection(f"paper_{thread_id}")
                collection_cache[thread_id] = collection
                corpus_cache[thread_id] = _collection_rows(collection)
                fingerprints[thread_id] = chunk_fingerprint(corpus_cache[thread_id])
            collection = collection_cache[thread_id]
            corpus = corpus_cache[thread_id]

            if case["expected_category"] == "B":
                candidates = []
                ranked = {variant: [] for variant in STAGE1_VARIANTS}
            else:
                query_vector = vector_by_case[case["id"]]
                candidates = _candidate_rows(collection, query_vector, budget.candidate_pool)
                ranked = {
                    "R0_vector": candidates[:budget.max_chunks],
                    "R1_mmr": _mmr_rows(
                        collection, candidates, query_vector, budget.max_chunks, rag_store
                    ),
                }

            corpus_metrics = _stage_metrics(case, corpus)
            candidate_metrics = _stage_metrics(case, candidates)
            for variant in STAGE1_VARIANTS:
                full_selected = ranked[variant]
                full_selected_metrics = _stage_metrics(case, full_selected)
                evidence, emitted = pack_chunks(
                    full_selected,
                    char_budget=budget.char_budget,
                    max_chunk_chars=budget.max_chunk_chars,
                    max_chunks=budget.max_chunks,
                )
                emitted_metrics = retrieval_metrics(case, evidence, _sections(emitted))
                row = {
                    "id": case["id"],
                    "thread_id": thread_id,
                    "expected_category": case["expected_category"],
                    "variant": variant,
                    **budget.to_dict(),
                    "candidate_chunk_ids": [item["id"] for item in candidates],
                    "ranked_chunk_ids": [item["id"] for item in full_selected],
                    "selected_chunk_ids": [item["id"] for item in emitted],
                    "not_emitted_ranked_ids": [
                        item["id"] for item in full_selected
                        if item["id"] not in {chunk["id"] for chunk in emitted}
                    ],
                    "evidence_chars": len(evidence),
                    "corpus_metrics": corpus_metrics,
                    "candidate_metrics": candidate_metrics,
                    "selected_full_metrics": full_selected_metrics,
                    "missing_terms": {
                        "corpus": _missing_terms(case, corpus),
                        "candidate_pool": _missing_terms(case, candidates),
                        "selected_full": _missing_terms(case, full_selected),
                        "emitted": _missing_terms(case, emitted, emitted=True),
                    },
                    "term_failure_stage": _diagnose_term_stage(
                        case, corpus, candidates, full_selected, emitted
                    ),
                    **emitted_metrics,
                    # Explicit aliases: these are deterministic proxy metrics,
                    # not claims that the evidence semantically answers the question.
                    "strict_proxy_hit": emitted_metrics["retrieval_hit"],
                    "section_label_hit": emitted_metrics["section_hit"],
                    "lexical_term_hit": emitted_metrics["term_hit"],
                }
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({
                "id": case["id"],
                "R0_term_hit": rows[-2]["term_hit"],
                "R1_term_hit": rows[-1]["term_hit"],
                "R0_failure": rows[-2]["term_failure_stage"],
                "R1_failure": rows[-1]["term_failure_stage"],
            }, ensure_ascii=False), flush=True)

    summary = _summarize(rows)
    manifest = build_manifest(
        cases_path=args.cases.resolve(),
        cases=cases,
        variants=STAGE1_VARIANTS,
        budget=budget,
        corpus_fingerprints=fingerprints,
    )
    manifest.update({
        "ablation": "R0 pure vector versus R1 MMR",
        "query_strategy": "original user question only",
        "candidate_pool_shared": True,
        "embedding_api_called": bool(eligible),
        "embedding_text_count": len(eligible),
        "embedding_http_request_count_expected": math.ceil(len(eligible) / 20),
        "embedding_payload": "question text only; no paper chunks sent to embedding API",
        "chroma_live_store_opened": False,
        "router_model_called": False,
        "answer_generation_called": False,
        "metric_contract": {
            "section_label_hit": "any expected section label equals an emitted parser section label",
            "lexical_term_hit": "every expected term is a case-insensitive substring of emitted evidence",
            "strict_proxy_hit": "non-empty evidence and both proxy checks pass",
            "semantic_answer_support_measured": False,
        },
        "limitations": [
            "Gold labels are AI-reviewed development labels, not human expert ground truth.",
            "Expected-term matching is lexical and may miss valid paraphrases.",
            "A lexical match may still occur in irrelevant, negated, or citation-only context.",
            "Expected-section matching compares parser labels, not semantic evidence support.",
            "R0 and R1 share vector candidates; this isolates selection, not embedding quality.",
        ],
    })
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
