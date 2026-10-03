"""Fair retrieval-only comparison on identical original chunks and budgets.

This script calls the embedding API, but never calls the answer-generation or
router model.  Both variants use the same query embedding, Chroma candidates,
600-character chunk cap and exact total evidence budget.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.experiments.common import digest, pack_chunks, proportion, retrieval_metrics


VARIANTS = ("vector_topk_same_chunks", "mmr_same_chunks")


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


def _mmr_rows(collection, candidates, query_embedding, k, rag_store):
    if len(candidates) <= k:
        return candidates
    stored = collection.get(ids=[row["id"] for row in candidates], include=["embeddings"])
    vector_by_id = dict(zip(stored["ids"], stored["embeddings"]))
    candidate_vectors = [rag_store._normalize(vector_by_id[row["id"]]) for row in candidates]
    chosen = rag_store._mmr_select(
        rag_store._normalize(query_embedding), candidate_vectors, k, 0.7
    )
    return [candidates[index] for index in chosen]


def _summarize(rows):
    result = {}
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            "cases": len(subset),
            "retrieval_hit": proportion([row["retrieval_hit"] for row in subset]),
            "section_hit": proportion([row["section_hit"] for row in subset]),
            "term_hit": proportion([row["term_hit"] for row in subset]),
            "mean_evidence_chars": (
                sum(row["evidence_chars"] for row in subset) / len(subset) if subset else None
            ),
            "mean_selected_chunks": (
                sum(len(row["selected_chunk_ids"]) for row in subset) / len(subset)
                if subset else None
            ),
        }
    by_key = {(row["variant"], row["id"]): row for row in rows}
    eligible_ids = sorted({row["id"] for row in rows})
    result["term_hit_changes"] = {
        "vector_miss_to_mmr_hit": [
            case_id for case_id in eligible_ids
            if by_key[(VARIANTS[0], case_id)]["term_hit"] is False
            and by_key[(VARIANTS[1], case_id)]["term_hit"] is True
        ],
        "vector_hit_to_mmr_miss": [
            case_id for case_id in eligible_ids
            if by_key[(VARIANTS[0], case_id)]["term_hit"] is True
            and by_key[(VARIANTS[1], case_id)]["term_hit"] is False
        ],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-chunk-chars", type=int, default=600)
    parser.add_argument("--char-budget", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.candidate_pool, args.max_chunks, args.max_chunk_chars, args.char_budget) <= 0:
        parser.error("all budgets and pool sizes must be positive")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if any("split" not in case for case in cases):
        parser.error("every case must have an explicit split")
    cases = [case for case in cases if case["split"] == args.split]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    elif not args.confirm_full:
        parser.error("use --limit for a smoke test or --confirm-full for the complete run")
    if not cases or len({case["id"] for case in cases}) != len(cases):
        parser.error("selected cases must be nonempty with unique ids")

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "corpus_snapshot" / "chroma"
    shutil.copytree(ROOT / "resources" / "chroma_db", snapshot)

    import rag_store

    rag = rag_store.PaperRAG(str(snapshot))
    rows = []
    output_path = args.output / "retrieval_rows.jsonl"
    with output_path.open("x", encoding="utf-8") as output:
        for case in cases:
            collection = rag.client.get_collection(f"paper_{case['thread_id']}")
            if case["expected_category"] == "B":
                ranked_by_variant = {variant: [] for variant in VARIANTS}
            else:
                query_embedding = rag_store._embed([case["question"]])[0]
                candidates = _candidate_rows(collection, query_embedding, args.candidate_pool)
                ranked_by_variant = {
                    VARIANTS[0]: candidates[:args.max_chunks],
                    VARIANTS[1]: _mmr_rows(
                        collection, candidates, query_embedding, args.max_chunks, rag_store
                    ),
                }
            for variant, ranked in ranked_by_variant.items():
                evidence, selected = pack_chunks(
                    ranked,
                    char_budget=args.char_budget,
                    max_chunk_chars=args.max_chunk_chars,
                    max_chunks=args.max_chunks,
                )
                sections = sorted({row.get("section", "unknown") for row in selected})
                row = {
                    "id": case["id"],
                    "thread_id": case["thread_id"],
                    "expected_category": case["expected_category"],
                    "variant": variant,
                    "selected_chunk_ids": [item["id"] for item in selected],
                    "actual_sections": sections,
                    "evidence_chars": len(evidence),
                    **retrieval_metrics(case, evidence, sections),
                }
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({
                "id": case["id"],
                "vector_term_hit": rows[-2]["term_hit"],
                "mmr_term_hit": rows[-1]["term_hit"],
            }, ensure_ascii=False), flush=True)

    report = {
        "protocol": {
            "description": "retrieval-only; identical original Chroma chunks and exact evidence budgets",
            "cases_sha256": digest(args.cases),
            "split": args.split,
            "candidate_pool": args.candidate_pool,
            "max_chunks": args.max_chunks,
            "max_chunk_chars": args.max_chunk_chars,
            "char_budget": args.char_budget,
            "generation_model_called": False,
            "router_model_called": False,
            "embedding_api_called": True,
            "limitations": [
                "This isolates pure-vector ranking versus MMR; it is not a full V0/V1 system comparison.",
                "Gold labels are AI-reviewed development labels, not human expert ground truth.",
            ],
        },
        "summary": _summarize(rows),
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
