"""Offline 2x2 character-budget ablation on frozen R0/R1 rankings.

Uses only the Python standard library and offline evaluation primitives.
Reads the archived Chroma SQLite metadata with mode=ro, without starting Chroma,
loading .env, embedding queries, or generating answers. Gold is used only to
score already packed evidence, never to select or reorder it.
"""

from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark.experiments.ablation_protocol import chunk_fingerprint, sha256_file
from benchmark.experiments.common import pack_chunks, proportion, read_jsonl, retrieval_metrics

CONFIGS = {"c600_b4000": (600, 4000), "c1200_b4000": (1200, 4000),
           "c600_b8000": (600, 8000), "c1200_b8000": (1200, 8000)}
BASE = "c600_b4000"
VARIANTS = ("R0_vector", "R1_mmr")
METRICS = ("retrieval_hit", "section_hit", "term_hit")


def load_snapshot(db_path, thread_ids):
    """Read the known Chroma metadata schema; fingerprints reject schema drift."""
    corpus = {}
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only = ON")
        for thread_id in sorted(thread_ids):
            collections = db.execute("SELECT id FROM collections WHERE name=?",
                                     (f"paper_{thread_id}",)).fetchall()
            if len(collections) != 1:
                raise ValueError(f"Missing or ambiguous collection: {thread_id}")
            records = db.execute(
                "SELECT e.id,e.embedding_id,m.key,m.string_value,m.int_value "
                "FROM embeddings e JOIN segments s ON e.segment_id=s.id "
                "LEFT JOIN embedding_metadata m ON m.id=e.id "
                "WHERE s.collection=? AND s.scope='METADATA'", collections[0]
            ).fetchall()
            grouped = {}
            for internal_id, chunk_id, key, text, number in records:
                item = grouped.setdefault(internal_id, {"id": chunk_id, "metadata": {}})
                item["metadata"][key] = text if text is not None else number
            chunks = []
            for item in grouped.values():
                md = item["metadata"]
                if "chroma:document" not in md:
                    raise ValueError(f"Document missing in snapshot: {item['id']}")
                chunks.append({"id": item["id"], "content": md["chroma:document"],
                               "section": md.get("section", "unknown"),
                               "page_start": md.get("page_start", 0),
                               "page_end": md.get("page_end", 0)})
            if not chunks or len({r['id'] for r in chunks}) != len(chunks):
                raise ValueError(f"Empty or duplicate snapshot chunks: {thread_id}")
            corpus[thread_id] = chunks
    return corpus


def replay_row(case, source_row, chunks_by_id, config):
    cap, budget = CONFIGS[config]
    ranked = [chunks_by_id[cid] for cid in source_row["ranked_chunk_ids"]]
    evidence, selected = pack_chunks(ranked, max_chunk_chars=cap,
                                     char_budget=budget, max_chunks=8)
    sections = sorted({r.get("section") or "unknown" for r in selected})
    # Scoring happens after selection; gold never affects packing.
    metrics = retrieval_metrics(case, evidence, sections)
    missing = []
    for term in case.get("expected_terms", []):
        if term.lower() in evidence.lower():
            continue
        locations = []
        for rank, chunk in enumerate(ranked, 1):
            start = (chunk.get("content") or "").lower().find(term.lower())
            if start >= 0:
                locations.append({"chunk_id": chunk["id"], "rank": rank,
                                  "first_match_start": start,
                                  "fits_per_chunk_cap": start + len(term) <= cap})
        reason = ("not_in_frozen_top8" if not locations else
                  "per_chunk_cap_loss" if not any(x["fits_per_chunk_cap"] for x in locations)
                  else "total_budget_loss")
        missing.append({"term": term, "reason": reason, "locations": locations})
    return {"id": case["id"], "thread_id": case["thread_id"],
            "expected_category": case["expected_category"], "question": case["question"],
            "variant": source_row["variant"], "config": config,
            "max_chunk_chars": cap, "char_budget": budget, "max_chunks": 8,
            "ranked_chunk_ids": source_row["ranked_chunk_ids"],
            "selected_chunk_ids": [r["id"] for r in selected],
            "actual_sections": sections, "evidence_chars": len(evidence),
            "body_chars": sum(len(r["content_emitted"]) for r in selected),
            "evidence": evidence, "missing_terms": missing, **metrics}


def verify_baseline(case, row, chunks):
    replay = replay_row(case, row, chunks, BASE)
    for key in (*METRICS, "selected_chunk_ids", "evidence_chars"):
        if row[key] != replay[key]:
            raise ValueError(f"Baseline mismatch: {row['id']}/{row['variant']}/{key}")
    return replay


def load_verified_source(source, cases_path=None):
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    cases_path = Path(cases_path or manifest["cases_path"])
    if sha256_file(cases_path) != manifest["cases_sha256"]:
        raise ValueError("Cases hash differs from frozen source run")
    if manifest["shared_budget"] != {"candidate_pool": 20, "max_chunks": 8,
                                      "max_chunk_chars": 600, "char_budget": 4000}:
        raise ValueError("Source run must use the frozen 20/8/600/4000 budget")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    cases = {c["id"]: c for c in cases}
    ids = manifest["case_ids"]
    if len(set(ids)) != len(ids) or len(ids) != manifest["case_count"]:
        raise ValueError("Invalid manifest case IDs/count")
    rows = read_jsonl(source / "retrieval_rows.jsonl")
    pairs = {(r["id"], r["variant"]): r for r in rows}
    if len(pairs) != len(rows) or set(pairs) != {(i, v) for i in ids for v in VARIANTS}:
        raise ValueError("Source must contain one R0/R1 pair per manifest case")
    threads = {cases[i]["thread_id"] for i in ids}
    corpus = load_snapshot(source / "corpus_snapshot/chroma/chroma.sqlite3", threads)
    lookup = {}
    for tid, chunks in corpus.items():
        if chunk_fingerprint(chunks) != manifest["corpus_fingerprints"].get(tid):
            raise ValueError(f"Corpus fingerprint mismatch: {tid}")
        lookup[tid] = {r["id"]: r for r in chunks}
    for case_id in ids:
        left, right = (pairs[(case_id, v)] for v in VARIANTS)
        if left["candidate_chunk_ids"] != right["candidate_chunk_ids"]:
            raise ValueError(f"Candidate pool drift: {case_id}")
    for row in rows:
        case = cases[row["id"]]
        if row["thread_id"] != case["thread_id"] or row["expected_category"] != case["expected_category"]:
            raise ValueError(f"Case metadata mismatch: {row['id']}")
        if any(row[k] != value for k, value in manifest["shared_budget"].items()):
            raise ValueError(f"Row budget drift: {row['id']}")
        ranked, candidates = row["ranked_chunk_ids"], row["candidate_chunk_ids"]
        if len(ranked) > 8 or len(set(ranked)) != len(ranked) or not set(ranked) <= set(candidates):
            raise ValueError(f"Invalid frozen ranking: {row['id']}")
        verify_baseline(case, row, lookup[case["thread_id"]])
    return manifest, cases, rows, lookup


def summarize(rows):
    result = {}
    for variant in VARIANTS:
        result[variant] = {}
        baseline = {r["id"]: r for r in rows if r["variant"] == variant and r["config"] == BASE}
        for config in CONFIGS:
            subset = [r for r in rows if r["variant"] == variant and r["config"] == config]
            changes = {}
            for metric in METRICS:
                changes[metric] = {
                    "miss_to_hit": sorted(r["id"] for r in subset if baseline[r["id"]][metric] is False and r[metric] is True),
                    "hit_to_miss": sorted(r["id"] for r in subset if baseline[r["id"]][metric] is True and r[metric] is False)}
            result[variant][config] = {
                "cases": len(subset), **{m: proportion([r[m] for r in subset]) for m in METRICS},
                "mean_evidence_chars": sum(r["evidence_chars"] for r in subset) / len(subset),
                "mean_emitted_chunks": sum(len(r["selected_chunk_ids"]) for r in subset) / len(subset),
                "changes_from_600_4000": changes}
    return result


def render_report(summary):
    lines = ["# 证据预算消融（固定已有排名，离线重放）", "",
             "本报告衡量章节/关键词代理覆盖及证据长度，不衡量答案正确率、真实 token 或延迟。",
             "四组均保持原来的最多八块及顺序；打包遇到超预算块即停止，沿用基线逻辑。",
             "单块变长会减少同一总预算内能装下的块，因此覆盖可能退化。8000 是上限，不保证实际用满。", "",
             "| 排序 | 单块/总预算 | 严格代理命中 | 关键词命中 | 章节命中 | 平均字符 | 严格改善/退化 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for variant, configs in summary.items():
        for config, data in configs.items():
            cap, budget = CONFIGS[config]
            counts = [f"{data[m]['successes']}/{data[m]['n']}" for m in ("retrieval_hit", "term_hit", "section_hit")]
            changes = data["changes_from_600_4000"]["retrieval_hit"]
            lines.append(f"| {variant} | {cap}/{budget} | " + " | ".join(counts) +
                         f" | {data['mean_evidence_chars']:.0f} | {len(changes['miss_to_hit'])}/{len(changes['hit_to_miss'])} |")
    lines += ["", "## 决策前检查", "",
              "结合覆盖变化、实际字符开销、改善和退化样例决定暂定配置，不自动选择最高分。",
              "budget_rows.jsonl 保留每题完整最终证据与缺失词在原排名中的位置，供逐题审查。",
              "这些位置仅用于事后诊断，真实 Agent 不能使用标准关键词决定展开或检索。",
              "暂定配置的选择与语义核验需在运行后完成；当前报告没有宣称答案质量提升。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="Optional relocated cases file; hash must match")
    parser.add_argument("--check-only", action="store_true", help="Verify all original baselines without producing new comparisons")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only")
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new output directory")
    source = args.source_run.resolve()
    manifest, cases, source_rows, lookup = load_verified_source(source, args.cases)
    print(json.dumps({"status": "baseline_verified", "case_count": manifest["case_count"],
                      "baseline_rows_verified": len(source_rows), "network_calls": 0}), flush=True)
    if args.check_only:
        return
    rows = [replay_row(cases[row["id"]], row, lookup[row["thread_id"]], config)
            for row in source_rows for config in CONFIGS]
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "budget_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {"experiment": "frozen_rankings_character_budget_2x2", "source_run": str(source),
                  "source_manifest_sha256": sha256_file(source / "manifest.json"),
                  "source_rows_sha256": sha256_file(source / "retrieval_rows.jsonl"),
                  "runner_sha256": sha256_file(Path(__file__)),
                  "cases_sha256": manifest["cases_sha256"],
                  "corpus_fingerprints": manifest["corpus_fingerprints"],
                  "case_count": manifest["case_count"], "row_count": len(rows),
                  "baseline_rows_verified": len(source_rows), "configs": CONFIGS,
                  "fixed_rankings": True, "max_chunks": 8, "network_calls": 0,
                  "snapshot_access": "sqlite mode=ro and query_only",
                  "gold_used_for_selection": False, "answer_quality_measured": False,
                  "cost_unit": "characters_not_tokens", "configuration_selected": False}
    (args.output / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"status": "complete", "rows": len(rows), "network_calls": 0,
                      "report": str(args.output / "report.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
