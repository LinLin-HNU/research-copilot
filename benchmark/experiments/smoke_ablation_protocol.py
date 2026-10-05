"""Offline smoke test for the retrieval-ablation protocol.

The command uses synthetic chunks and vectors only.  It never reads .env,
imports the production model, calls an API, or touches the live Chroma store.
Its output proves that manifests, evidence budgets, paired rows and gold-field
handling are wired consistently before a real ablation is allowed to run.
"""

import argparse
import json
from pathlib import Path
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
from benchmark.experiments.common import pack_chunks, retrieval_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    raw_cases = [
        {
            "id": "synthetic-a1",
            "split": "dev",
            "question": "Which component performs normalization?",
            "thread_id": "synthetic-paper",
            "expected_category": "A1",
            "expected_sections": ["method"],
            "expected_terms": ["normalization"],
        }
    ]
    cases_path = output / "cases.synthetic.json"
    cases_path.write_text(json.dumps(raw_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    cases = validate_cases(raw_cases, split="dev")
    budget = EvidenceBudget()

    chunks = [
        {
            "id": "chunk-method",
            "content": "The method applies normalization before the encoder.",
            "section": "method",
            "page_start": 3,
            "page_end": 3,
        },
        {
            "id": "chunk-results",
            "content": "The results compare three forecasting horizons.",
            "section": "results",
            "page_start": 6,
            "page_end": 6,
        },
        {
            "id": "chunk-intro",
            "content": "The introduction motivates long-term forecasting.",
            "section": "introduction",
            "page_start": 1,
            "page_end": 1,
        },
    ]
    ranked = {
        "R0_vector": [chunks[0], chunks[1], chunks[2]],
        "R1_mmr": [chunks[0], chunks[2], chunks[1]],
    }

    rows = []
    for variant in STAGE1_VARIANTS:
        evidence, selected = pack_chunks(
            ranked[variant],
            char_budget=budget.char_budget,
            max_chunk_chars=budget.max_chunk_chars,
            max_chunks=budget.max_chunks,
        )
        sections = sorted({row["section"] for row in selected})
        rows.append({
            "id": cases[0]["id"],
            "variant": variant,
            **budget.to_dict(),
            "selected_chunk_ids": [row["id"] for row in selected],
            "evidence_chars": len(evidence),
            **retrieval_metrics(cases[0], evidence, sections),
        })

    assert_shared_evidence_budget(rows, STAGE1_VARIANTS)
    manifest = build_manifest(
        cases_path=cases_path,
        cases=cases,
        variants=STAGE1_VARIANTS,
        budget=budget,
        corpus_fingerprints={"synthetic-paper": chunk_fingerprint(chunks)},
    )
    manifest["synthetic_smoke_only"] = True
    manifest["embedding_api_called"] = False

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "retrieval_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "status": "ok",
        "synthetic_smoke_only": True,
        "case_count": len(cases),
        "row_count": len(rows),
        "variants": list(STAGE1_VARIANTS),
        "shared_budget": budget.to_dict(),
        "network_calls": 0,
        "warning": "Synthetic smoke output is not a project quality metric.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
