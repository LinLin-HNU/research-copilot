"""Shared, offline-safe protocol primitives for retrieval ablations.

This module owns the experiment invariants.  It has no project imports and
performs no network or model calls, so it can be exercised by unit tests before
any paid retrieval run is started.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROTOCOL_VERSION = "retrieval-ablation-v1"
STAGE1_VARIANTS = ("R0_vector", "R1_mmr")
STAGE2_VARIANTS = ("R2_query_expansion", "R3_structural_fallback")


@dataclass(frozen=True)
class EvidenceBudget:
    candidate_pool: int = 20
    max_chunks: int = 8
    max_chunk_chars: int = 600
    char_budget: int = 4000

    def validate(self) -> None:
        values = asdict(self)
        invalid = {name: value for name, value in values.items() if value <= 0}
        if invalid:
            raise ValueError(f"budget values must be positive: {invalid}")
        if self.candidate_pool < self.max_chunks:
            raise ValueError("candidate_pool must be greater than or equal to max_chunks")
        if self.char_budget < self.max_chunk_chars:
            raise ValueError("char_budget must allow at least one capped chunk")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cases(cases: Sequence[Mapping], split: str = "dev") -> list[dict]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    selected = [dict(case) for case in cases if case.get("split", "dev") == split]
    if not selected:
        raise ValueError(f"no cases selected for split={split}")
    ids = [case.get("id") for case in selected]
    if any(not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("case ids must be nonempty and unique")
    required = {"question", "thread_id", "expected_category"}
    for case in selected:
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"case {case['id']} missing fields: {missing}")
        if case["expected_category"] not in {"A1", "A2", "B", "C"}:
            raise ValueError(f"case {case['id']} has unsupported expected_category")
    return selected


def chunk_fingerprint(chunks: Iterable[Mapping]) -> str:
    canonical = []
    for chunk in chunks:
        canonical.append({
            "id": chunk.get("id"),
            "content": chunk.get("content", ""),
            "section": chunk.get("section", "unknown"),
            "page_start": chunk.get("page_start", 0),
            "page_end": chunk.get("page_end", 0),
        })
    canonical.sort(key=lambda row: (str(row["id"]), row["page_start"], row["page_end"]))
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_shared_evidence_budget(rows: Sequence[Mapping], variants: Sequence[str]) -> None:
    """Fail when paired variants silently use different evidence constraints."""
    expected_variants = set(variants)
    by_case: dict[str, list[Mapping]] = {}
    for row in rows:
        by_case.setdefault(str(row["id"]), []).append(row)
    for case_id, paired in by_case.items():
        present = {str(row["variant"]) for row in paired}
        if present != expected_variants:
            raise ValueError(f"case {case_id} variant mismatch: {sorted(present)}")
        budgets = {
            (
                row.get("candidate_pool"),
                row.get("max_chunks"),
                row.get("max_chunk_chars"),
                row.get("char_budget"),
            )
            for row in paired
        }
        if len(budgets) != 1:
            raise ValueError(f"case {case_id} does not share one evidence budget")


def build_manifest(
    *,
    cases_path: Path,
    cases: Sequence[Mapping],
    variants: Sequence[str],
    budget: EvidenceBudget,
    corpus_fingerprints: Mapping[str, str] | None = None,
) -> dict:
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be nonempty and unique")
    budget.validate()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "retrieval_only": True,
        "answer_generation_called": False,
        "router_model_called": False,
        "cases_path": str(cases_path),
        "cases_sha256": sha256_file(cases_path),
        "case_count": len(cases),
        "case_ids": [case["id"] for case in cases],
        "variants": list(variants),
        "shared_budget": budget.to_dict(),
        "corpus_fingerprints": dict(corpus_fingerprints or {}),
        "gold_policy": "read_only; never modify expected fields to fit outputs",
    }
