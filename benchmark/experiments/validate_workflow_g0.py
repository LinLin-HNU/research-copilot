"""Offline structural check for the draft single-Agent workflow pilot.

This deliberately does not load .env, PDF text, Chroma, or any model client.
It prints only counts and file hashes, never questions or private thread IDs.
"""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


ALLOWED_MODES = {"paper_deep", "paper_deep_no_paper", "general"}
ALLOWED_SPLITS = {"dev", "reserve"}
FORBIDDEN_RUNTIME_KEYS = {"thread_id", "gold", "expected_terms", "expected_sections", "expected_category"}
REQUIRED_KEYS = {"id", "split", "mode", "paper_alias", "question", "subquestions", "source_case_ids"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(cases: list, source_cases: list) -> dict:
    if not isinstance(cases, list) or not isinstance(source_cases, list):
        raise ValueError("Both inputs must contain JSON arrays")
    source_by_id = {row["id"]: row for row in source_cases}
    if len(source_by_id) != len(source_cases):
        raise ValueError("Duplicate source case id")
    seen = set()
    counts = Counter()
    papers = set()
    for row in cases:
        if not isinstance(row, dict) or set(row) != REQUIRED_KEYS:
            raise ValueError("Workflow case fields differ from the runtime-safe schema")
        if set(row) & FORBIDDEN_RUNTIME_KEYS:
            raise ValueError("Gold or thread metadata leaked into workflow cases")
        case_id = row["id"]
        if not isinstance(case_id, str) or not case_id.startswith("wf-") or case_id in seen:
            raise ValueError("Missing, malformed, or duplicate workflow case id")
        seen.add(case_id)
        split, mode, alias = row["split"], row["mode"], row["paper_alias"]
        if split not in ALLOWED_SPLITS or mode not in ALLOWED_MODES:
            raise ValueError(f"Invalid split or mode in {case_id}")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"Missing question in {case_id}")
        sub = row["subquestions"]
        if not isinstance(sub, list) or not 1 <= len(sub) <= 2 or any(
            not isinstance(item, str) or not item.strip() for item in sub
        ):
            raise ValueError(f"Expected one or two nonempty subquestions in {case_id}")
        source_ids = row["source_case_ids"]
        if not isinstance(source_ids, list) or len(source_ids) > 2 or len(set(source_ids)) != len(source_ids):
            raise ValueError(f"Invalid source references in {case_id}")
        if mode == "paper_deep":
            if not isinstance(alias, str) or not alias:
                raise ValueError(f"Paper task lacks paper or source cases: {case_id}")
            papers.add(alias)
            if alias == "survey":
                if source_ids:
                    raise ValueError("Survey uses its local PDF, not an old AI-reviewed case")
            else:
                if not source_ids:
                    raise ValueError(f"Paper task lacks source cases: {case_id}")
                for source_id in source_ids:
                    source = source_by_id.get(source_id)
                    if source is None or source.get("split") != "dev" or not source_id.startswith(alias + "-"):
                        raise ValueError(f"Unknown or cross-paper source reference in {case_id}")
            counts[f"paper_{split}"] += 1
        else:
            if alias is not None or source_ids or split != "dev":
                raise ValueError(f"Control task must not carry paper sources: {case_id}")
            counts[mode] += 1
    if (counts["paper_dev"], counts["paper_reserve"], counts["paper_deep_no_paper"], counts["general"]) != (8, 2, 1, 1):
        raise ValueError("Expected eight dev paper tasks, two reserve paper tasks, and two controls")
    return {"status": "ok", "cases": len(cases), "paper_dev": counts["paper_dev"],
            "paper_reserve": counts["paper_reserve"], "controls": 2,
            "distinct_papers": len(papers), "network_calls": 0,
            "warning": "Structural check only; not an answer-quality result."}


def validate_survey_source(source_path: Path) -> dict:
    """Read-only provenance check; never returns private collection or session IDs."""
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("paper_alias") != "survey":
        raise ValueError("Wrong survey source binding")
    pdf = Path(source["local_pdf"])
    if not pdf.is_file():
        raise ValueError("Survey PDF is missing or invalid")
    with pdf.open("rb") as stream:
        header = stream.read(5)
    if header != b"%PDF-":
        raise ValueError("Survey PDF is missing or invalid")
    if sha256(pdf) != source["pdf_sha256"]:
        raise ValueError("Survey PDF fingerprint changed")
    import fitz
    with fitz.open(pdf) as doc:
        if len(doc) != source["pdf_pages"]:
            raise ValueError("Survey PDF page count changed")
    snapshot = (source_path.parent / source["snapshot"] / "chroma.sqlite3").resolve()
    conn = sqlite3.connect(f"file:{snapshot.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT d.string_value, p.int_value FROM collections c "
            "JOIN segments s ON s.collection=c.id "
            "JOIN embeddings e ON e.segment_id=s.id "
            "JOIN embedding_metadata t ON t.id=e.id AND t.key='title' "
            "JOIN embedding_metadata d ON d.id=e.id AND d.key='chroma:document' "
            "JOIN embedding_metadata p ON p.id=e.id AND p.key='page_start' "
            "WHERE c.name=? AND t.string_value LIKE 'Time Series Forecasting Methods%'",
            (source["collection_name"],),
        ).fetchall()
    finally:
        conn.close()
    if not rows or len(rows) != source["indexed_chunks"]:
        raise ValueError("Survey indexed chunk count changed")
    pages = [page for _, page in rows]
    if [min(pages), max(pages)] != source["indexed_page_range"]:
        raise ValueError("Survey indexed page range changed")
    content_hash = hashlib.sha256("".join(sorted(text for text, _ in rows)).encode()).hexdigest()
    if not content_hash.startswith(source["indexed_content_sha256_prefix"]):
        raise ValueError("Survey indexed content changed")
    return {"pdf_sha256": source["pdf_sha256"], "pdf_pages": source["pdf_pages"],
            "indexed_chunks": len(rows), "indexed_page_range": [min(pages), max(pages)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--survey-source", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    source_cases = json.loads(args.source_cases.read_text(encoding="utf-8"))
    result = validate(cases, source_cases)
    if any(row["paper_alias"] == "survey" for row in cases):
        if args.survey_source is None:
            parser.error("Survey task requires --survey-source")
        result["survey_source"] = validate_survey_source(args.survey_source)
    result["cases_sha256"] = sha256(args.cases)
    result["source_cases_sha256"] = sha256(args.source_cases)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
