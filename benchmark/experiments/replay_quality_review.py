"""Replay a stored reviewer response against a read-only local source. No API."""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quality_review import claim_table, validate_findings


def replay(trace_path, source_path):
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    event = next(row for row in trace["events"] if row["step"] == "quality_review")
    selected = event.get("provided_segments")
    if selected is None:
        source = sqlite3.connect(Path(source_path).resolve().as_uri() + "?mode=ro", uri=True)
        selected, remaining = [], 12000
        hashes = {row["id"]: row["sha256"] for step in trace["events"]
                  if step["step"] == "full_context" for row in step["provided_segments"]}
        try:
            for cid in event["provided_evidence_ids"]:
                rows = source.execute(
                    "SELECT m.string_value FROM embedding_metadata m JOIN embeddings e ON m.id=e.id "
                    "WHERE e.embedding_id=? AND m.key='chroma:document'", (cid,)).fetchall()
                if len(rows) != 1:
                    raise ValueError("Source ID missing or ambiguous")
                full = rows[0][0]
                if hashlib.sha256(full.encode("utf-8")).hexdigest() != hashes.get(cid):
                    raise ValueError("Source changed since historical request")
                text = full[:min(1600, remaining)]
                selected.append({"id": cid, "text": text, "source_start": 0, "source_end": len(text),
                                 "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
                remaining -= len(text)
        finally:
            source.close()
    issues = json.loads(event["raw"])["issues"]
    accepted, decisions = validate_findings(issues, trace["answer"], selected,
                                            claim_table(trace["answer"], trace), legacy=True)
    return {"source_request_id": trace["request_id"], "network_calls": 0,
            "semantic_accuracy_proven": False, "text_matched": len(accepted),
            "rejected": len(decisions) - len(accepted), "decisions": decisions,
            "provided_segments": selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--chroma-db", default="resources/chroma_db/chroma.sqlite3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = replay(args.trace, args.chroma_db)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("Output exists; choose a new path to preserve previous results")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("network_calls", "text_matched", "rejected", "semantic_accuracy_proven")}))


if __name__ == "__main__":
    main()
