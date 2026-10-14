import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from benchmark.experiments.ablation_protocol import chunk_fingerprint, sha256_file
from benchmark.experiments.common import pack_chunks, retrieval_metrics
from benchmark.experiments.replay_evidence_budget import (
    BASE, VARIANTS, load_verified_source, main, replay_row, verify_baseline,
)


CASE = {"id": "q", "thread_id": "p", "question": "What is the method?",
        "expected_category": "A1", "expected_terms": ["needle"], "expected_sections": ["method"]}


def chunk(cid, text):
    return {"id": cid, "content": text, "section": "method", "page_start": 1, "page_end": 1}


def make_source(root):
    source = root / "source"
    folder = source / "corpus_snapshot/chroma"
    folder.mkdir(parents=True)
    chunks = [chunk("p_0", "x" * 650 + " needle"), chunk("p_1", "background")]
    db_path = folder / "chroma.sqlite3"
    with contextlib.closing(sqlite3.connect(db_path)) as db, db:
        db.executescript("CREATE TABLE collections(id TEXT,name TEXT);"
                         "CREATE TABLE segments(id TEXT,collection TEXT,scope TEXT);"
                         "CREATE TABLE embeddings(id INTEGER,segment_id TEXT,embedding_id TEXT);"
                         "CREATE TABLE embedding_metadata(id INTEGER,key TEXT,string_value TEXT,int_value INTEGER);")
        db.execute("INSERT INTO collections VALUES('c','paper_p')")
        db.execute("INSERT INTO segments VALUES('s','c','METADATA')")
        for i, row in enumerate(chunks):
            db.execute("INSERT INTO embeddings VALUES(?,?,?)", (i, "s", row["id"]))
            for key, value in {"chroma:document": row["content"], "section": row["section"],
                               "page_start": 1, "page_end": 1}.items():
                db.execute("INSERT INTO embedding_metadata VALUES(?,?,?,?)",
                           (i, key, value if isinstance(value, str) else None,
                            value if isinstance(value, int) else None))
    cases = root / "cases.json"
    cases.write_text(json.dumps([CASE]), encoding="utf-8")
    budget = {"candidate_pool": 20, "max_chunks": 8, "max_chunk_chars": 600, "char_budget": 4000}
    manifest = {"case_count": 1, "case_ids": ["q"], "cases_path": str(cases),
                "cases_sha256": sha256_file(cases), "shared_budget": budget,
                "corpus_fingerprints": {"p": chunk_fingerprint(chunks)}}
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence, selected = pack_chunks(chunks)
    rows = [{"id": "q", "thread_id": "p", "expected_category": "A1", "variant": v,
             **budget, "candidate_chunk_ids": [r["id"] for r in chunks],
             "ranked_chunk_ids": [r["id"] for r in chunks],
             "selected_chunk_ids": [r["id"] for r in selected], "evidence_chars": len(evidence),
             **retrieval_metrics(CASE, evidence, ["method"])} for v in VARIANTS]
    (source / "retrieval_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return source, cases, db_path


class EvidenceBudgetTests(unittest.TestCase):
    def test_offline_end_to_end_preserves_snapshot_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, db_path = make_source(root)
            before = sha256_file(db_path)
            output = root / "output"
            argv = ["replay", "--source-run", str(source), "--output", str(output)]
            with patch("sys.argv", argv), patch("socket.socket", side_effect=AssertionError("network")), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(before, sha256_file(db_path))
            rows = [json.loads(line) for line in (output / "budget_rows.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["evidence_chars"] <= r["char_budget"] for r in rows))
            base = next(r for r in rows if r["config"] == BASE)
            expanded = next(r for r in rows if r["config"] == "c1200_b4000")
            self.assertFalse(base["term_hit"])
            self.assertEqual(base["missing_terms"][0]["reason"], "per_chunk_cap_loss")
            self.assertTrue(expanded["term_hit"])
            report_hash = sha256_file(output / "report.md")
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertEqual(report_hash, sha256_file(output / "report.md"))

    def test_source_cases_and_snapshot_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cases, db_path = make_source(Path(tmp))
            original = cases.read_text(encoding="utf-8")
            cases.write_text(original + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cases hash"):
                load_verified_source(source)
            cases.write_text(original, encoding="utf-8")
            with contextlib.closing(sqlite3.connect(db_path)) as db, db:
                db.execute("UPDATE embedding_metadata SET string_value='changed' WHERE key='chroma:document'")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_verified_source(source)

    def test_baseline_metrics_drift_and_missing_pair_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _, _ = make_source(Path(tmp))
            _, cases, rows, lookup = load_verified_source(source)
            with self.assertRaisesRegex(ValueError, "Baseline mismatch"):
                verify_baseline(cases["q"], {**rows[0], "term_hit": True}, lookup["p"])
            (source / "retrieval_rows.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one R0/R1 pair"):
                load_verified_source(source)

    def test_larger_chunk_cap_can_crowd_out_later_relevant_block(self):
        chunks = [chunk(f"p_{i}", "x" * 1200) for i in range(3)] + [chunk("p_3", "needle" + "x" * 1194)]
        lookup = {r["id"]: r for r in chunks}
        row = {"ranked_chunk_ids": list(lookup), "variant": "R0_vector"}
        small = replay_row(CASE, row, lookup, BASE)
        large = replay_row(CASE, row, lookup, "c1200_b4000")
        roomy = replay_row(CASE, row, lookup, "c1200_b8000")
        self.assertTrue(small["term_hit"])
        self.assertFalse(large["term_hit"])
        self.assertEqual(large["missing_terms"][0]["reason"], "total_budget_loss")
        self.assertTrue(roomy["term_hit"])
        alternate_gold = {**CASE, "expected_terms": ["different"], "expected_sections": ["conclusion"]}
        changed = replay_row(alternate_gold, row, lookup, BASE)
        self.assertEqual(small["evidence"], changed["evidence"])
        self.assertEqual(small["selected_chunk_ids"], changed["selected_chunk_ids"])


if __name__ == "__main__":
    unittest.main()
