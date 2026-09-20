import tempfile
import unittest
from pathlib import Path

import fitz

import database
from app import MAX_EVIDENCE_CHARS, MAX_EVIDENCE_CHUNKS, build_summary_evidence
from paper_parser import chunk_sections, detect_sections, extract_text_from_bytes


class ParserTests(unittest.TestCase):
    def test_removes_repeated_margin_noise_and_preserves_chunk_pages(self):
        pdf = fitz.open()
        for number in range(1, 5):
            page = pdf.new_page()
            page.insert_text((40, 25), "Example Author et al.", fontsize=9)
            if number == 1:
                page.insert_text((40, 140), "Abstract", fontsize=14)
                page.insert_text((40, 170), "Abstract body.", fontsize=10)
            elif number == 2:
                page.insert_text((40, 140), "2 Method", fontsize=14)
                page.insert_text((40, 170), "Method body.", fontsize=10)
            elif number == 3:
                page.insert_text((40, 140), "3 Results", fontsize=14)
                page.insert_text((40, 170), "Results body.", fontsize=10)
            else:
                page.insert_text((40, 140), "References", fontsize=14)
            page.insert_text((40, 815), f"365:{number}", fontsize=9)
        raw = pdf.tobytes()
        pdf.close()

        text, pages = extract_text_from_bytes(raw)
        sections, section_pages = detect_sections(text, pages)
        chunks = chunk_sections(sections, section_pages)

        self.assertNotIn("Example Author", text)
        self.assertNotIn("365:", text)
        self.assertTrue(any(c["section"] == "method" and c["page_start"] == 2 for c in chunks))
        self.assertTrue(any(c["section"] == "results" and c["page_start"] == 3 for c in chunks))


class EvidenceTests(unittest.TestCase):
    def test_structural_fallback_prefers_semantic_candidate_before_page_order(self):
        class FakeRag:
            def search(self, thread_id, query, k):
                return [
                    {"id": "selected_1", "content": "A" * 300, "section": "abstract", "page_start": 1, "page_end": 1},
                    {"id": "selected_2", "content": "B" * 300, "section": "abstract", "page_start": 1, "page_end": 1},
                    {"id": "relevant_method", "content": "C" * 300, "section": "method", "page_start": 9, "page_end": 9},
                ]

            def list_chunks(self, thread_id):
                return [
                    {"id": "early_method", "content": "E" * 300, "section": "method", "page_start": 2, "page_end": 2, "_order": 0},
                    {"id": "relevant_method", "content": "C" * 300, "section": "method", "page_start": 9, "page_end": 9, "_order": 1},
                ]

        evidence = build_summary_evidence(
            FakeRag(), "thread", queries=["q"], section_tiers=[{"method"}]
        )
        parts = [part for part in evidence.split("\n\n---\n\n") if part]
        self.assertLess(evidence.index("C" * 80), evidence.index("E" * 80))
        self.assertLessEqual(len(parts), MAX_EVIDENCE_CHUNKS)
        self.assertLessEqual(len(evidence), MAX_EVIDENCE_CHARS)


class MetricsTests(unittest.TestCase):
    def test_metric_storage_excludes_request_content(self):
        original_db_path = database.DB_PATH
        with tempfile.TemporaryDirectory() as tmp:
            database.DB_PATH = str(Path(tmp) / "metrics.db")
            try:
                database.init_db()
                database.record_request_metric({
                    "request_id": "request-1",
                    "thread_id": "thread-1",
                    "request_kind": "question",
                    "route_category": "A1",
                    "success": True,
                    "total_ms": 12.5,
                })
                conn = database.get_db_connection()
                try:
                    columns = {row[1] for row in conn.execute("PRAGMA table_info(request_metrics)")}
                    row = conn.execute("SELECT request_kind, route_category, success FROM request_metrics").fetchone()
                finally:
                    conn.close()
            finally:
                database.DB_PATH = original_db_path

        self.assertEqual(tuple(row), ("question", "A1", 1))
        self.assertNotIn("user_text", columns)
        self.assertNotIn("evidence", columns)
        self.assertNotIn("answer", columns)


if __name__ == "__main__":
    unittest.main()
