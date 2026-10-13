import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark.experiments import three_way_pilot as pilot
from benchmark.experiments.three_way_cases import CASES, TITLE_CUES
from benchmark.experiments.three_way_legacy import definitions


class ThreeWayTests(unittest.TestCase):
    def fixture(self, base):
        source = base / "source"
        source.mkdir()
        with sqlite3.connect(source / "chroma.sqlite3") as db:
            db.execute("CREATE TABLE embeddings(id INTEGER PRIMARY KEY,embedding_id TEXT)")
            db.execute("CREATE TABLE embedding_metadata(id INTEGER,key TEXT,string_value TEXT,int_value INTEGER)")
            for n, (alias, cue) in enumerate(TITLE_CUES.items(), 1):
                db.execute("INSERT INTO embeddings VALUES (?,?)", (n, f"{alias}_0"))
                for key, value in {"title": cue, "section": "method", "chroma:document": "Source text " * 20}.items():
                    db.execute("INSERT INTO embedding_metadata VALUES (?,?,?,NULL)", (n, key, value))
                db.execute("INSERT INTO embedding_metadata VALUES (?, 'page_start',NULL,1)", (n,))
        db.close()
        output = base / "pilot"
        frozen = {name: "# frozen test source" for name in pilot.FROZEN_FILES}
        def git(args, **kwargs):
            return "frozen-test-commit\n" if args[1] == "rev-parse" else frozen[args[-1].split(":", 1)[1]].encode()
        with patch.object(pilot, "private", side_effect=lambda p: Path(p)), \
             patch.object(pilot, "model_settings", return_value=SimpleNamespace(name="test-model", host="example.invalid", base_url="https://example.invalid/v1")), \
             patch.object(pilot.subprocess, "check_output", side_effect=git):
            result = pilot.prepare(source, output, "test-ref")
        return output, result

    def test_prepare_is_offline_and_no_approval_is_inferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, result = self.fixture(Path(tmp))
            self.assertEqual(result["network_calls"], 0)
            self.assertEqual(result["first_batch_max_model_calls"], 24)
            manifest = pilot.checked_manifest(output)
            self.assertFalse(manifest["review_enabled"])
            self.assertFalse(manifest["structured_enabled"])
            self.assertEqual(len(manifest["cases"]), 8)
            self.assertFalse((output / "approval.json").exists())
            with self.assertRaises(ValueError):
                pilot.run_batch(output, 1, False)
            self.assertFalse((output / "batch_1.started").exists())

    def test_manifest_drift_and_approval_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self.fixture(Path(tmp))
            self.assertEqual(pilot.approve(output)["network_calls"], 0)
            with self.assertRaises(ValueError):
                pilot.approve(output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["question"] = "changed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                pilot.checked_manifest(output)

    def test_review_keeps_missing_rows_without_disclosing_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, _ = self.fixture(Path(tmp))
            result = pilot.render(output, 1)
            self.assertEqual(result["review_rows"], 12)
            page = (output / "review_batch_1/answers.html").read_text(encoding="utf-8")
            self.assertIn("尚未运行", page)
            self.assertNotIn("legacy_v1", page)
            sheet = json.loads((output / "review_batch_1/review.json").read_text(encoding="utf-8"))
            self.assertNotIn("variant", sheet[0])
            self.assertIsNone(sheet[0]["supported_core_facts"])
            with self.assertRaises(FileExistsError):
                pilot.render(output, 1)

    def test_definition_loading_skips_imports_calls_and_decorators(self):
        source = "import forbidden\nunsafe=forbidden()\nCAP=6000\n@unavailable\ndef use(): return CAP\n"
        scope = definitions(source, {"use"}, {})
        self.assertEqual(scope["use"](), 6000)
        self.assertNotIn("unsafe", scope)

    def test_call_budget_journals_failure_without_another_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("deep_research_runtime.AliyunResearchModel"):
                model = pilot.RecordingModel(Path(tmp), "case", "variant", 1)
            result = model.dispatch("fake", lambda: {"usage": {"input_tokens": 1, "output_tokens": 1}})
            self.assertEqual(result["usage"]["input_tokens"], 1)
            with self.assertRaises(RuntimeError):
                model.dispatch("again", lambda: self.fail("second dispatch"))
            record = json.loads((Path(tmp) / "case_variant_calls.json").read_text(encoding="utf-8"))
            self.assertEqual(record[0]["state"], "returned")

    def test_draft_questions_have_no_gold_and_known_batch_members(self):
        self.assertEqual(len({c["id"] for c in CASES}), 8)
        self.assertEqual([c["paper"] for c in CASES if c["batch"] == 1], ["informer", "autoformer", "patchtst", "fedformer"])
        self.assertFalse(any("expected_terms" in c or "expected_category" in c for c in CASES))
