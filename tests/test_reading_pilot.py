import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.experiments.run_reading_pilot import prepare, run, recovery_output
from tests.test_paper_structure import Model, ROWS


class ReadingPilotTests(unittest.TestCase):
    def prepare_fixture(self, path):
        cases = [{"id": "sundial", "thread_id": "paper", "question": "Explain"},
                 {"id": "autoformer", "thread_id": "paper2", "question": "Explain"}]
        case_path = path / "cases.json"
        case_path.write_text(json.dumps(cases), encoding="utf-8")
        with patch("benchmark.experiments.run_reading_pilot.read_rows", return_value=ROWS):
            prepare(case_path, "unused", path / "pilot")
        return path / "pilot"

    def test_run_requires_confirmation_and_never_repeats_attempted_case(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, "confirm-paid"):
                run(output, "sundial", False)
            model = Model()
            model._base = "https://example.invalid/v1"
            result = run(output, "sundial", True, model_factory=lambda: model)
            self.assertEqual(result["attempted_calls"], 3)
            self.assertEqual([call[0] for call in model.calls], ["full_context", "source_structure", "structured_context"])
            with self.assertRaisesRegex(ValueError, "already attempted"):
                run(output, "sundial", True, model_factory=lambda: model)
            self.assertTrue((output / "sundial_comparison.html").is_file())
            self.assertEqual(len(model.calls), 3)

    def test_tampered_frozen_cases_abort_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["cases"][0]["question"] = "changed"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch("deep_research_runtime.AliyunResearchModel") as model:
                with self.assertRaisesRegex(ValueError, "manifest changed"):
                    run(output, "sundial", True)
                model.assert_not_called()

    def test_call_journal_and_response_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            model = Model()
            model._base = "https://example.invalid/v1"
            original_call = model.call

            def inspect_before_dispatch(stage, payload, **limits):
                state = json.loads((output / "sundial_attempt.json").read_text(encoding="utf-8"))
                self.assertEqual(state["calls"][-1]["status"], "dispatch_pending")
                self.assertEqual(state["attempted_calls"], len(model.calls) + 1)
                return original_call(stage, payload, **limits)

            model.call = inspect_before_dispatch
            run(output, "sundial", True, model_factory=lambda: model)
            state = json.loads((output / "sundial_attempt.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["attempted_calls"], 3)
            self.assertEqual(len(list(output.glob("sundial_call_*_response.json"))), 3)
            self.assertTrue(all(row["status"] == "returned" for row in state["calls"]))

    def test_failed_initialization_records_zero_calls_and_error_type(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            def fail():
                raise RuntimeError("do not persist secrets in exception messages")
            with self.assertRaises(RuntimeError):
                run(output, "sundial", True, model_factory=fail)
            raw = (output / "sundial_attempt.json").read_text(encoding="utf-8")
            state = json.loads(raw)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["attempted_calls"], 0)
            self.assertEqual(state["error_type"], "RuntimeError")
            self.assertNotIn("secrets", raw)

    def test_failed_call_is_logged_and_does_not_start_next_variant(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            model = Model()
            model._base = "https://example.invalid/v1"
            def fail(*args, **kwargs):
                raise TimeoutError("secret detail")
            model.call = fail
            result = run(output, "sundial", True, model_factory=lambda: model)
            state = json.loads((output / "sundial_attempt.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(state["attempted_calls"], 1)
            self.assertEqual(state["calls"][0]["error_type"], "TimeoutError")
            self.assertEqual(state["stop_reason"], "model_call_failed")
            self.assertFalse((output / "sundial_structured.json").exists())

    def test_recovery_preserves_legacy_attempt_and_is_single_use(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            legacy = '{"case":"sundial","status":"started","max_calls":3}'
            old = output / "sundial_attempt.json"
            old.write_text(legacy, encoding="utf-8")
            retry = recovery_output(output, "sundial")
            self.assertEqual(old.read_text(encoding="utf-8"), legacy)
            self.assertEqual((retry / "manifest.json").read_bytes(), (output / "manifest.json").read_bytes())
            model = Model()
            model._base = "https://example.invalid/v1"
            result = run(retry, "sundial", True, model_factory=lambda: model)
            self.assertEqual(result["attempted_calls"], 3)
            with self.assertRaises(FileExistsError):
                recovery_output(output, "sundial")
            self.assertTrue((retry / "recovery.json").is_file())
            with self.assertRaisesRegex(ValueError, "Nested"):
                recovery_output(retry, "sundial")

    def test_completed_case_cannot_be_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            output = self.prepare_fixture(Path(temp))
            (output / "sundial_attempt.json").write_text('{"status":"complete"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Completed"):
                recovery_output(output, "sundial")


if __name__ == "__main__":
    unittest.main()
