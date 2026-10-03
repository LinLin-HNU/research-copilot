import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from benchmark.experiments.run_llm_judge import parse_judgement
from benchmark.experiments.score_llm_judge import consensus

ROOT = Path(__file__).resolve().parents[1]


class LlmJudgeTests(unittest.TestCase):
    def test_parser_accepts_na_but_rejects_nonbinary_score(self):
        parsed = parse_judgement(json.dumps({
            "citation_correct": None,
            "grounded": 1,
            "hallucination": 0,
            "completeness": 1,
            "confidence": "medium",
            "insufficient_evidence": True,
            "reason": "Synthetic evidence is incomplete.",
        }))
        self.assertIsNone(parsed["citation_correct"])
        bad = json.dumps({**parsed, "completeness": 2})
        with self.assertRaises(ValueError):
            parse_judgement(bad)

    def test_consensus_is_strict(self):
        self.assertEqual(consensus([1, 1]), 1)
        self.assertIsNone(consensus([1, 0]))
        self.assertIsNone(consensus([1, None]))

    def test_scoring_is_labeled_ai_not_human(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            mapping = folder / "mapping.json"
            judgements = folder / "judgements.jsonl"
            output = folder / "report.json"
            mapping.write_text(json.dumps({
                "blind1": {"id": "case1", "variant": "v0_naive", "expected_category": "A1"},
                "blind2": {"id": "case1", "variant": "v1_current", "expected_category": "A1"},
            }), encoding="utf-8")
            rows = []
            for review_id, grounded in (("blind1", 0), ("blind2", 1)):
                for pass_number in (1, 2):
                    rows.append({
                        "review_id": review_id,
                        "pass": pass_number,
                        "error": None,
                        "citation_correct": grounded,
                        "grounded": grounded,
                        "hallucination": 1 - grounded,
                        "completeness": grounded,
                        "insufficient_evidence": False,
                    })
            judgements.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            result = subprocess.run([
                sys.executable,
                str(ROOT / "benchmark" / "experiments" / "score_llm_judge.py"),
                "--judgements", str(judgements),
                "--mapping", str(mapping),
                "--output", str(output),
            ], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["kind"], "llm_as_a_judge_not_human_review")
            self.assertNotIn("human_scored_cells", report)
            delta = next(row for row in report["paired_deltas"] if row["metric"] == "grounded")
            self.assertEqual(delta["difference_percentage_points"], 100)


if __name__ == "__main__":
    unittest.main()
