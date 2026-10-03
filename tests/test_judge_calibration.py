import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmark' / 'experiments' / 'judge_calibration.py'
SPEC = importlib.util.spec_from_file_location('judge_calibration_for_test', SCRIPT)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


class JudgeCalibrationTests(unittest.TestCase):
    def test_expected_labels_are_private_and_missing_errors_cannot_pass_as_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            prepared = base / 'prepared'
            packets = calibration.prepare(prepared)
            self.assertEqual(len(packets), 8)
            self.assertTrue(all('expected' not in p and 'variant' not in p for p in packets))
            expected = json.loads((prepared / 'expected.private.json').read_text(encoding='utf-8'))
            manifest = {'protocol': calibration.JUDGE_PROTOCOL,
                        'review_ids': [p['review_id'] for p in packets], 'passes': 1,
                        'packets_sha256': calibration.sha(prepared / 'packets.json')}
            rows = [{'review_id': rid, 'pass': 1, 'error': None,
                     'metrics': {metric: {'score': value, 'normalization_reasons': [],
                                         'insufficient_evidence': value is None,
                                         'reason': 'Synthetic test fixture'}
                                 for metric, value in item['expected'].items()}}
                    for rid, item in expected.items()]
            run_dir = base / 'run'
            run_dir.mkdir()
            calibration.write_json(run_dir / 'manifest.json', manifest)
            (run_dir / 'judgements.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            result = calibration.check(run_dir, prepared)
            self.assertEqual(result['passed_checks'], 24)
            self.assertTrue(result['ready_for_small_comparison'])
            bad = copy.deepcopy(rows)
            bad[1]['metrics']['evidence_faithfulness']['score'] = 1
            self.assertFalse(calibration.summarize(bad, manifest, expected)['critical_faithfulness_passed'])
            bad = copy.deepcopy(rows)
            bad[6] = {'review_id': 'cal_07', 'pass': 1, 'error': {'type': 'TimeoutError'}}
            failed = calibration.summarize(bad, manifest, expected)
            self.assertEqual(failed['passed_checks'], 21)
            self.assertFalse(failed['all_checks_passed'])
            missing = calibration.summarize(rows[:-1], manifest, expected)
            self.assertEqual(missing['missing_calls'], 1)
            self.assertEqual(missing['passed_checks'], 21)
            with self.assertRaises(ValueError):
                calibration.summarize(rows + [rows[0]], manifest, expected)
            with (prepared / 'packets.json').open('a', encoding='utf-8') as handle:
                handle.write(' ')
            with self.assertRaises(ValueError):
                calibration.check(run_dir, prepared)


if __name__ == '__main__':
    unittest.main()
