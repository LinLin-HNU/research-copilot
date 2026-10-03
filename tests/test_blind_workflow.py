"""Synthetic fixtures only: tests are not empirical model results."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'benchmark/experiments'


class BlindWorkflowTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run([sys.executable, '-X', 'utf8', str(SCRIPTS/name), *map(str,args)], capture_output=True, text=True, encoding='utf-8')

    def test_roundtrip_blank_not_zero_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            inputs = []
            for variant in ('v0_naive','v1_current'):
                path = folder/f'{variant}.jsonl'
                path.write_text(json.dumps({'id':'fake1','question':'SYNTHETIC question','answer':'SYNTHETIC answer',
                    'variant':variant,'thread_id':'fake','expected_category':'A1','error':None,'retrieval_hit':True}), encoding='utf-8')
                inputs.append(path)
            made = self.run_script('make_blind_sheet.py', *inputs, '--output', folder/'blind')
            self.assertEqual(made.returncode, 0, made.stderr)
            sheet, mapping = folder/'blind/review.csv', folder/'blind/mapping.private.json'
            scored = self.run_script('score_blind_sheet.py', '--review',sheet,'--mapping',mapping,'--output',folder/'scores.json')
            self.assertEqual(scored.returncode,0,scored.stderr)
            report = json.loads((folder/'scores.json').read_text())
            self.assertEqual(report['human_scored_cells'],0)
            self.assertIsNone(report['overall']['v0_naive']['grounded']['rate'])
            with sheet.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                fields=reader.fieldnames
                rows=list(reader)
            rows[0]['answer']='altered answer'
            with sheet.open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            bad=self.run_script('score_blind_sheet.py','--review',sheet,'--mapping',mapping,'--output',folder/'bad.json')
            self.assertNotEqual(bad.returncode,0)


if __name__ == '__main__':
    unittest.main()
