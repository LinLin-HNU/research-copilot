import json
import tempfile
import unittest
from pathlib import Path

from benchmark.experiments.judge_evidence_v2 import METRICS, normalize, prepare, summarize


class EvidenceJudgeTests(unittest.TestCase):
    def packet(self):
        return {'answer': 'Traffic gain is 22x.', 'generation_evidence': 'Traffic 464 10040 x 22',
                'reference_key_points': 'Traffic gain'}

    def scores(self):
        return {m: {'score': 1, 'reason': 'Supported.', 'insufficient_evidence': False,
                    'answer_quote': 'Traffic gain', 'evidence_quote': 'x 22'} for m in METRICS}

    def test_real_evidence_and_insufficiency(self):
        raw = self.scores()
        self.assertEqual(normalize(json.dumps(raw), self.packet())['evidence_faithfulness']['score'], 1)
        raw['evidence_faithfulness']['insufficient_evidence'] = True
        out = normalize(json.dumps(raw), self.packet())
        self.assertIsNone(out['evidence_faithfulness']['score'])
        self.assertEqual(out['evidence_faithfulness']['raw_score'], 1)
        self.assertEqual(out['completeness']['score'], 1)

    def test_fabricated_quote_and_boolean_score(self):
        raw = self.scores()
        raw['citation_support']['evidence_quote'] = 'not in evidence'
        self.assertIsNone(normalize(json.dumps(raw), self.packet())['citation_support']['score'])
        raw['completeness']['score'] = True
        with self.assertRaises(ValueError):
            normalize(json.dumps(raw), self.packet())

    def test_empty_context_is_not_hallucination(self):
        packet = {**self.packet(), 'generation_evidence': ''}
        out = normalize(json.dumps(self.scores()), packet)
        self.assertIsNone(out['evidence_faithfulness']['score'])

    def test_missing_pass_cannot_pass_consensus(self):
        manifest = {'review_ids': ['r'], 'passes': 2}
        reveal = {'r': {'id': 'case', 'variant': 'v1'}}
        rows = [{'review_id': 'r', 'pass': 1, 'error': None,
                 'metrics': normalize(json.dumps(self.scores()), self.packet())}]
        report = summarize(rows, manifest, reveal)
        self.assertEqual(report['missing_calls'], 1)
        self.assertEqual(report['overall']['v1']['completeness']['scored_n'], 0)
        with self.assertRaises(ValueError):
            summarize(rows * 2, manifest, reveal)

    def test_prepare_preserves_full_evidence_and_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mapping = {str(i): {'id': 'case', 'variant': v, 'question': 'Q',
                'thread_id': 't', 'gold': ['same gold'], 'answer': 'A',
                'evidence': 'long context ' * 1000 + 'x 22'}
                for i,v in enumerate(('v0', 'v1'))}
            source = base/'mapping.json'
            source.write_text(json.dumps(mapping), encoding='utf-8')
            packets = prepare(source, base/'out', 1)
            self.assertEqual(len(packets), 2)
            self.assertTrue(all(p['generation_evidence'].endswith('x 22') for p in packets))
            self.assertTrue(all('variant' not in p for p in packets))
            self.assertEqual(packets[0]['reference_key_points'], packets[1]['reference_key_points'])
            mapping['1']['gold'] = ['different gold']
            source.write_text(json.dumps(mapping), encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare(source, base/'bad')


if __name__ == '__main__':
    unittest.main()
