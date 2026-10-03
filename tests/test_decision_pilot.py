import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmark' / 'experiments' / 'run_decision_pilot.py'
SPEC = importlib.util.spec_from_file_location('decision_pilot_for_test', SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


class DecisionPilotTests(unittest.TestCase):
    def chunk(self, identifier, content):
        return {'id': identifier, 'content': content, 'section': 'Results',
                'page_start': 3, 'page_end': 4}

    def test_budget_includes_literal_headers_and_separators(self):
        chunks = [self.chunk('one', 'Alpha result.'), self.chunk('two', 'Beta result.')]
        first = '【来源: Results | 第 3-4 页 | chunk_id: one】\nAlpha result.'
        second = '【来源: Results | 第 3-4 页 | chunk_id: two】\nBeta result.'
        expected = first + '\n\n---\n\n' + second
        text, selected = pilot.pack_complete(chunks, budget=len(expected))
        self.assertEqual(text, expected)
        self.assertEqual([c['id'] for c in selected], ['one', 'two'])
        text, selected = pilot.pack_complete(chunks, budget=len(expected) - 1)
        self.assertEqual(text, first)
        self.assertEqual([c['id'] for c in selected], ['one'])
        self.assertLessEqual(len(text), len(expected) - 1)

    def test_oversized_chunk_is_skipped_without_truncating_or_stopping(self):
        too_big = self.chunk('huge', 'A' * 1000 + 'IMPORTANT TAIL MUST NOT BE CUT')
        fitting = self.chunk('small', 'A complete supporting sentence with its tail intact.')
        text, selected = pilot.pack_complete([too_big, fitting], budget=150)
        self.assertEqual(selected, [fitting])
        self.assertNotIn('huge', text)
        self.assertTrue(text.endswith(fitting['content']))
        self.assertLessEqual(len(text), 150)
        self.assertEqual(too_big['content'], 'A' * 1000 + 'IMPORTANT TAIL MUST NOT BE CUT')

    def test_duplicate_ids_blank_content_and_max_chunks(self):
        first = self.chunk('one', 'First original chunk.')
        same_id = self.chunk('one', 'Duplicate id must not consume the next slot.')
        second = self.chunk('two', 'Second original chunk.')
        third = self.chunk('three', 'Third original chunk.')
        text, selected = pilot.pack_complete(
            [self.chunk('blank', '  \n'), first, same_id, second, third], budget=1000, max_chunks=2)
        self.assertEqual(selected, [first, second])
        self.assertEqual(text.count('chunk_id: one'), 1)
        self.assertNotIn('Duplicate id', text)
        self.assertNotIn('Third original', text)

    def test_prepare_freezes_18_cases_preserves_gold_and_separates_input_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project, output = base / 'fake_project', base / 'prepared'
            source = project / 'benchmark/drafts/timeseries_20260922_curated_release2/ai_reviewed_dev_cases.json'
            source.parent.mkdir(parents=True)
            original = []
            for i, identifier in enumerate(pilot.IDS):
                original.append({'id': identifier, 'question': 'Synthetic question ' + identifier,
                    'thread_id': 'thread-' + identifier.split('-')[0], 'split': 'dev',
                    'expected_category': 'B' if i == 0 else identifier.split('-')[1].upper(),
                    'expected_terms': ['term-' + identifier], 'expected_sections': ['Results'],
                    'gold': {'key_points': ['point-' + identifier], 'provenance': 'synthetic test fixture'},
                    'custom_metadata': {'nested': [identifier, i]}})
            source.write_text(json.dumps(original), encoding='utf-8')
            source_hash = pilot.sha(source)
            pilot.prepare(project, output)
            cases = json.loads((output / 'cases.json').read_text(encoding='utf-8'))
            protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
            self.assertEqual(len(cases), 18)
            self.assertEqual(len({c['id'] for c in cases}), 18)
            paper, general = cases[:12], cases[12:]
            self.assertEqual(len({c['thread_id'] for c in paper}), 9)
            self.assertEqual([c['id'] for c in paper], pilot.IDS)
            for prior, prepared in zip(original, paper):
                self.assertEqual(prepared, {**prior, 'input_mode': 'paper'})
            # User input mode and gold category are distinct: even a synthetic B-labelled
            # source case remains paper mode; added general cases are explicitly general.
            self.assertEqual(paper[0]['expected_category'], 'B')
            self.assertEqual(paper[0]['input_mode'], 'paper')
            self.assertTrue(all(c['input_mode'] == 'general' and c['expected_category'] == 'B'
                                and c['expected_sections'] == [] and c['expected_terms'] == []
                                for c in general))
            self.assertEqual([c['thread_id'] for c in general], [c['thread_id'] for c in paper[:6]])
            self.assertTrue(all('not human gold' in c['gold']['provenance'] for c in general))
            self.assertEqual(protocol['paper_n'], 12)
            self.assertEqual(protocol['general_n'], 6)
            self.assertEqual(protocol['source_sha256'], source_hash)
            self.assertEqual(protocol['cases_sha256'], pilot.sha(output / 'cases.json'))
            self.assertEqual(pilot.sha(source), source_hash)
            with self.assertRaises(FileExistsError):
                pilot.prepare(project, output)


if __name__ == '__main__':
    unittest.main()
