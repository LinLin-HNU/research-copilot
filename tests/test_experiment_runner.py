"""Run the real experiment control flow with synthetic model/RAG; no API calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from benchmark.experiments import run_experiment


class RunnerTests(unittest.TestCase):
    def test_both_paths_capture_answers_and_usage_without_live_writes(self):
        class FakeRag:
            def __init__(self, path):
                self.client = self
            def get_collection(self, name):
                return self
            def list_chunks(self, thread):
                return [{'id':'fake_0', 'content':'Pile synthetic facts. '*20,
                         'section':'method', 'page_start':1, 'page_end':1}]
            def search(self, thread, query, k):
                return self.list_chunks(thread)
        for variant in ('v0_naive', 'v1_current', 'v2_experimental'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                cases = folder/'cases.json'
                cases.write_text(json.dumps([{'id':'fake','question':'Synthetic question','thread_id':'fake',
                    'expected_category':'A1','expected_terms':['Pile'],'expected_sections':['method']}]),encoding='utf-8')
                answer = AIMessage(content='Synthetic Pile answer', usage_metadata={'input_tokens':11,'output_tokens':7,'total_tokens':18})
                route = AIMessage(content='{"category":"A1"}', usage_metadata={'input_tokens':3,'output_tokens':2,'total_tokens':5})
                model = GenericFakeChatModel(messages=iter([answer] if variant == 'v0_naive' else [route,answer]))
                argv = ['run_experiment','--cases',str(cases),'--variant',variant,'--limit','1','--output',str(folder/'out')]
                with patch('sys.argv',argv), patch('config.model',model), patch('rag_store.PaperRAG',FakeRag), patch('rag_store._embed',side_effect=lambda texts:[[1.,0.] for _ in texts]), patch('shutil.copytree'), patch('requests.post',side_effect=AssertionError('Network forbidden')):
                    run_experiment.main()
                row = json.loads((folder/'out'/f'answers_{variant}.jsonl').read_text(encoding='utf-8'))
                self.assertIsNone(row['error'])
                self.assertEqual(row['answer'],'Synthetic Pile answer')
                self.assertTrue(row['retrieval_hit'])
                self.assertEqual(row['input_tokens'],11)
                if variant != 'v0_naive':
                    self.assertEqual(row['routing_usage']['input_tokens'],3)


if __name__ == '__main__':
    unittest.main()
