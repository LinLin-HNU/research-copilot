import unittest
from deep_research_runtime import AliyunResearchModel


class ReadingStyleTests(unittest.TestCase):
    def test_all_paper_writers_share_overview_and_local_style(self):
        for stage in ("final", "full_context", "structured_context"):
            with self.subTest(stage=stage):
                instruction = AliyunResearchModel._instruction(stage)
                self.assertIn("背景最多一句", instruction)
                self.assertIn("问题→核心机制→输入输出流程→少量关键实验", instruction)
                self.assertIn("只有用户明确要求推导/数学细节", instruction)
                self.assertIn("不得伪装为paper_fact", instruction)
                self.assertNotIn("advanced 在 beginner 基础上补充公式", instruction)

    def test_plan_does_not_expand_local_question_scope(self):
        instruction = AliyunResearchModel._instruction("plan")
        self.assertIn("局部机制追问只规划本轮机制", instruction)

    def test_source_extraction_is_not_replaced_by_answer_style(self):
        instruction = AliyunResearchModel._instruction("source_structure")
        self.assertNotIn("讲解风格约束", instruction)
        self.assertIn("quote逐字复制", instruction)
