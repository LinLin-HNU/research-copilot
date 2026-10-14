import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import database


class AdaptiveMemoryTests(unittest.TestCase):
    def test_thread_history_roundtrip_is_ordered_bounded_and_isolated(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(database, "DB_PATH", str(Path(tmp) / "test.db")):
            database.init_db()
            database.create_session("paper-a", "A")
            database.create_session("paper-b", "B")
            for index in range(5):
                database.record_adaptive_turn({
                    "request_id": f"req-{index}",
                    "thread_id": "paper-a",
                    "question": f"问题{index}",
                    "answer": f"回答{index}",
                    "strategy": "auto",
                    "mode": "paper_deep",
                    "status": "ok",
                })
            database.record_adaptive_turn({
                "request_id": "other", "thread_id": "paper-b",
                "question": "其他论文", "answer": "隔离回答",
                "strategy": "deep", "mode": "paper_full", "status": "ok",
            })

            history = database.list_adaptive_turns("paper-a", limit=3)

        self.assertEqual([item["question"] for item in history], ["问题2", "问题3", "问题4"])
        self.assertTrue(all(item["thread_id"] == "paper-a" for item in history))

    def test_deleting_session_also_deletes_adaptive_history(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(database, "DB_PATH", str(Path(tmp) / "test.db")):
            database.init_db()
            database.create_session("paper-a", "A")
            database.record_adaptive_turn({
                "request_id": "req", "thread_id": "paper-a", "question": "问题",
                "answer": "回答", "strategy": "auto", "mode": "paper_full", "status": "ok",
            })
            database.delete_session("paper-a")
            self.assertEqual(database.list_adaptive_turns("paper-a"), [])


if __name__ == "__main__":
    unittest.main()
