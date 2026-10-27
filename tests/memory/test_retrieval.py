
"""Tests for user-scoped memory retrieval."""

import tempfile
import unittest
from pathlib import Path

from memory.retrieval import rank_memories, retrieve_memories
from memory.storage import initialize_database, add_memory


class TestMemoryRetrieval(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"
        initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_find_matching_memory(self):
        add_memory(
            100, "Любит читать книги",
            "preference", "message:1", self.db_path
        )
        result = retrieve_memories(
            100, "Какие книги любит?", db_path=self.db_path
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["user_id"], 100)

    def test_no_matching_memory(self):
        add_memory(
            100, "Любит читать книги",
            "preference", "message:1", self.db_path
        )
        result = retrieve_memories(
            100, "Погода сегодня", db_path=self.db_path
        )
        self.assertEqual(result, [])

    def test_user_isolation(self):
        add_memory(
            100, "Любит кошек",
            "preference", "message:1", self.db_path
        )
        add_memory(
            200, "Любит собак",
            "preference", "message:2", self.db_path
        )
        result = retrieve_memories(
            100, "Любит собак", db_path=self.db_path
        )
        self.assertEqual(result, [])

    def test_result_limit(self):
        for index in range(10):
            add_memory(
                100, f"Любит музыку номер {index}",
                "preference", f"message:{index}", self.db_path
            )
        result = retrieve_memories(
            100, "Любит музыку", limit=3, db_path=self.db_path
        )
        self.assertEqual(len(result), 3)

    def test_relevance_ranking(self):
        records = [
            {
                "id": 1,
                "content": "Любит книги",
                "kind": "preference",
            },
            {
                "id": 2,
                "content": "Любит книги и музыку",
                "kind": "preference",
            },
        ]
        result = rank_memories(
            "Любит книги музыку", records
        )
        self.assertEqual(result[0]["id"], 2)

    def test_hypothesis_is_not_promoted_to_fact(self):
        records = [
            {
                "id": 1,
                "content": "Любит путешествия",
                "kind": "hypothesis",
            }
        ]
        result = rank_memories("путешествия", records)
        self.assertEqual(result[0]["kind"], "hypothesis")

    def test_invalid_user_id(self):
        with self.assertRaises(ValueError):
            retrieve_memories(
                0, "книги", db_path=self.db_path
            )

    def test_invalid_query(self):
        with self.assertRaises(ValueError):
            retrieve_memories(
                100, "", db_path=self.db_path
            )

    def test_invalid_limit(self):
        with self.assertRaises(ValueError):
            retrieve_memories(
                100, "книги", limit=501,
                db_path=self.db_path
            )

    def test_database_not_modified(self):
        memory_id = add_memory(
            100, "Любит книги",
            "preference", "message:1", self.db_path
        )
        retrieve_memories(
            100, "книги", db_path=self.db_path
        )
        from memory.storage import get_memory
        record = get_memory(
            100, memory_id, self.db_path
        )
        self.assertEqual(record["content"], "Любит книги")


if __name__ == "__main__":
    unittest.main()
