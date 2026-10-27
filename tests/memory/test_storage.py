
"""Tests for the SQLite long-term memory storage module."""

import tempfile
import unittest
from pathlib import Path

from memory.storage import (
    initialize_database,
    add_memory,
    get_memory,
    list_memories,
    update_memory,
    delete_memory,
)


class TestMemoryStorage(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"
        initialize_database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_initialization(self):
        self.assertTrue(self.db_path.exists())
        initialize_database(self.db_path)
        self.assertTrue(self.db_path.exists())

    def test_add_and_read_memory(self):
        memory_id = add_memory(
            101, "Любит заниматься творчеством",
            "fact", "private_dialog", self.db_path
        )

        memory = get_memory(101, memory_id, self.db_path)

        self.assertIsNotNone(memory)
        self.assertEqual(memory["user_id"], 101)
        self.assertEqual(
            memory["content"], "Любит заниматься творчеством"
        )
        self.assertEqual(memory["kind"], "fact")
        self.assertEqual(memory["source"], "private_dialog")
        self.assertTrue(memory["created_at"])
        self.assertTrue(memory["updated_at"])

    def test_user_isolation(self):
        memory_id = add_memory(
            101, "Личная информация пользователя 101",
            "fact", "test", self.db_path
        )

        self.assertIsNotNone(
            get_memory(101, memory_id, self.db_path)
        )
        self.assertIsNone(
            get_memory(202, memory_id, self.db_path)
        )
        self.assertEqual(
            list_memories(202, db_path=self.db_path), []
        )
        self.assertFalse(
            update_memory(
                202, memory_id, "Чужое изменение", self.db_path
            )
        )
        self.assertFalse(
            delete_memory(202, memory_id, self.db_path)
        )
        self.assertEqual(
            get_memory(101, memory_id, self.db_path)["content"],
            "Личная информация пользователя 101"
        )

    def test_list_memories(self):
        first_id = add_memory(
            101, "Первое воспоминание",
            "fact", "test", self.db_path
        )
        second_id = add_memory(
            101, "Второе воспоминание",
            "preference", "test", self.db_path
        )
        add_memory(
            202, "Данные другого пользователя",
            "fact", "test", self.db_path
        )

        memories = list_memories(101, db_path=self.db_path)

        self.assertEqual(len(memories), 2)
        self.assertEqual(
            [item["id"] for item in memories],
            [second_id, first_id]
        )
        self.assertTrue(
            all(item["user_id"] == 101 for item in memories)
        )

    def test_update_memory(self):
        memory_id = add_memory(
            101, "Старая информация",
            "hypothesis", "test", self.db_path
        )

        self.assertTrue(
            update_memory(
                101, memory_id, "Новая информация", self.db_path
            )
        )
        self.assertEqual(
            get_memory(101, memory_id, self.db_path)["content"],
            "Новая информация"
        )

    def test_delete_memory(self):
        memory_id = add_memory(
            101, "Информация для удаления",
            "fact", "test", self.db_path
        )

        self.assertTrue(
            delete_memory(101, memory_id, self.db_path)
        )
        self.assertIsNone(
            get_memory(101, memory_id, self.db_path)
        )
        self.assertFalse(
            delete_memory(101, memory_id, self.db_path)
        )

    def test_invalid_user_id(self):
        with self.assertRaises(ValueError):
            add_memory(
                0, "Тест", "fact", "test", self.db_path
            )

        with self.assertRaises(ValueError):
            list_memories(-1, db_path=self.db_path)

    def test_invalid_memory_data(self):
        with self.assertRaises(ValueError):
            add_memory(
                101, "", "fact", "test", self.db_path
            )

        with self.assertRaises(ValueError):
            add_memory(
                101, "Тест", "unknown", "test", self.db_path
            )

        with self.assertRaises(ValueError):
            add_memory(
                101, "Тест", "fact", "", self.db_path
            )

    def test_list_limit(self):
        for number in range(5):
            add_memory(
                101, f"Воспоминание {number}",
                "fact", "test", self.db_path
            )

        self.assertEqual(
            len(list_memories(101, limit=2, db_path=self.db_path)),
            2
        )

        with self.assertRaises(ValueError):
            list_memories(101, limit=0, db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()
