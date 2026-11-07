
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

    def test_compound_recall_finds_separate_facts(self):
        records = [
            dict(id=1, content="Пользователь изучает биологию", kind="fact"),
            dict(id=2, content="Пользователь готовится к экзамену по биологии", kind="fact"),
            dict(id=3, content="Пользователь предпочитает краткие ответы", kind="preference"),
        ]
        question = "Что ты помнишь о моём изучении биологии и подготовке к экзамену? Как они связаны?"
        self.assertEqual({r["id"] for r in rank_memories(question, records)}, {1, 2})

    def test_compound_recall_is_not_specific_to_biology(self):
        records = [dict(id=1, content="Изучает французский язык", kind="fact"),
                   dict(id=2, content="Готовится к поездке во Францию", kind="fact"),
                   dict(id=3, content="Играет в футбол", kind="fact")]
        result = rank_memories("Вспомни о французском языке и поездке во Францию", records)
        self.assertEqual({r["id"] for r in result}, {1, 2})

    def test_more_complete_recall_match_ranks_first(self):
        records = [dict(id=2, content="Изучает биологию", kind="fact"),
                   dict(id=1, content="Экзамен по биологии", kind="fact")]
        self.assertEqual([r["id"] for r in rank_memories(
            "Помнишь о биологии и экзамене?", records)], [1, 2])

    def test_recall_preference_verb_alone_cannot_match_another_topic(self):
        records = [dict(id=1, content="Любит собак", kind="preference")]
        self.assertEqual(rank_memories("Помнишь, каких котов я люблю?", records), [])

    def test_recall_does_not_drop_color_or_match_only_an_activity(self):
        records = [dict(id=1, content="Любит выращивать жёлтые кактусы", kind="preference")]
        self.assertEqual(rank_memories("Помнишь мои красные кактусы?", records), [])
        records = [dict(id=1, content="Увлекается выращиванием роз", kind="preference")]
        self.assertEqual(rank_memories("Помнишь, какие кактусы мне нравится выращивать?", records), [])

    def test_recall_keeps_exact_identifiers(self):
        records = [dict(id=1, content="Project alpha100", kind="fact")]
        for query in ("Помнишь Project alpha101?", "Помнишь Project alpha 100?"):
            self.assertEqual(rank_memories(query, records), [])

    def test_recall_no_topic_overlap_has_no_recent_fact_fallback(self):
        records = [dict(id=1, content="Изучает биологию", kind="fact")]
        self.assertEqual(rank_memories("Что ты помнишь о работе?", records), [])
        self.assertEqual(rank_memories("Что ты помнишь обо мне?", records), [])

    def test_compound_recall_remains_user_scoped(self):
        add_memory(100, "Изучает биологию", "fact", "message:1", self.db_path)
        add_memory(200, "Готовится к экзамену по биологии", "fact", "message:2", self.db_path)
        result = retrieve_memories(100, "Помнишь биологию и экзамен?", db_path=self.db_path)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["user_id"], 100)

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

    def test_natural_question_finds_rephrased_russian_preference(self):
        memory_id = add_memory(
            100, "Пользователь увлекается выращиванием фиолетовых кактусов",
            "preference", "message:purple", self.db_path,
        )
        result = retrieve_memories(
            100, "Какие кактусы мне нравится выращивать?", db_path=self.db_path,
        )
        self.assertEqual([item["id"] for item in result], [memory_id])

    def test_preference_paraphrase_is_not_specific_to_cacti(self):
        memory_id = add_memory(
            100, "Пользователь увлекается рисованием акварелью",
            "preference", "message:painting", self.db_path,
        )
        result = retrieve_memories(
            100, "Мне нравится рисовать акварелью", db_path=self.db_path,
        )
        self.assertEqual([item["id"] for item in result], [memory_id])

    def test_recall_question_words_do_not_hide_the_topic(self):
        memory_id = add_memory(
            100, "Любит читать книги", "preference", "message:books", self.db_path,
        )
        result = retrieve_memories(
            100, "Ты помнишь, какие книги я люблю?", db_path=self.db_path,
        )
        self.assertEqual([item["id"] for item in result], [memory_id])

    def test_shared_preference_and_activity_do_not_replace_missing_topic(self):
        add_memory(
            100, "Пользователь увлекается выращиванием роз",
            "preference", "message:roses", self.db_path,
        )
        self.assertEqual(retrieve_memories(
            100, "Какие кактусы мне нравится выращивать?", db_path=self.db_path,
        ), [])

    def test_short_shared_prefix_does_not_match_different_topics(self):
        records = [{"id": 1, "content": "Любит коттеджи", "kind": "preference"}]
        self.assertEqual(rank_memories("Любит котов", records), [])

    def test_latin_identifiers_and_numbers_require_exact_words(self):
        records = [{"id": 1, "content": "Project alpha100", "kind": "fact"}]
        self.assertEqual(rank_memories("Project alpha101", records), [])
        self.assertEqual(rank_memories("Project alpha 100", records), [])

    def test_exact_word_forms_rank_above_approximate_forms(self):
        records = [
            {"id": 1, "content": "Любит выращивать кактусы", "kind": "preference"},
            {"id": 2, "content": "Увлекается выращиванием кактусов", "kind": "preference"},
        ]
        result = rank_memories("Любит выращивать кактусы", records)
        self.assertEqual([item["id"] for item in result], [1, 2])

    def test_russian_yo_and_e_are_search_equivalent(self):
        records = [{"id": 1, "content": "Любит ёлки", "kind": "preference"}]
        self.assertEqual([item["id"] for item in rank_memories("Любит елки", records)], [1])

    def test_paraphrased_query_cannot_retrieve_another_users_preference(self):
        add_memory(
            100, "Пользователь увлекается выращиванием роз",
            "preference", "message:100", self.db_path,
        )
        add_memory(
            200, "Пользователь увлекается выращиванием фиолетовых кактусов",
            "preference", "message:200", self.db_path,
        )
        self.assertEqual(retrieve_memories(
            100, "Какие кактусы мне нравится выращивать?", db_path=self.db_path,
        ), [])


if __name__ == "__main__":
    unittest.main()
