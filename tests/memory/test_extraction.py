
import unittest

from memory.extraction import (
    MemoryCandidate,
    validate_candidate,
    extract_candidates,
    candidate_to_storage_record,
    has_statement_for_extraction,
)


class TestMemoryExtraction(unittest.TestCase):

    def test_question_only_messages_are_not_memory_statements(self):
        for text in ("Какие модели кораблей мне нравятся?",
                     "Что я люблю? Что я предпочитаю теперь?",
                     "Спасибо! Какие кактусы мне нравятся?",
                     "Какие кактусы мне нравятся?\nСпасибо! 🙏",
                     '"Какие кактусы мне нравятся?"',
                     "«Какие кактусы мне нравятся?».",
                     "What do I prefer now?", "", "   ", "/reset"):
            with self.subTest(text=text):
                self.assertFalse(has_statement_for_extraction(text))

    def test_mixed_message_still_allows_a_new_fact_or_correction(self):
        for text in ("Я изучаю биологию. Как организовать занятия?",
                     "Как лучше заниматься? Я изучаю биологию",
                     "Теперь предпочитаю металлические модели! Что посоветуешь?",
                     "Мне нравятся металлические модели, какие посоветуешь?",
                     "Я изучаю биологию; как лучше заниматься?",
                     "Я люблю чай", "Моя любимая книга «Что делать?».",
                     'Моя любимая книга "Что делать?".'):
            with self.subTest(text=text):
                self.assertTrue(has_statement_for_extraction(text))

    def test_valid_candidate(self):
        result = validate_candidate({
            "content": "Пользователь любит читать",
            "kind": "preference",
            "source": "message:123",
            "confidence": 0.95,
        })
        self.assertIsInstance(result, MemoryCandidate)
        self.assertEqual(result.kind, "preference")

    def test_empty_content(self):
        with self.assertRaises(ValueError):
            validate_candidate({
                "content": "   ",
                "kind": "fact",
                "source": "message:123",
                "confidence": 0.9,
            })

    def test_invalid_kind(self):
        with self.assertRaises(ValueError):
            validate_candidate({
                "content": "Тест",
                "kind": "unknown",
                "source": "message:123",
                "confidence": 0.9,
            })

    def test_invalid_confidence(self):
        with self.assertRaises(ValueError):
            validate_candidate({
                "content": "Тест",
                "kind": "fact",
                "source": "message:123",
                "confidence": 1.5,
            })

    def test_confidence_filter(self):
        result = extract_candidates([
            {
                "content": "Важный факт",
                "kind": "fact",
                "source": "message:1",
                "confidence": 0.9,
            },
            {
                "content": "Сомнительный факт",
                "kind": "hypothesis",
                "source": "message:2",
                "confidence": 0.4,
            },
        ])
        self.assertEqual(len(result), 1)

    def test_duplicate_filter(self):
        result = extract_candidates([
            {
                "content": "Любит музыку",
                "kind": "preference",
                "source": "message:1",
                "confidence": 0.9,
            },
            {
                "content": "любит музыку",
                "kind": "preference",
                "source": "message:2",
                "confidence": 0.95,
            },
        ])
        self.assertEqual(len(result), 1)

    def test_storage_record(self):
        candidate = MemoryCandidate(
            content="Любит книги",
            kind="preference",
            source="message:1",
            confidence=0.9,
        )
        result = candidate_to_storage_record(candidate)
        self.assertEqual(result["content"], "Любит книги")
        self.assertEqual(result["kind"], "preference")
        self.assertEqual(result["source"], "message:1")
        self.assertNotIn("confidence", result)

    def test_invalid_candidates_collection(self):
        with self.assertRaises(ValueError):
            extract_candidates("not a collection")

    def test_too_many_candidates(self):
        with self.assertRaises(ValueError):
            extract_candidates([{}] * 31)


if __name__ == "__main__":
    unittest.main()
