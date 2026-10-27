
"""Unit tests for long-term memory update decisions."""

import unittest

from memory.updating import (
    UpdateDecision,
    decision_to_record,
    make_update_decision,
)


class TestMemoryUpdating(unittest.TestCase):

    def test_unchanged_content(self):
        result = make_update_decision(
            "Живу в Москве",
            "Живу в Москве",
            "message:1",
        )
        self.assertEqual(result.action, "keep")

    def test_explicit_correction(self):
        result = make_update_decision(
            "Живу в Москве",
            "Живу в Казани",
            "message:2",
            explicitly_corrected=True,
        )
        self.assertEqual(result.action, "update")

    def test_unconfirmed_change(self):
        result = make_update_decision(
            "Люблю чай",
            "Люблю кофе",
            "message:3",
        )
        self.assertEqual(result.action, "review")

    def test_conflict(self):
        result = make_update_decision(
            "Работаю учителем",
            "Работаю инженером",
            "message:4",
            conflict_detected=True,
        )
        self.assertEqual(result.action, "conflict")

    def test_conflict_has_priority(self):
        result = make_update_decision(
            "У меня одна собака",
            "У меня две собаки",
            "message:5",
            explicitly_corrected=True,
            conflict_detected=True,
        )
        self.assertEqual(result.action, "conflict")

    def test_invalid_empty_content(self):
        with self.assertRaises(ValueError):
            make_update_decision(
                "Старый факт",
                " ",
                "message:6",
            )

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            make_update_decision(
                "Старый факт",
                "Новый факт",
                "",
            )

    def test_invalid_boolean(self):
        with self.assertRaises(ValueError):
            make_update_decision(
                "Старый факт",
                "Новый факт",
                "message:7",
                explicitly_corrected="yes",
            )

    def test_record_conversion(self):
        decision = make_update_decision(
            "Старое значение",
            "Новое значение",
            "message:8",
            explicitly_corrected=True,
        )
        record = decision_to_record(decision)

        self.assertEqual(record["action"], "update")
        self.assertEqual(record["source"], "message:8")
        self.assertEqual(record["old_content"], "Старое значение")
        self.assertEqual(record["new_content"], "Новое значение")

    def test_invalid_decision_object(self):
        with self.assertRaises(ValueError):
            decision_to_record({"action": "update"})

    def test_decision_is_immutable(self):
        decision = make_update_decision(
            "Старый факт",
            "Новый факт",
            "message:9",
        )
        self.assertIsInstance(decision, UpdateDecision)

        with self.assertRaises(AttributeError):
            decision.action = "update"


if __name__ == "__main__":
    unittest.main()
