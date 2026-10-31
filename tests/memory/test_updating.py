
"""Unit tests for long-term memory update decisions."""

import unittest

from memory.updating import (
    UpdateDecision,
    decision_to_record,
    make_update_decision,
    memory_match_key, preference_key, source_supports_memory, current_memory_content,
)


class TestMemoryUpdating(unittest.TestCase):

    def test_preference_paraphrases_match_without_sorting_objects(self):
        first = "Пользователь теперь предпочитает выращивать жёлтые кактусы вместо фиолетовых"
        second = "Пользователь увлекается выращиванием желтых кактусов."
        self.assertEqual(memory_match_key(first, "preference"),
                         memory_match_key(second, "preference"))
        self.assertTrue(source_supports_memory("Мне нравится выращивать жёлтые кактусы",
                                              first, "preference"))

    def test_temporal_ship_preference_matches_only_current_value(self):
        old = ("Пользователь раньше предпочитал собирать деревянные модели кораблей, "
               "теперь предпочитает собирать металлические модели кораблей")
        current = "Пользователь предпочитает собирать металлические модели кораблей"
        self.assertEqual(memory_match_key(old, "preference"), memory_match_key(current, "preference"))
        self.assertNotEqual(memory_match_key(old, "preference"),
                            memory_match_key("Любит собирать деревянные модели кораблей", "preference"))

    def test_conflicts_subjects_negations_and_numbers_are_not_equivalent(self):
        base = memory_match_key("Любит выращивать жёлтые кактусы", "preference")
        for other in ("Не любит выращивать жёлтые кактусы", "Мама любит выращивать жёлтые кактусы",
                      "Любит выращивать фиолетовые кактусы", "Любит поливать жёлтые кактусы",
                      "Любит выращивать 2 жёлтых кактуса", "Любит выращивать кактусы и розы",
                      "Возможно любит выращивать жёлтые кактусы"):
            with self.subTest(other=other):
                self.assertNotEqual(base, memory_match_key(other, "preference"))
        self.assertNotEqual(memory_match_key("Живёт в доме 12", "fact"),
                            memory_match_key("Живёт в доме 21", "fact"))
        self.assertNotEqual(memory_match_key("Любит Анну больше Марии", "preference"),
                            memory_match_key("Любит Марию больше Анны", "preference"))

    def test_question_or_other_person_is_not_support_for_reconfirmation(self):
        fact = "Предпочитает собирать металлические модели кораблей"
        for text in ("Какие модели кораблей мне нравятся?", "Брат любит собирать металлические модели кораблей",
                     "Сегодня я ел апельсин", "Я не люблю собирать металлические модели кораблей"):
            with self.subTest(text=text):
                self.assertFalse(source_supports_memory(text, fact, "preference"))

    def test_names_and_brands_do_not_enter_inflection_matching(self):
        for left, right in (("Любит рассказы Полины", "Любит рассказы Полин"),
                            ("Любит модели Airbus", "Любит модели Airbu")):
            with self.subTest(left=left):
                self.assertIsNone(preference_key(left))
                self.assertNotEqual(memory_match_key(left, "preference"),
                                    memory_match_key(right, "preference"))

    def test_separating_old_wording_preserves_identity_and_source_support(self):
        for text in ("Раньше мне нравилось собирать деревянные модели кораблей, теперь я предпочитаю собирать металлические модели кораблей",
                     "Пользователь раньше любил собирать деревянные модели кораблей, теперь пользователь предпочитает собирать металлические модели кораблей"):
            with self.subTest(text=text):
                current = current_memory_content(text, "preference")
                self.assertNotIn("деревянные", current)
                self.assertEqual(memory_match_key(text, "preference"), memory_match_key(current, "preference"))
                self.assertTrue(source_supports_memory("Я предпочитаю собирать металлические модели кораблей", current, "preference"))

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
