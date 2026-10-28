"""Unit tests for independent personalization decision logic."""
import unittest
from memory.personalization import PersonalizationItem, build_personalization


def item(user=1, category="communication", content="Краткие ответы",
         confirmed=True, explicit=False, status="active"):
    return PersonalizationItem(user, category, content, "message:1",
                               confirmed, status, explicit)


class PersonalizationTests(unittest.TestCase):
    def test_confirmed_preference(self):
        result = build_personalization(1, [item()])
        self.assertIn("Краткие ответы", " ".join(result["instructions"]))

    def test_unconfirmed_ignored(self):
        result = build_personalization(1, [item(confirmed=False)])
        self.assertNotIn("Краткие ответы", " ".join(result["instructions"]))

    def test_boundary_priority(self):
        result = build_personalization(1, [
            item(), item(category="boundary", content="Не обсуждать семью", explicit=True)])
        self.assertIn("Explicit boundary: Не обсуждать семью", result["instructions"][3])

    def test_inferred_boundary_not_enforced(self):
        result = build_personalization(1, [item(category="boundary", explicit=False)])
        self.assertEqual(len(result["instructions"]), 3)

    def test_other_user_rejected(self):
        with self.assertRaises(ValueError):
            build_personalization(1, [item(user=2)])

    def test_unfinished_not_unsolicited(self):
        records = [item(category="unfinished", content="Тема работы")]
        self.assertEqual(build_personalization(1, records)["unfinished_topics"], [])
        self.assertEqual(build_personalization(1, records, topic_invited=True)["unfinished_topics"],
                         ["Тема работы"])

    def test_resolved_ignored(self):
        self.assertEqual(len(build_personalization(1, [item(status="resolved")])["instructions"]), 3)

    def test_invalid_id(self):
        with self.assertRaises(ValueError):
            build_personalization(True, [])

    def test_invalid_category(self):
        with self.assertRaises(ValueError):
            item(category="diagnosis")

    def test_invalid_list(self):
        with self.assertRaises(ValueError):
            build_personalization(1, [item()] * 31)

    def test_sources(self):
        self.assertEqual(build_personalization(1, [item()])["sources"], ["message:1"])


if __name__ == "__main__":
    unittest.main()
