
import unittest

from memory.relationships import (
    MemoryRelationship,
    PersonReference,
    filter_user_relationships,
    make_person_reference,
    make_relationship,
    person_to_record,
    relationship_to_record,
)


class TestMemoryRelationships(unittest.TestCase):

    def test_create_confirmed_relationship(self):
        relation = make_relationship(
            user_id=100,
            source_memory_id=1,
            target_memory_id=2,
            relation_type="same_event",
            evidence="confirmed",
            source="message:10",
        )
        self.assertEqual(relation.user_id, 100)
        self.assertEqual(relation.relation_type, "same_event")
        self.assertEqual(relation.evidence, "confirmed")

    def test_hypothesis_stays_hypothesis(self):
        relation = make_relationship(
            100, 1, 2,
            "related_event",
            "hypothesis",
            "message:11",
        )
        self.assertEqual(relation.evidence, "hypothesis")

    def test_cannot_link_memory_to_itself(self):
        with self.assertRaises(ValueError):
            make_relationship(
                100, 1, 1,
                "same_topic",
                "confirmed",
                "message:12",
            )

    def test_invalid_relation_type(self):
        with self.assertRaises(ValueError):
            make_relationship(
                100, 1, 2,
                "unknown_relation",
                "confirmed",
                "message:13",
            )

    def test_invalid_evidence(self):
        with self.assertRaises(ValueError):
            make_relationship(
                100, 1, 2,
                "same_topic",
                "certain",
                "message:14",
            )

    def test_invalid_user_id(self):
        with self.assertRaises(ValueError):
            make_relationship(
                0, 1, 2,
                "same_topic",
                "confirmed",
                "message:15",
            )

    def test_missing_source(self):
        with self.assertRaises(ValueError):
            make_relationship(
                100, 1, 2,
                "same_topic",
                "confirmed",
                "",
            )

    def test_create_person_reference(self):
        person = make_person_reference(
            user_id=100,
            person_key="person_1",
            role="friend",
            source="message:16",
            display_name="Алексей",
        )
        self.assertEqual(person.role, "friend")
        self.assertEqual(person.display_name, "Алексей")

    def test_invalid_person_role(self):
        with self.assertRaises(ValueError):
            make_person_reference(
                100,
                "person_1",
                "unrecognized_role",
                "message:17",
            )

    def test_relationship_to_record(self):
        relation = make_relationship(
            100, 1, 2,
            "same_topic",
            "confirmed",
            "message:18",
        )
        record = relationship_to_record(relation)
        self.assertEqual(record["user_id"], 100)
        self.assertEqual(record["target_memory_id"], 2)

    def test_person_to_record(self):
        person = make_person_reference(
            100, "person_1", "relative", "message:19"
        )
        record = person_to_record(person)
        self.assertEqual(record["person_key"], "person_1")
        self.assertEqual(record["role"], "relative")

    def test_user_isolation_filter(self):
        first = make_relationship(
            100, 1, 2,
            "same_topic",
            "confirmed",
            "message:20",
        )
        second = make_relationship(
            200, 3, 4,
            "same_event",
            "confirmed",
            "message:21",
        )

        result = filter_user_relationships(
            100, [first, second]
        )
        self.assertEqual(result, [first])
        self.assertNotIn(second, result)


if __name__ == "__main__":
    unittest.main()
