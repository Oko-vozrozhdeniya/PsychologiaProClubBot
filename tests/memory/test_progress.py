import unittest
from memory.progress import Action, Result, Checkpoint, update, summary

T = "2026-10-28T10:00:00+00:00"


def initial():
    return Checkpoint(1, "course", "Lesson", "active", (), (), "Next", "source", T)


class ProgressTests(unittest.TestCase):
    def test_plan_and_fact(self):
        planned = update(initial(), 1, "source", T, action=Action("a", "Practice", "planned", "source"))
        self.assertEqual(summary(planned, 1)["planned"], ["Practice"])
        done = update(planned, 1, "source", T, action=Action("a", "Practice", "done", "source"))
        self.assertEqual(summary(done, 1)["completed"], ["Practice"])
        self.assertEqual(len(done.actions), 1)

    def test_reported_not_confirmed(self):
        value = update(initial(), 1, "source", T, result=Result("Better", "reported", "source"))
        self.assertEqual(summary(value, 1)["reported"], ["Better"])
        self.assertEqual(summary(value, 1)["confirmed"], [])

    def test_confirmed(self):
        value = update(initial(), 1, "source", T, result=Result("Passed", "confirmed", "source"))
        self.assertEqual(summary(value, 1)["confirmed"], ["Passed"])

    def test_idempotent(self):
        action = Action("a", "Practice", "done", "source")
        value = update(initial(), 1, "source", T, action=action)
        self.assertEqual(update(value, 1, "source", T, action=action), value)

    def test_cross_user(self):
        with self.assertRaises(ValueError):
            update(initial(), 2, "source", T)
        with self.assertRaises(ValueError):
            summary(initial(), 2)

    def test_version(self):
        with self.assertRaises(ValueError):
            update(initial(), 1, "source", T, expected_version=2)

    def test_old_event(self):
        with self.assertRaises(ValueError):
            update(initial(), 1, "source", "2026-10-27T10:00:00+00:00")

    def test_next_step(self):
        value = update(initial(), 1, "source", T, next_step="Lesson 2")
        self.assertEqual(summary(value, 1)["next_step"], "Lesson 2")

    def test_deferred(self):
        value = update(initial(), 1, "source", T, action=Action("a", "Practice", "deferred", "source"))
        self.assertEqual(summary(value, 1)["deferred"], ["Practice"])

    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            Action("a", "Practice", "unknown", "source")


if __name__ == "__main__":
    unittest.main()
