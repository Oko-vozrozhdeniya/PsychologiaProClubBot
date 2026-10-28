import unittest
from memory.progress import Checkpoint, summary


class ProgressTests(unittest.TestCase):
    def test_empty_checkpoint(self):
        point = Checkpoint(1, 'course', 'goal', 'active', (), (), 'next', 'source', '2026-10-28T10:00:00+00:00')
        self.assertEqual(summary(point, 1)['planned'], [])


if __name__ == '__main__':
    unittest.main()
