import unittest

from session_events import SessionEventStream


class SessionEventStreamTests(unittest.TestCase):
    def test_events_are_replayable_for_multiple_readers(self):
        stream = SessionEventStream()
        stream.put({"type": "step", "value": 1})
        stream.put({"type": "step", "value": 2})

        first, cursor, closed = stream.wait_after(0, timeout=0)
        second, _, _ = stream.wait_after(0, timeout=0)
        self.assertEqual([event["value"] for _, event in first], [1, 2])
        self.assertEqual(first, second)
        self.assertEqual(cursor, 2)
        self.assertFalse(closed)

        stream.put({"__sentinel__": True})
        remaining, cursor, closed = stream.wait_after(cursor, timeout=0)
        self.assertEqual(remaining, [])
        self.assertEqual(cursor, 2)
        self.assertTrue(closed)


if __name__ == "__main__":
    unittest.main()
