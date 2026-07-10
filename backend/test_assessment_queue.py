import threading
import time
import unittest

from assessment_queue import AssessmentQueue


class AssessmentQueueTests(unittest.TestCase):
    def test_fifo_execution_and_queue_positions(self) -> None:
        scheduler = AssessmentQueue()
        release_first = threading.Event()
        first_started = threading.Event()
        finished = threading.Event()
        seen = []

        def worker(value: str) -> None:
            seen.append(value)
            if value == "first":
                first_started.set()
                release_first.wait(timeout=2)
            if value == "third":
                finished.set()

        scheduler.start(worker)
        scheduler.enqueue("s1", "first")
        self.assertTrue(first_started.wait(timeout=1))

        second = scheduler.enqueue("s2", "second")
        third = scheduler.enqueue("s3", "third")
        self.assertEqual((second.queue_position, second.queue_ahead), (2, 1))
        self.assertEqual((third.queue_position, third.queue_ahead), (3, 2))
        self.assertEqual(scheduler.snapshot("s1").state, "running")

        release_first.set()
        self.assertTrue(finished.wait(timeout=2))
        deadline = time.time() + 1
        while scheduler.snapshot("s3") is not None and time.time() < deadline:
            time.sleep(0.01)
        scheduler.stop()

        self.assertEqual(seen, ["first", "second", "third"])

    def test_duplicate_enqueue_is_ignored(self) -> None:
        scheduler = AssessmentQueue()
        release = threading.Event()
        started = threading.Event()
        seen = []

        def worker(value: str) -> None:
            seen.append(value)
            started.set()
            release.wait(timeout=2)

        scheduler.start(worker)
        scheduler.enqueue("same", "payload")
        self.assertTrue(started.wait(timeout=1))
        duplicate = scheduler.enqueue("same", "other")
        self.assertEqual(duplicate.state, "running")
        release.set()
        scheduler.stop()
        self.assertEqual(seen, ["payload"])


if __name__ == "__main__":
    unittest.main()
