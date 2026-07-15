"""Single-process FIFO scheduler for GPU-backed assessment sessions.

The production deployment runs one FastAPI worker on one GPU.  This scheduler
keeps the complete inference/report pipeline serial so browser and device jobs
cannot compete for the same model memory.
"""
from __future__ import annotations

import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional, Tuple


@dataclass(frozen=True)
class QueueSnapshot:
    state: str
    queue_position: int
    queue_ahead: int


class AssessmentQueue:
    """A small FIFO queue with observable positions and one worker thread."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: Deque[Tuple[str, Any]] = deque()
        self._active_id: Optional[str] = None
        self._worker: Optional[Callable[[Any], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = False

    def start(self, worker: Callable[[Any], None]) -> None:
        with self._condition:
            self._worker = worker
            if self._thread and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name="assessment-queue",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        if thread:
            thread.join(timeout=timeout)

    def enqueue(self, session_id: str, state: Any) -> QueueSnapshot:
        with self._condition:
            existing = self._snapshot_locked(session_id)
            if existing is not None:
                return existing
            ahead = len(self._pending) + (1 if self._active_id else 0)
            self._pending.append((session_id, state))
            self._condition.notify()
            return QueueSnapshot("queued", ahead + 1, ahead)

    def snapshot(self, session_id: str) -> Optional[QueueSnapshot]:
        with self._condition:
            return self._snapshot_locked(session_id)

    def has_work(self) -> bool:
        with self._condition:
            return self._active_id is not None or bool(self._pending)

    def pending_count(self) -> int:
        with self._condition:
            return len(self._pending) + (1 if self._active_id else 0)

    def cancel_pending(self, session_id: str) -> bool:
        """Remove a queued (not active) session."""
        with self._condition:
            for index, (pending_id, _) in enumerate(self._pending):
                if pending_id == session_id:
                    del self._pending[index]
                    self._condition.notify_all()
                    return True
            return False

    def _snapshot_locked(self, session_id: str) -> Optional[QueueSnapshot]:
        if self._active_id == session_id:
            return QueueSnapshot("running", 0, 0)
        active_ahead = 1 if self._active_id else 0
        for index, (pending_id, _) in enumerate(self._pending):
            if pending_id == session_id:
                ahead = active_ahead + index
                return QueueSnapshot("queued", ahead + 1, ahead)
        return None

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                session_id, state = self._pending.popleft()
                self._active_id = session_id
                worker = self._worker
            try:
                if worker is None:
                    raise RuntimeError("assessment queue worker is not configured")
                worker(state)
            except Exception:  # keep later assessments runnable after one unexpected failure
                traceback.print_exc()
            finally:
                with self._condition:
                    self._active_id = None
                    self._condition.notify_all()


__all__ = ["AssessmentQueue", "QueueSnapshot"]
