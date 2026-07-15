"""Replayable in-process event stream for one browser assessment session."""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple


class SessionEventStream:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._events: List[Dict[str, Any]] = []
        self._closed = False

    def put(self, event: Dict[str, Any]) -> None:
        with self._condition:
            if event.get("__sentinel__"):
                self._closed = True
            elif not self._closed:
                self._events.append(dict(event))
            self._condition.notify_all()

    def wait_after(
        self,
        cursor: int,
        timeout: float = 15.0,
    ) -> Tuple[List[Tuple[int, Dict[str, Any]]], int, bool]:
        """Return events after a zero-based cursor, next cursor, and closed flag."""
        with self._condition:
            if cursor >= len(self._events) and not self._closed:
                self._condition.wait(timeout=timeout)
            start = max(0, min(int(cursor), len(self._events)))
            rows = [(index + 1, dict(self._events[index])) for index in range(start, len(self._events))]
            return rows, len(self._events), self._closed

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


__all__ = ["SessionEventStream"]
