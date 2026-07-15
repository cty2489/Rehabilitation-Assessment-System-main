"""Pure deterministic sampling helpers shared by serving tests and inference."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def deterministic_bag_indices(
    n_trials: int,
    bag_size: int,
    bag_count: int,
    seed: int,
) -> np.ndarray:
    """Reproduce ``BagDS(..., deterministic=True)`` for one served subject."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    rows: List[np.ndarray] = []
    for bag_index in range(bag_count):
        if bag_size == 1:
            picked = np.array([bag_index % n_trials], dtype=np.int64)
        else:
            rng = np.random.default_rng(seed + 9176 * bag_index)
            picked = rng.choice(n_trials, size=bag_size, replace=n_trials < bag_size)
        rows.append(np.asarray(picked, dtype=np.int64))
    return np.stack(rows, axis=0)


def trial_embedding_indices(
    trial_details: Optional[Sequence[Dict[str, Any]]],
    n_trials: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return zero-based model task/trial embedding IDs in signal order."""
    if trial_details and len(trial_details) == n_trials:
        task_ids = [max(int(item.get("model_task_index", i)), 0) for i, item in enumerate(trial_details)]
        trial_ids = [max(int(item.get("model_trial_index", 0)), 0) for item in trial_details]
    else:
        task_ids = list(range(n_trials))
        trial_ids = [0] * n_trials
    return np.asarray(task_ids, dtype=np.int64), np.asarray(trial_ids, dtype=np.int64)


__all__ = ["deterministic_bag_indices", "trial_embedding_indices"]
