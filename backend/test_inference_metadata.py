import unittest

import numpy as np

from inference_sampling import deterministic_bag_indices, trial_embedding_indices


class InferenceMetadataTests(unittest.TestCase):
    def test_deterministic_bags_match_training_sampler(self):
        actual = deterministic_bag_indices(3, bag_size=4, bag_count=2, seed=2031)
        expected = np.stack([
            np.random.default_rng(2031).choice(3, size=4, replace=True),
            np.random.default_rng(2031 + 9176).choice(3, size=4, replace=True),
        ])
        np.testing.assert_array_equal(actual, expected)

    def test_explicit_embedding_ids_are_used(self):
        task, trial = trial_embedding_indices(
            [
                {"model_task_index": 0, "model_trial_index": 0},
                {"model_task_index": 2, "model_trial_index": 1},
            ],
            2,
        )
        self.assertEqual(task.tolist(), [0, 2])
        self.assertEqual(trial.tolist(), [0, 1])

    def test_legacy_hospital_fallback_is_one_trial_per_action(self):
        task, trial = trial_embedding_indices(None, 3)
        self.assertEqual(task.tolist(), [0, 1, 2])
        self.assertEqual(trial.tolist(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
