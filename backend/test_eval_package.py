import json
import tempfile
import unittest
from pathlib import Path

from eval_package import read_eval_package


class EvalPackageTests(unittest.TestCase):
    def _bundle(self, root: Path, eeg_path: str = "active/a1/eeg.bdf") -> None:
        manifest = {
            "patient_id": "P001",
            "assessments": [
                {
                    "assessment_type": "active",
                    "action_id": "action_SS2",
                    "action_name": "伸食指",
                    "trials": [
                        {"trial_index": 1, "eeg_file": eeg_path, "emg_imu_file": "active/a1/emg.csv"},
                        {"trial_index": 2, "eeg_file": "active/a2/eeg.bdf", "emg_imu_file": "active/a2/emg.csv"},
                    ],
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        for relative in ("active/a1/eeg.bdf", "active/a1/emg.csv", "active/a2/eeg.bdf", "active/a2/emg.csv"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")

    def test_model_embedding_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._bundle(root)
            package = read_eval_package(root, "device")

        self.assertEqual([row["model_task_index"] for row in package.trial_details], [1, 1])
        self.assertEqual([row["model_trial_index"] for row in package.trial_details], [0, 1])
        self.assertEqual(package.patient_prefill["sex"], "")
        self.assertEqual(package.patient_prefill["paralysis_side"], "")

    def test_manifest_file_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._bundle(root, eeg_path="../../etc/passwd")
            with self.assertRaisesRegex(ValueError, "路径越界"):
                read_eval_package(root, "device")


if __name__ == "__main__":
    unittest.main()
