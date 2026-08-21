import json
import tempfile
import unittest
from pathlib import Path

from eval_package import read_eval_package


class EvalPackageTests(unittest.TestCase):
    def _bundle(
        self,
        root: Path,
        eeg_path: str = "active/a1/eeg.bdf",
        clinical_profile: dict | None = None,
    ) -> None:
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
        if clinical_profile is not None:
            manifest["hospital_patient_id"] = clinical_profile.get("hospital_patient_id")
            manifest["clinical_profile"] = clinical_profile
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

    def test_enriched_clinical_profile_prefills_basic_fields(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._bundle(
                root,
                clinical_profile={
                    "hospital_patient_id": "89732",
                    "diagnosis": "脑梗死",
                    "disease_days": 180,
                    "paralysis_side": "左",
                    "age_from_clinical_sheet": 69,
                },
            )
            package = read_eval_package(root, "hospital")

        self.assertEqual(package.patient_prefill["hospital_patient_id"], "89732")
        self.assertEqual(package.patient_prefill["diagnosis"], "脑梗死")
        self.assertEqual(package.patient_prefill["disease_days"], 180)
        self.assertEqual(package.patient_prefill["paralysis_side"], "左")
        self.assertEqual(package.patient_prefill["clinical_age"], 69)


if __name__ == "__main__":
    unittest.main()
