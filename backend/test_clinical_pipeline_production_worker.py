from __future__ import annotations

import importlib
import sys
import threading
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from clinical_pipeline import production_adapter
from clinical_pipeline.production_adapter import ProductionPipelineBlockedError
from schemas import PatientInfo
from session_events import SessionEventStream


_MAIN_RUNTIME_MODULES = (
    "main",
    "admin_auth",
    "assessment_export",
    "assessment_queue",
    "device_auth",
    "device_patient_policy",
    "eval_package",
    "llm_settings",
    "mysql_db",
    "session_events",
)
_MODULE_MISSING = object()


def _load_main_with_lightweight_runtime():
    sys.modules.pop("main", None)

    inference = types.ModuleType("inference")
    inference.CHECKPOINTS = {}
    inference.SENTINEL = {"__sentinel__": True}

    class AssessmentCancelled(Exception):
        pass

    class ModelRegistry:
        pass

    inference.AssessmentCancelled = AssessmentCancelled
    inference.ModelRegistry = ModelRegistry
    inference.error_event = lambda message: {"type": "error", "message": message}
    inference.run_pipeline = lambda *args, **kwargs: {}

    report = types.ModuleType("report")

    class ReportModel:
        loaded = True

        def load(self):
            self.loaded = True

        def reset(self):
            self.loaded = False

    report.REPORT_MODEL = ReportModel()
    report.llm_model_name = lambda: "qwen3_8b_hf"
    report.llm_provider = lambda: "local"
    report.remote_url = lambda: ""

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None

    mysql_db = types.ModuleType("mysql_db")
    mysql_db.MySQLUnavailable = type("MySQLUnavailable", (Exception,), {})
    mysql_db.save_assessment_bundle = lambda *args, **kwargs: 1
    mysql_db.update_device_job = lambda *args, **kwargs: {}

    session_events = types.ModuleType("session_events")
    session_events.SessionEventStream = SessionEventStream

    with patch.dict(
        sys.modules,
        {
            "dotenv": dotenv,
            "inference": inference,
            "mysql_db": mysql_db,
            "report": report,
            "session_events": session_events,
        },
    ):
        return importlib.import_module("main")


def _biomarkers() -> dict:
    return {
        "groups": [
            {
                "key": "imu",
                "markers": [
                    {
                        "key": "movement_smoothness_sparc",
                        "name": "运动平滑度SPARC",
                        "value": -1.4,
                        "unit": "",
                        "available": True,
                        "n_valid": 3,
                    }
                ],
            }
        ],
        "coverage": {"available": 1, "total": 26, "missing_keys": []},
    }


def _predictions(institution: str) -> dict:
    return {
        "FMA_UE": 8.0,
        "hand_tone": "2",
        "hand_function": 3,
        "_biomarkers": _biomarkers(),
        "_quality": {"status": "pass", "trial_count": 1},
        "_validation_status": (
            "engineering_validation_only"
            if institution == "device"
            else "research_assessment"
        ),
    }


class ProductionWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._module_snapshot = {
            name: sys.modules.get(name, _MODULE_MISSING)
            for name in _MAIN_RUNTIME_MODULES
        }
        cls.main = _load_main_with_lightweight_runtime()

    @classmethod
    def tearDownClass(cls) -> None:
        for name, previous in cls._module_snapshot.items():
            if previous is _MODULE_MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    def _state(self, institution: str):
        patient_id = "DEV001_0001" if institution == "device" else "HOSP001"
        return SimpleNamespace(
            session_id=f"session-{institution}",
            patient=PatientInfo(
                patient_id=patient_id,
                name="张三",
                sex="男",
                age=62,
                diagnosis="脑梗死",
                disease_days=120,
                paralysis_side="左",
            ),
            eeg_paths=[Path("trial.bdf")],
            emg_paths=[Path("trial.csv")],
            institution=institution,
            trial_details=[{"trial_index": 1}],
            cancel_event=threading.Event(),
            device_job_id="devjob-1" if institution == "device" else None,
            assessment_id=f"assessment-{institution}",
            report_model_id="qwen3_8b_hf",
            report_provider="local",
            queue=self.main.SessionEventStream(),
            package_name="bundle.zip",
            assessment_time=None,
            n_trials=1,
            package_hash="abc123",
            parse_warnings=[],
            assessment_db_id=None,
            result=None,
            temporary_work_dir=None,
            finished_monotonic=None,
        )

    @staticmethod
    def _pipeline_result(validation_status: str = "passed"):
        return SimpleNamespace(
            retrieval=SimpleNamespace(status=SimpleNamespace(value="complete")),
            validation=SimpleNamespace(status=SimpleNamespace(value=validation_status)),
        )

    def _run_success(self, institution: str, validation_status: str = "passed"):
        main = self.main
        state = self._state(institution)
        pipeline_result = self._pipeline_result(validation_status)
        orchestrator = Mock()
        orchestrator.run.return_value = pipeline_result
        markdown = (
            "# 智能康复评估报告\n\n> **人工复核要求：** 需要人工复核。\n"
            if validation_status == "manual_review"
            else "# 智能康复评估报告\n\n报告内容。\n"
        )
        metadata = {
            "mode": "planner_rag",
            "run_id": "pipeline-worker-test",
            "run_status": "completed",
            "quality_gate": "pass",
            "planner_generation_mode": "llm",
            "retrieval_status": "complete",
            "report_id": "report-worker-test",
            "validation_status": validation_status,
        }

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    main,
                    "_load_production_adapter",
                    return_value=production_adapter,
                )
            )
            run_pipeline = stack.enter_context(
                patch.object(main, "run_pipeline", return_value=_predictions(institution))
            )
            build = stack.enter_context(
                patch.object(
                    production_adapter,
                    "build_production_orchestrator",
                    return_value=orchestrator,
                )
            )
            stack.enter_context(
                patch.object(
                    production_adapter,
                    "require_completed_report",
                    return_value=(object(), pipeline_result.validation),
                )
            )
            stack.enter_context(
                patch.object(
                    production_adapter,
                    "render_compatible_markdown",
                    return_value=markdown,
                )
            )
            stack.enter_context(
                patch.object(
                    production_adapter,
                    "orchestration_metadata",
                    return_value=metadata,
                )
            )
            save = stack.enter_context(
                patch.object(main.mysql_db, "save_assessment_bundle", return_value=41)
            )
            update_job = stack.enter_context(
                patch.object(main.mysql_db, "update_device_job", return_value={})
            )
            main.app.state.dl_model_version = "fake-dl-model"
            main._worker(state, object(), main.REPORT_MODEL)

        return state, pipeline_result, orchestrator, run_pipeline, build, save, update_job

    def test_browser_task_runs_the_new_pipeline_and_persists_markdown(self) -> None:
        state, pipeline_result, orchestrator, run_pipeline, build, save, update_job = (
            self._run_success("hospital")
        )

        run_pipeline.assert_called_once()
        build.assert_called_once_with("qwen3_8b_hf")
        orchestrator.run.assert_called_once()
        save.assert_called_once()
        update_job.assert_not_called()
        self.assertIsNotNone(state.result)
        self.assertIn("智能康复评估报告", state.result.report)
        self.assertEqual(
            state.result.quality["clinical_pipeline"]["mode"], "planner_rag"
        )
        self.assertEqual(state.assessment_db_id, 41)
        self.assertIsNotNone(orchestrator.run.call_args.args[0].patient.patient_id)
        self.assertIs(pipeline_result, orchestrator.run.return_value)

    def test_device_task_runs_the_same_pipeline_and_completes_job(self) -> None:
        state, _, orchestrator, _, build, save, update_job = self._run_success("device")

        build.assert_called_once_with("qwen3_8b_hf")
        orchestrator.run.assert_called_once()
        save.assert_called_once()
        self.assertTrue(
            any(call.kwargs.get("status") == "completed" for call in update_job.call_args_list)
        )
        self.assertEqual(
            state.result.validation_status, "engineering_validation_only"
        )
        self.assertEqual(
            state.result.quality["clinical_pipeline"]["validation_status"], "passed"
        )

    def test_manual_review_report_is_returned_and_marked(self) -> None:
        state, *_ = self._run_success("hospital", validation_status="manual_review")

        self.assertIn("人工复核要求", state.result.report)
        self.assertEqual(
            state.result.quality["clinical_pipeline"]["validation_status"],
            "manual_review",
        )

    def test_quality_gate_block_fails_without_persistence_or_report(self) -> None:
        main = self.main
        state = self._state("hospital")
        orchestrator = Mock()
        orchestrator.run.return_value = SimpleNamespace(retrieval=None)

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    main,
                    "_load_production_adapter",
                    return_value=production_adapter,
                )
            )
            stack.enter_context(
                patch.object(main, "run_pipeline", return_value=_predictions("hospital"))
            )
            stack.enter_context(
                patch.object(
                    production_adapter,
                    "build_production_orchestrator",
                    return_value=orchestrator,
                )
            )
            stack.enter_context(
                patch.object(
                    production_adapter,
                    "require_completed_report",
                    side_effect=ProductionPipelineBlockedError(
                        "planner_rag质量门控阻断：缺少患者标识"
                    ),
                )
            )
            save = stack.enter_context(
                patch.object(main.mysql_db, "save_assessment_bundle")
            )
            main._worker(state, object(), main.REPORT_MODEL)

        events, _, closed = state.queue.wait_after(0, timeout=0)
        payloads = [event for _, event in events]
        self.assertTrue(closed)
        self.assertIsNone(state.result)
        save.assert_not_called()
        self.assertTrue(
            any(
                event.get("type") == "error" and "质量门控阻断" in event.get("message", "")
                for event in payloads
            )
        )


if __name__ == "__main__":
    unittest.main()
