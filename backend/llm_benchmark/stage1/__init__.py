"""Clinical-only Stage 1 benchmark path.

This package deliberately does not call signal processing, DL, RAG, or KG.
It adapts the fixed clinician workbook into the existing v2 report contract.
"""

from .clinical_input import (
    DOCTOR_SHEET_NAME,
    Stage1Case,
    Stage1ClinicalInput,
    Stage1WorkbookError,
    read_doctor_sheet,
)
from .adapters import (
    ExistingModelAdapter,
    ExistingModelCatalog,
    ExistingModelSpec,
    ModelAdapterError,
    Stage1ModelRouter,
    build_existing_stage1_config,
    discover_existing_models,
)
from .doctor_reference import (
    build_reference_payload,
    parse_doctor_review,
    validate_doctor_sheet_progress,
    write_gold_references,
)
from .output import (
    Stage1CandidateOutput,
    Stage1OutputParseError,
    parse_stage1_model_output,
)
from .runner import (
    STAGE1_NAME,
    STAGE1_PROMPT_VERSION,
    Stage1GenerationError,
    Stage1ExperimentRun,
    Stage1RunConfig,
    generate_stage1_batch_reports,
    generate_stage1_report,
    run_stage1_experiment,
    write_stage1_manifest,
    write_stage1_result,
)

__all__ = [
    "DOCTOR_SHEET_NAME",
    "ExistingModelAdapter",
    "ExistingModelCatalog",
    "ExistingModelSpec",
    "ModelAdapterError",
    "STAGE1_NAME",
    "STAGE1_PROMPT_VERSION",
    "Stage1GenerationError",
    "Stage1Case",
    "Stage1CandidateOutput",
    "Stage1ExperimentRun",
    "Stage1ClinicalInput",
    "Stage1OutputParseError",
    "Stage1RunConfig",
    "Stage1ModelRouter",
    "Stage1WorkbookError",
    "build_reference_payload",
    "build_existing_stage1_config",
    "discover_existing_models",
    "generate_stage1_batch_reports",
    "generate_stage1_report",
    "parse_doctor_review",
    "parse_stage1_model_output",
    "read_doctor_sheet",
    "run_stage1_experiment",
    "validate_doctor_sheet_progress",
    "write_gold_references",
    "write_stage1_result",
    "write_stage1_manifest",
]
