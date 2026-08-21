"""Isolated infrastructure for the future rehabilitation LLM benchmark.

This package is deliberately not imported by the production assessment path.
It defines the fixed benchmark contracts, prompt, report renderer, audit log,
and blind-review records without starting any model or retrieval run.
"""

from .schemas import (
    AssessmentInputMode,
    ClinicalScoreSource,
    BenchmarkBiomarker,
    BenchmarkClinicalScores,
    BenchmarkEvidenceCard,
    BenchmarkEvaluationInput,
    BenchmarkExperimentLog,
    BenchmarkGraphContext,
    BenchmarkKnowledgeContext,
    BenchmarkLlmOutput,
    BenchmarkPatientInfo,
    BenchmarkRegistryMapping,
    BenchmarkRunConfig,
    BENCHMARK_PRESETS,
    benchmark_preset,
    BlindReviewPacket,
    DoctorReviewScore,
    RehabilitationAction,
)
from .blind_review import (
    anonymous_report_id,
    build_blind_review_packet,
    build_registry_mapping,
)
from .prompt import PROMPT_VERSION, build_messages, load_prompt
from .runner import (
    BenchmarkOutputParseError,
    build_evaluation_input,
    build_experiment_log,
    parse_model_output,
    render_report,
)
from .batch import (
    BenchmarkBatch,
    BenchmarkBatchValidationError,
    BenchmarkPatientPackage,
    existing_biomarker_extractor,
    prepare_batch_inputs,
    prepare_single_input,
    read_benchmark_batch,
    read_benchmark_batch_zip,
    read_clinical_scores,
)
from .service import (
    BenchmarkGenerationError,
    BenchmarkGenerationResult,
    generate_benchmark_batch_reports,
    generate_benchmark_report,
)
from .storage import (
    materialize_batch_manifest,
    materialize_benchmark_generation,
    materialize_benchmark_input,
)

__all__ = [
    "AssessmentInputMode",
    "ClinicalScoreSource",
    "BenchmarkBiomarker",
    "BenchmarkClinicalScores",
    "BenchmarkEvidenceCard",
    "BenchmarkEvaluationInput",
    "BenchmarkExperimentLog",
    "BenchmarkGraphContext",
    "BenchmarkKnowledgeContext",
    "BenchmarkLlmOutput",
    "BenchmarkPatientInfo",
    "BenchmarkRegistryMapping",
    "BenchmarkRunConfig",
    "BENCHMARK_PRESETS",
    "benchmark_preset",
    "BenchmarkBatch",
    "BenchmarkBatchValidationError",
    "BenchmarkPatientPackage",
    "existing_biomarker_extractor",
    "BenchmarkGenerationError",
    "BenchmarkGenerationResult",
    "BenchmarkOutputParseError",
    "BlindReviewPacket",
    "DoctorReviewScore",
    "PROMPT_VERSION",
    "anonymous_report_id",
    "build_blind_review_packet",
    "build_evaluation_input",
    "build_experiment_log",
    "build_messages",
    "build_registry_mapping",
    "load_prompt",
    "parse_model_output",
    "RehabilitationAction",
    "render_report",
    "generate_benchmark_report",
    "generate_benchmark_batch_reports",
    "prepare_batch_inputs",
    "prepare_single_input",
    "read_benchmark_batch",
    "read_benchmark_batch_zip",
    "read_clinical_scores",
    "materialize_batch_manifest",
    "materialize_benchmark_generation",
    "materialize_benchmark_input",
]
