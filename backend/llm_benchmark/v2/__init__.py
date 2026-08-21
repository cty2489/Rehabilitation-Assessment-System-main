"""Version-isolated rehabilitation LLM benchmark v2 contracts and runner."""

from .prompt import PROMPT_PATH, PROMPT_VERSION, build_messages, load_prompt
from .runner import (
    BenchmarkOutputParseErrorV2,
    build_evaluation_input,
    build_experiment_log,
    parse_model_output,
    render_report,
)
from .schemas import (
    BENCHMARK_PRESETS_V2,
    BenchmarkEvaluationInput,
    BenchmarkExperimentLog,
    BenchmarkLlmOutput,
    BenchmarkRunConfig,
    benchmark_preset,
)
from .service import (
    BenchmarkGenerationError,
    BenchmarkGenerationResult,
    generate_benchmark_batch_reports,
    generate_benchmark_report,
)

__all__ = [
    "BENCHMARK_PRESETS_V2",
    "BenchmarkEvaluationInput",
    "BenchmarkExperimentLog",
    "BenchmarkGenerationError",
    "BenchmarkGenerationResult",
    "BenchmarkLlmOutput",
    "BenchmarkOutputParseErrorV2",
    "BenchmarkRunConfig",
    "PROMPT_PATH",
    "PROMPT_VERSION",
    "benchmark_preset",
    "build_evaluation_input",
    "build_experiment_log",
    "build_messages",
    "generate_benchmark_batch_reports",
    "generate_benchmark_report",
    "load_prompt",
    "parse_model_output",
    "render_report",
]
