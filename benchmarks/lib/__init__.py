"""Reusable library for piper-kernels development benchmarks."""

from .attention import AttentionConfig, AttentionShape
from .environment import EnvironmentInfo, capture_environment
from .profiling import (
    CaptureController,
    CudaProfilerController,
    ProfilePhase,
    ProviderProfile,
    add_profile_arguments,
    profile_provider,
)
from .providers import BenchmarkProvider, ProviderMeasurement, measure_provider
from .quality import QualityMetrics, QuantizerSaturation, measure_quality, measure_saturation
from .reporting import (
    BenchmarkRecord,
    OutputFormat,
    OutputTarget,
    SerializableRecord,
    add_output_arguments,
    output_target,
    write_records,
)
from .timing import (
    ClockDomain,
    PhaseTimings,
    Timing,
    synchronized_wall_benchmark,
    time_first_call,
    triton_benchmark,
)
from .triton_inspection import (
    NvdisasmUnavailableError,
    TritonArtifactUnavailableError,
    TritonCompatibilityError,
    TritonCompilerRecord,
    TritonInspectionError,
    TritonSpecializationReport,
    add_compiler_inspection_arguments,
    format_compiler_report,
    inspect_provider,
)

__all__ = [
    "AttentionConfig",
    "AttentionShape",
    "BenchmarkProvider",
    "BenchmarkRecord",
    "CaptureController",
    "ClockDomain",
    "CudaProfilerController",
    "EnvironmentInfo",
    "NvdisasmUnavailableError",
    "OutputFormat",
    "OutputTarget",
    "PhaseTimings",
    "ProfilePhase",
    "ProviderMeasurement",
    "ProviderProfile",
    "QualityMetrics",
    "QuantizerSaturation",
    "SerializableRecord",
    "Timing",
    "TritonArtifactUnavailableError",
    "TritonCompatibilityError",
    "TritonCompilerRecord",
    "TritonInspectionError",
    "TritonSpecializationReport",
    "add_compiler_inspection_arguments",
    "add_output_arguments",
    "add_profile_arguments",
    "capture_environment",
    "format_compiler_report",
    "inspect_provider",
    "measure_provider",
    "measure_quality",
    "measure_saturation",
    "output_target",
    "profile_provider",
    "synchronized_wall_benchmark",
    "time_first_call",
    "triton_benchmark",
    "write_records",
]
