"""Reusable support for piper-kernels development benchmarks."""

from .attention import AttentionConfig, AttentionShape
from .environment import EnvironmentInfo, capture_environment
from .providers import BenchmarkProvider, ProviderMeasurement, measure_provider
from .quality import QualityMetrics, QuantizerSaturation, measure_quality, measure_saturation
from .reporting import (
    BenchmarkRecord,
    OutputFormat,
    OutputTarget,
    add_output_arguments,
    output_target,
    write_records,
)
from .timing import PhaseTimings, Timing, time_first_call, triton_benchmark

__all__ = [
    "AttentionConfig",
    "AttentionShape",
    "BenchmarkProvider",
    "BenchmarkRecord",
    "EnvironmentInfo",
    "OutputFormat",
    "OutputTarget",
    "PhaseTimings",
    "ProviderMeasurement",
    "QualityMetrics",
    "QuantizerSaturation",
    "Timing",
    "add_output_arguments",
    "capture_environment",
    "measure_provider",
    "measure_quality",
    "measure_saturation",
    "output_target",
    "time_first_call",
    "triton_benchmark",
    "write_records",
]
