"""Machine-readable benchmark result records and CLI output options."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .environment import EnvironmentInfo
from .quality import QualityMetrics
from .timing import PhaseTimings

SCHEMA_VERSION = 1
type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None


class OutputFormat(Enum):
    """Supported machine-readable formats."""

    JSON = "json"
    JSONL = "jsonl"


@dataclass(frozen=True, slots=True)
class OutputTarget:
    """A path and serialization format selected on the command line."""

    path: Path
    format: OutputFormat


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One provider, shape, configuration, timing, and quality observation."""

    benchmark: str
    provider: str
    shape: Mapping[str, Any]
    configuration: Mapping[str, Any]
    timings: PhaseTimings
    environment: EnvironmentInfo
    quality: QualityMetrics | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Return the stable versioned result schema."""
        return {
            "schema_version": self.schema_version,
            "benchmark": self.benchmark,
            "provider": self.provider,
            "shape": dict(self.shape),
            "configuration": dict(self.configuration),
            "timings": self.timings.as_dict(),
            "quality": None if self.quality is None else self.quality.as_dict(),
            "environment": self.environment.as_dict(),
            "extra": dict(self.extra),
        }


def _json_safe(value: object) -> JSONValue:
    if isinstance(value, float) and not math.isfinite(value):
        result: JSONValue = None
    elif isinstance(value, Path):
        result = str(value)
    elif isinstance(value, Enum):
        result = str(value.value)
    elif isinstance(value, Mapping):
        result = {str(key): _json_safe(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_json_safe(item) for item in value]
    elif value is None or isinstance(value, (str, int, float, bool)):
        result = value
    else:
        result = str(value)
    return result


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive optional JSON and JSONL output arguments."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="write the complete result set as a JSON array",
    )
    group.add_argument(
        "--jsonl",
        type=Path,
        metavar="PATH",
        help="write one JSON result object per line",
    )


def output_target(arguments: argparse.Namespace) -> OutputTarget | None:
    """Resolve arguments populated by :func:`add_output_arguments`."""
    json_path = getattr(arguments, "json", None)
    jsonl_path = getattr(arguments, "jsonl", None)
    if json_path is not None:
        return OutputTarget(json_path, OutputFormat.JSON)
    if jsonl_path is not None:
        return OutputTarget(jsonl_path, OutputFormat.JSONL)
    return None


def write_records(records: Iterable[BenchmarkRecord], target: OutputTarget | None) -> None:
    """Write records when a machine-readable output target was requested."""
    if target is None:
        return
    values = [_json_safe(record.as_dict()) for record in records]
    target.path.parent.mkdir(parents=True, exist_ok=True)
    if target.format is OutputFormat.JSON:
        content = json.dumps(values, indent=2, allow_nan=False) + "\n"
    else:
        content = "".join(
            json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
            for value in values
        )
    target.path.write_text(content, encoding="utf-8")
