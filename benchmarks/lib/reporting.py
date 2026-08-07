"""Machine-readable benchmark records and CLI output options."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .environment import EnvironmentInfo
from .quality import QualityMetrics
from .timing import PhaseTimings

SCHEMA_VERSION = 1
type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None


class SerializableRecord(Protocol):
    """A versioned development record accepted by the common output writer."""

    def as_dict(self) -> Mapping[str, object]: ...


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
    """Convert nested benchmark values to strict JSON-compatible values."""
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


def add_output_arguments(
    parser: argparse.ArgumentParser,
    *,
    option_prefix: str | None = None,
    record_name: str = "result",
) -> None:
    """Add optional JSON and JSONL output arguments, with an optional prefix."""
    option_stem = "" if option_prefix is None else f"{option_prefix}-"
    destination_stem = "" if option_prefix is None else f"{option_prefix}_"
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{option_stem}json",
        dest=f"{destination_stem}json",
        type=Path,
        metavar="PATH",
        help=f"write all {record_name} records as a JSON array",
    )
    group.add_argument(
        f"--{option_stem}jsonl",
        dest=f"{destination_stem}jsonl",
        type=Path,
        metavar="PATH",
        help=f"write one {record_name} record per JSON line",
    )


def output_target(
    arguments: argparse.Namespace,
    *,
    option_prefix: str | None = None,
) -> OutputTarget | None:
    """Resolve arguments populated by :func:`add_output_arguments`."""
    destination_stem = "" if option_prefix is None else f"{option_prefix}_"
    json_path = getattr(arguments, f"{destination_stem}json", None)
    jsonl_path = getattr(arguments, f"{destination_stem}jsonl", None)
    if json_path is not None:
        return OutputTarget(json_path, OutputFormat.JSON)
    if jsonl_path is not None:
        return OutputTarget(jsonl_path, OutputFormat.JSONL)
    return None


def write_records(records: Iterable[SerializableRecord], target: OutputTarget | None) -> None:
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
