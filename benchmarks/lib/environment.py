"""Reproducibility environment metadata for benchmark records."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class EnvironmentInfo:
    """Software, hardware, and repository state captured with a benchmark."""

    captured_at_utc: str
    python_version: str
    platform: str
    torch_version: str
    triton_version: str | None
    accelerator_backend: str | None
    accelerator_runtime_version: str | None
    accelerator_driver_version: str | None
    gpu_name: str | None
    gpu_architecture: str | None
    gpu_index: int | None
    git_revision: str | None
    git_dirty: bool | None

    def as_dict(self) -> dict[str, str | int | bool | None]:
        """Return stable machine-readable field names."""
        return {
            "captured_at_utc": self.captured_at_utc,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "accelerator_backend": self.accelerator_backend,
            "accelerator_runtime_version": self.accelerator_runtime_version,
            "accelerator_driver_version": self.accelerator_driver_version,
            "gpu_name": self.gpu_name,
            "gpu_architecture": self.gpu_architecture,
            "gpu_index": self.gpu_index,
            "git_revision": self.git_revision,
            "git_dirty": self.git_dirty,
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_state(repository: Path | None) -> tuple[str | None, bool | None]:
    if repository is None:
        return None, None
    revision = _run(["git", "rev-parse", "HEAD"], repository)
    if revision is None:
        return None, None
    status = _run(["git", "status", "--porcelain", "--untracked-files=normal"], repository)
    return revision, bool(status)


def _accelerator_build() -> tuple[str | None, str | None]:
    hip_version = getattr(torch.version, "hip", None)
    if hip_version is not None:
        return "rocm", str(hip_version)
    if torch.version.cuda is not None:
        return "cuda", torch.version.cuda
    return None, None


def _nvidia_driver_version() -> str | None:
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    return output.splitlines()[0] if output else None


def capture_environment(repository: Path | None = None) -> EnvironmentInfo:
    """Capture the current process, CUDA device, and Git checkout metadata."""
    accelerator_backend, runtime_version = _accelerator_build()
    gpu_index = None
    gpu_name = None
    gpu_architecture = None
    driver_version = None
    if torch.cuda.is_available():
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_index)
        if accelerator_backend == "rocm":
            properties = torch.cuda.get_device_properties(gpu_index)
            architecture = getattr(properties, "gcnArchName", None)
            gpu_architecture = str(architecture) if architecture is not None else None
        else:
            major, minor = torch.cuda.get_device_capability(gpu_index)
            gpu_architecture = f"SM{major}{minor}"
            driver_version = _nvidia_driver_version()

    git_revision, git_dirty = _git_state(repository)
    return EnvironmentInfo(
        captured_at_utc=datetime.now(UTC).isoformat(),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        torch_version=str(torch.__version__),
        triton_version=_package_version("triton"),
        accelerator_backend=accelerator_backend,
        accelerator_runtime_version=runtime_version,
        accelerator_driver_version=driver_version,
        gpu_name=gpu_name,
        gpu_architecture=gpu_architecture,
        gpu_index=gpu_index,
        git_revision=git_revision,
        git_dirty=git_dirty,
    )
