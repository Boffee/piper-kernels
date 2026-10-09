"""Hardware-independent workloads for comparable operator measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fnmatch import fnmatchcase
from typing import ClassVar, Literal

CATALOG_VERSION = 1
type Family = Literal["attention", "sparse_attention", "linear", "ffn", "conv3d", "pipeline"]


@dataclass(frozen=True, kw_only=True)
class Case:
    """A stable workload identity; changing its meaning requires a catalog revision."""

    id: str
    dtype: str = "bfloat16"
    seed: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class AttentionCase(Case):
    sequence: int
    heads: int
    kv_heads: int
    head_dim: int
    batch: int = 1
    causal: bool = False
    keep_ratio: float | None = None

    @property
    def family(self) -> Family:
        return "attention" if self.keep_ratio is None else "sparse_attention"


@dataclass(frozen=True, kw_only=True)
class LinearCase(Case):
    family: ClassVar[Family] = "linear"
    rows: int
    in_features: int
    out_features: int
    bias: bool = False


@dataclass(frozen=True, kw_only=True)
class FFNCase(Case):
    family: ClassVar[Family] = "ffn"
    rows: int
    width: int
    intermediate: int
    activation: Literal["gelu", "swiglu"] = "gelu"


@dataclass(frozen=True, kw_only=True)
class Conv3DCase(Case):
    family: ClassVar[Family] = "conv3d"
    channels: int
    frames: int
    height: int
    width: int
    out_channels: int
    batch: int = 1
    group_norm_silu: bool = False


@dataclass(frozen=True, kw_only=True)
class PipelineCase(Case):
    family: ClassVar[Family] = "pipeline"
    sequence: int
    heads: int
    kv_heads: int
    head_dim: int
    width: int
    batch: int = 1
    rotary_dim: int = 64
    keep_ratio: float | None = None


type BenchmarkCase = AttentionCase | LinearCase | FFNCase | Conv3DCase | PipelineCase


@dataclass(frozen=True)
class _Profile:
    name: str
    sequences: tuple[int, int]
    heads: int
    kv_heads: int
    head_dim: int
    width: int
    intermediate: int
    linears: tuple[tuple[int, int], ...]


# Rounded latent-space anchors, not exact model executions. See benchmarks/README.md.
_PROFILES = (
    _Profile("video", (20480, 110592), 56, 56, 128, 5376, 14336, ((5376, 7168), (14336, 5376))),
    _Profile("image", (2816, 16896), 48, 12, 128, 6144, 16384, ((6144, 1536), (6144, 16384))),
    _Profile("decoder", (10240, 20480), 32, 32, 64, 2048, 8192, ((2048, 2048), (8192, 2048))),
)


def standard_cases() -> tuple[BenchmarkCase, ...]:
    """Expand the same fixed suite on every device; no capability-dependent cases."""
    cases: list[BenchmarkCase] = []
    for profile in _PROFILES:
        for size, sequence in zip(("low", "high"), profile.sequences, strict=True):
            suffix = f"{profile.name}-{size}"
            attention = AttentionCase(
                id=f"attention-{suffix}",
                sequence=sequence,
                heads=profile.heads,
                kv_heads=profile.kv_heads,
                head_dim=profile.head_dim,
            )
            pipeline = PipelineCase(
                id=f"pipeline-{suffix}",
                sequence=sequence,
                heads=profile.heads,
                kv_heads=profile.kv_heads,
                head_dim=profile.head_dim,
                width=profile.width,
                rotary_dim=128 if profile.name == "image" else 64,
            )
            cases.extend((attention, pipeline))
            if profile.name == "video":
                for ratio, label in ((0.125, "eighth"), (0.5, "half")):
                    cases.extend(
                        (
                            replace(
                                attention, id=f"sparse-attention-{suffix}-{label}", keep_ratio=ratio
                            ),
                            replace(
                                pipeline, id=f"sparse-pipeline-{suffix}-{label}", keep_ratio=ratio
                            ),
                        )
                    )
            for inputs, outputs in profile.linears:
                cases.append(
                    LinearCase(
                        id=f"linear-{suffix}-{inputs}x{outputs}",
                        rows=sequence,
                        in_features=inputs,
                        out_features=outputs,
                    )
                )
            for activation in ("gelu", "swiglu"):
                cases.append(
                    FFNCase(
                        id=f"ffn-{suffix}-{activation}",
                        rows=sequence,
                        width=profile.width,
                        intermediate=profile.intermediate,
                        activation=activation,
                    )
                )
    for size, height, width in (("low", 544, 960), ("high", 768, 1344)):
        for stage, channels, outputs, frames, divisor in (
            (1, 256, 256, 9, 4),
            (2, 256, 512, 5, 8),
            (3, 512, 1024, 5, 16),
        ):
            for fused in (False, True):
                suffix = "-norm-silu" if fused else ""
                cases.append(
                    Conv3DCase(
                        id=f"conv3d-{size}-stage{stage}{suffix}",
                        dtype="float16",
                        channels=channels,
                        out_channels=outputs,
                        frames=frames,
                        height=height // divisor,
                        width=width // divisor,
                        group_norm_silu=fused,
                    )
                )
    return tuple(cases)


def diagnostic_cases() -> tuple[BenchmarkCase, ...]:
    """Small and ragged workloads for regression checks, outside the standard suite."""
    return (
        AttentionCase(id="attention-small", sequence=257, heads=4, kv_heads=2, head_dim=64),
        AttentionCase(
            id="sparse-attention-small",
            sequence=257,
            heads=2,
            kv_heads=2,
            head_dim=128,
            keep_ratio=0.5,
        ),
        LinearCase(id="linear-small", rows=64, in_features=256, out_features=256),
        LinearCase(id="linear-tail", rows=65, in_features=256, out_features=272),
        FFNCase(id="ffn-small-gelu", rows=64, width=256, intermediate=512),
        FFNCase(id="ffn-small-swiglu", rows=64, width=256, intermediate=512, activation="swiglu"),
        Conv3DCase(
            id="conv3d-small",
            dtype="float16",
            channels=64,
            out_channels=64,
            frames=3,
            height=8,
            width=8,
        ),
        Conv3DCase(
            id="conv3d-small-norm-silu",
            dtype="float16",
            channels=64,
            out_channels=64,
            frames=3,
            height=8,
            width=8,
            group_norm_silu=True,
        ),
        PipelineCase(
            id="pipeline-small", sequence=257, heads=4, kv_heads=2, head_dim=64, width=256
        ),
        PipelineCase(
            id="sparse-pipeline-small",
            sequence=257,
            heads=2,
            kv_heads=2,
            head_dim=128,
            width=256,
            keep_ratio=0.5,
        ),
    )


def select_cases(
    patterns: tuple[str, ...] = (),
    *,
    families: tuple[str, ...] = (),
) -> tuple[BenchmarkCase, ...]:
    """Select stable identities; diagnostic cases are included only by explicit pattern."""
    catalog = standard_cases() + diagnostic_cases() if patterns else standard_cases()
    for pattern in patterns:
        if not any(fnmatchcase(case.id, pattern) for case in catalog):
            raise ValueError(f"no benchmark case matches {pattern!r}")
    result = tuple(
        case
        for case in catalog
        if (not patterns or any(fnmatchcase(case.id, pattern) for pattern in patterns))
        and (not families or case.family in families)
    )
    if not result:
        raise ValueError("no benchmark cases selected")
    return result


def catalog_metadata(case_id: str | None) -> dict[str, str | int | None]:
    """Identify the selected catalog revision; custom workloads have no catalog identity."""
    return {"case_id": case_id, "catalog_version": CATALOG_VERSION if case_id is not None else None}


def named_case(identity: str) -> BenchmarkCase:
    """Look up an exact case for a tuner or stage diagnostic."""
    for case in standard_cases() + diagnostic_cases():
        if case.id == identity:
            return case
    raise ValueError(f"unknown benchmark case {identity!r}")
