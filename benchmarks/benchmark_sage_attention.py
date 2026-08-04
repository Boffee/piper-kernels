"""Benchmark end-to-end Triton SageAttention2++ against canonical Sage and SDPA."""

import argparse
import importlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
import triton.testing

from piper_kernels.attention import sage_attention

_CANONICAL_REVISION = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
CanonicalSage = Callable[..., torch.Tensor]


@dataclass(slots=True, frozen=True)
class Timing:
    """Median latency with a central 60% interval."""

    median_ms: float
    p20_ms: float
    p80_ms: float

    def display(self) -> str:
        """Format p50 followed by the p20/p80 interval."""
        return f"{self.median_ms:.3f} [{self.p20_ms:.3f}, {self.p80_ms:.3f}]"


@dataclass(slots=True, frozen=True)
class Result:
    """Timing and numerical result for one sequence length."""

    query_sequence: int
    key_sequence: int
    cold_ms: float
    sage: Timing
    sdpa: Timing
    mean_error: float
    max_error: float
    canonical_sage2pp: Timing | None = None
    canonical_sage2: Timing | None = None
    canonical_sage2pp_mean_error: float | None = None
    canonical_sage2pp_max_error: float | None = None
    canonical_sage2_mean_error: float | None = None
    canonical_sage2_max_error: float | None = None

    @property
    def speedup(self) -> float:
        """Return the median SDPA-to-SageAttention speed ratio."""
        return self.sdpa.median_ms / self.sage.median_ms

    @property
    def canonical_sage2pp_speedup(self) -> float | None:
        """Return median canonical Sage2++ latency divided by Piper latency."""
        if self.canonical_sage2pp is None:
            return None
        return self.canonical_sage2pp.median_ms / self.sage.median_ms

    @property
    def canonical_accumulator_speedup(self) -> float | None:
        """Return canonical Sage2 latency divided by canonical Sage2++ latency."""
        if self.canonical_sage2 is None or self.canonical_sage2pp is None:
            return None
        return self.canonical_sage2.median_ms / self.canonical_sage2pp.median_ms


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def _do_bench(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> Timing:
    median, p20, p80 = triton.testing.do_bench(
        function,
        warmup=warmup_ms,
        rep=repeat_ms,
        quantiles=[0.5, 0.2, 0.8],
    )
    return Timing(float(median), float(p20), float(p80))


def _load_canonical(capability: tuple[int, int]) -> CanonicalSage:
    try:
        module = importlib.import_module("sageattention")
    except (ImportError, OSError) as exc:
        architecture = f"{capability[0]}.{capability[1]}"
        raise SystemExit(
            "Canonical SageAttention is unavailable. Build the pinned benchmark dependency with:\n"
            f"  TORCH_CUDA_ARCH_LIST={architecture} uv sync --group benchmark"
        ) from exc
    return cast(CanonicalSage, module.sageattn_qk_int8_pv_fp8_cuda)


@torch.inference_mode()
def _run_shape(
    sequences: tuple[int, int],
    batch: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    warmup_ms: int,
    repeat_ms: int,
    canonical_sage: CanonicalSage | None,
    canonical_qk_gran: str,
) -> Result:
    query_sequence, key_sequence = sequences
    query = torch.randn(batch, heads, query_sequence, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(batch, heads, key_sequence, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)

    def optimized() -> torch.Tensor:
        return sage_attention(query, key, value, is_causal=is_causal)

    def sdpa() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
        )

    def canonical(pv_accum_dtype: str) -> torch.Tensor:
        assert canonical_sage is not None
        return canonical_sage(
            query,
            key,
            value,
            tensor_layout="HND",
            is_causal=is_causal,
            qk_quant_gran=canonical_qk_gran,
            sm_scale=head_dim**-0.5,
            pv_accum_dtype=pv_accum_dtype,
            smooth_k=True,
            smooth_v=False,
            return_lse=False,
        )

    expected = sdpa()
    torch.cuda.synchronize()
    started = time.perf_counter()
    actual = optimized()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - started) * 1_000
    error = (actual.float() - expected.float()).abs()
    canonical_sage2pp_timing = None
    canonical_sage2_timing = None
    canonical_sage2pp_mean_error = None
    canonical_sage2pp_max_error = None
    canonical_sage2_mean_error = None
    canonical_sage2_max_error = None
    if canonical_sage is not None:

        def canonical_sage2pp() -> torch.Tensor:
            return canonical("fp32+fp16")

        def canonical_sage2() -> torch.Tensor:
            return canonical("fp32+fp32")

        canonical_sage2pp_error = (canonical_sage2pp().float() - expected.float()).abs()
        canonical_sage2_error = (canonical_sage2().float() - expected.float()).abs()
        canonical_sage2pp_timing = _do_bench(canonical_sage2pp, warmup_ms, repeat_ms)
        canonical_sage2_timing = _do_bench(canonical_sage2, warmup_ms, repeat_ms)
        canonical_sage2pp_mean_error = canonical_sage2pp_error.mean().item()
        canonical_sage2pp_max_error = canonical_sage2pp_error.max().item()
        canonical_sage2_mean_error = canonical_sage2_error.mean().item()
        canonical_sage2_max_error = canonical_sage2_error.max().item()

    return Result(
        query_sequence=query_sequence,
        key_sequence=key_sequence,
        cold_ms=cold_ms,
        sage=_do_bench(optimized, warmup_ms, repeat_ms),
        sdpa=_do_bench(sdpa, warmup_ms, repeat_ms),
        mean_error=error.mean().item(),
        max_error=error.max().item(),
        canonical_sage2pp=canonical_sage2pp_timing,
        canonical_sage2=canonical_sage2_timing,
        canonical_sage2pp_mean_error=canonical_sage2pp_mean_error,
        canonical_sage2pp_max_error=canonical_sage2pp_max_error,
        canonical_sage2_mean_error=canonical_sage2_mean_error,
        canonical_sage2_max_error=canonical_sage2_max_error,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument(
        "--kv-sequence",
        type=int,
        help="fixed key/value sequence length (defaults to each query sequence length)",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="also benchmark the pinned official SageAttention2++ CUDA implementation",
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the requested benchmark matrix and print a Markdown table."""
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("SageAttention benchmarking requires a CUDA-capable GPU")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9) and capability[0] != 12:
        raise SystemExit("The optimized SageAttention backend requires SM89 or SM12x")
    if (
        args.causal
        and args.kv_sequence is not None
        and any(sequence != args.kv_sequence for sequence in args.sequence)
    ):
        raise SystemExit("causal attention requires equal query and key/value lengths")

    canonical_sage = _load_canonical(capability) if args.canonical else None
    canonical_qk_gran = "per_warp" if capability[0] == 12 else "per_thread"

    dtype = _dtype(args.dtype)
    print(f"GPU: {torch.cuda.get_device_name()}; capability: SM{capability[0]}{capability[1]}")
    print(
        f"Torch: {torch.__version__}; dtype: {dtype}; batch: {args.batch}; "
        f"heads: {args.heads}; head dim: {args.head_dim}; causal: {args.causal}"
    )
    if canonical_sage is not None:
        print(
            f"Canonical SageAttention: 2.2.0 @ {_CANONICAL_REVISION[:12]}; "
            f"Q/K granularity: {canonical_qk_gran}; "
            "PV accumulation: Sage2++ fp32+fp16, Sage2 fp32+fp32"
        )
    print()
    if canonical_sage is None:
        print(
            "| query | key/value | cold Piper (ms) | Piper p50 [p20, p80] (ms) | "
            "SDPA p50 [p20, p80] (ms) | SDPA / Piper | "
            "Piper mean error | Piper max error |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        print(
            "| query | key/value | cold Piper (ms) | Piper p50 [p20, p80] (ms) | "
            "canonical Sage2++ p50 [p20, p80] (ms) | "
            "canonical Sage2 p50 [p20, p80] (ms) | SDPA p50 [p20, p80] (ms) | "
            "Sage2 / Sage2++ | Sage2++ / Piper | SDPA / Piper | Piper mean/max error | "
            "Sage2++ mean/max error | Sage2 mean/max error |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for query_sequence in args.sequence:
        key_sequence = args.kv_sequence or query_sequence
        result = _run_shape(
            (query_sequence, key_sequence),
            args.batch,
            args.heads,
            args.head_dim,
            dtype,
            args.causal,
            args.warmup_ms,
            args.repeat_ms,
            canonical_sage,
            canonical_qk_gran,
        )
        if result.canonical_sage2pp is None:
            print(
                f"| {result.query_sequence} | {result.key_sequence} | {result.cold_ms:.3f} "
                f"| {result.sage.display()} "
                f"| {result.sdpa.display()} | {result.speedup:.2f}x "
                f"| {result.mean_error:.6f} | {result.max_error:.6f} |"
            )
        else:
            assert result.canonical_sage2 is not None
            assert result.canonical_sage2pp_speedup is not None
            assert result.canonical_accumulator_speedup is not None
            assert result.canonical_sage2pp_mean_error is not None
            assert result.canonical_sage2pp_max_error is not None
            assert result.canonical_sage2_mean_error is not None
            assert result.canonical_sage2_max_error is not None
            print(
                f"| {result.query_sequence} | {result.key_sequence} | {result.cold_ms:.3f} "
                f"| {result.sage.display()} "
                f"| {result.canonical_sage2pp.display()} | {result.canonical_sage2.display()} "
                f"| {result.sdpa.display()} | {result.canonical_accumulator_speedup:.2f}x "
                f"| {result.canonical_sage2pp_speedup:.2f}x | {result.speedup:.2f}x "
                f"| {result.mean_error:.6f}/{result.max_error:.6f} "
                f"| {result.canonical_sage2pp_mean_error:.6f}/"
                f"{result.canonical_sage2pp_max_error:.6f} "
                f"| {result.canonical_sage2_mean_error:.6f}/"
                f"{result.canonical_sage2_max_error:.6f} |"
            )


if __name__ == "__main__":
    main()
