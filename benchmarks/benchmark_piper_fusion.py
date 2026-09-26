"""Compare compiled dense ConvRot INT8 QKV fusion with its output pipeline.

Defaults to H3 transformer dimensions: B1/H56/D128, hidden/output width 5376,
group 256, BF16, noncausal self-attention, and 75% rotary dimensions. Inputs
and quantized weights are seeded synthetic data, not a model checkpoint.
Both providers include input preparation, projections, norm/RoPE, attention,
and output projection. Compilation and tensor construction are excluded.
"""

import argparse
import gc
import random
import uuid
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from lib.environment import capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.timing import ClockDomain, SampleTimings, Timing, time_first_call
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import piper_attention
from piper_kernels._triton.runtime import device_context
from piper_kernels.fusions.convrot_int8_piper import (
    _backend,
    _schedule,
    convrot_int8_piper_compile_options,
    output,
)
from piper_kernels.weights.convrot._rotation import SUPPORTED_GROUP_SIZES
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_FUSED_OUTPUT = "piper_kernels.convrot_int8_piper_projected_query_attention_output.default"
_QUANTIZED_ATTENTION = "piper_kernels.piper_attention_from_quantized.default"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 99968, 100000]
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--rotary-dim", type=int, help="rotary width; defaults to 75%% of head dim")
    parser.add_argument("--width", type=int, default=5376, help="hidden and output feature width")
    parser.add_argument("--group-size", type=int, choices=SUPPORTED_GROUP_SIZES, default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--chunk-rows", type=int, help="override the maximum output chunk rows")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=881)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if args.kv_heads is None:
        args.kv_heads = args.heads
    if args.rotary_dim is None:
        args.rotary_dim = args.head_dim * 3 // 4
    if not 0 < args.rotary_dim <= args.head_dim or args.rotary_dim % 2:
        parser.error("rotary dim must be positive, even, and at most head dim")
    if min(*args.sequence, args.batch, args.heads, args.kv_heads, args.width, args.samples) < 1:
        parser.error("sequence, batch, heads, width, and samples must be positive")
    if args.device < 0 or args.heads % args.kv_heads:
        parser.error("requires a nonnegative device and heads divisible by KV heads")
    if args.width % args.group_size or (args.heads * args.head_dim) % args.group_size:
        parser.error("hidden and merged query widths must be divisible by group size")
    if args.chunk_rows is not None and (args.chunk_rows < 128 or args.chunk_rows % 128):
        parser.error("chunk rows must be a positive multiple of 128")
    return args


def _projection(
    inputs: int, outputs: int, group_size: int, dtype: torch.dtype, generator: torch.Generator
) -> torch.nn.Linear:
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(
            -128,
            128,
            (outputs, inputs),
            device=generator.device,
            dtype=torch.int8,
            generator=generator,
        ),
        torch.rand((outputs, 1), device=generator.device, generator=generator) * 0.01 + 0.001,
        group_size=group_size,
        logical_dtype=dtype,
    )
    layer = torch.nn.Linear(inputs, outputs, bias=False, device="meta", dtype=dtype)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


class _Attention(torch.nn.Module):
    query_norm: torch.Tensor
    key_norm: torch.Tensor

    def __init__(self, args: argparse.Namespace, generator: torch.Generator) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = args.heads, args.kv_heads, args.head_dim
        self.causal, self.dtype = args.causal, getattr(torch, args.dtype)
        self.rotary_dim = args.rotary_dim
        projection = partial(
            _projection, group_size=args.group_size, dtype=self.dtype, generator=generator
        )
        self.query = projection(args.width, self.heads * self.head_dim)
        self.key = projection(args.width, self.kv_heads * self.head_dim)
        self.value = projection(args.width, self.kv_heads * self.head_dim)
        self.output = projection(self.heads * self.head_dim, args.width)
        for name in ("query_norm", "key_norm"):
            self.register_buffer(
                name,
                (torch.rand(self.head_dim, device=generator.device, generator=generator) + 0.5).to(
                    self.dtype
                ),
            )

    def _norm_rope(
        self,
        projected: torch.Tensor,
        norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        batch, sequence, _ = projected.shape
        normalized = F.rms_norm(
            projected.view(batch, sequence, heads, self.head_dim), (self.head_dim,), norm, 1e-5
        )
        rotary = normalized[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        rotary = rotary * cos.to(self.dtype)[None, :, None, :]
        rotary = rotary + rotated * sin.to(self.dtype)[None, :, None, :]
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        q = self._norm_rope(self.query(hidden), self.query_norm, cos, sin, self.heads)
        k = self._norm_rope(self.key(hidden), self.key_norm, cos, sin, self.kv_heads)
        v = self.value(hidden).view(batch, sequence, self.kv_heads, self.head_dim).transpose(1, 2)
        attended = piper_attention(q, k, v, scale=self.head_dim**-0.5, is_causal=self.causal)
        return self.output(
            attended.transpose(1, 2).reshape(batch, sequence, self.heads * self.head_dim)
        )


class _CaptureFusion(CustomInferenceAwareGraphPass):
    """Check advertised fusions and optionally set a benchmark-only chunk override."""

    def __init__(self, *, fuse_output: bool, chunk_rows: int | None) -> None:
        self.fuse_output, self.chunk_rows = fuse_output, chunk_rows
        self.calls = 0
        self.targets: list[str] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        for node in graph.nodes:
            if str(node.target) == _FUSED_OUTPUT and self.chunk_rows is not None:
                node.kwargs = dict(node.kwargs, query_chunk_rows=self.chunk_rows)
        self.targets = [str(node.target) for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid

    def check(self) -> None:
        assert self.calls == 1, f"expected one dynamic graph, got {self.calls}"
        expected = _FUSED_OUTPUT if self.fuse_output else _QUANTIZED_ATTENTION
        assert self.targets.count(expected) == 1, self.targets
        prefix = "piper_kernels.convrot_int8_piper_project_"
        for name in ("key", "value"):
            assert self.targets.count(f"{prefix}{name}.default") == 1, self.targets
        assert (f"{prefix}query.default" in self.targets) is (not self.fuse_output), self.targets


def _check_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check correctness outside timing without a sequence-sized FP32 copy."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    for start in range(0, actual.shape[1], 4096):
        left, right = actual[:, start : start + 4096], expected[:, start : start + 4096]
        assert bool(torch.isfinite(left).all())
        assert bool(torch.isfinite(right).all())
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def _measure(
    functions: dict[str, Callable[[], torch.Tensor]], args: argparse.Namespace
) -> dict[str, tuple[SampleTimings, int, dict[str, object] | None]]:
    order, rng = list(functions), random.Random(args.seed)
    timings: dict[str, list[float]] = {name: [] for name in order}
    peaks: dict[str, list[int]] = {name: [] for name in order}
    for function in functions.values():
        function()
    torch.cuda.synchronize()
    for _ in range(args.samples):
        rng.shuffle(order)
        for name in order:
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result, elapsed = time_first_call(functions[name], synchronize=torch.cuda.synchronize)
            timings[name].append(elapsed)
            peaks[name].append(torch.cuda.max_memory_allocated() - before)
            del result
    graph_timings = {} if args.no_cuda_graph else _measure_graphs(functions, args.samples, rng)
    return {
        name: (SampleTimings(1, tuple(timings[name])), max(peaks[name]), graph_timings.get(name))
        for name in functions
    }


def _measure_graphs(
    functions: dict[str, Callable[[], torch.Tensor]], samples: int, rng: random.Random
) -> dict[str, dict[str, object]]:
    graphs, outputs = {}, {}
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = function()
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    _check_equal(outputs["qkv_output"], outputs["qkv_only"])
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    order = list(functions)
    timings: dict[str, list[float]] = {name: [] for name in order}
    for _ in range(samples):
        rng.shuffle(order)
        for name in order:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end))
    return {
        name: {
            **Timing.from_samples(values, ClockDomain.DEVICE_EVENT).as_dict(),
            "warmup_replays": 1,
            "samples_ms": values,
        }
        for name, values in timings.items()
    }


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device("cuda", args.device)
    if not torch.cuda.is_available():
        raise SystemExit("requires a native dense fusion backend")
    with device_context(device):
        if (
            _backend.select_projection_backend(
                torch.empty(0, device=device), head_dim=args.head_dim
            )
            is None
        ):
            raise SystemExit("requires a native dense fusion backend")
        model = _Attention(args, torch.Generator(device=device).manual_seed(args.seed)).eval()
        compiled, captures = {}, {}
        for name, fuse_output in (("qkv_only", False), ("qkv_output", True)):
            capture = _CaptureFusion(fuse_output=fuse_output, chunk_rows=args.chunk_rows)
            options = convrot_int8_piper_compile_options(fuse_output=fuse_output)
            passes = options["post_grad_custom_pre_pass"]
            assert isinstance(passes, tuple)
            options["post_grad_custom_pre_pass"] = (*passes, capture)
            compiled[name] = torch.compile(
                model, dynamic=True, fullgraph=True, options=cast(Any, options)
            )
            captures[name] = capture
        environment = capture_environment(Path(__file__).resolve().parents[1])
        records: list[BenchmarkRecord[SampleTimings]] = []
        for sequence in args.sequence:
            generator = torch.Generator(device=device).manual_seed(args.seed + sequence)
            hidden = torch.randn(
                (args.batch, sequence, args.width),
                device=device,
                dtype=model.dtype,
                generator=generator,
            )
            angles = torch.rand(
                (sequence, model.rotary_dim), device=device, generator=generator
            ) * (2 * torch.pi)
            cos, sin = angles.cos(), angles.sin()
            del angles
            # Prevent an initial sequence equal to a feature width from being
            # duck-shaped with that static dimension by Dynamo.
            torch._dynamo.mark_dynamic(hidden, 1)
            torch._dynamo.mark_dynamic(cos, 0)
            torch._dynamo.mark_dynamic(sin, 0)
            functions = {name: partial(fn, hidden, cos, sin) for name, fn in compiled.items()}
            expected, actual = functions["qkv_only"](), functions["qkv_output"]()
            _check_equal(actual, expected)
            del expected, actual
            measurements = _measure(functions, args)
            maximum_rows = args.chunk_rows or output.DEFAULT_QUERY_CHUNK_ROWS
            selected_rows = _schedule.select_query_chunk_rows(
                (args.batch, args.heads, sequence, args.head_dim),
                device,
                maximum_rows,
                is_causal=args.causal,
            )
            for name, (timings, peak, graph_timings) in measurements.items():
                captures[name].check()
                records.append(
                    BenchmarkRecord(
                        benchmark="piper_fusion",
                        provider=name,
                        shape={
                            "batch": args.batch,
                            "sequence": sequence,
                            "heads": args.heads,
                            "kv_heads": args.kv_heads,
                            "head_dim": args.head_dim,
                            "hidden_width": args.width,
                            "output_width": args.width,
                        },
                        configuration={
                            "seed": args.seed,
                            "group_size": args.group_size,
                            "dtype": args.dtype,
                            "is_causal": args.causal,
                            "rotary_dim": model.rotary_dim,
                            "chunk_rows_override": args.chunk_rows,
                            "maximum_chunk_rows": maximum_rows if name == "qkv_output" else None,
                            "selected_chunk_rows": selected_rows if name == "qkv_output" else None,
                            "input_source": "synthetic_bf16_or_fp16_activations_and_int8_weights",
                            "scope": (
                                "input preparation + QKV + norm/RoPE + attention "
                                "+ output projection"
                            ),
                            "measurement_order": "shuffled_paired_calls",
                            "compile_time_included": False,
                        },
                        timings=timings,
                        environment=environment,
                        extra={
                            "exact_output_match": True,
                            "compile_calls": captures[name].calls,
                            "graph_targets": captures[name].targets,
                            "peak_extra_allocated_bytes": peak,
                            "peak_scope": (
                                "output and workspace, excluding resident inputs and weights"
                            ),
                            "cuda_graph": graph_timings,
                        },
                    )
                )
                graph_summary = (
                    ""
                    if graph_timings is None
                    else (f", graph {graph_timings['median_ms']:.3f} ms")
                )
                chunk_summary = (
                    f", output chunk rows={selected_rows}" if name == "qkv_output" else ""
                )
                print(
                    f"S={sequence} {name}: wall {timings.operator_end_to_end.display()} ms"
                    f"{graph_summary}, peak extra {peak / 2**30:.3f} GiB{chunk_summary}",
                    flush=True,
                )
            write_records(records, output_target(args))
            del functions, hidden, cos, sin
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
