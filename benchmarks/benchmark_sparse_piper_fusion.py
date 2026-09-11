"""Compare compiled fused and materialized ConvRot INT8 sparse attention at H3 size.

B1/H56/D128, hidden/output width 5376, group 256, minmax routing, 25% keep.
Inputs are seeded synthetic BF16 activations and INT8 weights, not a checkpoint.
Both paths include input preparation, QKV projections, routing, attention, and
output projection. The materialized reference uses the same quantized operator
boundaries; it is not an unfused BF16 model. Compilation and weight creation are
excluded. Shuffled, synchronized wall samples include allocation and host work.
Peak extra allocated bytes include the returned output and execution workspace.
"""

import argparse
import random
import uuid
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from lib.environment import capture_environment
from lib.reporting import BenchmarkRecord, add_output_arguments, output_target, write_records
from lib.timing import SampleTimings, time_first_call
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import SparsePiperAttention
from piper_kernels._triton.runtime import device_context
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    _backend,
    convrot_int8_sparse_piper_compile_options,
    key,
    query,
    value,
)
from piper_kernels.linear.convrot.int8 import _backend as linear_backend
from piper_kernels.linear.convrot.int8 import _ops
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_FUSED_OUTPUT = "piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, nargs="+", default=[8192, 32768, 100000])
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=881)
    add_output_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.sequence) < 64 or args.samples < 1 or args.device < 0:
        parser.error("requires sequence >= 64, positive samples, and a nonnegative device")
    if max(args.sequence) // 64 > 65536:
        parser.error("sparse key prefixes must fit UINT16 route indices")
    return args


def _projection(inputs: int, outputs: int, generator: torch.Generator) -> torch.nn.Linear:
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
        group_size=256,
    )
    layer = torch.nn.Linear(inputs, outputs, bias=False, device="meta", dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


class _H3Attention(torch.nn.Module):
    query_norm: torch.Tensor
    key_norm: torch.Tensor

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.query, self.key, self.value = [_projection(5376, 7168, generator) for _ in range(3)]
        self.output = _projection(7168, 5376, generator)
        for name in ("query_norm", "key_norm"):
            self.register_buffer(
                name,
                (torch.rand(128, device=generator.device, generator=generator) + 0.5).bfloat16(),
            )
        self.attention = SparsePiperAttention((0.25,) * 56)

    def _norm_rope(
        self, projected: torch.Tensor, norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, sequence, _ = projected.shape
        normalized = F.rms_norm(projected.view(batch, sequence, 56, 128), (128,), norm, 1e-5)
        rotary = normalized[..., :96]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        rotary = rotary * cos.to(torch.bfloat16)[None, :, None, :]
        rotary = rotary + rotated * sin.to(torch.bfloat16)[None, :, None, :]
        return torch.cat((rotary, normalized[..., 96:]), dim=-1).contiguous()

    def forward(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sparse_blocks: int
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        q = self._norm_rope(self.query(hidden), self.query_norm, cos, sin)
        k = self._norm_rope(self.key(hidden), self.key_norm, cos, sin)
        v = self.value(hidden).view(batch, sequence, 56, 128)
        attended = self.attention(q, k, v, sparse_key_blocks=sparse_blocks)
        return self.output(attended.reshape(batch, sequence, 7168))

    def materialized(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sparse_blocks: int
    ) -> torch.Tensor:
        qw, kw, vw, ow = (
            cast(ConvRotInt8Tensor, layer.weight)
            for layer in (self.query, self.key, self.value, self.output)
        )
        data, scale = _ops.prepare_input(hidden, 256)
        q = query._project_query_op(
            data,
            scale,
            qw.qdata,
            qw.scale,
            self.query_norm,
            cos,
            sin,
            1e-5,
            128**-0.5,
            _MINMAX_ROUTING,
        )
        k = key._project_key_op(
            data, scale, kw.qdata, kw.scale, self.key_norm, cos, sin, 1e-5, _MINMAX_ROUTING
        )
        mean = _ops.dequantized_input_mean(data, scale)
        v = value._project_value_op(data, scale, mean, vw.qdata, vw.scale)
        attended = _sparse_piper_attention_from_quantized_op(
            *q,
            *k,
            *v,
            [250_000] * 56,
            sparse_blocks,
            hidden.shape[1],
            _MINMAX_ROUTING,
        )
        # Match the materialized attention boundary's lifetimes before output
        # projection; retaining Q/K/V here would inflate its memory footprint.
        del q, k, v, data, scale, mean
        return linear_backend.require_linear_backend(attended).linear(
            attended.reshape(hidden.shape[0], hidden.shape[1], 7168),
            ow.qdata,
            ow.scale,
            None,
            256,
        )


class _CaptureFusion(CustomInferenceAwareGraphPass):
    """Fail the benchmark if the advertised fusion is absent or retraces."""

    def __init__(self) -> None:
        self.calls = 0
        self.targets: list[str] = []
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [str(node.target) for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid

    def check(self) -> None:
        assert self.calls == 1, f"expected one dynamic graph, got {self.calls}"
        assert self.targets.count(_FUSED_OUTPUT) == 1, self.targets
        assert "piper_kernels.convrot_int8_sparse_piper_project_query.default" not in self.targets


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    error, energy = 0.0, 0.0
    for start in range(0, actual.shape[1], 4096):
        left, right = (
            actual[:, start : start + 4096].float(),
            expected[:, start : start + 4096].float(),
        )
        assert bool(torch.isfinite(left).all())
        assert bool(torch.isfinite(right).all())
        error += float((left - right).square().sum())
        energy += float(right.square().sum())
    return (error / max(energy, 1e-30)) ** 0.5


def _measure(
    functions: dict[str, Callable[[], torch.Tensor]], args: argparse.Namespace
) -> dict[str, tuple[SampleTimings, int]]:
    for function in functions.values():
        function()
    torch.cuda.synchronize()
    timings: dict[str, list[float]] = {name: [] for name in functions}
    peaks: dict[str, list[int]] = {name: [] for name in functions}
    rng, order = random.Random(args.seed), list(functions)
    for _ in range(args.samples):
        rng.shuffle(order)
        for name in order:
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output, elapsed = time_first_call(functions[name], synchronize=torch.cuda.synchronize)
            timings[name].append(elapsed)
            peaks[name].append(torch.cuda.max_memory_allocated() - before)
            del output
    return {
        name: (SampleTimings(warmup_calls=1, samples_ms=tuple(timings[name])), max(peaks[name]))
        for name in functions
    }


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("requires native sparse projection, attention, and output backends")
    device = torch.device("cuda", args.device)
    with device_context(device):
        probe = torch.empty(0, device=device)
        _backend.require_projection_backend(probe)
        _backend.require_output_backend(probe)
        model = _H3Attention(torch.Generator(device=device).manual_seed(args.seed)).eval()
        capture = _CaptureFusion()
        options = convrot_int8_sparse_piper_compile_options()
        passes = options["post_grad_custom_pre_pass"]
        assert isinstance(passes, tuple)
        options["post_grad_custom_pre_pass"] = (*passes, capture)
        # Torch's options annotation does not include tuples of graph passes.
        compiled = torch.compile(model, dynamic=True, fullgraph=True, options=cast(Any, options))
        environment = capture_environment(Path(__file__).resolve().parents[1])
        records: list[BenchmarkRecord[SampleTimings]] = []
        for sequence in args.sequence:
            generator = torch.Generator(device=device).manual_seed(args.seed + sequence)
            hidden = torch.randn(
                (1, sequence, 5376), device=device, dtype=torch.bfloat16, generator=generator
            )
            angles = torch.rand((sequence, 96), device=device, generator=generator) * (2 * torch.pi)
            cos, sin = angles.cos(), angles.sin()
            del angles
            functions = {
                "materialized": partial(model.materialized, hidden, cos, sin, sequence // 64),
                "compiled_fused": partial(compiled, hidden, cos, sin, sequence // 64),
            }
            expected, actual = functions["materialized"](), functions["compiled_fused"]()
            error = _relative_l2(actual, expected)
            assert error < 0.015, error
            capture.check()
            del expected, actual
            measurements = _measure(functions, args)
            capture.check()
            for name, (timings, peak) in measurements.items():
                record = BenchmarkRecord(
                    benchmark="sparse_piper_fusion",
                    provider=name,
                    shape={
                        "batch": 1,
                        "sequence": sequence,
                        "heads": 56,
                        "head_dim": 128,
                        "hidden_width": 5376,
                        "output_width": 5376,
                    },
                    configuration={
                        "seed": args.seed,
                        "group_size": 256,
                        "routing": "minmax",
                        "keep_ratio": 0.25,
                        "input_source": "synthetic_bf16_activations_and_int8_weights",
                        "scope": (
                            "input preparation + QKV + routing + attention + output projection"
                        ),
                        "measurement_order": "shuffled_paired_calls",
                        "compile_time_included": False,
                    },
                    timings=timings,
                    environment=environment,
                    extra={
                        "relative_l2_vs_materialized": error if name == "compiled_fused" else 0.0,
                        "compile_calls": capture.calls if name == "compiled_fused" else 0,
                        "fused_operator": _FUSED_OUTPUT if name == "compiled_fused" else None,
                        "peak_extra_allocated_bytes": peak,
                    },
                )
                records.append(record)
                print(
                    f"S={sequence} {name}: {timings.operator_end_to_end.display()} ms "
                    f"(synchronized_wall; p50 [p20, p80]), peak extra {peak} bytes",
                    flush=True,
                )
            del functions, hidden, cos, sin
        write_records(records, output_target(args))


if __name__ == "__main__":
    main()
