"""Complete ConvRot INT8 projection/attention/output pipeline comparisons.

Both implementations use the same prepared projection arithmetic. The fused
implementation bounds query/output scratch; the materialized implementation
retains the attention boundary. Quality checks sample the latter outside timing,
without retaining a second full output or constructing quadratic FP references.
"""

from __future__ import annotations

import importlib.util
import uuid
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import SparsePiperAttention, piper_attention
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _backend as dense_attention_backend
from piper_kernels.attention.piper_attention._quantized_dispatch import (
    _piper_attention_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
)
from piper_kernels.attention.sparse_piper_attention._routing_modes import _MINMAX_ROUTING

from .cases import PipelineCase
from .quality import measure_quality
from .suite_types import Implementation, Operation, QualityCheck, normal_tensor, sample_indices

# Projection imports transitively require optional TorchAO; resolve them after support checks.
# ruff: noqa: PLC0415

if TYPE_CHECKING:
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_GROUP_SIZE = 64


def _projection(
    inputs: int, outputs: int, *, dtype: torch.dtype, device: torch.device, seed: int
) -> torch.nn.Linear:
    """Create identical packed weights on every device, without a dense GPU copy."""
    from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

    generator = torch.Generator(device="cpu").manual_seed(seed)
    qdata = torch.randint(-128, 128, (outputs, inputs), dtype=torch.int8, generator=generator)
    scale = (torch.rand((outputs, 1), generator=generator) + 0.5) / (64 * inputs**0.5)
    weight = ConvRotInt8Tensor.from_quantized(
        qdata.to(device), scale.to(device), group_size=_GROUP_SIZE, logical_dtype=dtype
    )
    layer = torch.nn.Linear(inputs, outputs, bias=False, device="meta", dtype=dtype)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    return layer


class Pipeline(torch.nn.Module):
    """The same model-shaped operation for dense and block-sparse attention."""

    query_norm: torch.Tensor
    key_norm: torch.Tensor

    def __init__(self, case: PipelineCase, device: torch.device) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = case.heads, case.kv_heads, case.head_dim
        self.rotary_dim, self.dtype = case.rotary_dim, getattr(torch, case.dtype)
        self.keep_ratio = case.keep_ratio
        self.query = _projection(
            case.width,
            case.heads * case.head_dim,
            dtype=self.dtype,
            device=device,
            seed=case.seed + 1,
        )
        self.key = _projection(
            case.width,
            case.kv_heads * case.head_dim,
            dtype=self.dtype,
            device=device,
            seed=case.seed + 2,
        )
        self.value = _projection(
            case.width,
            case.kv_heads * case.head_dim,
            dtype=self.dtype,
            device=device,
            seed=case.seed + 3,
        )
        self.output = _projection(
            case.heads * case.head_dim,
            case.width,
            dtype=self.dtype,
            device=device,
            seed=case.seed + 4,
        )
        for offset, name in enumerate(("query_norm", "key_norm"), start=5):
            self.register_buffer(
                name,
                normal_tensor(
                    (case.head_dim,),
                    dtype=self.dtype,
                    device=device,
                    seed=case.seed + offset,
                    scale=0.1,
                )
                + 1,
            )
        self.sparse = (
            None
            if case.keep_ratio is None
            else SparsePiperAttention((case.keep_ratio,) * case.heads, routing="minmax")
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
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        q = self._norm_rope(self.query(hidden), self.query_norm, cos, sin, self.heads)
        k = self._norm_rope(self.key(hidden), self.key_norm, cos, sin, self.kv_heads)
        v = self.value(hidden).view(batch, sequence, self.kv_heads, self.head_dim)
        if self.sparse is None:
            attended = piper_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            attended = attended.transpose(1, 2)
        else:
            attended = self.sparse(
                q.contiguous(), k.contiguous(), v, sparse_key_blocks=sequence // 64
            )
        return self.output(attended.reshape(batch, sequence, self.heads * self.head_dim))

    def materialized(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Use the fused projection arithmetic, retaining a complete attention output."""
        from piper_kernels.linear.convrot.int8 import _backend as linear_backend
        from piper_kernels.linear.convrot.int8 import _ops

        qw, kw, vw, ow = (
            cast("ConvRotInt8Tensor", layer.weight)
            for layer in (self.query, self.key, self.value, self.output)
        )
        data, scale = _ops.prepare_input(hidden, _GROUP_SIZE)
        if self.keep_ratio is None:
            from piper_kernels.fusions.convrot_int8_piper import (
                key as dense_key,
            )
            from piper_kernels.fusions.convrot_int8_piper import (
                query as dense_query,
            )
            from piper_kernels.fusions.convrot_int8_piper import (
                value as dense_value,
            )

            q = dense_query._project_query_op(
                data,
                scale,
                qw.qdata,
                qw.scale,
                self.query_norm,
                cos,
                sin,
                1e-5,
                self.head_dim**-0.5,
            )
            k = dense_key._project_key_op(
                data,
                scale,
                kw.qdata,
                kw.scale,
                self.key_norm,
                cos,
                sin,
                1e-5,
            )
            v = dense_value._project_value_op(
                data,
                scale,
                vw.qdata,
                vw.scale,
                head_dim=self.head_dim,
                is_causal=False,
            )
        else:
            from piper_kernels.fusions.convrot_int8_sparse_piper import (
                key as sparse_key,
            )
            from piper_kernels.fusions.convrot_int8_sparse_piper import (
                query as sparse_query,
            )
            from piper_kernels.fusions.convrot_int8_sparse_piper import (
                value as sparse_value,
            )

            q = sparse_query._project_query_op(
                data,
                scale,
                qw.qdata,
                qw.scale,
                self.query_norm,
                cos,
                sin,
                1e-5,
                self.head_dim**-0.5,
                _MINMAX_ROUTING,
            )
            k = sparse_key._project_key_op(
                data,
                scale,
                kw.qdata,
                kw.scale,
                self.key_norm,
                cos,
                sin,
                1e-5,
                _MINMAX_ROUTING,
            )
            mean = _ops.dequantized_input_mean(data, scale)
            v = sparse_value._project_value_op(
                data,
                scale,
                mean,
                vw.qdata,
                vw.scale,
                head_dim=self.head_dim,
            )
            del mean
        # Do not charge the materialized path for storage its next stage no longer uses.
        del data, scale
        if self.keep_ratio is None:
            attended = _piper_attention_from_quantized_op(
                *q,
                *k,
                *v,
                hidden.shape[1],
                hidden.shape[1],
                False,
                self.dtype,
            ).transpose(1, 2)
        else:
            attended = _sparse_piper_attention_from_quantized_op(
                *q,
                *k,
                *v,
                [round(self.keep_ratio * 1_000_000)] * self.heads,
                hidden.shape[1] // 64,
                hidden.shape[1],
                _MINMAX_ROUTING,
                output_dtype=self.dtype,
            )
        del q, k, v
        return linear_backend.require_linear_backend(attended).linear(
            attended.reshape(hidden.shape[0], hidden.shape[1], self.heads * self.head_dim),
            ow.qdata,
            ow.scale,
            None,
            _GROUP_SIZE,
        )


class CaptureFusion(CustomInferenceAwareGraphPass):
    """Require the advertised full pipeline rewrite, and report its actual window."""

    def __init__(self, *, sparse: bool) -> None:
        family = "convrot_int8_sparse_piper" if sparse else "convrot_int8_piper"
        self.prefix = f"piper_kernels.{family}"
        self.output_target = f"{self.prefix}_projected_query_attention_output.default"
        self.calls = 0
        self.targets: list[str] = []
        self.query_chunk_rows: int | None = None
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [str(node.target) for node in graph.nodes if node.op == "call_function"]
        for node in graph.nodes:
            if str(node.target) != self.output_target:
                continue
            # FX targets expose their schema dynamically; contain that introspection here.
            schema = cast(Any, node.target)._schema
            index = next(
                i
                for i, argument in enumerate(schema.arguments)
                if argument.name == "query_chunk_rows"
            )
            rows = (
                node.args[index]
                if index < len(node.args)
                else node.kwargs.get("query_chunk_rows", schema.arguments[index].default_value)
            )
            assert isinstance(rows, int)
            self.query_chunk_rows = rows

    def uuid(self) -> bytes:
        return self._uuid

    def check(self) -> None:
        assert self.calls == 1, f"expected one compiled graph, got {self.calls}"
        assert self.targets.count(self.output_target) == 1, self.targets
        assert f"{self.prefix}_project_query.default" not in self.targets, self.targets
        for name in ("key", "value"):
            assert self.targets.count(f"{self.prefix}_project_{name}.default") == 1, self.targets


def _unsupported_reason(case: PipelineCase, device: torch.device) -> str | None:  # noqa: PLR0911
    if importlib.util.find_spec("torchao") is None:
        return "quantized pipeline weight construction requires TorchAO"
    if device.type != "cuda" or not torch.cuda.is_available():
        return "requires native projection and attention backends on a supported accelerator"
    if case.dtype not in ("float16", "bfloat16"):
        return "pipeline projections require FP16 or BF16"
    if case.head_dim not in (64, 128) or case.width % _GROUP_SIZE:
        return "requires D64/D128 and hidden width divisible by the rotation group size 64"
    if case.rotary_dim % 2 or not 0 < case.rotary_dim <= case.head_dim:
        return "rotary dimension must be positive, even, and at most head dimension"
    if case.keep_ratio is not None and (case.sequence < 64 or case.sequence // 64 > 65536):
        return "sparse pipeline requires 1 through 65536 complete K64 prefix blocks"
    from piper_kernels.fusions.convrot_int8_piper import _backend as dense_backend
    from piper_kernels.fusions.convrot_int8_sparse_piper import _backend as sparse_backend
    from piper_kernels.linear.convrot.int8 import _backend as linear_backend

    probe = torch.empty(0, device=device)
    backend = dense_backend if case.keep_ratio is None else sparse_backend
    if backend.select_projection_backend(probe, head_dim=case.head_dim) is None:
        return "native projection backend is unavailable for this device/head dimension"
    if case.keep_ratio is not None:
        if sparse_backend.select_output_backend(probe) is None:
            return "native sparse attention/output backend is unavailable"
    elif (
        dense_attention_backend.select_backend(AcceleratorTarget.from_device(device)) is None
        or linear_backend.select_linear_backend(probe) is None
    ):
        return "native dense attention/output backend is unavailable"
    return None


@torch.no_grad()
def _build(case: PipelineCase, device: torch.device, *, fused: bool) -> Operation:
    model = Pipeline(case, device).eval()
    hidden = normal_tensor(
        (case.batch, case.sequence, case.width),
        dtype=model.dtype,
        device=device,
        seed=case.seed,
    )
    # Compute the rotary constants once on CPU so cross-vendor inputs agree exactly.
    angles = normal_tensor(
        (case.sequence, case.rotary_dim),
        dtype=torch.float32,
        device=torch.device("cpu"),
        seed=case.seed + 7,
    )
    cos, sin = angles.cos().to(device), angles.sin().to(device)
    rows = sample_indices(case.sequence, device=device)
    reference = model.materialized(hidden, cos, sin).index_select(1, rows).cpu()
    configuration: dict[str, Any] = {
        "weight_format": "convrot_int8",
        "group_size": _GROUP_SIZE,
        "input_source": "cpu_seeded_synthetic_activations_and_packed_weights",
        "scope": "input preparation + QKV + RMSNorm/RoPE + attention + output projection",
        "routing": "minmax" if case.keep_ratio is not None else None,
        "keep_ratio": case.keep_ratio,
        "is_causal": False,
        "rotary_dim": case.rotary_dim,
        "compile_time_included": False,
        "compiled": fused,
        "materialized_projection_arithmetic": "same FP32 boundaries as fused projections",
        "quality_rows": rows.cpu().tolist(),
    }
    capture: CaptureFusion | None = None
    if fused:
        from piper_kernels.fusions.convrot_int8_piper import convrot_int8_piper_compile_options
        from piper_kernels.fusions.convrot_int8_sparse_piper import (
            convrot_int8_sparse_piper_compile_options,
        )

        capture = CaptureFusion(sparse=case.keep_ratio is not None)
        options = (
            convrot_int8_piper_compile_options(fuse_output=True)
            if case.keep_ratio is None
            else convrot_int8_sparse_piper_compile_options()
        )
        passes = options["post_grad_custom_pre_pass"]
        assert isinstance(passes, tuple)
        options["post_grad_custom_pre_pass"] = (*passes, capture)
        # Compiler options contain graph-pass objects outside Torch's public type annotation.
        compiled = torch.compile(model, fullgraph=True, options=cast(Any, options))
        run = partial(compiled, hidden, cos, sin)
        run()
        capture.check()
        configuration.update(
            graph_targets=capture.targets,
            compile_calls=capture.calls,
            query_chunk_rows=capture.query_chunk_rows,
        )
    else:
        run = partial(model.materialized, hidden, cos, sin)

    def check(actual: torch.Tensor) -> QualityCheck:
        if capture is not None:
            capture.check()
        assert actual.shape == (case.batch, case.sequence, case.width)
        assert actual.dtype == model.dtype
        return QualityCheck(
            metrics=measure_quality(actual.index_select(1, rows).cpu(), reference),
            reference="materialized ConvRot INT8 pipeline with identical projection arithmetic",
            sample_count=reference.numel(),
            total_count=actual.numel(),
            relative_l2_limit=0.015,
        )

    return Operation(run=run, check=check, configuration=configuration)


def implementations(case: PipelineCase, device: torch.device) -> list[Implementation]:
    """List native providers without constructing their resident inputs or weights."""
    reason = _unsupported_reason(case, device)
    fused_reason = reason
    if reason is None and case.keep_ratio is not None and case.heads != case.kv_heads:
        fused_reason = "sparse projection/output compiler fusion requires equal Q/KV heads"
    return [
        Implementation(
            "convrot_int8_materialized", partial(_build, case, device, fused=False), reason
        ),
        Implementation(
            "convrot_int8_fused", partial(_build, case, device, fused=True), fused_reason
        ),
    ]
