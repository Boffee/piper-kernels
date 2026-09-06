"""Tests for automatic ConvRot INT8-to-sparse-Piper graph fusion."""

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import (
    SparsePiperAttention,
    sparse_piper_coarse_residual,
)
from piper_kernels.attention.sparse_piper_attention._quantized_dispatch import (
    _sparse_piper_attention_from_quantized_op,
    _sparse_piper_attention_with_coarse_residual_from_quantized_op,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import (
    convrot_int8_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_int8_sparse_piper import key as fused_key
from piper_kernels.fusions.convrot_int8_sparse_piper import query as fused_query
from piper_kernels.fusions.convrot_int8_sparse_piper import value as fused_value
from piper_kernels.fusions.convrot_int8_sparse_piper._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_compile_options
from piper_kernels.linear.convrot.int8 import _ops as int8_ops
from piper_kernels.linear.convrot.int8._compile import compile_pass as convrot_int8_compile_pass
from piper_kernels.linear.convrot.int8._nvidia import triton as int8_nvidia

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"


class _SparseProjectionAttention(torch.nn.Module):
    """Small canonical H3 attention region used to exercise the fusion pass."""

    batch = 1
    sequence_length = 192
    sparse_key_blocks = 2
    input_features = 256
    heads = 2
    head_dim = 128
    rotary_dim = 96

    def __init__(
        self,
        *,
        value_bias: bool = False,
        strided_rope: bool = False,
        routing: str = "minmax",
    ) -> None:
        super().__init__()
        projections = []
        for bias in (False, False, value_bias):
            weight = ConvRotInt8Tensor.from_quantized(
                torch.randint(
                    -127,
                    128,
                    (self.heads * self.head_dim, self.input_features),
                    device="cuda",
                    dtype=torch.int8,
                ),
                torch.rand(
                    self.heads * self.head_dim,
                    1,
                    device="cuda",
                    dtype=torch.float32,
                )
                .mul_(0.01)
                .add_(0.001),
                group_size=self.input_features,
            )
            projection = torch.nn.Linear(
                self.input_features,
                self.heads * self.head_dim,
                bias=bias,
                device="cuda",
                dtype=torch.bfloat16,
            )
            projection.weight = torch.nn.Parameter(weight, requires_grad=False)
            if projection.bias is not None:
                projection.bias.requires_grad_(False)
            projections.append(projection)
        self.query, self.key, self.value = projections
        self.query_norm = torch.nn.Parameter(
            torch.rand(self.head_dim, device="cuda", dtype=torch.float32).add_(0.5).bfloat16(),
            requires_grad=False,
        )
        self.key_norm = torch.nn.Parameter(
            torch.rand(self.head_dim, device="cuda", dtype=torch.float32).add_(0.5).bfloat16(),
            requires_grad=False,
        )
        angles = torch.rand(
            self.sequence_length,
            self.rotary_dim,
            device="cuda",
            dtype=torch.float32,
        ).mul_(2 * torch.pi)
        cos = angles.cos()
        sin = angles.sin()
        if strided_rope:
            cos = torch.stack((cos, cos), dim=-1)[..., 0]
            sin = torch.stack((sin, sin), dim=-1)[..., 0]
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.sparse_attention = SparsePiperAttention((0.5, 1.0), routing=routing)

    def _norm_rope(self, projected: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        normalized = F.rms_norm(
            projected.view(
                self.batch,
                self.sequence_length,
                self.heads,
                self.head_dim,
            ),
            (self.head_dim,),
            norm,
            1e-5,
        )
        rotary = normalized[..., : self.rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        cos = self.cos.to(torch.bfloat16)[None, :, None, :]
        sin = self.sin.to(torch.bfloat16)[None, :, None, :]
        rotary = rotary * cos + rotated * sin
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1).contiguous()

    def _project_inputs(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self._norm_rope(self.query(hidden_states), self.query_norm)
        key = self._norm_rope(self.key(hidden_states), self.key_norm)
        value = self.value(hidden_states).view(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
    ) -> torch.Tensor:
        query, key, value = self._project_inputs(hidden_states)
        return self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=self.sparse_key_blocks,
            block_lengths=block_lengths,
            sparse_query_blocks=sparse_query_blocks,
        )


class _CoarseSparseProjectionAttention(_SparseProjectionAttention):
    coarse_scale = 0.125
    coarse_key_blocks = 3

    def __init__(
        self,
        *,
        sparse_routing: str,
        coarse_routing: str | None = None,
    ) -> None:
        super().__init__(routing=sparse_routing)
        self.coarse_routing = sparse_routing if coarse_routing is None else coarse_routing

    def forward(
        self,
        hidden_states: torch.Tensor,
        coarse_gate: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
    ) -> torch.Tensor:
        query, key, value = self._project_inputs(hidden_states)
        fine_output = self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=self.sparse_key_blocks,
            block_lengths=block_lengths,
            sparse_query_blocks=sparse_query_blocks,
        )
        coarse_output = sparse_piper_coarse_residual(
            query,
            key,
            value,
            coarse_gate,
            routing=self.coarse_routing,
            coarse_key_blocks=self.coarse_key_blocks,
            coarse_scale=self.coarse_scale,
            block_lengths=block_lengths,
        )
        return fine_output + coarse_output


class _TargetCapturePass(CustomInferenceAwareGraphPass):
    def __init__(self) -> None:
        self.targets: list[object] = []
        self.calls = 0
        self._uuid = uuid.uuid4().bytes

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    def uuid(self) -> bytes:
        return self._uuid


class _DynamicSparseProjectionAttention(_SparseProjectionAttention):
    def _dynamic_norm_rope(
        self,
        projected: torch.Tensor,
        norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _features = projected.shape
        normalized = F.rms_norm(
            projected.view(batch, sequence, self.heads, self.head_dim),
            (self.head_dim,),
            norm,
            1e-5,
        )
        rotary_dim = cos.shape[1]
        rotary = normalized[..., :rotary_dim]
        first, second = rotary.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        rotary = rotary * cos.to(torch.bfloat16)[None, :, None, :]
        rotary = rotary + rotated * sin.to(torch.bfloat16)[None, :, None, :]
        return torch.cat((rotary, normalized[..., rotary_dim:]), dim=-1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sparse_key_blocks: int,
    ) -> torch.Tensor:
        batch, sequence, _features = hidden_states.shape
        query = self._dynamic_norm_rope(
            self.query(hidden_states),
            self.query_norm,
            cos,
            sin,
        )
        key = self._dynamic_norm_rope(
            self.key(hidden_states),
            self.key_norm,
            cos,
            sin,
        )
        value = self.value(hidden_states).view(batch, sequence, self.heads, self.head_dim)
        return self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
        )


class _DynamicCoarseSparseProjectionAttention(_DynamicSparseProjectionAttention):
    coarse_scale = 0.125

    def __init__(self) -> None:
        super().__init__()
        self.sparse_attention = SparsePiperAttention((0.5, 1.0), routing="mean")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sparse_key_blocks: int,
        coarse_gate: torch.Tensor,
        coarse_key_blocks: int | None = None,
    ) -> torch.Tensor:
        batch, sequence, _features = hidden_states.shape
        query = self._dynamic_norm_rope(
            self.query(hidden_states),
            self.query_norm,
            cos,
            sin,
        )
        key = self._dynamic_norm_rope(
            self.key(hidden_states),
            self.key_norm,
            cos,
            sin,
        )
        value = self.value(hidden_states).view(batch, sequence, self.heads, self.head_dim)
        fine_output = self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=sparse_key_blocks,
        )
        coarse_output = sparse_piper_coarse_residual(
            query,
            key,
            value,
            coarse_gate,
            routing="mean",
            coarse_key_blocks=(
                (sequence + 63) // 64 if coarse_key_blocks is None else coarse_key_blocks
            ),
            coarse_scale=self.coarse_scale,
        )
        return fine_output + coarse_output


def _make_output_projection(
    input_features: int,
    output_features: int,
    *,
    bias: bool,
    bias_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Linear:
    projection = torch.nn.Linear(
        input_features,
        output_features,
        bias=bias,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = ConvRotInt8Tensor.from_quantized(
        torch.randint(
            -127,
            128,
            (output_features, input_features),
            device="cuda",
            dtype=torch.int8,
        ),
        torch.rand(
            output_features,
            1,
            device="cuda",
            dtype=torch.float32,
        )
        .mul_(0.01)
        .add_(0.001),
        group_size=input_features,
    )
    projection.weight = torch.nn.Parameter(weight, requires_grad=False)
    if projection.bias is not None:
        projection.bias = torch.nn.Parameter(
            projection.bias.to(bias_dtype),
            requires_grad=False,
        )
    return projection


class _SparseProjectionAttentionOutput(_SparseProjectionAttention):
    output_features = 320

    def __init__(
        self,
        *,
        escape_attention: bool = False,
        bias_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.output = _make_output_projection(
            self.heads * self.head_dim,
            self.output_features,
            bias=True,
            bias_dtype=bias_dtype,
        )
        self.escape_attention = escape_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attention = super().forward(hidden_states, block_lengths, sparse_query_blocks)
        projected = self.output(
            attention.reshape(
                self.batch,
                self.sequence_length,
                self.heads * self.head_dim,
            )
        )
        return (projected, attention) if self.escape_attention else projected


class _MeanPoolSparseProjectionAttentionOutput(_SparseProjectionAttentionOutput):
    def __init__(self) -> None:
        super().__init__()
        self.sparse_attention = SparsePiperAttention((0.5, 1.0), routing="mean")


class _CoarseSparseProjectionAttentionOutput(_CoarseSparseProjectionAttention):
    output_features = 320

    def __init__(self, *, routing: str) -> None:
        super().__init__(sparse_routing=routing)
        self.output = _make_output_projection(
            self.heads * self.head_dim,
            self.output_features,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        coarse_gate: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
    ) -> torch.Tensor:
        attention = super().forward(
            hidden_states,
            coarse_gate,
            block_lengths,
            sparse_query_blocks,
        )
        return self.output(
            attention.reshape(
                self.batch,
                self.sequence_length,
                self.heads * self.head_dim,
            )
        )


class _ProjectedGateCoarseSparseAttentionOutput(_CoarseSparseProjectionAttentionOutput):
    """Canonical VSA region whose coarse gate is another ConvRot INT8 linear."""

    def __init__(self, *, routing: str) -> None:
        super().__init__(routing=routing)
        self.gate = _make_output_projection(
            self.input_features,
            self.heads * self.head_dim,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
        sparse_query_blocks: int | None = None,
    ) -> torch.Tensor:
        coarse_gate = self.gate(hidden_states).reshape(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
        attention = _CoarseSparseProjectionAttention.forward(
            self,
            hidden_states,
            coarse_gate,
            block_lengths,
            sparse_query_blocks,
        )
        return self.output(attention.flatten(2))


class _DynamicSparseProjectionAttentionOutput(_DynamicSparseProjectionAttention):
    output_features = 320

    def __init__(self) -> None:
        super().__init__()
        self.output = _make_output_projection(
            self.heads * self.head_dim,
            self.output_features,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        sparse_key_blocks: int,
    ) -> torch.Tensor:
        attention = super().forward(hidden_states, cos, sin, sparse_key_blocks)
        return self.output(
            attention.reshape(
                hidden_states.shape[0],
                hidden_states.shape[1],
                self.heads * self.head_dim,
            )
        )


def _run_explicit_fused_projection(
    model: _DynamicSparseProjectionAttention | _SparseProjectionAttention,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sparse_key_blocks: int,
    *,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> torch.Tensor:
    input_qdata, input_scale = int8_ops.prepare_input(
        hidden_states,
        model.query.weight.group_size,
    )
    routing_mode = model.sparse_attention._routing_mode
    query = fused_query._project_query_op(
        input_qdata,
        input_scale,
        model.query.weight.qdata,
        model.query.weight.scale,
        model.query_norm,
        cos,
        sin,
        1e-5,
        model.head_dim**-0.5,
        routing_mode,
        block_lengths,
    )
    key = fused_key._project_key_op(
        input_qdata,
        input_scale,
        model.key.weight.qdata,
        model.key.weight.scale,
        model.key_norm,
        cos,
        sin,
        1e-5,
        routing_mode,
        block_lengths,
    )
    input_mean = int8_ops.dequantized_input_mean(
        input_qdata,
        input_scale,
        block_lengths,
    )
    projection_arguments = (
        input_qdata,
        input_scale,
        input_mean,
        model.value.weight.qdata,
        model.value.weight.scale,
    )
    if coarse_gate is None:
        value = fused_value._project_value_op(*projection_arguments, block_lengths)
        return _sparse_piper_attention_from_quantized_op(
            *query,
            *key,
            *value,
            list(model.sparse_attention._head_keep_ratio_units),
            sparse_key_blocks,
            hidden_states.shape[1],
            routing_mode,
            block_lengths,
            sparse_query_blocks,
        )
    assert coarse_scale is not None
    assert coarse_key_blocks is not None
    value = fused_value._project_value_with_block_means_op(
        *projection_arguments,
        block_lengths,
    )
    return _sparse_piper_attention_with_coarse_residual_from_quantized_op(
        *query,
        *key,
        *value,
        coarse_gate,
        list(model.sparse_attention._head_keep_ratio_units),
        sparse_key_blocks,
        hidden_states.shape[1],
        routing_mode,
        coarse_scale,
        block_lengths,
        coarse_key_blocks,
        sparse_query_blocks,
    )


def _run_explicit_attention_output(
    model: _DynamicSparseProjectionAttentionOutput | _SparseProjectionAttentionOutput,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sparse_key_blocks: int,
    *,
    coarse_gate: torch.Tensor | None = None,
    coarse_scale: float | None = None,
    coarse_key_blocks: int | None = None,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    attention = _run_explicit_fused_projection(
        model,
        hidden_states,
        cos,
        sin,
        sparse_key_blocks,
        coarse_gate=coarse_gate,
        coarse_scale=coarse_scale,
        coarse_key_blocks=coarse_key_blocks,
        block_lengths=block_lengths,
        sparse_query_blocks=sparse_query_blocks,
    )
    projected = int8_nvidia.run_linear(
        attention.reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            model.heads * model.head_dim,
        ),
        model.output.weight.qdata,
        model.output.weight.scale,
        model.output.bias,
        model.output.weight.group_size,
    )
    return projected, attention


def test_compile_options_install_fusion_before_convrot() -> None:
    options = convrot_int8_sparse_piper_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (fusion_compile_pass, convrot_int8_compile_pass)


def test_compile_options_reorder_existing_convrot_pass_without_duplication() -> None:
    base_options = convrot_int8_compile_options()
    options = convrot_int8_sparse_piper_compile_options(base_options)
    reapplied = convrot_int8_sparse_piper_compile_options(options)

    assert options[_POST_GRAD_PRE_PASS] == (fusion_compile_pass, convrot_int8_compile_pass)
    assert reapplied == options


def test_compile_options_preserve_unrelated_pass_order() -> None:
    before_convrot = object()
    after_convrot = object()
    options = convrot_int8_sparse_piper_compile_options(
        {
            _POST_GRAD_PRE_PASS: (
                before_convrot,
                convrot_int8_compile_pass,
                after_convrot,
            )
        }
    )

    assert options[_POST_GRAD_PRE_PASS] == (
        before_convrot,
        fusion_compile_pass,
        convrot_int8_compile_pass,
        after_convrot,
    )
    assert convrot_int8_sparse_piper_compile_options(options) == options


def test_fusion_compiler_pass_uuid_is_versioned_and_stable() -> None:
    first = fusion_compile_pass.uuid()
    second = fusion_compile_pass.uuid()

    assert isinstance(first, bytes)
    assert first
    assert first == second


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_compile_options_fuse_sparse_piper_projection_region() -> None:
    torch.manual_seed(701)
    model = _SparseProjectionAttention().eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True)(hidden_states)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states)

    difference = actual.float() - expected.float()
    relative_l2 = difference.norm() / expected.float().norm()
    assert relative_l2 < 0.025
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 1
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default
        )
        == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_sparse_piper_project_key.default)
        == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_dequantized_input_mean.default)
        == 1
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_value.default
        )
        == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_cuda_projection_fusion_respects_internal_block_lengths(routing: str) -> None:
    torch.manual_seed(703)
    model = _SparseProjectionAttention(routing=routing).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        expected = _run_explicit_fused_projection(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
            block_lengths=block_lengths,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states, block_lengths)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_cuda_compile_options_fuse_sparse_piper_coarse_residual(routing: str) -> None:
    torch.manual_seed(713)
    model = _CoarseSparseProjectionAttention(sparse_routing=routing).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = torch.randn(
        model.batch,
        model.sequence_length,
        model.heads,
        model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        semantic = model(hidden_states, coarse_gate)
        expected = _run_explicit_fused_projection(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
            coarse_gate=coarse_gate,
            coarse_scale=model.coarse_scale,
            coarse_key_blocks=model.coarse_key_blocks,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states, coarse_gate)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    relative_l2 = (actual.float() - semantic.float()).norm() / semantic.float().norm()
    assert relative_l2 < 0.025
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_project_value_with_block_means.default
        )
        == 1
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        )
        == 1
    )
    absent_targets = (
        torch.ops.piper_kernels.convrot_int8_linear.default,
        torch.ops.piper_kernels.convrot_int8_sparse_piper_project_value.default,
        torch.ops.piper_kernels.sparse_piper_attention.default,
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default,
        torch.ops.piper_kernels.sparse_piper_coarse_residual.default,
    )
    assert all(target not in capture.targets for target in absent_targets)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_cuda_coarse_projection_fusion_respects_internal_block_lengths(routing: str) -> None:
    torch.manual_seed(715)
    model = _CoarseSparseProjectionAttention(sparse_routing=routing).eval()
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    valid_rows = torch.arange(model.sequence_length, device="cuda") % 64
    valid_rows = valid_rows < block_lengths.repeat_interleave(64)
    hidden_states[:, ~valid_rows] = torch.randn_like(hidden_states[:, ~valid_rows]).mul_(100)
    coarse_gate = torch.randn(
        model.batch,
        model.sequence_length,
        model.heads,
        model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        semantic = model(hidden_states, coarse_gate, block_lengths)
        expected = _run_explicit_fused_projection(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
            coarse_gate=coarse_gate,
            coarse_scale=model.coarse_scale,
            coarse_key_blocks=model.coarse_key_blocks,
            block_lengths=block_lengths,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states, coarse_gate, block_lengths)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    relative_l2 = (
        actual[:, valid_rows].float() - semantic[:, valid_rows].float()
    ).norm() / semantic[:, valid_rows].float().norm()
    assert relative_l2 < 0.025
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_coarse_residual.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_padded_coarse_fusion_reuses_graph_for_changed_block_lengths() -> None:
    torch.manual_seed(716)
    model = _CoarseSparseProjectionAttention(sparse_routing="mean").eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = torch.randn(
        model.batch,
        model.sequence_length,
        model.heads,
        model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    compiled = torch.compile(model, fullgraph=True, options=options)

    with torch.no_grad():
        for lengths in ([64, 17, 51], [33, 64, 11]):
            block_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
            expected = _run_explicit_fused_projection(
                model,
                hidden_states,
                model.cos,
                model.sin,
                model.sparse_key_blocks,
                coarse_gate=coarse_gate,
                coarse_scale=model.coarse_scale,
                coarse_key_blocks=model.coarse_key_blocks,
                block_lengths=block_lengths,
            )
            actual = compiled(hidden_states, coarse_gate, block_lengths)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    assert capture.calls == 1


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_coarse_residual_fusion_fails_closed_for_mismatched_routing() -> None:
    torch.manual_seed(717)
    model = _CoarseSparseProjectionAttention(
        sparse_routing="minmax",
        coarse_routing="mean",
    ).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = torch.randn(
        model.batch,
        model.sequence_length,
        model.heads,
        model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True)(hidden_states, coarse_gate)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states, coarse_gate)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.sparse_piper_coarse_residual.default) == 1


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_compile_options_fuse_attention_output_boundary() -> None:
    torch.manual_seed(719)
    model = _SparseProjectionAttentionOutput(bias_dtype=torch.float32).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        expected, _attention = _run_explicit_attention_output(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in capture.targets
    )
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_compile_fuses_padded_mixed_query_attention_output() -> None:
    torch.manual_seed(722)
    model = _SparseProjectionAttentionOutput().eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        expected, _attention = _run_explicit_attention_output(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
            block_lengths=block_lengths,
            sparse_query_blocks=2,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states, block_lengths, 2)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in capture.targets
    )
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_compile_options_fuse_mean_pool_attention_and_output() -> None:
    torch.manual_seed(720)
    model = _MeanPoolSparseProjectionAttentionOutput().eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        expected, _attention = _run_explicit_attention_output(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert (
        torch.ops.piper_kernels.convrot_int8_sparse_piper_project_query.default
        not in capture.targets
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_sparse_piper_project_key.default)
        == 1
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in capture.targets
    )
    assert torch.ops.piper_kernels.convrot_int8_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_cuda_compile_fuses_every_bounded_attention_feature(routing: str) -> None:
    torch.manual_seed(721)
    model = _CoarseSparseProjectionAttentionOutput(routing=routing).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = torch.randn(
        model.batch,
        model.sequence_length,
        model.heads,
        model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True, options=options)

    with torch.no_grad():
        for lengths in ([64, 17, 51], [33, 64, 11]):
            block_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
            expected, _attention = _run_explicit_attention_output(
                model,
                hidden_states,
                model.cos,
                model.sin,
                model.sparse_key_blocks,
                coarse_gate=coarse_gate,
                coarse_scale=model.coarse_scale,
                coarse_key_blocks=model.coarse_key_blocks,
                block_lengths=block_lengths,
                sparse_query_blocks=2,
            )
            actual = compiled(hidden_states, coarse_gate, block_lengths, 2)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    assert capture.calls == 1
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    absent_targets = (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
        torch.ops.piper_kernels.convrot_int8_linear.default,
    )
    assert all(target not in capture.targets for target in absent_targets)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize("routing", ["mean", "minmax"])
def test_cuda_compile_lifetime_chunks_a_projected_coarse_gate(routing: str) -> None:
    torch.manual_seed(725)
    model = _ProjectedGateCoarseSparseAttentionOutput(routing=routing).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = model.gate(hidden_states).reshape(
        model.batch,
        model.sequence_length,
        model.heads,
        model.head_dim,
    )
    expected, _attention = _run_explicit_attention_output(
        model,
        hidden_states,
        model.cos,
        model.sin,
        model.sparse_key_blocks,
        coarse_gate=coarse_gate,
        coarse_scale=model.coarse_scale,
        coarse_key_blocks=model.coarse_key_blocks,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_prepare_input.default) == 1
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    absent_targets = (
        torch.ops.piper_kernels.convrot_int8_linear.default,
        torch.ops.piper_kernels.convrot_int8_linear_prepared.default,
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default,
    )
    assert all(target not in capture.targets for target in absent_targets)


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_attention_output_fusion_fails_closed_when_attention_escapes() -> None:
    torch.manual_seed(727)
    model = _SparseProjectionAttentionOutput(escape_attention=True).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        expected_projected, expected_attention = _run_explicit_attention_output(
            model,
            hidden_states,
            model.cos,
            model.sin,
            model.sparse_key_blocks,
        )
        torch._dynamo.reset()
        actual_projected, actual_attention = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states)

    torch.testing.assert_close(actual_projected, expected_projected, atol=0, rtol=0)
    torch.testing.assert_close(actual_attention, expected_attention, atol=0, rtol=0)
    assert (
        torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        not in capture.targets
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_fused_projection_reuses_one_dynamic_shape_route_capacity_graph() -> None:
    torch.manual_seed(709)
    model = _DynamicSparseProjectionAttention().eval()
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    compiled = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        options=options,
    )

    with torch.no_grad():
        cases = (
            (192, 2),
            (193, 2),
            (256, 2),
            (256, 3),
            (257, 3),
            (1024, 8),
            (1024, 9),
        )
        for sequence, sparse_key_blocks in cases:
            hidden_states = torch.randn(
                model.batch,
                sequence,
                model.input_features,
                device="cuda",
                dtype=torch.bfloat16,
            )
            angles = torch.rand(
                sequence,
                model.rotary_dim,
                device="cuda",
                dtype=torch.float32,
            ).mul_(2 * torch.pi)
            cos = angles.cos().contiguous()
            sin = angles.sin().contiguous()
            expected = _run_explicit_fused_projection(
                model,
                hidden_states,
                cos,
                sin,
                sparse_key_blocks,
            )
            output = compiled(
                hidden_states,
                cos,
                sin,
                sparse_key_blocks,
            )
            assert output.shape == (model.batch, sequence, model.heads, model.head_dim)
            assert bool(torch.isfinite(output).all())
            torch.testing.assert_close(output, expected, atol=0, rtol=0)

    assert capture.calls == 1
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_fused_coarse_projection_reuses_one_dynamic_shape_graph() -> None:
    torch.manual_seed(711)
    model = _DynamicCoarseSparseProjectionAttention().eval()
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    compiled = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        options=options,
    )

    with torch.no_grad():
        for sequence, sparse_key_blocks in ((193, 2), (256, 3), (257, 3)):
            hidden_states = torch.randn(
                model.batch,
                sequence,
                model.input_features,
                device="cuda",
                dtype=torch.bfloat16,
            )
            coarse_gate = torch.randn(
                model.batch,
                sequence,
                model.heads,
                model.head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            angles = torch.rand(
                sequence,
                model.rotary_dim,
                device="cuda",
                dtype=torch.float32,
            ).mul_(2 * torch.pi)
            cos = angles.cos().contiguous()
            sin = angles.sin().contiguous()
            expected = _run_explicit_fused_projection(
                model,
                hidden_states,
                cos,
                sin,
                sparse_key_blocks,
                coarse_gate=coarse_gate,
                coarse_scale=model.coarse_scale,
                coarse_key_blocks=(sequence + 63) // 64,
            )
            actual = compiled(
                hidden_states,
                cos,
                sin,
                sparse_key_blocks,
                coarse_gate,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    assert capture.calls == 1
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        )
        == 1
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_dynamic_coarse_scope_recompiles_without_invalid_fusion() -> None:
    torch.manual_seed(712)
    model = _DynamicCoarseSparseProjectionAttention().eval()
    sequence = 192
    hidden_states = torch.randn(
        model.batch,
        sequence,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = torch.randn(
        model.batch,
        sequence,
        model.heads,
        model.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    angles = torch.rand(
        sequence,
        model.rotary_dim,
        device="cuda",
        dtype=torch.float32,
    ).mul_(2 * torch.pi)
    cos = angles.cos().contiguous()
    sin = angles.sin().contiguous()
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)

    with torch.no_grad():
        expected_narrow = model(
            hidden_states,
            cos,
            sin,
            3,
            coarse_gate,
            2,
        )
        torch._dynamo.reset()
        compiled = torch.compile(
            model,
            dynamic=True,
            fullgraph=True,
            options=options,
        )
        wide = compiled(hidden_states, cos, sin, 2, coarse_gate, 3)
        assert bool(torch.isfinite(wide).all())
        assert capture.calls == 1
        assert (
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
            in capture.targets
        )

        narrow = compiled(hidden_states, cos, sin, 3, coarse_gate, 2)

    relative_l2 = (narrow.float() - expected_narrow.float()).norm() / expected_narrow.float().norm()
    assert relative_l2 < 0.025
    assert capture.calls == 2
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        not in capture.targets
    )
    assert torch.ops.piper_kernels.sparse_piper_attention.default in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_coarse_residual.default in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
def test_cuda_attention_output_fusion_reuses_one_dynamic_shape_graph() -> None:
    torch.manual_seed(733)
    model = _DynamicSparseProjectionAttentionOutput().eval()
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    compiled = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        options=options,
    )

    with torch.no_grad():
        for sequence, sparse_key_blocks in ((193, 2), (256, 3), (257, 3)):
            hidden_states = torch.randn(
                model.batch,
                sequence,
                model.input_features,
                device="cuda",
                dtype=torch.bfloat16,
            )
            angles = torch.rand(
                sequence,
                model.rotary_dim,
                device="cuda",
                dtype=torch.float32,
            ).mul_(2 * torch.pi)
            cos = angles.cos().contiguous()
            sin = angles.sin().contiguous()
            expected, _attention = _run_explicit_attention_output(
                model,
                hidden_states,
                cos,
                sin,
                sparse_key_blocks,
            )
            actual = compiled(hidden_states, cos, sin, sparse_key_blocks)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    assert capture.calls == 1
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.convrot_int8_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires exact NVIDIA SM120",
)
@pytest.mark.parametrize(
    "model_options",
    [{"value_bias": True}, {"strided_rope": True}],
    ids=("projection-bias", "strided-rope"),
)
def test_cuda_sparse_piper_projection_fails_closed_for_incompatible_operands(
    model_options: dict[str, bool],
) -> None:
    torch.manual_seed(703)
    model = _SparseProjectionAttention(**model_options).eval()
    hidden_states = torch.randn(
        model.batch,
        model.sequence_length,
        model.input_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = _TargetCapturePass()
    options = convrot_int8_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True)(hidden_states)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=options,
        )(hidden_states)

    assert torch.equal(actual, expected)
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in capture.targets
    )
    assert capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention.default) == 1
    assert capture.targets.count(torch.ops.piper_kernels.convrot_int8_linear_prepared.default) == 3
