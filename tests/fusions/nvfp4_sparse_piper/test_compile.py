"""Tests for automatic NVFP4 and ConvRot NVFP4 sparse-Piper graph fusion."""

from __future__ import annotations

from typing import TypedDict, Unpack

import pytest
import torch
from _accelerators import exact_sm120_available, mock_cuda_target
from _compile_capture import TargetCapturePass
from torch.nn import functional as F  # noqa: N812
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor as TorchAONVFP4Tensor,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    QuantizeTensorToNVFP4Kwargs,
    per_tensor_amax_to_scale,
)

from piper_kernels import (
    SparsePiperAttention,
    compile_options,
    sparse_piper_coarse_residual,
)
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.nvfp4_sparse_piper import _output
from piper_kernels.fusions.nvfp4_sparse_piper import key as fused_key
from piper_kernels.fusions.nvfp4_sparse_piper import query as fused_query
from piper_kernels.fusions.nvfp4_sparse_piper import value as fused_value
from piper_kernels.fusions.nvfp4_sparse_piper._compile import fold_sparse_attention
from piper_kernels.fusions.nvfp4_sparse_piper._output_compile import (
    _fold_attention_output,
)
from piper_kernels.linear.convrot.nvfp4 import _ops as convrot_nvfp4_ops
from piper_kernels.linear.convrot.nvfp4._compile import fold_linear as fold_convrot_nvfp4_linear
from piper_kernels.linear.nvfp4 import _ops as nvfp4_ops
from piper_kernels.linear.nvfp4._compile import fold_linear as fold_nvfp4_linear
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.nvfp4 import ConvRotNVFP4Tensor
from piper_kernels.weights.nvfp4 import PiperNVFP4Tensor

from .._accuracy import assert_fusion_output_close
from .._nvfp4_sparse_piper_graphs import (
    build_attention_graph,
    build_mixed_quantized_attention_output_graph,
    build_quantized_attention_output_graph,
    run_nvfp4_passes,
)

_POST_GRAD_PRE_PASS = "post_grad_custom_pre_pass"
_GROUP_SIZES = [pytest.param(None, id="nvfp4"), pytest.param(64, id="convrot64")]


def _convert_weight(weight: torch.Tensor, group_size: int) -> ConvRotNVFP4Tensor:
    assert isinstance(weight, TorchAONVFP4Tensor)
    dense = weight.dequantize(torch.bfloat16)
    rotated = rotate_groups(dense, group_size)
    converted = TorchAONVFP4Tensor.to_nvfp4(
        rotated,
        per_tensor_scale=per_tensor_amax_to_scale(rotated.abs().amax()),
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=True,
        act_quant_kwargs=weight.act_quant_kwargs,
    )
    return ConvRotNVFP4Tensor.from_torchao(converted, group_size=group_size)


class _SparseAttentionConfig(TypedDict, total=False):
    dynamic: bool
    routing: str
    group_size: int | None
    head_dim: int
    sequence_length: int
    sparse_key_blocks: int


class _SparseOutputConfig(_SparseAttentionConfig, total=False):
    output_dynamic: bool
    output_features: int


class _SparseProjectionAttention(torch.nn.Module):
    """Small canonical H3 attention region used to exercise the fusion pass."""

    sequence_length = 192
    sparse_key_blocks = 2
    input_features = 256
    heads = 2
    head_dim = 128
    rotary_dim = 96

    def __init__(
        self,
        *,
        batch: int = 1,
        dynamic: bool = False,
        routing: str = "minmax",
        group_size: int | None = None,
        head_dim: int | None = None,
        sequence_length: int | None = None,
        sparse_key_blocks: int | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.group_size = group_size
        if head_dim is not None:
            self.head_dim = head_dim
            self.rotary_dim = head_dim * 3 // 4
        if sequence_length is not None:
            self.sequence_length = sequence_length
        if sparse_key_blocks is not None:
            self.sparse_key_blocks = sparse_key_blocks
        calibration = torch.full(
            (1, self.sequence_length, self.input_features),
            3.0,
            device="cuda",
            dtype=torch.bfloat16,
        )
        base_scale = per_tensor_amax_to_scale(calibration.abs().amax())
        quantization = QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=dynamic,
        )
        projections = []
        for index in range(3):
            dense = torch.randn(
                (self.heads * self.head_dim, self.input_features),
                device="cuda",
                dtype=torch.bfloat16,
            )
            torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
                dense,
                per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
                act_per_tensor_scale=(None if dynamic else base_scale * (1.0 + index * 0.01)),
                is_swizzled_scales=True,
                act_quant_kwargs=quantization,
            )
            projection = torch.nn.Linear(
                self.input_features,
                self.heads * self.head_dim,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
            weight: torch.Tensor = PiperNVFP4Tensor.from_torchao(torchao_weight)
            if group_size is not None:
                weight = _convert_weight(weight, group_size)
            projection.weight = torch.nn.Parameter(weight, requires_grad=False)
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
            (self.sequence_length, self.rotary_dim),
            device="cuda",
            dtype=torch.float32,
        ).mul_(2 * torch.pi)
        self.register_buffer("cos", angles.cos().contiguous())
        self.register_buffer("sin", angles.sin().contiguous())
        self.sparse_attention = SparsePiperAttention((0.5, 1.0), routing=routing)

    def set_activation_dtype(self, dtype: torch.dtype) -> None:
        for projection in self.modules():
            if isinstance(projection, torch.nn.Linear):
                projection.weight = torch.nn.Parameter(
                    projection.weight.to(dtype), requires_grad=False
                )
                if projection.bias is not None:
                    projection.bias = torch.nn.Parameter(
                        projection.bias.to(dtype), requires_grad=False
                    )
        for name in ("query_norm", "key_norm"):
            setattr(
                self, name, torch.nn.Parameter(getattr(self, name).to(dtype), requires_grad=False)
            )

    def _norm_rope(self, projected: torch.Tensor, norm: torch.Tensor | None) -> torch.Tensor:
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
        cos = self.cos.to(projected.dtype)[None, :, None, :]
        sin = self.sin.to(projected.dtype)[None, :, None, :]
        rotary = rotary * cos + rotated * sin
        return torch.cat((rotary, normalized[..., self.rotary_dim :]), dim=-1).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        force_tile_ids: torch.Tensor | None = None,
        force_tile_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self._norm_rope(self.query(hidden_states), self.query_norm)
        key = self._norm_rope(self.key(hidden_states), self.key_norm)
        value = self.value(hidden_states).view(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
        return self.sparse_attention(
            query,
            key,
            value,
            sparse_key_blocks=self.sparse_key_blocks,
            force_tile_ids=force_tile_ids,
            force_tile_counts=force_tile_counts,
        )


class _SparseProjectionAttentionOutput(_SparseProjectionAttention):
    """Canonical H3 attention with one NVFP4 or ConvRot NVFP4 output projection."""

    output_features = 320

    def __init__(
        self,
        *,
        output_dynamic: bool = False,
        output_features: int | None = None,
        **config: Unpack[_SparseAttentionConfig],
    ) -> None:
        super().__init__(**config)
        if output_features is not None:
            self.output_features = output_features
        calibration = torch.full(
            (1, self.sequence_length, self.heads * self.head_dim),
            3.0,
            device="cuda",
            dtype=torch.bfloat16,
        )
        activation_scale = per_tensor_amax_to_scale(calibration.abs().amax())
        quantization = QuantizeTensorToNVFP4Kwargs(
            block_size=16,
            is_swizzled_scales=True,
            use_triton_kernel=False,
            use_dynamic_per_tensor_scale=output_dynamic,
        )
        dense = torch.randn(
            (self.output_features, self.heads * self.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
            dense,
            per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
            act_per_tensor_scale=None if output_dynamic else activation_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=quantization,
        )
        self.output = torch.nn.Linear(
            self.heads * self.head_dim,
            self.output_features,
            bias=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight: torch.Tensor = PiperNVFP4Tensor.from_torchao(torchao_weight)
        if self.group_size is not None:
            weight = _convert_weight(weight, self.group_size)
        self.output.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        force_tile_ids: torch.Tensor | None = None,
        force_tile_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention = super().forward(hidden_states, force_tile_ids, force_tile_counts)
        return self.output(attention.flatten(2))


class _BoundedSparseProjectionAttentionOutput(_SparseProjectionAttentionOutput):
    """NVFP4 projection with padding, coarse attention, and a mixed query scope."""

    coarse_scale = 0.125
    coarse_key_blocks = 3

    def __init__(self, **config: Unpack[_SparseOutputConfig]) -> None:
        super().__init__(**config)
        self.routing = config.get("routing", "minmax")

    def forward(
        self,
        hidden_states: torch.Tensor,
        coarse_gate: torch.Tensor,
        block_lengths: torch.Tensor,
        sparse_query_blocks: int,
    ) -> torch.Tensor:
        query = self._norm_rope(self.query(hidden_states), self.query_norm)
        key = self._norm_rope(self.key(hidden_states), self.key_norm)
        value = self.value(hidden_states).view(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
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
            routing=self.routing,
            coarse_key_blocks=self.coarse_key_blocks,
            coarse_scale=self.coarse_scale,
            block_lengths=block_lengths,
        )
        return self.output((fine_output + coarse_output).flatten(2))


class _ProjectedGateCoarseSparseAttentionOutput(_BoundedSparseProjectionAttentionOutput):
    """Bounded coarse attention whose gate is projected from hidden states."""

    def __init__(self, **config: Unpack[_SparseOutputConfig]) -> None:
        super().__init__(**config)
        query_weight = self.query.weight
        assert isinstance(query_weight, TorchAONVFP4Tensor)
        dense = torch.randn(
            (self.heads * self.head_dim, self.input_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        torchao_weight = TorchAONVFP4Tensor.to_nvfp4(
            dense,
            per_tensor_scale=per_tensor_amax_to_scale(dense.abs().amax()),
            act_per_tensor_scale=query_weight.act_per_tensor_scale,
            is_swizzled_scales=True,
            act_quant_kwargs=query_weight.act_quant_kwargs,
        )
        self.gate = torch.nn.Linear(
            self.input_features,
            self.heads * self.head_dim,
            bias=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight: torch.Tensor = PiperNVFP4Tensor.from_torchao(torchao_weight)
        if self.group_size is not None:
            weight = _convert_weight(weight, self.group_size)
        self.gate.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_lengths: torch.Tensor,
        sparse_query_blocks: int,
    ) -> torch.Tensor:
        coarse_gate = self.gate(hidden_states).view(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
        return super().forward(
            hidden_states,
            coarse_gate,
            block_lengths,
            sparse_query_blocks,
        )


def _nvfp4_storage(weight: torch.Tensor) -> tuple[torch.Tensor, ...]:
    assert isinstance(weight, TorchAONVFP4Tensor)
    return weight.qdata, weight.scale, weight.per_tensor_scale


def _prepare_nvfp4_input(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    assert isinstance(weight, TorchAONVFP4Tensor)
    quantization = weight.act_quant_kwargs
    assert quantization is not None
    if isinstance(weight, ConvRotNVFP4Tensor):
        return convrot_nvfp4_ops.prepare_input(
            hidden_states,
            weight.act_per_tensor_scale,
            quantization.use_dynamic_per_tensor_scale,
            weight.group_size,
            None,
            weight.high_first,
        )
    assert isinstance(weight, PiperNVFP4Tensor)
    return nvfp4_ops.prepare_input(
        hidden_states,
        weight.act_per_tensor_scale,
        quantization.use_dynamic_per_tensor_scale,
        None,
        weight.high_first,
    )


def _run_explicit_attention_output(
    model: _SparseProjectionAttentionOutput,
    hidden_states: torch.Tensor,
    *,
    coarse_gate: torch.Tensor | None = None,
    block_lengths: torch.Tensor | None = None,
    sparse_query_blocks: int | None = None,
    force_tile_ids: torch.Tensor | None = None,
    force_tile_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    q_input = _prepare_nvfp4_input(hidden_states, model.query.weight)
    k_input = _prepare_nvfp4_input(hidden_states, model.key.weight)
    v_input = _prepare_nvfp4_input(hidden_states, model.value.weight)
    query = fused_query.project_query(
        *q_input,
        *_nvfp4_storage(model.query.weight),
        model.query.bias,
        model.query_norm,
        model.cos,
        model.sin,
        1e-5,
        model.head_dim**-0.5,
        4_096,
        model.sparse_attention._routing_mode,
        block_lengths,
        head_dim=model.head_dim,
    )
    key = fused_key.project_key(
        *k_input,
        *_nvfp4_storage(model.key.weight),
        model.key.bias,
        model.key_norm,
        model.cos,
        model.sin,
        1e-5,
        4_096,
        model.sparse_attention._routing_mode,
        block_lengths,
        head_dim=model.head_dim,
    )
    value = fused_value.project_value(
        *v_input,
        *_nvfp4_storage(model.value.weight),
        model.value.bias,
        4_096,
        block_lengths,
        head_dim=model.head_dim,
    )
    attention_arguments = (*query, *key, *value)
    if coarse_gate is None:
        optional_arguments = (
            (block_lengths, sparse_query_blocks)
            if sparse_query_blocks is not None
            else ((block_lengths,) if block_lengths is not None else ())
        )
        attention = torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default(
            *attention_arguments,
            model.sparse_attention._ratio_profile,
            model.sparse_key_blocks,
            model.sequence_length,
            model.sparse_attention._routing_mode,
            *optional_arguments,
            output_dtype=hidden_states.dtype,
            force_tile_ids=force_tile_ids,
            force_tile_counts=force_tile_counts,
        )
    else:
        assert isinstance(model, _BoundedSparseProjectionAttentionOutput)
        coarse_attention = (
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized
        ).default
        attention = coarse_attention(
            *attention_arguments,
            coarse_gate,
            model.sparse_attention._ratio_profile,
            model.sparse_key_blocks,
            model.sequence_length,
            model.sparse_attention._routing_mode,
            model.coarse_scale,
            model.coarse_key_blocks,
            block_lengths,
            sparse_query_blocks,
            output_dtype=hidden_states.dtype,
            force_tile_ids=force_tile_ids,
            force_tile_counts=force_tile_counts,
        )
    output_weight = model.output.weight
    if isinstance(output_weight, ConvRotNVFP4Tensor):
        assert output_weight.act_quant_kwargs is not None
        return convrot_nvfp4_ops.linear(
            attention.flatten(2),
            output_weight.qdata,
            output_weight.scale,
            output_weight.per_tensor_scale,
            output_weight.act_per_tensor_scale,
            model.output.bias,
            output_weight.act_quant_kwargs.use_dynamic_per_tensor_scale,
            output_weight.group_size,
            output_weight.high_first,
        )
    assert isinstance(output_weight, PiperNVFP4Tensor)
    quantization = output_weight.act_quant_kwargs
    assert quantization is not None
    output_qdata, output_scale, output_per_tensor_scale = nvfp4_ops.prepare_input(
        attention.flatten(2),
        output_weight.act_per_tensor_scale,
        quantization.use_dynamic_per_tensor_scale,
        None,
        output_weight.high_first,
    )
    assert output_weight.per_tensor_scale is not None
    scaling_type = F.ScalingType
    swizzle_type = F.SwizzleType
    result = F.scaled_mm(
        output_qdata.view(torch.float4_e2m1fn_x2),
        output_weight.qdata.t().view(torch.float4_e2m1fn_x2),
        [output_scale.view(torch.float8_e4m3fn), output_per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [output_weight.scale.view(torch.float8_e4m3fn), output_weight.per_tensor_scale],
        [scaling_type.BlockWise1x16, scaling_type.TensorWise],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        [swizzle_type.SWIZZLE_32_4_4, swizzle_type.NO_SWIZZLE],
        bias=None if hidden_states.dtype is torch.float32 else model.output.bias,
        output_dtype=hidden_states.dtype,
    )
    if hidden_states.dtype is torch.float32 and model.output.bias is not None:
        result.add_(model.output.bias)
    return result.view(model.batch, model.sequence_length, model.output_features)


def test_compile_options_install_idempotent_passes() -> None:
    options = compile_options({"max_autotune": True}, attention=True)

    assert options["max_autotune"] is True
    passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    assert len(passes) == 1
    assert compile_options(options, attention=True) == options


@pytest.mark.parametrize("sparse_first", [False, True])
def test_compile_options_compose_with_linear_options(sparse_first: bool) -> None:
    if sparse_first:
        options = compile_options(compile_options(attention=True))
    else:
        options = compile_options(compile_options(), attention=True)

    assert options == compile_options(attention=True)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize(("dynamic", "preparation_count"), [(False, 3), (True, 1)])
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
@pytest.mark.parametrize("q_affine", [True, False])
@pytest.mark.parametrize("k_affine", [True, False])
def test_prepared_projection_family_fuses_without_materializing_linears(
    group_size: int | None,
    q_affine: bool,
    k_affine: bool,
    monkeypatch: pytest.MonkeyPatch,
    dynamic: bool,
    preparation_count: int,
    routing_mode: int,
) -> None:
    graph = build_attention_graph(
        dynamic=dynamic,
        routing_mode=routing_mode,
        q_affine=q_affine,
        k_affine=k_affine,
        group_size=group_size,
    )
    mock_cuda_target(monkeypatch, (12, 0))

    fold_nvfp4_linear(graph)
    fold_convrot_nvfp4_linear(graph)
    expected_prepare = (
        torch.ops.piper_kernels.nvfp4_prepare_input.default
        if group_size is None
        else torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default
    )
    prepared_targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert prepared_targets.count(expected_prepare) == preparation_count
    assert prepared_targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 3

    fold_sparse_attention(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(expected_prepare) == preparation_count
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert torch.ops.piper_kernels.nvfp4_linear_mean.default not in targets
    assert targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default) == 1
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize(
    ("linear_enabled", "distinct_query_input", "expected_prepares"),
    [
        (False, False, 1),
        (True, True, 2),
        (False, True, 2),
    ],
    ids=["semantic-only", "cross-attention", "cross-attention-semantic-only"],
)
def test_semantic_and_cross_attention_projections_fuse_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    linear_enabled: bool,
    distinct_query_input: bool,
    expected_prepares: int,
) -> None:
    graph = build_attention_graph(
        dynamic=True,
        output_dynamic=False,
        group_size=group_size,
        distinct_query_input=distinct_query_input,
    )
    run_nvfp4_passes(graph, monkeypatch, linear=linear_enabled)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    expected_prepare = (
        torch.ops.piper_kernels.nvfp4_prepare_input.default
        if group_size is None
        else torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default
    )
    assert targets.count(expected_prepare) == expected_prepares
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert (
        targets.count(
            torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("with_block_lengths", [False, True])
@pytest.mark.parametrize("with_coarse", [False, True])
@pytest.mark.parametrize("with_sparse_query_blocks", [False, True])
def test_projection_fold_supports_every_bounded_attention_variant(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> None:
    graph = build_attention_graph(
        dynamic=False,
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
        group_size=group_size,
    )
    run_nvfp4_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    expected_attention = (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        if with_coarse
        else torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default
    )
    assert targets.count(expected_attention) == 1
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_coarse_residual.default not in targets
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize(("dynamic", "preparation_count"), [(False, 3), (True, 1)])
@pytest.mark.parametrize("routing_mode", [_MINMAX_ROUTING, _MEAN_ROUTING])
@pytest.mark.parametrize("with_sparse_query_blocks", [False, True])
def test_static_output_fuses_after_prepared_sparse_projection(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    dynamic: bool,
    preparation_count: int,
    routing_mode: int,
    with_sparse_query_blocks: bool,
) -> None:
    graph = build_attention_graph(
        dynamic=dynamic,
        output_dynamic=False,
        routing_mode=routing_mode,
        with_sparse_query_blocks=with_sparse_query_blocks,
        group_size=group_size,
    )
    run_nvfp4_passes(graph, monkeypatch)

    call_nodes = [node for node in graph.nodes if node.op == "call_function"]
    targets = [node.target for node in call_nodes]
    expected_prepare = (
        torch.ops.piper_kernels.nvfp4_prepare_input.default
        if group_size is None
        else torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default
    )
    assert targets.count(expected_prepare) == preparation_count
    projected_output = (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    )
    assert targets.count(projected_output) == 1
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in targets
    output_node = next(node for node in call_nodes if node.target is projected_output)
    assert output_node.args[30] == 8_192
    assert output_node.args[-1] == (2 if with_sparse_query_blocks else None)
    assert output_node.kwargs["group_size"] == group_size
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("with_block_lengths", [False, True])
@pytest.mark.parametrize("with_coarse", [False, True])
@pytest.mark.parametrize("with_sparse_query_blocks", [False, True])
def test_complete_fold_supports_every_bounded_attention_variant(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> None:
    graph = build_attention_graph(
        dynamic=False,
        output_dynamic=False,
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
        group_size=group_size,
    )
    run_nvfp4_passes(graph, monkeypatch)

    call_nodes = [node for node in graph.nodes if node.op == "call_function"]
    targets = [node.target for node in call_nodes]
    projected_output = (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    )
    assert torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default not in targets
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert targets.count(projected_output) == 1
    output_node = next(node for node in call_nodes if node.target is projected_output)
    assert output_node.kwargs["group_size"] == group_size
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in targets
    assert torch.ops.piper_kernels.sparse_piper_coarse_residual.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("with_block_lengths", [False, True])
@pytest.mark.parametrize("with_coarse", [False, True])
@pytest.mark.parametrize("with_sparse_query_blocks", [False, True])
def test_output_fold_supports_every_bounded_attention_variant(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    with_block_lengths: bool,
    with_coarse: bool,
    with_sparse_query_blocks: bool,
) -> None:
    graph = build_quantized_attention_output_graph(
        with_block_lengths=with_block_lengths,
        with_coarse=with_coarse,
        with_sparse_query_blocks=with_sparse_query_blocks,
        group_size=group_size,
    )
    mock_cuda_target(monkeypatch, (12, 0))

    assert _fold_attention_output(graph)

    call_nodes = [node for node in graph.nodes if node.op == "call_function"]
    targets = [node.target for node in call_nodes]
    fused = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    assert targets.count(fused) == 1
    node = next(node for node in call_nodes if node.target is fused)
    assert node.kwargs["group_size"] == group_size
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert (
        torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
        not in targets
    )
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_high_first_output_fold_preserves_order_metadata(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
) -> None:
    graph = build_quantized_attention_output_graph(
        with_block_lengths=False,
        with_coarse=False,
        with_sparse_query_blocks=False,
        high_first=True,
        group_size=group_size,
    )
    mock_cuda_target(monkeypatch, (12, 0))

    assert _fold_attention_output(graph)

    fused = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    node = next(node for node in graph.nodes if node.target is fused)
    assert node.kwargs == {
        "high_first": True,
        "output_dtype": torch.bfloat16,
        "dynamic_activation_scale": False,
        "group_size": group_size,
    }
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_dynamic_output_fuses(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
) -> None:
    graph = build_attention_graph(
        dynamic=False,
        output_dynamic=True,
        group_size=group_size,
    )
    run_nvfp4_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    fused = torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    assert targets.count(fused) == 1
    node = next(node for node in graph.nodes if node.target is fused)
    assert node.kwargs["dynamic_activation_scale"] is True
    assert node.kwargs["group_size"] == group_size
    assert node.args[28] is None
    assert node.args[30] == 32_768
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert torch.ops.piper_kernels.nvfp4_linear.default not in targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in targets
    graph.lint()


def test_mixed_plain_and_convrot_output_blocks_fold_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_mixed_quantized_attention_output_graph(group_size=64)
    run_nvfp4_passes(graph, monkeypatch)

    fused = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    fused_nodes = [node for node in graph.nodes if node.target is fused]
    assert len(fused_nodes) == 2
    assert [node.kwargs["group_size"] for node in fused_nodes] == [None, 64]
    assert all(
        node.target
        not in (
            torch.ops.piper_kernels.nvfp4_linear.default,
            torch.ops.piper_kernels.convrot_nvfp4_linear.default,
        )
        for node in graph.nodes
    )
    graph.lint()


def test_output_fold_fails_closed_for_invalid_group_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_quantized_attention_output_graph(
        with_block_lengths=False,
        with_coarse=False,
        with_sparse_query_blocks=False,
        group_size=32,
    )
    mock_cuda_target(monkeypatch, (12, 0))

    assert not _fold_attention_output(graph)
    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default not in targets
    assert targets.count(torch.ops.piper_kernels.convrot_nvfp4_linear.default) == 1
    graph.lint()


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_static_output_fails_closed_when_attention_escapes(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
) -> None:
    graph = build_attention_graph(
        dynamic=False,
        output_dynamic=False,
        escape_attention=True,
        group_size=group_size,
    )
    run_nvfp4_passes(graph, monkeypatch)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    expected_linear = (
        torch.ops.piper_kernels.nvfp4_linear.default
        if group_size is None
        else torch.ops.piper_kernels.convrot_nvfp4_linear.default
    )
    assert torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default not in targets
    assert targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default) == 1
    assert targets.count(expected_linear) == 1
    graph.lint()


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("dynamic", "preparation_count", "routing"),
    [(False, 3, "minmax"), (True, 1, "minmax"), (False, 3, "mean")],
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_cuda_compile_fuses_nvfp4_sparse_projection_region(
    head_dim: int,
    dynamic: bool,
    preparation_count: int,
    routing: str,
) -> None:
    torch.manual_seed(823)
    model = _SparseProjectionAttention(dynamic=dynamic, routing=routing, head_dim=head_dim).eval()
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = TargetCapturePass()
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True)(hidden_states)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(attention=True)),
        )(hidden_states)

    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert relative_l2 < 0.025
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_prepare_input.default)
        == preparation_count
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    )
    assert torch.ops.piper_kernels.nvfp4_linear_mean.default not in capture.targets
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize(
    ("dynamic", "routing", "preparation_count"),
    [(False, "minmax", 3), (True, "minmax", 1), (False, "mean", 3)],
)
# test_projection.py::test_chunked_qkv_epilogues_match_materialized_fp32_contract
# covers the full head-size/affine arithmetic. Retain every pair
# of dtype, head size, and affine mode here, across both formats and scale modes.
@pytest.mark.parametrize(
    ("dtype", "head_dim", "affine"),
    [
        (torch.float16, 64, True),
        (torch.float16, 128, False),
        (torch.bfloat16, 64, False),
        (torch.bfloat16, 128, True),
        (torch.float32, 64, True),
        (torch.float32, 128, False),
    ],
)
@pytest.mark.parametrize("output_dynamic", [False, True])
def test_cuda_compile_fuses_nvfp4_attention_output(
    group_size: int | None,
    output_dynamic: bool,
    affine: bool,
    dtype: torch.dtype,
    head_dim: int,
    dynamic: bool,
    routing: str,
    preparation_count: int,
) -> None:
    torch.manual_seed(829 + (0 if group_size is None else group_size) + dynamic)
    model = _SparseProjectionAttentionOutput(
        dynamic=dynamic,
        routing=routing,
        output_dynamic=output_dynamic,
        group_size=group_size,
        output_features=128 if output_dynamic else 320,
        head_dim=head_dim,
    ).eval()
    model.set_activation_dtype(dtype)
    if not affine:
        model.query_norm = None
        model.key_norm = None
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=dtype,
    )
    capture = TargetCapturePass()
    with torch.no_grad():
        expected = _run_explicit_attention_output(model, hidden_states)
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(attention=True)),
        )(hidden_states)

    expected_prepare = (
        torch.ops.piper_kernels.nvfp4_prepare_input.default
        if group_size is None
        else torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default
    )
    assert actual.dtype is dtype
    assert_fusion_output_close(actual, expected)
    assert capture.targets.count(expected_prepare) == preparation_count
    assert torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default not in capture.targets
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in (
        capture.targets
    )
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize(
    ("dynamic", "routing"),
    [(False, "minmax"), (False, "mean"), (True, "minmax")],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("output_dynamic", [False, True])
def test_cuda_compile_fuses_every_bounded_nvfp4_attention_feature(
    output_dynamic: bool,
    monkeypatch,
    head_dim: int,
    dynamic: bool,
    routing: str,
) -> None:
    torch.manual_seed(831)
    model = _BoundedSparseProjectionAttentionOutput(
        dynamic=dynamic, routing=routing, output_dynamic=output_dynamic, head_dim=head_dim
    ).eval()
    if output_dynamic:
        monkeypatch.setattr(_output, "DYNAMIC_QUERY_CHUNK_ROWS", 128)
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    coarse_gate = torch.randn(
        (model.batch, model.sequence_length, model.heads, model.head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    valid_rows = (torch.arange(64, device="cuda")[None, :] < block_lengths[:, None]).flatten()
    capture = TargetCapturePass()
    with torch.no_grad():
        expected = _run_explicit_attention_output(
            model,
            hidden_states,
            coarse_gate=coarse_gate,
            block_lengths=block_lengths,
            sparse_query_blocks=2,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(attention=True)),
        )(
            hidden_states,
            coarse_gate,
            block_lengths,
            2,
        )

    assert_fusion_output_close(actual[:, valid_rows], expected[:, valid_rows])
    assert torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default not in capture.targets
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    )
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.sparse_piper_attention.default not in capture.targets
    assert torch.ops.piper_kernels.sparse_piper_coarse_residual.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize(
    ("dynamic", "routing", "preparation_count"),
    [(False, "minmax", 4), (False, "mean", 4), (True, "minmax", 1)],
)
# test_prepared_projection_family_fuses_without_materializing_linears covers Q/K norms;
# test_cuda_compile_fuses_nvfp4_attention_output covers their dtype/head-size pairs.
# Retain both formats, gate layouts, all dtypes, routing, preparation sharing, and chunk boundaries.
@pytest.mark.parametrize(
    ("dtype", "key_affine", "query_chunk_rows", "output_dynamic"),
    [
        (torch.float16, False, 128, False),
        (torch.bfloat16, True, 128, False),
        (torch.float32, False, 128, False),
        (torch.bfloat16, True, 8_192, False),
        (torch.bfloat16, False, 128, True),
    ],
)
def test_cuda_compile_lifetime_chunks_a_projected_coarse_gate(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    key_affine: bool,
    dtype: torch.dtype,
    dynamic: bool,
    routing: str,
    preparation_count: int,
    query_chunk_rows: int,
    output_dynamic: bool,
) -> None:
    # Exercise eight windows with a short tail and the single-window path.
    monkeypatch.setattr(_output, "DEFAULT_QUERY_CHUNK_ROWS", query_chunk_rows)
    if output_dynamic:
        monkeypatch.setattr(_output, "DYNAMIC_QUERY_CHUNK_ROWS", query_chunk_rows)
    torch.manual_seed(839 + (0 if group_size is None else group_size) + dynamic)
    model = _ProjectedGateCoarseSparseAttentionOutput(
        dynamic=dynamic,
        routing=routing,
        output_dynamic=output_dynamic,
        group_size=group_size,
        sequence_length=960,
    ).eval()
    model.set_activation_dtype(dtype)
    if not key_affine:
        model.key_norm = None
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=dtype,
    )
    block_lengths = torch.tensor([64, 17, 51] * 5, device="cuda", dtype=torch.int32)
    valid_rows = (torch.arange(64, device="cuda")[None, :] < block_lengths[:, None]).flatten()
    gate_chunks = []
    project_gate = _output.PreparedGateProjection.project

    def record_gate_chunk(projection, out, start, rows):
        gate_chunks.append((start, rows, out.data_ptr()))
        project_gate(projection, out, start, rows)

    monkeypatch.setattr(_output.PreparedGateProjection, "project", record_gate_chunk)
    capture = TargetCapturePass()
    with torch.no_grad():
        gate_input = _prepare_nvfp4_input(hidden_states, model.gate.weight)
        coarse_gate = nvfp4_ops.linear_prepared(
            *gate_input,
            *_nvfp4_storage(model.gate.weight),
            model.gate.bias,
            dtype,
        ).view(
            model.batch,
            model.sequence_length,
            model.heads,
            model.head_dim,
        )
        expected = _run_explicit_attention_output(
            model,
            hidden_states,
            coarse_gate=coarse_gate,
            block_lengths=block_lengths,
            sparse_query_blocks=2,
        )
        torch._dynamo.reset()
        actual = torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(attention=True)),
        )(hidden_states, block_lengths, 2)

    expected_prepare = (
        torch.ops.piper_kernels.nvfp4_prepare_input.default
        if group_size is None
        else torch.ops.piper_kernels.convrot_nvfp4_prepare_input.default
    )
    assert actual.dtype is dtype
    # 1-ULP FP32 differences across chunked coarse_attention can flip one 4-bit
    # NVFP4 block on dynamic outputs; static outputs retain the 1e-3 bound.
    assert_fusion_output_close(
        actual[:, valid_rows],
        expected[:, valid_rows],
        relative_l2=5e-3 if output_dynamic else 1e-3,
    )
    assert [(start, rows) for start, rows, _ in gate_chunks] == [
        (start, min(query_chunk_rows, model.sequence_length - start))
        for start in range(0, model.sequence_length, query_chunk_rows)
    ]
    assert len({pointer for _, _, pointer in gate_chunks}) == (
        1 if output_dynamic else min(2, len(gate_chunks))
    )
    assert capture.targets.count(expected_prepare) == preparation_count
    assert (
        capture.targets.count(
            torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert torch.ops.piper_kernels.nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.convrot_nvfp4_linear.default not in capture.targets
    assert torch.ops.piper_kernels.nvfp4_linear_prepared.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_fails_closed_for_batch_two() -> None:
    torch.manual_seed(827)
    model = _SparseProjectionAttention(batch=2).eval()
    hidden_states = torch.randn(
        (model.batch, model.sequence_length, model.input_features),
        device="cuda",
        dtype=torch.bfloat16,
    )
    capture = TargetCapturePass()

    with torch.no_grad():
        torch.compile(
            model,
            fullgraph=True,
            options=capture.wrap_options(compile_options(attention=True)),
        )(hidden_states)

    assert torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default not in capture.targets
    assert capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention.default) == 1


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("output", [False, True])
def test_forced_inputs_survive_projection_and_output_fusion(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int | None,
    bounded: bool,
    output: bool,
) -> None:
    graph = build_attention_graph(
        dynamic=True,
        output_dynamic=False if output else None,
        with_block_lengths=bounded,
        with_coarse=bounded,
        with_sparse_query_blocks=bounded,
        with_force_tiles=True,
        group_size=group_size,
    )
    ids = next(node for node in graph.nodes if node.target == "force_tile_ids")
    counts = next(node for node in graph.nodes if node.target == "force_tile_counts")
    run_nvfp4_passes(graph, monkeypatch)
    if output:
        target = torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    else:
        target = (
            torch.ops.piper_kernels.sparse_piper_attention_with_coarse_residual_from_quantized.default
            if bounded
            else torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default
        )
    calls = [node for node in graph.nodes if node.target == target]
    assert len(calls) == 1
    if output:
        assert calls[0].kwargs["group_size"] == group_size
        assert calls[0].kwargs["force_tile_ids"] is ids
        assert calls[0].kwargs["force_tile_counts"] is counts
    else:
        assert calls[0].args[-2:] == (ids, counts)
    assert all(
        node.target != torch.ops.piper_kernels.sparse_piper_attention.default
        for node in graph.nodes
    )
    graph.lint()
