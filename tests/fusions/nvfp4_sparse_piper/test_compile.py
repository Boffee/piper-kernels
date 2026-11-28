"""Tests for automatic NVFP4 and ConvRot NVFP4 sparse-Piper graph fusion."""

from __future__ import annotations

from collections.abc import Container
from typing import TypedDict, Unpack

import pytest
import torch
from _accelerators import (
    ensure_convrot_nvfp4_linear_ops,
    ensure_nvfp4_linear_ops,
    exact_sm120_available,
    forbid_device_probes,
    mock_cuda_target,
    mock_triton_driver,
)
from _compile_capture import FunctionalTargetCapturePass, TargetCapturePass
from torch._dynamo.utils import ReinplaceCounters
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F  # noqa: N812

from piper_kernels import (
    SparsePiperAttention,
    compile_options,
    sparse_piper_coarse_residual,
)
from piper_kernels.attention.sparse_piper_attention._contracts import (
    _MEAN_ROUTING,
    _MINMAX_ROUTING,
)
from piper_kernels.fusions.nvfp4_sparse_piper import key as fused_key
from piper_kernels.fusions.nvfp4_sparse_piper import output as nvfp4_output
from piper_kernels.fusions.nvfp4_sparse_piper import query as fused_query
from piper_kernels.fusions.nvfp4_sparse_piper import value as fused_value
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


def _run_attention_pass(graph: torch.fx.Graph) -> None:
    passes = compile_options(linear=False, attention=True)[_POST_GRAD_PRE_PASS]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)


def _op_absent(targets: Container[object], name: str) -> bool:
    packet = getattr(torch.ops.piper_kernels, name, None)
    return packet is None or packet.default not in targets


def _amax_to_scale(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().abs().amax() / (448.0 * 6.0)


def _convert_weight(weight: torch.Tensor, group_size: int) -> ConvRotNVFP4Tensor:
    assert isinstance(weight, PiperNVFP4Tensor)
    dense = weight.dequantize(torch.bfloat16)
    return ConvRotNVFP4Tensor.from_hp(
        dense,
        group_size=group_size,
        compute_per_tensor_scale=True,
        act_per_tensor_scale=weight.act_per_tensor_scale,
        is_swizzled_scales=True,
        act_quant=weight.act_quant,
    )


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
        base_scale = _amax_to_scale(calibration)
        act_quant = "dynamic" if dynamic else "static"
        projections = []
        for index in range(3):
            dense = torch.randn(
                (self.heads * self.head_dim, self.input_features),
                device="cuda",
                dtype=torch.bfloat16,
            )
            weight: torch.Tensor = PiperNVFP4Tensor.from_hp(
                dense,
                compute_per_tensor_scale=True,
                act_per_tensor_scale=(None if dynamic else base_scale * (1.0 + index * 0.01)),
                is_swizzled_scales=True,
                act_quant=act_quant,
            )
            projection = torch.nn.Linear(
                self.input_features,
                self.heads * self.head_dim,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
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
        activation_scale = _amax_to_scale(calibration)
        dense = torch.randn(
            (self.output_features, self.heads * self.head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight: torch.Tensor = PiperNVFP4Tensor.from_hp(
            dense,
            compute_per_tensor_scale=True,
            act_per_tensor_scale=None if output_dynamic else activation_scale,
            is_swizzled_scales=True,
            act_quant="dynamic" if output_dynamic else "static",
        )
        self.output = torch.nn.Linear(
            self.heads * self.head_dim,
            self.output_features,
            bias=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
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
        assert isinstance(query_weight, PiperNVFP4Tensor)
        dense = torch.randn(
            (self.heads * self.head_dim, self.input_features),
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight: torch.Tensor = PiperNVFP4Tensor.from_hp(
            dense,
            compute_per_tensor_scale=True,
            act_per_tensor_scale=query_weight.act_per_tensor_scale,
            is_swizzled_scales=True,
            act_quant=query_weight.act_quant,
        )
        self.gate = torch.nn.Linear(
            self.input_features,
            self.heads * self.head_dim,
            bias=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
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
    assert isinstance(weight, PiperNVFP4Tensor)
    return weight.qdata, weight.scale, weight.per_tensor_scale


def _prepare_nvfp4_input(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    assert isinstance(weight, (PiperNVFP4Tensor, ConvRotNVFP4Tensor))
    assert weight.act_quant is not None
    ensure_nvfp4_linear_ops()
    return torch.ops.piper_kernels.nvfp4_prepare_input(
        hidden_states,
        weight.act_per_tensor_scale,
        weight.act_quant == "dynamic",
        high_first=weight.high_first,
        group_size=weight.group_size if isinstance(weight, ConvRotNVFP4Tensor) else None,
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
    optional_arguments = (
        (block_lengths, sparse_query_blocks)
        if sparse_query_blocks is not None
        else ((block_lengths,) if block_lengths is not None else ())
    )
    if coarse_gate is None:
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
        attention = torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default(
            *attention_arguments,
            model.sparse_attention._ratio_profile,
            model.sparse_key_blocks,
            model.sequence_length,
            model.sparse_attention._routing_mode,
            *optional_arguments,
            coarse_gate=coarse_gate,
            coarse_scale=model.coarse_scale,
            coarse_key_blocks=model.coarse_key_blocks,
            output_dtype=hidden_states.dtype,
            force_tile_ids=force_tile_ids,
            force_tile_counts=force_tile_counts,
        )
    output_weight = model.output.weight
    if isinstance(output_weight, ConvRotNVFP4Tensor):
        assert output_weight.act_quant is not None
        ensure_convrot_nvfp4_linear_ops()
        return torch.ops.piper_kernels.convrot_nvfp4_linear(
            attention.flatten(2),
            output_weight.qdata,
            output_weight.scale,
            output_weight.per_tensor_scale,
            output_weight.act_per_tensor_scale,
            model.output.bias,
            output_weight.act_quant == "dynamic",
            output_weight.group_size,
            output_weight.high_first,
        )
    assert isinstance(output_weight, PiperNVFP4Tensor)
    assert output_weight.act_quant is not None
    ensure_nvfp4_linear_ops()
    output_qdata, output_scale, output_per_tensor_scale = (
        torch.ops.piper_kernels.nvfp4_prepare_input(
            attention.flatten(2),
            output_weight.act_per_tensor_scale,
            output_weight.act_quant == "dynamic",
            None,
            output_weight.high_first,
        )
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

    (linear_pass,) = compile_options()[_POST_GRAD_PRE_PASS]
    linear_pass(graph, is_inference=True)
    expected_prepare = torch.ops.piper_kernels.nvfp4_prepare_input.default
    prepared_targets = [node.target for node in graph.nodes if node.op == "call_function"]
    preparations = graph.find_nodes(op="call_function", target=expected_prepare)
    assert len(preparations) == preparation_count
    for node in preparations:
        fake_args = torch.fx.map_arg(node.args, lambda arg: arg.meta["val"])
        assert node.kwargs.get("group_size") == group_size
        expected = expected_prepare(*fake_args, **node.kwargs)
        for value, output in zip(node.meta["val"], expected, strict=True):
            assert (value.shape, value.dtype) == (output.shape, output.dtype)
        for getitem in node.users:
            value = getitem.meta["val"]
            output = expected[getitem.args[1]]
            assert (value.shape, value.dtype) == (output.shape, output.dtype)
    assert prepared_targets.count(torch.ops.piper_kernels.nvfp4_linear_prepared.default) == 3

    _run_attention_pass(graph)

    targets = [node.target for node in graph.nodes if node.op == "call_function"]
    assert targets.count(expected_prepare) == preparation_count
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_query.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert _op_absent(targets, "nvfp4_linear_mean")
    assert targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default) == 1
    assert _op_absent(targets, "nvfp4_linear")
    assert _op_absent(targets, "convrot_nvfp4_linear")
    assert _op_absent(targets, "nvfp4_linear_prepared")
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
    expected_prepare = torch.ops.piper_kernels.nvfp4_prepare_input.default
    assert targets.count(expected_prepare) == expected_prepares
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_key.default) == 1
    assert targets.count(torch.ops.piper_kernels.nvfp4_sparse_piper_project_value.default) == 1
    assert (
        targets.count(
            torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        )
        == 1
    )
    assert _op_absent(targets, "nvfp4_linear")
    assert _op_absent(targets, "convrot_nvfp4_linear")
    assert _op_absent(targets, "nvfp4_linear_prepared")
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
    expected_attention = torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default
    assert targets.count(expected_attention) == 1
    attention_call = next(node for node in graph.nodes if node.target is expected_attention)
    assert ("coarse_gate" in attention_call.kwargs) is with_coarse
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
    expected_prepare = torch.ops.piper_kernels.nvfp4_prepare_input.default
    assert targets.count(expected_prepare) == preparation_count
    projected_output = (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
    )
    assert targets.count(projected_output) == 1
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert _op_absent(targets, "nvfp4_linear")
    assert _op_absent(targets, "convrot_nvfp4_linear")
    assert _op_absent(targets, "nvfp4_linear_prepared")
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
    assert _op_absent(targets, "nvfp4_linear")
    assert _op_absent(targets, "convrot_nvfp4_linear")
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

    _run_attention_pass(graph)

    call_nodes = [node for node in graph.nodes if node.op == "call_function"]
    targets = [node.target for node in call_nodes]
    fused = torch.ops.piper_kernels.nvfp4_sparse_piper_attention_output.default
    assert targets.count(fused) == 1
    node = next(node for node in call_nodes if node.target is fused)
    assert node.kwargs["group_size"] == group_size
    assert torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default not in targets
    assert _op_absent(targets, "nvfp4_linear")
    assert _op_absent(targets, "convrot_nvfp4_linear")
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

    _run_attention_pass(graph)

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
    assert _op_absent(targets, "nvfp4_linear")
    assert _op_absent(targets, "convrot_nvfp4_linear")
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
    forbid_device_probes(monkeypatch)

    _run_attention_pass(graph)
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
    assert _op_absent(capture.targets, "nvfp4_linear_mean")
    assert (
        capture.targets.count(torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default)
        == 1
    )
    assert _op_absent(capture.targets, "nvfp4_linear")
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

    expected_prepare = torch.ops.piper_kernels.nvfp4_prepare_input.default
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
    assert _op_absent(capture.targets, "nvfp4_linear")
    assert _op_absent(capture.targets, "convrot_nvfp4_linear")


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
        monkeypatch.setattr(nvfp4_output, "_DYNAMIC_QUERY_CHUNK_ROWS", 128)
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
    monkeypatch.setattr(nvfp4_output, "_DEFAULT_QUERY_CHUNK_ROWS", query_chunk_rows)
    if output_dynamic:
        monkeypatch.setattr(nvfp4_output, "_DYNAMIC_QUERY_CHUNK_ROWS", query_chunk_rows)
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
    project_gate = nvfp4_output._PreparedGateProjection.project

    def record_gate_chunk(projection, out, start, rows):
        gate_chunks.append((start, rows, out.data_ptr()))
        project_gate(projection, out, start, rows)

    monkeypatch.setattr(nvfp4_output._PreparedGateProjection, "project", record_gate_chunk)
    capture = TargetCapturePass()
    with torch.no_grad():
        gate_input = _prepare_nvfp4_input(hidden_states, model.gate.weight)
        coarse_gate = torch.ops.piper_kernels.nvfp4_linear_prepared(
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

    expected_prepare = torch.ops.piper_kernels.nvfp4_prepare_input.default
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
    assert _op_absent(capture.targets, "nvfp4_linear")
    assert _op_absent(capture.targets, "convrot_nvfp4_linear")
    assert _op_absent(capture.targets, "nvfp4_linear_prepared")


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
        target = torch.ops.piper_kernels.sparse_piper_attention_from_quantized.default
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


class _SparseNvfp4ResidualBlock(torch.nn.Module):
    """Copy-back gated residual block over sparse NVFP4 / ConvRot NVFP4 attention and a linear."""

    width = _SparseProjectionAttention.input_features

    def __init__(
        self,
        *,
        group_size: int | None = None,
        output_dynamic: bool = False,
        sequence_length: int = 192,
        with_second_update: bool = True,
        coarse: bool = False,
        escape_attention: bool = False,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.escape_attention = escape_attention
        self.coarse = coarse
        self.sparse_query_blocks = 2 if coarse else None
        if coarse:
            self.attention: _SparseProjectionAttentionOutput = (
                _ProjectedGateCoarseSparseAttentionOutput(
                    dynamic=True,
                    output_dynamic=output_dynamic,
                    output_features=self.width,
                    group_size=group_size,
                    sequence_length=sequence_length,
                )
            )
        else:
            self.attention = _SparseProjectionAttentionOutput(
                dynamic=True,
                output_dynamic=output_dynamic,
                output_features=self.width,
                group_size=group_size,
                sequence_length=sequence_length,
            )
        gates = torch.randn(3, 6 * self.width, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        gates[:, 5 * self.width :].mul_(0.05)
        self.register_buffer("modulation", gates)
        self.with_second_update = with_second_update
        if with_second_update:
            self.post_linear = self._make_nvfp4_linear(
                self.width, self.width, group_size=group_size
            )

    @staticmethod
    def _make_nvfp4_linear(
        in_features: int,
        out_features: int,
        *,
        group_size: int | None,
    ) -> torch.nn.Linear:
        calibration = torch.full((1, 16, in_features), 3.0, device="cuda", dtype=torch.bfloat16)
        dense = torch.randn((out_features, in_features), device="cuda", dtype=torch.bfloat16)
        weight: torch.Tensor = PiperNVFP4Tensor.from_hp(
            dense,
            compute_per_tensor_scale=True,
            act_per_tensor_scale=_amax_to_scale(calibration),
            is_swizzled_scales=True,
            act_quant="static",
        )
        layer = torch.nn.Linear(
            in_features, out_features, bias=False, device="cuda", dtype=torch.bfloat16
        )
        if group_size is not None:
            weight = _convert_weight(weight, group_size)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        return layer

    def forward(
        self,
        hidden: torch.Tensor,
        gate_indices: torch.Tensor,
        block_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        shift, scale, attention_gate, post_shift, post_scale, post_gate = self.modulation.split(
            self.width, dim=-1
        )
        normalized = F.rms_norm(hidden, (self.width,), eps=1e-6)
        normalized = normalized * (1 + scale[gate_indices]) + shift[gate_indices]
        if self.coarse:
            assert isinstance(self.attention, _ProjectedGateCoarseSparseAttentionOutput)
            assert block_lengths is not None
            assert self.sparse_query_blocks is not None
            attended = self.attention(normalized, block_lengths, self.sparse_query_blocks)
        else:
            attended = self.attention(normalized)
        residual = hidden + attention_gate[gate_indices] * attended
        if self.with_second_update:
            post_input = F.rms_norm(residual, (self.width,), eps=1e-6)
            post_input = post_input * (1 + post_scale[gate_indices]) + post_shift[gate_indices]
            residual = residual + post_gate[gate_indices] * self.post_linear(post_input)
        return (residual, attended) if self.escape_attention else residual


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("output_dynamic", [False, True], ids=["static", "dynamic"])
def test_cuda_compile_folds_residual_add_into_nvfp4_sparse_attention_output(
    group_size: int | None,
    output_dynamic: bool,
) -> None:
    torch.manual_seed(1207 + (group_size or 0) + int(output_dynamic))
    model = _SparseNvfp4ResidualBlock(
        group_size=group_size,
        output_dynamic=output_dynamic,
        sequence_length=192,
        with_second_update=True,
    ).eval()
    hidden = torch.randn(1, model.sequence_length, model.width, device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.randint(-3, 3, (model.sequence_length,), device="cuda")

    base_options = compile_options(attention=True)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=base_options)(
            hidden.clone(), gate_indices
        )

    capture = FunctionalTargetCapturePass()
    fused_options = capture.wrap_options(compile_options(base_options, residual_updates=True))

    def copy_back(base: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return base.copy_(model(base, indices))

    torch._dynamo.reset()
    compiled = torch.compile(copy_back, fullgraph=True, options=fused_options)
    ReinplaceCounters.clear()
    with torch.no_grad():
        result = compiled(hidden, gate_indices)

    assert result.data_ptr() == hidden.data_ptr()
    assert ReinplaceCounters.get_total_missed() == 0
    assert_fusion_output_close(hidden, expected, relative_l2=5e-3)
    kernels = torch.ops.piper_kernels
    assert (
        capture.targets.count(
            kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default
        )
        == 1
    )
    assert capture.targets.count(kernels.residual_add_.default) == 1
    assert kernels.nvfp4_sparse_piper_projected_query_attention_output.default not in (
        capture.targets
    )


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_folds_residual_add_across_multi_chunk_stream_overlap() -> None:
    torch.manual_seed(1213)
    # Exceeds the 8192-row static window cap so producer/consumer stream overlap executes.
    model = _SparseNvfp4ResidualBlock(
        group_size=64,
        output_dynamic=False,
        sequence_length=8_320,
        with_second_update=False,
    ).eval()
    hidden = torch.randn(1, model.sequence_length, model.width, device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.randint(0, 3, (model.sequence_length,), device="cuda")

    base_options = compile_options(attention=True)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=base_options)(
            hidden.clone(), gate_indices
        )

    capture = FunctionalTargetCapturePass()

    def copy_back(base: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return base.copy_(model(base, indices))

    torch._dynamo.reset()
    compiled = torch.compile(
        copy_back,
        fullgraph=True,
        options=capture.wrap_options(compile_options(base_options, residual_updates=True)),
    )
    ReinplaceCounters.clear()
    with torch.no_grad():
        result = compiled(hidden, gate_indices)

    assert result.data_ptr() == hidden.data_ptr()
    assert ReinplaceCounters.get_total_missed() == 0
    assert_fusion_output_close(hidden, expected, relative_l2=5e-3)
    kernels = torch.ops.piper_kernels
    assert (
        capture.targets.count(
            kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default
        )
        == 1
    )
    assert kernels.nvfp4_sparse_piper_projected_query_attention_output.default not in (
        capture.targets
    )


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_folds_coarse_residual_add_with_block_lengths() -> None:
    torch.manual_seed(1217)
    model = _SparseNvfp4ResidualBlock(
        group_size=64,
        output_dynamic=False,
        sequence_length=192,
        with_second_update=False,
        coarse=True,
    ).eval()
    block_lengths = torch.tensor([64, 17, 51], device="cuda", dtype=torch.int32)
    valid_rows = (torch.arange(64, device="cuda")[None, :] < block_lengths[:, None]).flatten()
    hidden = torch.randn(1, model.sequence_length, model.width, device="cuda", dtype=torch.bfloat16)
    gate_indices = torch.randint(0, 3, (model.sequence_length,), device="cuda")

    base_options = compile_options(attention=True)
    with torch.no_grad():
        torch._dynamo.reset()
        expected = torch.compile(model, fullgraph=True, options=base_options)(
            hidden.clone(), gate_indices, block_lengths
        )

    capture = FunctionalTargetCapturePass()

    def copy_back(base: torch.Tensor, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return base.copy_(model(base, indices, lengths))

    torch._dynamo.reset()
    compiled = torch.compile(
        copy_back,
        fullgraph=True,
        options=capture.wrap_options(compile_options(base_options, residual_updates=True)),
    )
    ReinplaceCounters.clear()
    with torch.no_grad():
        result = compiled(hidden, gate_indices, block_lengths)

    assert result.data_ptr() == hidden.data_ptr()
    assert ReinplaceCounters.get_total_missed() == 0
    assert_fusion_output_close(hidden[:, valid_rows], expected[:, valid_rows], relative_l2=5e-3)
    kernels = torch.ops.piper_kernels
    assert (
        capture.targets.count(
            kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default
        )
        == 1
    )
    assert kernels.nvfp4_sparse_piper_projected_query_attention_output.default not in (
        capture.targets
    )
    assert kernels.residual_add_.default not in capture.targets


@pytest.mark.gpu
@pytest.mark.skipif(not exact_sm120_available(), reason="requires exact NVIDIA SM120")
def test_cuda_compile_residual_add_fails_closed_without_exclusive_copy_back() -> None:
    torch.manual_seed(1223)
    model = _SparseNvfp4ResidualBlock(
        group_size=None,
        output_dynamic=False,
        sequence_length=192,
        with_second_update=False,
        escape_attention=False,
    ).eval()
    hidden = torch.randn(1, model.sequence_length, model.width, device="cuda", dtype=torch.bfloat16)
    original = hidden.clone()
    gate_indices = torch.randint(0, 3, (model.sequence_length,), device="cuda")

    kernels = torch.ops.piper_kernels
    capture = TargetCapturePass()
    torch._dynamo.reset()
    compiled = torch.compile(
        model,
        fullgraph=True,
        options=capture.wrap_options(compile_options(attention=True, residual_updates=True)),
    )
    with torch.no_grad():
        compiled(hidden, gate_indices)

    torch.testing.assert_close(hidden, original, atol=0, rtol=0)
    assert kernels.nvfp4_sparse_piper_projected_query_attention_output.default in capture.targets
    assert kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default not in (
        capture.targets
    )

    escaping_model = _SparseNvfp4ResidualBlock(
        group_size=None,
        output_dynamic=False,
        sequence_length=192,
        with_second_update=False,
        escape_attention=True,
    ).eval()
    func_capture = FunctionalTargetCapturePass()

    def copy_back_escaped(
        base: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual, attended = escaping_model(base, indices)
        return base.copy_(residual), attended

    torch._dynamo.reset()
    compiled_escaped = torch.compile(
        copy_back_escaped,
        fullgraph=True,
        options=func_capture.wrap_options(compile_options(attention=True, residual_updates=True)),
    )
    with torch.no_grad():
        compiled_escaped(hidden.clone(), gate_indices)

    assert kernels.nvfp4_sparse_piper_projected_query_attention_output.default in (
        func_capture.targets
    )
    assert kernels.residual_add_.default in func_capture.targets
    assert kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default not in (
        func_capture.targets
    )


def _append_gated_residual_copy_back(
    graph: torch.fx.Graph,
    attention_output: torch.fx.Node,
) -> torch.fx.Node:
    base = graph.placeholder("base")
    base.meta["val"] = torch.empty((1, 192, 320), device="cuda", dtype=torch.bfloat16)
    gate = graph.placeholder("gate")
    gate.meta["val"] = torch.empty((3, 320), device="cuda", dtype=torch.bfloat16)
    indices = graph.placeholder("gate_indices")
    indices.meta["val"] = torch.empty((192,), device="cuda", dtype=torch.int64)
    selected = graph.call_function(torch.ops.aten.index.Tensor, (gate, [indices]))
    selected.meta["val"] = torch.empty((192, 320), device="cuda", dtype=torch.bfloat16)
    scaled = graph.call_function(torch.ops.aten.mul.Tensor, (selected, attention_output))
    scaled.meta["val"] = torch.empty((1, 192, 320), device="cuda", dtype=torch.bfloat16)
    summed = graph.call_function(torch.ops.aten.add.Tensor, (base, scaled))
    summed.meta["val"] = torch.empty((1, 192, 320), device="cuda", dtype=torch.bfloat16)
    copied = graph.call_function(torch.ops.aten.copy_.default, (base, summed))
    copied.meta["val"] = torch.empty((1, 192, 320), device="cuda", dtype=torch.bfloat16)
    return copied


def test_forced_inputs_survive_residual_add_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = build_attention_graph(
        dynamic=True,
        output_dynamic=False,
        with_block_lengths=True,
        with_coarse=True,
        with_sparse_query_blocks=True,
        with_force_tiles=True,
        group_size=64,
    )
    ids = next(node for node in graph.nodes if node.target == "force_tile_ids")
    counts = next(node for node in graph.nodes if node.target == "force_tile_counts")
    output_node = next(node for node in graph.nodes if node.op == "output")
    attention_output = output_node.args[0]
    assert isinstance(attention_output, torch.fx.Node)
    with FakeTensorMode(), graph.inserting_before(output_node):
        output_node.args = (_append_gated_residual_copy_back(graph, attention_output),)

    mock_cuda_target(monkeypatch, (12, 0))
    mock_triton_driver(monkeypatch, backend="cuda", arch=120)
    passes = compile_options(linear=True, attention=True, residual_updates=True)[
        "post_grad_custom_pre_pass"
    ]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)

    target = (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default
    )
    calls = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.auto_functionalized
        and node.args[0] is target
    ]
    assert len(calls) == 1
    assert calls[0].kwargs["group_size"] == 64
    assert calls[0].kwargs["force_tile_ids"] is ids
    assert calls[0].kwargs["force_tile_counts"] is counts
    assert (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        not in {node.target for node in graph.nodes}
    )
    graph.lint()


def test_prefolded_projected_query_attention_output_folds_residual_add_without_sparse_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = torch.fx.Graph()
    with FakeTensorMode():
        profile = graph.placeholder("ratio_profile")
        profile.meta["val"] = SparsePiperAttention((0.5, 1.0))._ratio_profile
        attention_output = graph.call_function(
            torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default,
            args=(
                graph.placeholder("q_in"),
                graph.placeholder("q_s"),
                graph.placeholder("q_g"),
                graph.placeholder("qw"),
                graph.placeholder("qws"),
                None,
                None,
                None,
                graph.placeholder("cos"),
                graph.placeholder("sin"),
                1e-5,
                128**-0.5,
                graph.placeholder("k"),
                graph.placeholder("ks"),
                graph.placeholder("kc"),
                graph.placeholder("kcs"),
                graph.placeholder("ksum"),
                graph.placeholder("kaux"),
                graph.placeholder("v"),
                graph.placeholder("vs"),
                graph.placeholder("vm"),
                profile,
                2,
                192,
                0,
                graph.placeholder("w"),
                graph.placeholder("ws"),
                None,
                None,
                None,
            ),
        )
        attention_output.meta["val"] = torch.empty(
            (1, 192, 320), device="cuda", dtype=torch.bfloat16
        )
        graph.output(_append_gated_residual_copy_back(graph, attention_output))

    mock_cuda_target(monkeypatch, (12, 0))
    mock_triton_driver(monkeypatch, backend="cuda", arch=120)
    passes = compile_options(linear=False, attention=True, residual_updates=True)[
        "post_grad_custom_pre_pass"
    ]
    assert isinstance(passes, tuple)
    for compiler_pass in passes:
        compiler_pass(graph, is_inference=True)

    target = (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_residual_add_.default
    )
    calls = [
        node
        for node in graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.auto_functionalized
        and node.args[0] is target
    ]
    assert len(calls) == 1
    assert (
        torch.ops.piper_kernels.nvfp4_sparse_piper_projected_query_attention_output.default
        not in {node.target for node in graph.nodes}
    )
    graph.lint()
