"""Tests for automatic ConvRot-to-sparse-Piper graph fusion."""

import uuid

import pytest
import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass
from torch.nn import functional as F  # noqa: N812

from piper_kernels import (
    SparsePiperAttentionPlan,
    prepare_sparse_piper_attention_plan,
    sparse_piper_attention,
)
from piper_kernels.fusions.convrot_sparse_piper import (
    convrot_sparse_piper_compile_options,
)
from piper_kernels.fusions.convrot_sparse_piper._compile import (
    compile_pass as fusion_compile_pass,
)
from piper_kernels.linear.convrot import ConvRotInt8Tensor, convrot_int8_compile_options
from piper_kernels.linear.convrot.int8._compile import compile_pass as convrot_compile_pass

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
        self.plan = prepare_sparse_piper_attention_plan(
            torch.full((self.heads,), 2, device="cuda", dtype=torch.int32),
        )

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = self._norm_rope(self.query(hidden_states), self.query_norm)
        key = self._norm_rope(self.key(hidden_states), self.key_norm)
        value = self.value(hidden_states).view(
            self.batch,
            self.sequence_length,
            self.heads,
            self.head_dim,
        )
        return sparse_piper_attention(
            query,
            key,
            value,
            self.plan,
            sparse_key_blocks=self.sparse_key_blocks,
        )


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
        plan: SparsePiperAttentionPlan,
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
        return sparse_piper_attention(
            query,
            key,
            value,
            plan,
            sparse_key_blocks=sparse_key_blocks,
        )


def test_compile_options_install_fusion_before_convrot() -> None:
    options = convrot_sparse_piper_compile_options({"max_autotune": True})

    assert options["max_autotune"] is True
    assert options[_POST_GRAD_PRE_PASS] == (fusion_compile_pass, convrot_compile_pass)


def test_compile_options_reorder_existing_convrot_pass_without_duplication() -> None:
    base_options = convrot_int8_compile_options()
    options = convrot_sparse_piper_compile_options(base_options)
    reapplied = convrot_sparse_piper_compile_options(options)

    assert options[_POST_GRAD_PRE_PASS] == (fusion_compile_pass, convrot_compile_pass)
    assert reapplied == options


def test_compile_options_preserve_unrelated_pass_order() -> None:
    before_convrot = object()
    after_convrot = object()
    options = convrot_sparse_piper_compile_options(
        {
            _POST_GRAD_PRE_PASS: (
                before_convrot,
                convrot_compile_pass,
                after_convrot,
            )
        }
    )

    assert options[_POST_GRAD_PRE_PASS] == (
        before_convrot,
        fusion_compile_pass,
        convrot_compile_pass,
        after_convrot,
    )
    assert convrot_sparse_piper_compile_options(options) == options


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
    options = convrot_sparse_piper_compile_options()
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
        capture.targets.count(torch.ops.piper_kernels.convrot_sparse_piper_project_query.default)
        == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_sparse_piper_project_key.default) == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_int8_dequantized_input_mean.default)
        == 1
    )
    assert (
        capture.targets.count(torch.ops.piper_kernels.convrot_sparse_piper_project_value.default)
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
def test_cuda_fused_projection_reuses_one_dynamic_shape_route_capacity_graph() -> None:
    torch.manual_seed(709)
    model = _DynamicSparseProjectionAttention().eval()
    capture = _TargetCapturePass()
    options = convrot_sparse_piper_compile_options()
    compiler_passes = options[_POST_GRAD_PRE_PASS]
    assert isinstance(compiler_passes, tuple)
    options[_POST_GRAD_PRE_PASS] = (*compiler_passes, capture)
    materialized = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
    )
    compiled = torch.compile(
        model,
        dynamic=True,
        fullgraph=True,
        options=options,
    )

    with torch.no_grad():
        cases = (
            (192, 2, (1, 1)),
            (256, 2, (1, 2)),
            (256, 3, (2, 2)),
            (256, 3, (2, 3)),
            (1024, 8, (8, 8)),
            (1024, 9, (9, 9)),
        )
        for sequence, sparse_key_blocks, keep_values in cases:
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
            plan = prepare_sparse_piper_attention_plan(
                torch.tensor(keep_values, device="cuda", dtype=torch.int32)
            )
            expected = materialized(hidden_states, cos, sin, sparse_key_blocks, plan)
            output = compiled(
                hidden_states,
                cos,
                sin,
                sparse_key_blocks,
                plan,
            )
            assert output.shape == (model.batch, sequence, model.heads, model.head_dim)
            assert bool(torch.isfinite(output).all())
            difference = output.float() - expected.float()
            relative_l2 = difference.norm() / expected.float().norm()
            assert relative_l2 < 0.025, (
                sequence,
                sparse_key_blocks,
                keep_values,
                relative_l2.item(),
            )

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
    options = convrot_sparse_piper_compile_options()
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
