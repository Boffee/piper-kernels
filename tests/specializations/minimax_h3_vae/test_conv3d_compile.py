"""H3 graph fusion around prequantized convolution modules."""

import pytest
import torch
from _compile_capture import TargetCapturePass
from torch.nn import functional

from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d, group_norm_silu_conv3d
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_compile_options,
    minimax_h3_vae_convrot_int8_conv3d_compile_options,
)
from piper_kernels.specializations.minimax_h3_vae.conv3d._compile import compile_pass, fuse_conv3d
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor

_CONV = torch.ops.piper_kernels.convrot_int8_conv3d.default
_FUSED = torch.ops.piper_kernels.convrot_int8_group_norm_silu_conv3d.default


class _Block(torch.nn.Module):
    def __init__(self, *, framewise=True, shared=False, padding="reflect_right", standard=False):
        super().__init__()
        self.norm = torch.nn.GroupNorm(8, 64, eps=1e-6).half().requires_grad_(False)
        self.conv = ConvRotInt8Conv3d(
            ConvRotInt8Tensor.from_quantized(
                torch.randint(-8, 8, (64, 3, 3, 3, 64), dtype=torch.int8),
                torch.full((64,), 0.005),
                group_size=64,
                logical_dtype=torch.float16,
                act_per_tensor_scale=torch.tensor(0.02),
            ),
            stride=(1, 2, 2) if padding == "reflect_right" else (1, 1, 1),
            padding="none" if padding == "reflect_right" else padding,
        )
        self.standard = standard
        if standard:
            weight, stride = self.conv.weight, self.conv.stride
            self.conv = torch.nn.Conv3d(64, 64, 3, stride=stride, bias=False, device="meta")
            self.conv.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.framewise = framewise
        self.shared = shared
        self.padding = padding

    def forward(self, x, residual):
        batch, channels, frames, height, width = x.shape
        if self.framewise:
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(batch * frames, channels, 1, height, width)
        x = self.norm(x)
        if self.framewise:
            x = x.view(batch, frames, channels, height, width)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
        activated = functional.silu(x)
        x = activated
        if self.padding == "reflect_right":
            x = functional.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        if self.standard:
            if self.padding == "reflect":
                x = functional.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
            x = functional.pad(x, (0, 0, 0, 0, 2, 0))
        output = self.conv(x) + residual
        return (output, activated) if self.shared else output


def _args(padding="reflect_right", device="cpu"):
    size = 2 if padding == "reflect_right" else 4
    return (
        torch.randn(2, 64, 3, 4, 4, device=device, dtype=torch.float16),
        torch.randn(2, 64, 3, size, size, device=device, dtype=torch.float16),
    )


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("padding", ["reflect", "reflect_right"])
@pytest.mark.parametrize("standard", [False, True])
def test_export_fuses_framewise_norm_silu_padding_and_residual(padding, standard, static):
    block = _Block(padding=padding, standard=standard)
    args = _args(padding)
    with torch.no_grad():
        graph_module = torch._dynamo.export(block, assume_static_by_default=static)(
            *args
        ).graph_module
        fuse_conv3d(graph_module.graph)
        graph_module.recompile()
        targets = [node.target for node in graph_module.graph.nodes]
        assert targets.count(_FUSED) == 1
        assert _CONV not in targets
        assert functional.group_norm not in targets
        assert functional.silu not in targets
        assert torch._C._nn.pad not in targets
        actual = graph_module(*args)
        expected = group_norm_silu_conv3d(
            args[0],
            block.norm.weight,
            block.norm.bias,
            8,
            1e-6,
            block.conv.weight,
            stride=block.conv.stride,
            padding=padding,
            residual=args[1],
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("options", [{"framewise": False}, {"shared": True}])
def test_pass_preserves_temporal_group_norm_and_shared_activation(options, static):
    block = _Block(**options)
    args = _args()
    with torch.no_grad():
        graph_module = torch._dynamo.export(block, assume_static_by_default=static)(
            *args
        ).graph_module
        fuse_conv3d(graph_module.graph)
        targets = [node.target for node in graph_module.graph.nodes]
        assert _FUSED not in targets
        assert _CONV in targets
        assert functional.group_norm in targets


def test_pass_does_not_rewrite_with_gradients_enabled():
    block = _Block()
    graph_module = torch._dynamo.export(block, assume_static_by_default=True)(*_args()).graph_module
    original = str(graph_module.graph)
    fuse_conv3d(graph_module.graph)
    assert str(graph_module.graph) == original


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("autocast_input_dtype", [None, torch.float16, torch.float32])
@pytest.mark.parametrize("standard", [False, True])
def test_fullgraph_compiled_module_uses_live_weight_and_activation_scale(
    device, autocast_input_dtype, standard, dynamic
):
    torch.compiler.reset()
    block = _Block(standard=standard).to(device=device)
    if autocast_input_dtype is not None:
        block.norm.float()
    args = _args(device=device)
    if autocast_input_dtype is not None:
        args = (args[0].to(autocast_input_dtype), args[1])
    capture = TargetCapturePass()
    options = minimax_h3_vae_convrot_int8_conv3d_compile_options(
        {"post_grad_custom_pre_pass": capture}
    )
    compiled = torch.compile(block, fullgraph=True, dynamic=dynamic, options=options)
    with (
        torch.no_grad(),
        torch.autocast(device, dtype=torch.float16, enabled=autocast_input_dtype is not None),
    ):
        actual = compiled(*args)
        assert capture.targets.count(_FUSED) == 1
        assert _CONV not in capture.targets
        expected = group_norm_silu_conv3d(
            args[0],
            block.norm.weight,
            block.norm.bias,
            8,
            1e-6,
            block.conv.weight,
            stride=block.conv.stride,
            padding="reflect_right",
            residual=args[1],
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Change the scale tensor without changing its metadata or recompiling.
        block.conv.weight.act_per_tensor_scale = torch.tensor(0.05, device=device)
        changed = compiled(*args)
        expected = group_norm_silu_conv3d(
            args[0],
            block.norm.weight,
            block.norm.bias,
            8,
            1e-6,
            block.conv.weight,
            stride=block.conv.stride,
            padding="reflect_right",
            residual=args[1],
        )
        torch.testing.assert_close(changed, expected, atol=0, rtol=0)
        assert not torch.equal(changed, actual)
        # Offload may also replace the whole managed weight.
        replacement = ConvRotInt8Tensor.from_quantized(
            torch.zeros_like(block.conv.weight.qdata),
            block.conv.weight.scale,
            group_size=64,
            logical_dtype=torch.float16,
            act_per_tensor_scale=block.conv.weight.act_per_tensor_scale,
        )
        block.conv.weight = torch.nn.Parameter(replacement, requires_grad=False)
        torch.testing.assert_close(compiled(*args), args[1], atol=0, rtol=0)
        assert capture.calls == 1


def test_compile_options_preserve_existing_pre_grad_pass_and_are_idempotent():
    def existing(graph):
        pass

    options = minimax_h3_vae_convrot_int8_conv3d_compile_options({"pre_grad_custom_pass": existing})
    options = minimax_h3_vae_convrot_int8_conv3d_compile_options(options)
    assert options["pre_grad_custom_pass"] == (existing, compile_pass)
    assert compile_pass.uuid() == compile_pass.uuid()


def test_encoder_and_decoder_options_are_independent_and_composable():
    decoder = minimax_h3_vae_convrot_int8_compile_options()
    encoder = minimax_h3_vae_convrot_int8_conv3d_compile_options()
    assert "pre_grad_custom_pass" not in decoder
    assert "post_grad_custom_pre_pass" not in encoder
    assert minimax_h3_vae_convrot_int8_conv3d_compile_options(decoder) == (
        minimax_h3_vae_convrot_int8_compile_options(encoder)
    )
    assert decoder["post_grad_custom_pre_pass"]
    assert encoder["pre_grad_custom_pass"] == (compile_pass,)


@pytest.mark.parametrize("standard", [False, True])
@pytest.mark.parametrize("tile_view", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
            ],
        ),
    ],
)
def test_dynamic_fusions_reuse_graphs_across_shapes_and_strides(device, tile_view, standard):
    torch.compiler.reset()
    block = _Block(standard=standard).to(device=device)
    capture = TargetCapturePass()
    options = minimax_h3_vae_convrot_int8_conv3d_compile_options(
        {"post_grad_custom_pre_pass": capture}
    )
    compiled = torch.compile(block, fullgraph=True, dynamic=True, options=options)

    # Export the established static rewrite once as the numerical oracle.
    # Its single fused operator accepts every shape exercised below.
    with torch.no_grad():
        baseline = torch._dynamo.export(block, assume_static_by_default=True)(
            *_args(device=device)
        ).graph_module
        fuse_conv3d(baseline.graph)
        baseline.recompile()

    def check(batch, frames, height, width, border):
        canvas = torch.randn(
            batch,
            64,
            frames,
            height + border,
            width + border,
            device=device,
            dtype=torch.float16,
        )
        x = canvas[:, :, :, :height, :width]
        if not tile_view:
            x = x.contiguous()
        residual = torch.randn(
            batch, 64, frames, height // 2, width // 2, device=device, dtype=torch.float16
        )
        expected = baseline(x, residual)
        actual = compiled(x, residual)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert capture.targets.count(_FUSED) == 1
        assert _CONV not in capture.targets
        assert torch.ops.aten.add.Tensor not in capture.targets
        assert torch.ops.aten.reflection_pad3d.default not in capture.targets

    with torch.no_grad(), torch._dynamo.config.patch(recompile_limit=8):
        # Singleton dimensions and independent tile strides have legitimate
        # separate guards. Warm image/video and single/multiple batch families.
        for batch, frames in [(2, 3), (1, 3), (2, 1), (1, 1)]:
            check(batch, frames, 8, 10, 4)
        warmed_calls = capture.calls
        assert 1 <= warmed_calls <= 4
        for index in range(1, 5):
            for batch, frames in [(3, 5), (1, 7), (3, 1), (1, 1)]:
                check(batch, frames, 8 + 2 * index, 10 + 4 * index, 4 + 2 * index)
        assert capture.calls == warmed_calls


@pytest.mark.parametrize("static", [False, True])
def test_residual_broadcast_stays_outside_fused_convolution(static):
    block = _Block()
    x, residual = _args()
    residual = residual[:, :, :, :1, :1]
    with torch.no_grad():
        gm = torch._dynamo.export(block, assume_static_by_default=static)(x, residual).graph_module
        fuse_conv3d(gm.graph)
        gm.recompile()
        fused = next(node for node in gm.graph.nodes if node.target == _FUSED)
        assert fused.args[-1] is None
        expected = (
            group_norm_silu_conv3d(
                x,
                block.norm.weight,
                block.norm.bias,
                8,
                1e-6,
                block.conv.weight,
                stride=block.conv.stride,
                padding="reflect_right",
            )
            + residual
        )
        torch.testing.assert_close(gm(x, residual), expected, atol=0, rtol=0)
