"""H3 graph fusion around prequantized convolution modules."""

import pytest
import torch
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
    def __init__(self, *, framewise=True, shared=False, padding="reflect_right"):
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
        output = self.conv(x) + residual
        return (output, activated) if self.shared else output


def _args(padding="reflect_right", device="cpu"):
    size = 2 if padding == "reflect_right" else 4
    return (
        torch.randn(2, 64, 3, 4, 4, device=device, dtype=torch.float16),
        torch.randn(2, 64, 3, size, size, device=device, dtype=torch.float16),
    )


@pytest.mark.parametrize("padding", ["reflect", "reflect_right"])
def test_export_fuses_framewise_norm_silu_padding_and_residual(padding):
    block = _Block(padding=padding)
    args = _args(padding)
    with torch.no_grad():
        graph_module = torch._dynamo.export(block, assume_static_by_default=True)(
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


@pytest.mark.parametrize("options", [{"framewise": False}, {"shared": True}])
def test_pass_preserves_temporal_group_norm_and_shared_activation(options):
    block = _Block(**options)
    args = _args()
    with torch.no_grad():
        graph_module = torch._dynamo.export(block, assume_static_by_default=True)(
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
def test_fullgraph_compiled_module_uses_live_weight_and_activation_scale(device):
    block = _Block().to(device=device)
    args = _args(device=device)
    options = minimax_h3_vae_convrot_int8_conv3d_compile_options()
    compiled = torch.compile(block, fullgraph=True, options=options)
    with torch.no_grad():
        actual = compiled(*args)
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
