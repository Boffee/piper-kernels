"""Tests for the Triton ConvRot rotation shared by weights and activations."""

import pytest
import torch

from piper_kernels.weights.convrot import triton as rotation_triton
from piper_kernels.weights.convrot._rotation import rotate_groups
from piper_kernels.weights.convrot.int8 import triton as int8_weight_triton


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_matches_gpu_reference(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(41 + group_size)
    activation = torch.randn(5, 2 * group_size, dtype=dtype, device="cuda")
    actual = torch.empty_like(activation)

    rotation_triton.rotate_input(activation, actual, group_size, num_warps=4)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize("group_size", [16, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_factorized_h4_rotation_handles_rounding_boundary_values(
    group_size: int,
    dtype: torch.dtype,
) -> None:
    one = torch.tensor(1.0, dtype=dtype, device="cuda")
    positive = torch.nextafter(one, torch.tensor(float("inf"), dtype=dtype, device="cuda"))
    negative = torch.nextafter(one, torch.tensor(float("-inf"), dtype=dtype, device="cuda"))
    pattern = torch.stack((one, positive, negative, -one, -positive, -negative, one, -one))
    width = 2 * group_size
    activation = pattern.repeat((width + pattern.numel() - 1) // pattern.numel())[:width]
    activation = activation.reshape(1, width)
    actual = torch.empty_like(activation)

    rotation_triton.rotate_input(activation, actual, group_size, num_warps=4)
    expected = rotate_groups(activation, group_size)

    torch.testing.assert_close(actual, expected)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU is not available")
@pytest.mark.parametrize(
    ("width", "blocks"),
    [
        (5376, (4096, 2048, 0)),
        (9216, (8192, 1024, 0)),
        (12288, (4096, 4096, 4096)),
    ],
)
@pytest.mark.parametrize("activation_fn", [None, "gelu_tanh", "swiglu"])
@pytest.mark.parametrize("static", [False, True])
def test_rotate_quantize_rows_unequal_chunks_and_folded_norm_match_unfolded(
    width: int,
    blocks: tuple[int, int, int],
    activation_fn: str | None,
    static: bool,
) -> None:
    torch.manual_seed(313)
    rows = 5
    raw_width = width * (2 if activation_fn == "swiglu" else 1)
    x = torch.randn(rows, raw_width, dtype=torch.bfloat16, device="cuda")
    input_scale = torch.tensor(0.03, dtype=torch.float32, device="cuda") if static else None
    backend = "hip" if torch.version.hip is not None else "cuda"

    folded_q = torch.empty((rows, width), dtype=torch.int8, device="cuda")
    folded_s = torch.empty(rows, dtype=torch.float32, device="cuda")
    unfolded_q = torch.empty_like(folded_q)
    unfolded_s = torch.empty_like(folded_s)

    int8_weight_triton.rotate_quantize_rows_kernel[(rows,)](
        x,
        folded_q,
        folded_s,
        width,
        block0=blocks[0],
        block1=blocks[1],
        block2=blocks[2],
        group_size=256,
        inverse_sqrt_group=256**-0.5,
        activation_fn=activation_fn,
        accelerator_backend=backend,
        gguf_quant_type=-1,
        static_scale_ptr=input_scale,
        fold_norm_into_scale=True,
        num_warps=4,
    )
    int8_weight_triton.rotate_quantize_rows_kernel[(rows,)](
        x,
        unfolded_q,
        unfolded_s,
        width,
        block0=blocks[0],
        block1=blocks[1],
        block2=blocks[2],
        group_size=256,
        inverse_sqrt_group=256**-0.5,
        activation_fn=activation_fn,
        accelerator_backend=backend,
        gguf_quant_type=-1,
        static_scale_ptr=input_scale,
        fold_norm_into_scale=False,
        num_warps=4,
    )

    assert torch.equal(folded_q, unfolded_q)
    assert torch.equal(folded_s, unfolded_s)
