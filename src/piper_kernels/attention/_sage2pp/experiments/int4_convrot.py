"""INT4-range ConvRot QK experiment using an unpacked INT8 Triton dot.

The quantized values occupy the symmetric INT4 range ``[-7, 7]`` but remain
stored in INT8 tensors because Triton's public ``tl.dot`` API does not accept
packed INT4 operands.  This implementation therefore measures quantization
quality and preprocessing overhead, not native INT4 MMA performance.
"""

import torch

from piper_kernels.attention._sage2pp.backends.triton import _run_sage_attention


def triton_sage_attention_int4_convrot(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    *,
    rotation_group: int = 64,
    grouped_qk: bool = False,
) -> torch.Tensor:
    """Run Sage2++ with rotated INT4-range Q/K and canonical FP8 P/V.

    ``grouped_qk=False`` selects Sage's finer per-thread scale organization,
    which is the accuracy-oriented default for 4-bit Q/K.  Setting it to true
    evaluates the coarser per-warp/per-block organization used by the current
    SM120 8+8 kernel.
    """
    return _run_sage_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        qk_quantization_range=7,
        grouped_qk=grouped_qk,
        rotation_group=rotation_group,
    )
