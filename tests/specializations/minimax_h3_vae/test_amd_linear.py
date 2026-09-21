"""On-device coverage of MiniMax-H3's measured RDNA4 linear schedules."""

import pytest
import torch

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.linear.convrot.int8._amd import triton as amd
from piper_kernels.specializations.minimax_h3_vae._ops import linear_prepared

_TARGET = AcceleratorTarget.from_device(torch.device("cuda")) if torch.cuda.is_available() else None

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _TARGET is None
        or not _TARGET.is_amd_hip
        or not _TARGET.is_architecture("gfx1200", "gfx1201"),
        reason="requires Linux RDNA4 ROCm",
    ),
]


def test_specialized_prepared_linear_matches_default_rdna4_execution() -> None:
    torch.manual_seed(1_797)
    input_qdata = torch.randint(-127, 128, (1_797, 2_048), device="cuda", dtype=torch.int8)
    input_scale = torch.rand(1_797, device="cuda", dtype=torch.float32) / 127
    weight_qdata = torch.randint(-127, 128, (2_048, 2_048), device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(2_048, 1, device="cuda", dtype=torch.float32) / 127

    expected = amd.linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        None,
        torch.bfloat16,
    )
    actual = linear_prepared(
        input_qdata,
        input_scale,
        weight_qdata,
        weight_scale,
        None,
        torch.bfloat16,
        [128, 128, 64, 8, 2],
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
