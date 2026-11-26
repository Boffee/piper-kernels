"""Run the RDNA4 hardware regression suite with an existing ROCm environment."""

import json
import os
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MIN_TRITON_VERSION = (3, 9)

# Use this checkout even when the environment has another Piper wheel installed.
sys.path.insert(0, str(_ROOT / "src"))

_SUITE = (
    "tests/attention/piper_attention/test_amd.py",
    "tests/attention/piper_attention/test_kv_cache.py",
    "tests/attention/piper_attention/test_kv_decode.py",
    "tests/attention/test_gqa.py::test_dense_gqa_matches_repeated_kv",
    "tests/attention/sparse_piper_attention/test_amd_fragments.py",
    "tests/conv3d/convrot/int8",
    "tests/specializations/minimax_h3_vae/test_conv3d_compile.py",
    "tests/linear/convrot/int8/test_amd.py",
    "tests/linear/convrot/int8/test_amd_residual.py",
    "tests/linear/convrot/int8/test_paired_projection.py",
    "tests/linear/convrot/int8/test_splitk.py",
    "tests/linear/convrot/int8/test_static_input.py",
    "tests/linear/convrot/int8/test_trellis_linear.py",
    "tests/linear/nvfp4/test_w4a16.py",
    "tests/moe/test_moe.py",
    "tests/trellis_format/test_decode.py",
    "tests/weights/convrot/int8/test_trellis.py",
    "tests/fusions/ffn",
    "tests/fusions/convrot_int8_sparse_piper/test_static.py",
    "tests/attention/test_gqa.py::test_sparse_gqa_matches_repeated_kv",
    "tests/attention/test_gqa.py::test_sparse_gqa_compile_and_prepared_storage",
    "tests/attention/test_gqa.py::test_gqa_minmax_score_kernel_preserves_query_heads",
    "tests/attention/sparse_piper_attention/test_dispatch.py",
    "tests/attention/sparse_piper_attention/test_amd_attention.py",
    "tests/fusions/convrot_int8_sparse_piper/test_compile.py"
    "::test_fused_projection_reuses_one_dynamic_shape_route_capacity_graph",
    "tests/fusions/convrot_int8_sparse_piper/test_compile.py"
    "::test_fused_coarse_projection_reuses_one_dynamic_shape_graph",
    "tests/fusions/convrot_int8_sparse_piper/test_compile.py"
    "::test_attention_output_fusion_reuses_one_dynamic_shape_graph",
    "tests/fusions/convrot_int8_piper/test_query.py",
    "tests/fusions/convrot_int8_piper/test_context.py",
    "tests/fusions/convrot_int8_piper/test_amd.py",
    "tests/fusions/convrot_int8_piper/test_output.py",
    "tests/fusions/convrot_int8_piper/test_residual_add.py",
    "tests/fusions/residual_updates",
)


def _check_environment() -> None:
    import torch  # noqa: PLC0415
    import triton  # noqa: PLC0415

    from piper_kernels._triton.targets import AcceleratorTarget  # noqa: PLC0415
    from piper_kernels.attention.piper_attention import _backend as dense  # noqa: PLC0415
    from piper_kernels.attention.sparse_piper_attention import (  # noqa: PLC0415
        _backend as attention,
    )
    from piper_kernels.conv3d.convrot.int8 import _ops as convolution  # noqa: PLC0415
    from piper_kernels.fusions.convrot_int8_sparse_piper import _backend as fusion  # noqa: PLC0415
    from piper_kernels.linear.convrot.int8 import _backend as linear  # noqa: PLC0415

    if (
        sys.platform not in ("linux", "win32")
        or torch.version.hip is None
        or not torch.cuda.is_available()
    ):
        raise SystemExit(
            "ROCm regressions require Linux or Windows ROCm PyTorch and a visible RDNA4 GPU."
        )
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not target.is_amd_hip or not target.is_architecture("gfx1200", "gfx1201"):
        raise SystemExit(f"ROCm regressions require RDNA4 (gfx1200/gfx1201), got {target}.")
    triton_version = tuple(int(part) for part in re.findall(r"\d+", triton.__version__)[:2])
    if triton_version < _MIN_TRITON_VERSION:
        raise SystemExit(
            f"ROCm regressions require Triton 3.9 or newer, got {triton.__version__}. "
            "See docs/environments.md."
        )
    probe = torch.empty(0, device="cuda")
    if dense.select_backend(target) is None:
        raise SystemExit("ROCm regressions require a native dense Piper backend.")
    if convolution._select_native_backend(probe) is None:
        raise SystemExit("ROCm regressions require a native ConvRot INT8 Conv3D backend.")
    if linear.select_linear_backend(probe) is None or fusion.select_output_backend(probe) is None:
        raise SystemExit("ROCm regressions require native INT8 linear and sparse output backends.")
    for head_dim in (64, 128):
        query = torch.empty(0, 0, 0, head_dim, device="cuda")
        if (
            attention.select_attention_backend(query) is None
            or fusion.select_projection_backend(probe, head_dim=head_dim) is None
        ):
            raise SystemExit(f"ROCm regressions require native D{head_dim} attention/projections.")
    environment = {
        "checkout": str(_ROOT),
        "python": platform.python_version(),
        "platform": sys.platform,
        "executable": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "pytest": version("pytest"),
        "gpu": torch.cuda.get_device_name(),
        "architecture": target.architecture,
    }
    print(json.dumps(environment, indent=2), flush=True)  # noqa: T201


def main() -> int:
    _check_environment()
    import pytest  # noqa: PLC0415

    os.chdir(_ROOT)
    # Serial execution bounds VRAM and isolates tests that count compiled kernels.
    # Clearing addopts also works in provisioned environments without pytest-xdist.
    return int(pytest.main(["-o", "addopts=", "-m", "gpu", "-ra", *_SUITE, *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
