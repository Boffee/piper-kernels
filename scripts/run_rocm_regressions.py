"""Run the RDNA4 hardware regression suite with an existing ROCm environment."""

import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Use this checkout even when the environment has another Piper wheel installed.
sys.path.insert(0, str(_ROOT / "src"))

_SUITE = (
    "tests/attention/piper_attention/test_amd.py",
    "tests/attention/test_gqa.py::test_dense_gqa_matches_repeated_kv",
    "tests/attention/sparse_piper_attention/test_amd_fragments.py",
    "tests/conv3d/convrot/int8",
    "tests/specializations/minimax_h3_vae/test_amd_linear.py",
    "tests/specializations/minimax_h3_vae/test_conv3d_compile.py",
    "tests/linear/convrot/int8/test_amd.py",
    "tests/linear/convrot/int8/test_static_input.py",
    "tests/fusions/convrot_int8_gelu_ffn",
    "tests/fusions/convrot_int8_swiglu_ffn/test_static.py",
    "tests/fusions/convrot_int8_sparse_piper/test_static.py",
    "tests/attention/test_gqa.py::test_sparse_gqa_matches_repeated_kv",
    "tests/attention/test_gqa.py::test_sparse_gqa_compile_and_prepared_storage",
    "tests/attention/test_gqa.py::test_gqa_minmax_score_kernel_preserves_query_heads",
    "tests/attention/sparse_piper_attention/test_amd_attention.py",
    "tests/fusions/convrot_int8_sparse_piper/test_compile.py"
    "::test_fused_projection_reuses_one_dynamic_shape_route_capacity_graph",
    "tests/fusions/convrot_int8_sparse_piper/test_compile.py"
    "::test_fused_coarse_projection_reuses_one_dynamic_shape_graph",
    "tests/fusions/convrot_int8_sparse_piper/test_compile.py"
    "::test_attention_output_fusion_reuses_one_dynamic_shape_graph",
)


def _check_environment() -> None:
    import torch  # noqa: PLC0415
    import triton  # noqa: PLC0415

    from piper_kernels._triton.targets import AcceleratorTarget  # noqa: PLC0415
    from piper_kernels.attention.piper_attention import _backend as dense  # noqa: PLC0415
    from piper_kernels.attention.sparse_piper_attention import (  # noqa: PLC0415
        _backend as attention,
    )
    from piper_kernels.conv3d.convrot.int8 import _backend as convolution  # noqa: PLC0415
    from piper_kernels.fusions.convrot_int8_sparse_piper import _backend as fusion  # noqa: PLC0415
    from piper_kernels.linear.convrot.int8 import _backend as linear  # noqa: PLC0415

    if sys.platform != "linux" or torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("ROCm regressions require Linux ROCm PyTorch and a visible RDNA4 GPU.")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not target.is_amd_hip or not target.is_architecture("gfx1200", "gfx1201"):
        raise SystemExit(f"ROCm regressions require RDNA4 (gfx1200/gfx1201), got {target}.")
    probe = torch.empty(0, device="cuda")
    if dense.select_backend(target) is None:
        raise SystemExit("ROCm regressions require a native dense Piper backend.")
    if convolution.select_backend(probe) is None:
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
        "executable": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "torchao": version("torchao"),
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
