"""Regenerate the independent GGUF decoder inputs and FP32 expectations."""

from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import gguf
import torch
from _fixtures import REFERENCE_PATH, finite_packed, reference_key

from piper_kernels.gguf import GGUFQuantizationType


def _save_case(
    archive: ZipFile,
    quant_type: GGUFQuantizationType,
    *,
    seed: int,
    rows: int = 2,
    features: int = 256,
) -> None:
    torch.manual_seed(seed)
    packed = finite_packed(quant_type, rows=rows, features=features).numpy()
    dense = gguf.dequantize(packed, gguf.GGMLQuantizationType(quant_type))
    dense = dense.astype("<f4", copy=False).reshape(rows, features)
    key = reference_key(quant_type, rows=rows, features=features, seed=seed)
    for name, array in (("packed", packed), ("decoded-fp32", dense)):
        archive.writestr(
            ZipInfo(f"{key}/{name}"), array.tobytes(), compress_type=ZIP_DEFLATED, compresslevel=9
        )


def main() -> None:
    """Capture the inputs and decoded values for each numerical case."""
    with ZipFile(REFERENCE_PATH, "w") as archive:
        for quant_type in GGUFQuantizationType:
            _save_case(archive, quant_type, seed=820 + int(quant_type))
            _save_case(archive, quant_type, seed=820 + int(quant_type), rows=3, features=1280)
            _save_case(archive, quant_type, seed=840 + int(quant_type))
        for quant_type, features in (
            (GGUFQuantizationType.F32, 16),
            (GGUFQuantizationType.Q4_0, 32),
            (GGUFQuantizationType.Q4_K, 768),
            (GGUFQuantizationType.Q5_K, 5376),
            (GGUFQuantizationType.Q6_K, 32768),
            (GGUFQuantizationType.IQ4_XS, 65792),
        ):
            _save_case(archive, quant_type, seed=75, rows=5, features=features)
        _save_case(archive, GGUFQuantizationType.BF16, seed=870, rows=128)


if __name__ == "__main__":
    main()
