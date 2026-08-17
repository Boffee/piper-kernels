"""Tests for the rotation shared by ConvRot storage formats."""

import hashlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from piper_kernels.linear.convrot._rotation import build_hadamard
from piper_kernels.linear.convrot._torch_compat import is_fake_mode_active


@pytest.mark.parametrize(
    ("size", "digest"),
    [
        (16, "9b46fb3c57a096bd73a10d9a089ce835fd22c80fc252f10576658e3f958d72ac"),
        (64, "42759176f7fc530ed4a0ae8acbfcf1bb3c4f86fc34ab1b89e3e50a4ed44383af"),
        (256, "de75fba5a830070acfbfec9cf3fba8a70ef4bfc05e9e6e4dc1fa9bb994f3bc5d"),
    ],
)
def test_regular_hadamard_order_matches_comfy_kitchen(size: int, digest: str) -> None:
    """Pin the full sign pattern; orthogonality alone cannot catch a basis reorder."""
    matrix = build_hadamard(size)
    signs = (matrix * size**0.5).to(torch.int8).view(torch.uint8).flatten().tolist()
    assert hashlib.sha256(bytes(signs)).hexdigest() == digest
    assert torch.equal(matrix @ matrix.T, torch.eye(size))


def test_fake_mode_detection_isolated_behind_compatibility_helper() -> None:
    assert not is_fake_mode_active()
    with FakeTensorMode():
        assert is_fake_mode_active()
