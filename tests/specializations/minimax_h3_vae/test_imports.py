"""Encoder and decoder consumers load only their own operator families."""

import subprocess
import sys

import pytest


def test_decoder_options_and_calibration_do_not_import_conv3d_operators():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_compile_options,
)

minimax_h3_vae_convrot_int8_compile_options()
assert 'piper_kernels.specializations.minimax_h3_vae.conv3d' not in sys.modules
assert 'piper_kernels.conv3d' not in sys.modules

from piper_kernels.specializations.minimax_h3_vae.conv3d import P995_ACTIVATION_SCALES
assert len(P995_ACTIVATION_SCALES) == 29
assert 'piper_kernels.conv3d' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("triton_available", [True, False])
def test_conv3d_weight_layer_and_compile_options_do_not_import_linear(triton_available):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import importlib.abc
import importlib.util
import sys

if sys.argv[1] == 'False':
    original_import = builtins.__import__
    original_find_spec = importlib.util.find_spec

    def without_triton(name, *args, **kwargs):
        if name == 'triton' or name.startswith('triton.'):
            raise ModuleNotFoundError('Triton intentionally unavailable', name='triton')
        return original_import(name, *args, **kwargs)

    def find_spec(name, *args, **kwargs):
        if name == 'triton' or name.startswith('triton.'):
            return None
        return original_find_spec(name, *args, **kwargs)

    builtins.__import__ = without_triton
    importlib.util.find_spec = find_spec

import torch

class BlockLinear(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'piper_kernels.linear' or fullname.startswith('piper_kernels.linear.'):
            raise AssertionError(f'Conv3D imported {fullname}')

sys.meta_path.insert(0, BlockLinear())

from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor
from piper_kernels.conv3d.convrot.int8 import ConvRotInt8Conv3d
from piper_kernels.specializations.minimax_h3_vae import (
    minimax_h3_vae_convrot_int8_conv3d_compile_options,
)

with torch.inference_mode():
    weight = ConvRotInt8Tensor.from_hp(
        torch.ones(8, 64, 3, 3, 3, dtype=torch.float16),
        group_size=64,
        act_per_tensor_scale=torch.tensor(0.02),
    )
    assert weight.dequantize().shape == (8, 64, 3, 3, 3)
    layer = ConvRotInt8Conv3d(weight, padding='reflect')
    output = layer(torch.ones(1, 64, 2, 4, 4, dtype=torch.float16))
    assert output.shape == (1, 8, 2, 4, 4)
    assert torch.isfinite(output).all()

options = minimax_h3_vae_convrot_int8_conv3d_compile_options()
for graph_pass in options['pre_grad_custom_pass']:
    assert graph_pass.uuid()
assert not any(name.startswith('piper_kernels.linear') for name in sys.modules)
if sys.argv[1] == 'False':
    assert 'triton' not in sys.modules
""",
            str(triton_available),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
