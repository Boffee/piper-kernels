"""Device and stream integration tests requiring the optional linear dependencies."""

import pytest
import torch

from piper_kernels._triton import runtime
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import triton as piper_attention
from piper_kernels.attention.sage_attention_2pp import triton as sage_attention
from piper_kernels.gguf import GGUFQuantizationType
from piper_kernels.linear.convrot.int8 import ConvRotInt8Tensor, _backend, _generic
from piper_kernels.linear.convrot.int8._generic import triton as generic
from piper_kernels.linear.convrot.nvfp4 import triton as convrot_nvfp4
from piper_kernels.linear.nvfp4 import triton as nvfp4

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA or ROCm"),
]


@pytest.fixture(params=[0, 1], ids=["current-gpu", "noncurrent-gpu"])
def execution_device(request):
    if torch.cuda.device_count() <= request.param:
        pytest.skip("requires a second GPU")
    device = torch.device("cuda", request.param)
    if not runtime.supports_device(device):
        pytest.skip("requires a supported Triton device")
    return device


@pytest.mark.parametrize(
    "operation", ["gguf_fused", "gguf_tiled", "prepare", "add", "addmm", "paired_projection"]
)
def test_operations_use_operand_device_and_its_current_stream(
    monkeypatch, execution_device, operation
):
    if operation == "gguf_tiled":
        monkeypatch.setattr(generic, "select_conversion_chunks", lambda target, width: None)

    with torch.cuda.device(0):
        original_stream = torch.cuda.current_stream(execution_device)
        stream = torch.cuda.Stream(device=execution_device)
        with torch.cuda.stream(stream):
            value = torch.randn(129, 256, device=execution_device, dtype=torch.bfloat16)
            qdata, scale = _generic.prepare_input(value, 256)
            update = torch.randn_like(value) * 0.01
            mat1, mat2 = value[:, :16].contiguous(), value[:16].contiguous()
            backend = _backend.select_linear_backend(value)
            if operation == "paired_projection" and backend is None:
                pytest.skip("requires a tuned INT8 matrix backend")

            def run():
                if operation.startswith("gguf_"):
                    weight = ConvRotInt8Tensor.from_gguf(
                        value.float(), quant_type=GGUFQuantizationType.F32, group_size=256
                    )
                    return weight.qdata, weight.scale
                if operation == "prepare":
                    return _generic.prepare_input(value, 256)
                if operation in ("add", "addmm"):
                    output, output_scale = qdata.clone(), scale.clone()
                    if operation == "add":
                        _generic.add_(output, output_scale, update, 256, 0.5)
                    else:
                        _generic.addmm_(output, output_scale, mat1, mat2, 256, 1.0, 0.5)
                    return output, output_scale
                assert backend is not None
                prepared = backend.prepare_input(value, 256)
                # Exercise paired projection, row tails, and caller-owned strided output.
                storage = torch.full((129, 260), 7.0, device=execution_device)
                output = storage[:, 1:259]
                result = backend.linear_prepared(
                    *prepared,
                    qdata,
                    scale,
                    None,
                    torch.float32,
                    out=output,
                    second_projection=(qdata, scale, None),
                )
                assert result is output
                return (storage,)

            expected = run()
            with torch.cuda.device(0):
                caller_stream = torch.cuda.current_stream()
                actual = run()
                assert torch.cuda.current_device() == 0
                assert torch.cuda.current_stream() == caller_stream
                assert torch.cuda.current_stream(execution_device) == stream
            stream.synchronize()
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)
            if operation == "paired_projection":
                assert (actual[0][:, [0, -1]] == 7).all()
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(execution_device) == original_stream


@pytest.mark.parametrize(
    "operation", ["piper_attention", "sage_attention", "nvfp4", "convrot_nvfp4"]
)
def test_nvidia_attention_and_preparation_use_operand_context(execution_device, operation):
    if not AcceleratorTarget.from_device(execution_device).is_architecture("sm120"):
        pytest.skip("requires NVIDIA SM120")
    with torch.cuda.device(0):
        original_stream = torch.cuda.current_stream(execution_device)
        stream = torch.cuda.Stream(device=execution_device)
        with torch.cuda.stream(stream):
            value = torch.randn(1, 2, 258, 128, device=execution_device, dtype=torch.bfloat16)
            key, query = torch.randn_like(value), torch.randn_like(value)

            def run():
                if operation == "piper_attention":
                    return (
                        piper_attention._run_piper_attention(query, key, value, 128**-0.5, False),
                    )
                if operation == "sage_attention":
                    return (
                        sage_attention._run_sage_attention_2pp(query, key, value, 128**-0.5, False),
                    )
                matrix = value.reshape(-1, 256)
                if operation == "convrot_nvfp4":
                    return convrot_nvfp4.prepare_dynamic(matrix, 256)
                return nvfp4.prepare_static(matrix, nvfp4.dynamic_scale(matrix))

            expected = run()
            with torch.cuda.device(0):
                caller_stream = torch.cuda.current_stream()
                actual = run()
                assert torch.cuda.current_device() == 0
                assert torch.cuda.current_stream() == caller_stream
                assert torch.cuda.current_stream(execution_device) == stream
            stream.synchronize()
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result.float(), reference.float(), rtol=0, atol=0)
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(execution_device) == original_stream


def test_execution_exception_restores_caller_device_and_stream(execution_device):
    def fail():
        assert torch.cuda.current_device() == execution_device.index
        raise RuntimeError("execution failed")

    with torch.cuda.device(0):
        caller_stream = torch.cuda.current_stream()
        target_stream = torch.cuda.current_stream(execution_device)
        with (
            pytest.raises(RuntimeError, match="execution failed"),
            runtime.device_context(execution_device),
        ):
            fail()
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream() == caller_stream
        assert torch.cuda.current_stream(execution_device) == target_stream
