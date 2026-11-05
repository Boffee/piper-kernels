"""Temporary CI diagnostic: compare compiled and eager FP16-autocast ConvRot INT8 Conv3D."""

import torch
from torch.nn import functional

from piper_kernels.conv3d.convrot.int8 import convrot_int8_conv3d_compile_options
from piper_kernels.weights.convrot.int8 import ConvRotInt8Tensor


def _weight() -> ConvRotInt8Tensor:
    return ConvRotInt8Tensor.from_quantized(
        torch.randint(-8, 8, (16, 3, 3, 3, 64), dtype=torch.int8),
        torch.full((16,), 0.003),
        group_size=64,
        logical_dtype=torch.float16,
        act_per_tensor_scale=torch.tensor(0.025),
    )


def main() -> None:
    from torch._inductor import cpu_vec_isa

    print("vec isa:", cpu_vec_isa.pick_vec_isa(), "threads:", torch.get_num_threads())
    for scope in ("outer", "inner", "disabled"):

        def run(x, weight, bias, scope=scope):
            def convolution():
                padded = functional.pad(x, (1, 1, 1, 1, 0, 0), mode="reflect")
                padded = functional.pad(padded, (0, 0, 0, 0, 2, 0))
                return functional.conv3d(padded, weight, bias)

            if scope == "outer":
                return convolution()
            with torch.autocast("cpu", dtype=torch.float16, enabled=scope == "inner"):
                return convolution()

        torch.manual_seed(0)
        x, bias, weight = torch.randn(1, 64, 2, 4, 6), torch.randn(16), _weight()
        backends = {
            "inductor+piper passes": {"options": convrot_int8_conv3d_compile_options()},
            "inductor": {},
            "aot_eager": {"backend": "aot_eager"},
        }
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16, enabled=scope != "inner"):
            expected = run(x, weight, bias)
            report = {}
            for name, kwargs in backends.items():
                torch._dynamo.reset()
                actual = torch.compile(run, fullgraph=True, **kwargs)(x, weight, bias)
                difference = (actual.float() - expected.float()).abs()
                report[name] = f"max {difference.max().item():.4g}, mismatched {(difference > 0).float().mean().item():.0%}"
        print(scope, "expected", expected.dtype, f"|max| {expected.abs().max().item():.3g}", report)


if __name__ == "__main__":
    main()
