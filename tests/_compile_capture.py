"""Inductor capture pass shared by compiler-integration tests."""

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass


class TargetCapturePass(CustomInferenceAwareGraphPass):
    """Record the call targets of each post-grad inference graph.

    By default the pass returns no uuid, which makes Inductor bypass its FX
    graph cache so the pass runs on every compile. A random uuid would also
    force it to run, but it misses on every run and writes a cache entry that
    nothing can ever read.

    A test that exercises the FX graph cache itself passes a stable
    ``cache_key`` instead. The pass then joins the cache key and a cache hit
    skips it, so such a test must not rely on what the pass records.
    """

    def __init__(self, *, cache_key: bytes | None = None) -> None:
        self.calls = 0
        self.targets: list[object] = []
        self._cache_key = cache_key

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    @property
    def target_names(self) -> list[str]:
        """Qualified names of the recorded targets, for string matching."""
        return [str(target) for target in self.targets]

    def uuid(self) -> bytes | None:
        return self._cache_key
