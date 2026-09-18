"""Inductor capture pass shared by compiler-integration tests."""

import torch
from torch._inductor.custom_graph_pass import CustomInferenceAwareGraphPass


class TargetCapturePass(CustomInferenceAwareGraphPass):
    """Record the call targets of each post-grad inference graph.

    The pass mutates per-test state, so it must run on every compile. Returning
    no uuid makes Inductor bypass its FX graph cache for the graph. A random
    uuid would also force the pass to run, but it misses on every run and
    writes a cache entry that nothing can ever read.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.targets: list[object] = []

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        assert is_inference
        self.calls += 1
        self.targets = [node.target for node in graph.nodes if node.op == "call_function"]

    @property
    def target_names(self) -> list[str]:
        """Qualified names of the recorded targets, for string matching."""
        return [str(target) for target in self.targets]

    def uuid(self) -> None:
        return None
