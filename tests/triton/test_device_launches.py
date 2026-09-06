"""Audit native launch ownership without importing optional linear dependencies."""

import ast
from pathlib import Path

import piper_kernels


def test_every_native_launch_and_descriptor_has_an_explicit_device_context():
    root = Path(piper_kernels.__file__).parent
    unguarded = []
    native_calls = []

    class Audit(ast.NodeVisitor):
        depth = 0

        def visit_FunctionDef(self, node):
            if any(isinstance(d, ast.Attribute) and d.attr == "jit" for d in node.decorator_list):
                return
            self.generic_visit(node)

        def visit_With(self, node):
            guarded = any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func) in ("device_context", "torch.cuda.device")
                for item in node.items
            )
            self.depth += guarded
            self.generic_visit(node)
            self.depth -= guarded

        def visit_Call(self, node):
            if isinstance(node.func, ast.Subscript) or (
                isinstance(node.func, ast.Name)
                and node.func.id in ("TensorDescriptor", "install_uint8_int8_dot_hook")
            ):
                location = f"{path.relative_to(root)}:{node.lineno}"
                native_calls.append(location)
                if not self.depth:
                    unguarded.append(location)
            self.generic_visit(node)

    for path in root.rglob("*.py"):
        Audit().visit(ast.parse(path.read_text(encoding="utf-8")))
    assert native_calls
    assert not unguarded, unguarded
