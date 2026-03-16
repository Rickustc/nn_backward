from __future__ import annotations

"""Function-based autograd: operators as edges, tensors as nodes."""

from typing import Iterable, List

import numpy as np


def _shape_of(data):
    return getattr(data, "shape", ())


def _is_scalar_data(data) -> bool:
    return _shape_of(data) == ()


def _ensure_array(data):
    return np.asarray(data)


def _data_of(value):
    return value.data if isinstance(value, Value) else value


def _ensure_value(value, requires_grad: bool = False) -> "Value":
    if isinstance(value, Value):
        return value
    return Value(value, requires_grad=requires_grad)


def _format_repr(value) -> str:
    if isinstance(value, Value):
        return f"Value(data={_format_repr(value.data)}, grad={_format_repr(value.grad)})"
    return np.array_repr(value, precision=3, suppress_small=True)


def _as_value_sequence(values, name: str) -> tuple["Value", ...]:
    if isinstance(values, Value):
        return (values,)
    if isinstance(values, (list, tuple)) and all(isinstance(item, Value) for item in values):
        return tuple(values)
    raise TypeError(f"{name} must be a Value or a sequence of Value objects")


def _topological_sort(outputs: Iterable["Value"]) -> list["Value"]:
    topo: list[Value] = []
    visited: set[Value] = set()

    def build(node: Value) -> None:
        if node in visited:
            return
        visited.add(node)
        if node.creator is not None:
            for parent in node.creator.inputs:
                build(parent)
        topo.append(node)

    for output in outputs:
        build(output)
    return topo


def _accumulate_grad(current, new):
    if new is None:
        return current
    if current is None:
        return new
    if isinstance(current, Value) or isinstance(new, Value):
        return add(_ensure_value(current, requires_grad=False), _ensure_value(new, requires_grad=False))
    return np.add(current, new)


def _transpose_data(data):
    shape = _shape_of(data)
    if len(shape) != 2:
        raise ValueError("transpose only supports 2D tensors")
    return data.T


def _normalize_grad_outputs(outputs: tuple["Value", ...], grad_outputs):
    if grad_outputs is None:
        provided = (None,) * len(outputs)
    elif len(outputs) == 1:
        provided = (grad_outputs,)
    else:
        if not isinstance(grad_outputs, (list, tuple)):
            raise TypeError("grad_outputs must match outputs")
        if len(grad_outputs) != len(outputs):
            raise ValueError("grad_outputs must have the same length as outputs")
        provided = tuple(grad_outputs)

    normalized = []
    for output, grad_output in zip(outputs, provided):
        if grad_output is None:
            if not _is_scalar_data(output.data):
                raise ValueError("grad_outputs is required for non-scalar outputs")
            normalized.append(np.ones_like(output.data))
        elif isinstance(grad_output, Value):
            normalized.append(grad_output)
        else:
            normalized.append(_ensure_array(grad_output))
    return tuple(normalized)


def _run_backward(outputs, grad_outputs, create_graph: bool, accumulate_into_grad: bool):
    node_to_grad: dict[Value, object] = {}
    topo = _topological_sort(outputs)

    for output, seed in zip(outputs, grad_outputs):
        node_to_grad[output] = _accumulate_grad(node_to_grad.get(output), seed)

    for node in reversed(topo):
        upstream_grad = node_to_grad.get(node)
        if upstream_grad is None or not node.requires_grad:
            continue

        if node.creator is None:
            if accumulate_into_grad:
                node.grad = _accumulate_grad(node.grad, upstream_grad)
            continue

        grads = node.creator.backward(upstream_grad, create_graph=create_graph)
        if not isinstance(grads, tuple):
            grads = (grads,)

        for parent, grad in zip(node.creator.inputs, grads):
            if grad is None or not parent.requires_grad:
                continue
            node_to_grad[parent] = _accumulate_grad(node_to_grad.get(parent), grad)

    return node_to_grad


class Function:
    def __init__(self):
        self.saved_tensors: List["Value"] = []
        self.inputs: tuple["Value", ...] = ()
        self.needs_input_grad: tuple[bool, ...] = ()

    def save_for_backward(self, *values: "Value") -> None:
        self.saved_tensors = list(values)

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output, create_graph: bool = False):
        raise NotImplementedError

    @staticmethod
    def apply(func_cls, *inputs):
        func = func_cls()
        normalized_inputs = tuple(
            value if isinstance(value, Value) else Value(value, requires_grad=False) for value in inputs
        )
        func.inputs = normalized_inputs
        func.needs_input_grad = tuple(value.requires_grad for value in normalized_inputs)
        output = func.forward(*normalized_inputs)
        if not isinstance(output, Value):
            output = Value(output, requires_grad=any(func.needs_input_grad))
        output.requires_grad = output.requires_grad and any(func.needs_input_grad)
        output.creator = func if output.requires_grad else None
        return output


class Value:
    def __init__(self, data, requires_grad: bool = True):
        self.data = _ensure_array(data)
        self.grad = np.zeros_like(self.data)
        self.creator: Function | None = None
        self.requires_grad = requires_grad

    def backward(self, grad_output=None, create_graph: bool = False):
        if grad_output is None:
            grads = (np.ones_like(self.data),)
        else:
            grads = _normalize_grad_outputs((self,), grad_output)
        _run_backward((self,), grads, create_graph=create_graph, accumulate_into_grad=True)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __repr__(self) -> str:
        return f"Value(data={_format_repr(self.data)}, grad={_format_repr(self.grad)})"


class Add(Function):
    def forward(self, a: Value, b: Value):
        self.save_for_backward(a, b)
        return Value(np.add(a.data, b.data))

    def backward(self, grad_output, create_graph: bool = False):
        if create_graph:
            grad_value = _ensure_value(grad_output, requires_grad=False)
            return grad_value, grad_value
        return _data_of(grad_output), _data_of(grad_output)


class Mul(Function):
    def forward(self, a: Value, b: Value):
        self.save_for_backward(a, b)
        return Value(np.multiply(a.data, b.data))

    def backward(self, grad_output, create_graph: bool = False):
        a, b = self.saved_tensors
        if create_graph:
            grad_value = _ensure_value(grad_output, requires_grad=False)
            return mul(grad_value, b), mul(grad_value, a)
        grad_data = _data_of(grad_output)
        return np.multiply(b.data, grad_data), np.multiply(a.data, grad_data)


class Dot(Function):
    def forward(self, a: Value, b: Value):
        self.save_for_backward(a, b)
        return Value(np.dot(a.data, b.data))

    def backward(self, grad_output, create_graph: bool = False):
        a, b = self.saved_tensors
        if create_graph:
            grad_value = _ensure_value(grad_output, requires_grad=False)
            return mul(grad_value, b), mul(grad_value, a)
        grad_data = _data_of(grad_output)
        return np.multiply(grad_data, b.data), np.multiply(grad_data, a.data)


class Transpose(Function):
    def forward(self, a: Value):
        self.save_for_backward(a)
        return Value(_transpose_data(a.data))

    def backward(self, grad_output, create_graph: bool = False):
        if create_graph:
            return transpose(_ensure_value(grad_output, requires_grad=False))
        return _transpose_data(_data_of(grad_output))


class MatMul(Function):
    def forward(self, a: Value, b: Value):
        self.save_for_backward(a, b)
        return Value(np.matmul(a.data, b.data))

    def backward(self, grad_output, create_graph: bool = False):
        a, b = self.saved_tensors
        if create_graph:
            grad_value = _ensure_value(grad_output, requires_grad=False)
            return matmul(grad_value, transpose(b)), matmul(transpose(a), grad_value)
        grad_data = _data_of(grad_output)
        return np.matmul(grad_data, b.data.T), np.matmul(a.data.T, grad_data)


def grad(outputs, inputs, grad_outputs=None, create_graph: bool = False, allow_unused: bool = False):
    normalized_outputs = _as_value_sequence(outputs, "outputs")
    normalized_inputs = _as_value_sequence(inputs, "inputs")
    normalized_grad_outputs = _normalize_grad_outputs(normalized_outputs, grad_outputs)
    node_to_grad = _run_backward(
        normalized_outputs,
        normalized_grad_outputs,
        create_graph=create_graph,
        accumulate_into_grad=False,
    )

    grads = []
    for input_value in normalized_inputs:
        if input_value not in node_to_grad:
            if allow_unused:
                grads.append(None)
                continue
            raise ValueError("One of the requested inputs was not used to compute the outputs")
        grads.append(node_to_grad[input_value])

    if isinstance(inputs, Value):
        return grads[0]
    return tuple(grads)


def vjp(func, *primals, v=None, create_graph: bool = False):
    outputs = func(*primals)
    primals_as_values = tuple(_ensure_value(primal, requires_grad=False) for primal in primals)
    grads = grad(outputs, primals_as_values, grad_outputs=v, create_graph=create_graph, allow_unused=False)
    if isinstance(grads, Value) or grads is None:
        grads = (grads,)
    return outputs, grads


def add(a, b) -> Value:
    return Add.apply(Add, a, b)


def mul(a, b) -> Value:
    return Mul.apply(Mul, a, b)


def dot(a, b) -> Value:
    return Dot.apply(Dot, a, b)


def transpose(a) -> Value:
    return Transpose.apply(Transpose, a)


def matmul(a, b) -> Value:
    return MatMul.apply(MatMul, a, b)
