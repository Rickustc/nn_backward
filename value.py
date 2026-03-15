from __future__ import annotations

"""Minimal autograd ``Value`` node for scalars with backprop support."""

import math
from typing import Callable, Set


class Value:
    def __init__(self, data: float):
        self.data = data
        self.grad = 0.0
        self._children: Set["Value"] = set()
        self.op = ""
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        out._children.update({self, other})
        out.op = "+"

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data)
        out._children.update({self, other})
        out.op = "-"

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __rsub__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        out._children.update({self, other})
        out.op = "*"

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data)
        out._children.update({self, other})
        out.op = "/"

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= (self.data / (other.data * other.data)) * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def __neg__(self) -> "Value":
        out = Value(-self.data)
        out._children.add(self)
        out.op = "neg"

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data))
        out._children.add(self)
        out.op = "exp"

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t)
        out._children.add(self)
        out.op = "tanh"

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def _topological_order(self) -> list["Value"]:
        topo: list[Value] = []
        visited: Set[Value] = set()

        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._children:
                    build(parent)
                topo.append(v)

        build(self)
        return topo

    def zero_grad(self) -> None:
        for node in self._topological_order():
            node.grad = 0.0

    def backward(self) -> None:
        topo = self._topological_order()
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad}, op={self.op})"

    def __hash__(self) -> int:
        return id(self)
