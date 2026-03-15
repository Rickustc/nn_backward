from __future__ import annotations

"""Minimal autograd ``Value`` node for scalars with backprop support."""

import math
from typing import Callable, Set

import numpy as np


class Value:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self._children: Set["Value"] = set()
        self.op = ""
        self._backward: Callable[[], None] = lambda: None

    def __add__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.add(self.data, other.data))
        out._children.update({self, other})
        out.op = "+"

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __sub__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.subtract(self.data, other.data))
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

    def __mul__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.multiply(self.data, other.data))
        out._children.update({self, other})
        out.op = "*"

        def _backward():
            self.grad += np.multiply(other.data, out.grad)
            other.grad += np.multiply(self.data, out.grad)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.divide(self.data, other.data))
        out._children.update({self, other})
        out.op = "/"

        def _backward():
            self.grad += np.divide(out.grad, other.data)
            other.grad -= np.multiply(np.divide(self.data, np.multiply(other.data, other.data)), out.grad)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def __neg__(self) -> "Value":
        out = Value(np.negative(self.data))
        out._children.add(self)
        out.op = "neg"

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(np.exp(self.data))
        out._children.add(self)
        out.op = "exp"

        def _backward():
            self.grad += np.multiply(out.data, out.grad)

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = np.tanh(self.data)
        out = Value(t)
        out._children.add(self)
        out.op = "tanh"

        def _backward():
            self.grad += np.multiply(1 - np.multiply(t, t), out.grad)

        out._backward = _backward
        return out

    def dot(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.dot(self.data, other.data))
        out._children.update({self, other})
        out.op = "dot"

        def _backward():
            self.grad += np.dot(out.grad, other.data)
            other.grad += np.dot(self.data, out.grad)

        out._backward = _backward
        return out

    def matmul(self, other) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(np.matmul(self.data, other.data))
        out._children.update({self, other})
        out.op = "matmul"

        def _backward():
            self.grad += np.matmul(out.grad, other.data.T)
            other.grad += np.matmul(self.data.T, out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False) -> "Value":
        out = Value(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._children.add(self)
        out.op = "sum"

        def _backward():
            self.grad += np.broadcast_to(out.grad, self.data.shape)

        out._backward = _backward
        return out

    def reshape(self, shape) -> "Value":
        out = Value(np.reshape(self.data, shape))
        out._children.add(self)
        out.op = "reshape"

        def _backward():
            self.grad += np.reshape(out.grad, self.data.shape)

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
            node.grad = np.zeros_like(node.data)

    def backward(self) -> None:
        topo = self._topological_order()
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad}, op={self.op})"

    def __hash__(self) -> int:
        return id(self)
