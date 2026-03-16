from __future__ import annotations

"""Function-based autograd: operators as edges, tensors as nodes."""

import numpy as np
from typing import List


class Function:
    def __init__(self):
        self.saved_tensors: List["Value"] = []

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    @staticmethod
    def apply(func_cls, *inputs):
        func = func_cls()
        output = func.forward(*inputs)
        output.creator = func
        return output


class Value:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.grad = np.zeros_like(self.data)
        self.creator: Function | None = None

    def backward(self):
        if self.creator is None:
            self.grad = np.ones_like(self.data)
        else:
            grads = self.creator.backward(np.ones_like(self.data))
            if isinstance(grads, tuple):
                for inp, grad in zip(self.creator.saved_tensors, grads):
                    inp.grad += grad
                    inp.backward()
            else:
                for inp in self.creator.saved_tensors:
                    inp.grad += grads
                    inp.backward()

    def __repr__(self) -> str:
        data_str = np.array_repr(self.data, precision=3, suppress_small=True)
        grad_str = np.array_repr(self.grad, precision=3, suppress_small=True)
        return f"Value(data={data_str}, grad={grad_str})"


class Add(Function):
    def forward(self, a: Value, b: Value):
        self.saved_tensors = [a, b]
        return Value(np.add(a.data, b.data))

    def backward(self, grad_output):
        return grad_output, grad_output


class Mul(Function):
    def forward(self, a: Value, b: Value):
        self.saved_tensors = [a, b]
        return Value(np.multiply(a.data, b.data))

    def backward(self, grad_output):
        a, b = self.saved_tensors
        return np.multiply(b.data, grad_output), np.multiply(a.data, grad_output)


class Dot(Function):
    def forward(self, a: Value, b: Value):
        self.saved_tensors = [a, b]
        return Value(np.dot(a.data, b.data))

    def backward(self, grad_output):
        a, b = self.saved_tensors
        return np.dot(grad_output, b.data), np.dot(a.data, grad_output)


# Convenience functions
def add(a: Value, b: Value) -> Value:
    return Add.apply(Add, a, b)

def mul(a: Value, b: Value) -> Value:
    return Mul.apply(Mul, a, b)

def dot(a: Value, b: Value) -> Value:
    return Dot.apply(Dot, a, b)