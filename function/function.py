from __future__ import annotations
import numpy as np


def as_value(x, requires_grad=False):
    if isinstance(x, Value):
        return x
    return Value(x, requires_grad=requires_grad)


def grad_add(a, b):
    if isinstance(a, Value) or isinstance(b, Value):
        return add(as_value(a), as_value(b))
    return a + b


class Value:
    def __init__(self, data, requires_grad=True):
        self.data = np.asarray(data, dtype=float)
        self.grad = None
        self.creator = None
        self.requires_grad = requires_grad

    def backward(self, grad=None, create_graph=False):
        if grad is None:
            grad = np.ones_like(self.data)

        node_to_grad = {self: grad}
        topo = topo_sort(self)

        for node in reversed(topo):
            if node not in node_to_grad:
                continue

            g = node_to_grad[node]

            if node.creator is None:
                if node.requires_grad:
                    if node.grad is None:
                        node.grad = g
                    else:
                        node.grad = grad_add(node.grad, g)
                continue

            grads = node.creator.backward(g, create_graph=create_graph)
            for parent, parent_grad in zip(node.creator.inputs, grads):
                if not parent.requires_grad:
                    continue
                if parent in node_to_grad:
                    node_to_grad[parent] = grad_add(node_to_grad[parent], parent_grad)
                else:
                    node_to_grad[parent] = parent_grad

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

    @property
    def T(self):
        return transpose(self)

    def sum(self):
        return sum_op(self)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


def topo_sort(output):
    topo = []
    visited = set()

    def build(node):
        if node in visited:
            return
        visited.add(node)
        if node.creator is not None:
            for parent in node.creator.inputs:
                build(parent)
        topo.append(node)

    build(output)
    return topo


class Function:
    def __init__(self):
        self.inputs = ()
        self.saved_tensors = ()

    def save_for_backward(self, *xs):
        self.saved_tensors = xs

    @classmethod
    def apply(cls, *inputs):
        inputs = tuple(as_value(x, requires_grad=False) for x in inputs)
        func = cls()
        func.inputs = inputs
        out = func.forward(*inputs)
        out.creator = func
        out.requires_grad = any(x.requires_grad for x in inputs)
        if not out.requires_grad:
            out.creator = None
        return out


class Add(Function):
    def forward(self, a, b):
        return Value(a.data + b.data)

    def backward(self, grad_output, create_graph=False):
        if create_graph:
            g = as_value(grad_output, requires_grad=False)
            return g, g
        return grad_output, grad_output


class Mul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return Value(a.data * b.data)

    def backward(self, grad_output, create_graph=False):
        a, b = self.saved_tensors
        if create_graph:
            g = as_value(grad_output, requires_grad=False)
            return g * b, g * a
        return grad_output * b.data, grad_output * a.data


class Transpose(Function):
    def forward(self, a):
        return Value(a.data.T)

    def backward(self, grad_output, create_graph=False):
        if create_graph:
            g = as_value(grad_output, requires_grad=False)
            return (transpose(g),)
        return (grad_output.T,)


class MatMul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return Value(a.data @ b.data)

    def backward(self, grad_output, create_graph=False):
        a, b = self.saved_tensors
        if create_graph:
            g = as_value(grad_output, requires_grad=False)
            return g @ b.T, a.T @ g
        return grad_output @ b.data.T, a.data.T @ grad_output


class Sum(Function):
    def forward(self, a):
        self.save_for_backward(a)
        return Value(np.array(a.data.sum()))

    def backward(self, grad_output, create_graph=False):
        (a,) = self.saved_tensors
        if create_graph:
            g = as_value(grad_output, requires_grad=False)
            ones = Value(np.ones_like(a.data), requires_grad=False)
            return (g * ones,)
        return (grad_output * np.ones_like(a.data),)


def add(a, b):
    return Add.apply(a, b)


def mul(a, b):
    return Mul.apply(a, b)


def transpose(a):
    return Transpose.apply(a)


def matmul(a, b):
    return MatMul.apply(a, b)


def sum_op(a):
    return Sum.apply(a)


def grad(output, inputs, grad_output=None, create_graph=False):
    if isinstance(inputs, Value):
        inputs = (inputs,)

    if grad_output is None:
        grad_output = np.ones_like(output.data)

    node_to_grad = {output: grad_output}
    topo = topo_sort(output)

    for node in reversed(topo):
        if node not in node_to_grad:
            continue

        g = node_to_grad[node]

        if node.creator is None:
            continue

        grads = node.creator.backward(g, create_graph=create_graph)
        for parent, parent_grad in zip(node.creator.inputs, grads):
            if not parent.requires_grad:
                continue
            if parent in node_to_grad:
                node_to_grad[parent] = grad_add(node_to_grad[parent], parent_grad)
            else:
                node_to_grad[parent] = parent_grad

    result = tuple(node_to_grad[x] for x in inputs)
    if len(result) == 1:
        return result[0]
    return result


def vjp(func, *primals, v=None, create_graph=False):
    output = func(*primals)
    if v is None:
        v = np.ones_like(output.data)
    grads = grad(output, primals, grad_output=v, create_graph=create_graph)
    if isinstance(grads, Value):
        grads = (grads,)
    return output, grads