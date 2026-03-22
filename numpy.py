"""A tiny subset of NumPy used by this repository's examples and tests."""

from __future__ import annotations

import builtins
import math
from typing import Any

pi = math.pi


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    return value


def _clone_data(value: Any) -> Any:
    if isinstance(value, list):
        return [_clone_data(item) for item in value]
    return _normalize_scalar(value)


def _shape_of(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        if not value:
            return (0,)
        return (len(value),) + _shape_of(value[0])
    return ()


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _build_from_flat(flat: list[Any], shape: tuple[int, ...]) -> Any:
    if not shape:
        return flat.pop(0)
    size = shape[0]
    return [_build_from_flat(flat, shape[1:]) for _ in range(size)]


def _map_data(value: Any, func) -> Any:
    if isinstance(value, list):
        return [_map_data(item, func) for item in value]
    return func(value)


def _zip_map(left: Any, right: Any, func) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [_zip_map(a, b, func) for a, b in zip(left, right)]
    if isinstance(left, list):
        return [_zip_map(item, right, func) for item in left]
    if isinstance(right, list):
        return [_zip_map(left, item, func) for item in right]
    return func(left, right)


def _transpose_2d(value: list[list[Any]]) -> list[list[Any]]:
    if not value:
        return []
    rows = len(value)
    cols = len(value[0])
    return [[value[row][col] for row in range(rows)] for col in range(cols)]


class NDArray:
    def __init__(self, data: Any):
        self._data = _clone_data(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape_of(self._data)

    @property
    def T(self) -> "NDArray":
        if len(self.shape) <= 1:
            return NDArray(self._data)
        if len(self.shape) != 2:
            raise NotImplementedError("Transpose is only implemented for 2D arrays")
        return NDArray(_transpose_2d(self._data))

    def copy(self) -> "NDArray":
        return NDArray(self._data)

    def tolist(self) -> Any:
        return _clone_data(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __add__(self, other: Any):
        return add(self, other)

    def __radd__(self, other: Any):
        return add(other, self)

    def __sub__(self, other: Any):
        return subtract(self, other)

    def __rsub__(self, other: Any):
        return subtract(other, self)

    def __mul__(self, other: Any):
        return multiply(self, other)

    def __rmul__(self, other: Any):
        return multiply(other, self)

    def __truediv__(self, other: Any):
        return divide(self, other)

    def __rtruediv__(self, other: Any):
        return divide(other, self)

    def __neg__(self):
        return negative(self)

    def __iadd__(self, other: Any):
        result = add(self, other)
        self._data = result._data
        return self

    def __isub__(self, other: Any):
        result = subtract(self, other)
        self._data = result._data
        return self

    def __imul__(self, other: Any):
        result = multiply(self, other)
        self._data = result._data
        return self

    def __itruediv__(self, other: Any):
        result = divide(self, other)
        self._data = result._data
        return self

    def __gt__(self, other: Any):
        return _wrap(_zip_map(_unwrap(self), _unwrap(other), lambda a, b: a > b))

    def __repr__(self) -> str:
        return array_repr(self)


def _unwrap(value: Any) -> Any:
    if isinstance(value, NDArray):
        return value._data
    if isinstance(value, tuple):
        return [_unwrap(item) for item in value]
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return _normalize_scalar(value)


def _wrap(value: Any) -> Any:
    if isinstance(value, list):
        return NDArray(value)
    return _normalize_scalar(value)


def array(data: Any) -> Any:
    return asarray(data)


def asarray(data: Any) -> Any:
    value = _unwrap(data)
    if isinstance(value, list):
        return NDArray(value)
    return _normalize_scalar(value)


def zeros_like(data: Any) -> Any:
    value = _unwrap(data)
    return _wrap(_map_data(value, lambda _: 0.0))


def ones_like(data: Any) -> Any:
    value = _unwrap(data)
    return _wrap(_map_data(value, lambda _: 1.0))


def add(a: Any, b: Any) -> Any:
    return _wrap(_zip_map(_unwrap(a), _unwrap(b), lambda x, y: x + y))


def subtract(a: Any, b: Any) -> Any:
    return _wrap(_zip_map(_unwrap(a), _unwrap(b), lambda x, y: x - y))


def multiply(a: Any, b: Any) -> Any:
    return _wrap(_zip_map(_unwrap(a), _unwrap(b), lambda x, y: x * y))


def divide(a: Any, b: Any) -> Any:
    return _wrap(_zip_map(_unwrap(a), _unwrap(b), lambda x, y: x / y))


def negative(a: Any) -> Any:
    return _wrap(_map_data(_unwrap(a), lambda x: -x))


def exp(a: Any) -> Any:
    return _wrap(_map_data(_unwrap(a), math.exp))


def tanh(a: Any) -> Any:
    return _wrap(_map_data(_unwrap(a), math.tanh))


def sin(a: Any) -> Any:
    return _wrap(_map_data(_unwrap(a), math.sin))


def cos(a: Any) -> Any:
    return _wrap(_map_data(_unwrap(a), math.cos))


def maximum(a: Any, b: Any) -> Any:
    return _wrap(_zip_map(_unwrap(a), _unwrap(b), lambda x, y: x if x >= y else y))


def where(condition: Any, x: Any, y: Any) -> Any:
    pairs = _zip_map(_unwrap(x), _unwrap(y), lambda left, right: (left, right))
    return _wrap(_zip_map(_unwrap(condition), pairs, lambda cond, pair: pair[0] if cond else pair[1]))


def dot(a: Any, b: Any) -> Any:
    left = _unwrap(a)
    right = _unwrap(b)
    left_shape = _shape_of(left)
    right_shape = _shape_of(right)

    if left_shape == () or right_shape == ():
        return multiply(left, right)
    if len(left_shape) == 1 and len(right_shape) == 1:
        return sum(multiply(left, right))
    if len(left_shape) == 2 and len(right_shape) == 1:
        return _wrap([sum(multiply(row, right)) for row in left])
    if len(left_shape) == 1 and len(right_shape) == 2:
        cols = _transpose_2d(right)
        return _wrap([sum(multiply(left, col)) for col in cols])
    if len(left_shape) == 2 and len(right_shape) == 2:
        right_t = _transpose_2d(right)
        return _wrap([[sum(multiply(row, col)) for col in right_t] for row in left])
    raise NotImplementedError("dot only supports scalars, vectors, and matrices")


def matmul(a: Any, b: Any) -> Any:
    left = _unwrap(a)
    right = _unwrap(b)
    left_shape = _shape_of(left)
    right_shape = _shape_of(right)
    if len(left_shape) == 2 and len(right_shape) == 2:
        return dot(left, right)
    if len(left_shape) == 2 and len(right_shape) == 1:
        return dot(left, right)
    if len(left_shape) == 1 and len(right_shape) == 2:
        return dot(left, right)
    raise NotImplementedError("matmul only supports matrix/vector multiplication")


def sum(a: Any, axis: int | None = None, keepdims: bool = False) -> Any:
    value = _unwrap(a)
    shape = _shape_of(value)
    if axis is None:
        total = 0.0
        for item in _flatten(value):
            total += item
        return total
    if len(shape) == 1 and axis == 0:
        total = builtins.sum(value)
        if keepdims:
            return _wrap([total])
        return total
    if len(shape) == 2 and axis == 0:
        cols = _transpose_2d(value)
        result = [builtins.sum(col) for col in cols]
        return _wrap([result] if keepdims else result)
    if len(shape) == 2 and axis == 1:
        result = [builtins.sum(row) for row in value]
        if keepdims:
            return _wrap([[item] for item in result])
        return _wrap(result)
    raise NotImplementedError("sum only supports axis=None, 0, or 1 for 1D/2D inputs")


def broadcast_to(a: Any, shape: tuple[int, ...]) -> Any:
    value = _unwrap(a)
    if _shape_of(value) == shape:
        return _wrap(value)
    if _shape_of(value) == ():
        flat = [value for _ in range(math.prod(shape))]
        return _wrap(_build_from_flat(flat, shape))
    raise NotImplementedError("broadcast_to only supports scalar broadcasting in this shim")


def reshape(a: Any, shape: tuple[int, ...]) -> Any:
    flat = _flatten(_unwrap(a))
    if math.prod(shape) != len(flat):
        raise ValueError("cannot reshape array to requested shape")
    return _wrap(_build_from_flat(flat, shape))


def array_repr(a: Any, precision: int = 3, suppress_small: bool = True) -> str:
    def format_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        number = float(value)
        if suppress_small and abs(number) < 10 ** (-precision):
            number = 0.0
        text = f"{number:.{precision}f}"
        text = text.rstrip("0").rstrip(".")
        if "." not in text:
            text += ".0"
        return text

    def format_data(value: Any) -> str:
        if isinstance(value, list):
            return "[" + ", ".join(format_data(item) for item in value) + "]"
        return format_scalar(value)

    return format_data(_unwrap(a))


class _TestingModule:
    @staticmethod
    def assert_array_almost_equal(actual: Any, desired: Any, decimal: int = 6) -> None:
        actual_value = _unwrap(actual)
        desired_value = _unwrap(desired)
        if _shape_of(actual_value) != _shape_of(desired_value):
            raise AssertionError(f"shape mismatch: {_shape_of(actual_value)} != {_shape_of(desired_value)}")

        tolerance = 1.5 * 10 ** (-decimal)
        actual_flat = _flatten(actual_value)
        desired_flat = _flatten(desired_value)
        for index, (left, right) in enumerate(zip(actual_flat, desired_flat)):
            if abs(left - right) > tolerance:
                raise AssertionError(f"arrays differ at index {index}: {left} != {right}")


testing = _TestingModule()
