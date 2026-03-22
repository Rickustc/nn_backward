# nn_backward

A minimal autograd library exploring two design paradigms for automatic differentiation.

The repository is self-contained. It ships with a small local `numpy`-compatible shim in the project root, so the examples and tests run without installing external dependencies.

## Structure

- `value/`: Node-based design (Value as nodes, operations as edges).
  - `value.py`: Core Value class with vector support.
  - `test_value.py`: Unit tests for Value operations.
  - `test.py`: Demo and interactive entrypoint.

- `function/`: Operator-based design (Function as edges, Value as nodes).
  - `function.py`: Function-based autograd implementation.
  - `test_function.py`: Unit tests for Function operations.

## Usage

### Value-based (Simple)
```python
from value import Value
import numpy as np

a = Value(np.array([1.0, 2.0]))
b = Value(np.array([3.0, 4.0]))
out = a.dot(b)
out.backward()
print(a.grad)  # [3. 4.]
```

what value do:
  1. 前向时创建计算图
  2. 每个节点记住它从哪来
  3. 反向时从输出往回传梯度
  4. 梯度在共享路径上累加

### Function-based (Modular)
```python
from function import Value, dot, grad, vjp

a = Value([1.0, 2.0])
b = Value([3.0, 4.0])
out = dot(a, b)
out.backward()
print(a.grad)  # [3. 4.]

da, db = grad(out, (a, b))
print(da, db)  # [3. 4.] [1. 2.]

out2, (vjp_a, vjp_b) = vjp(lambda left, right: dot(left, right), a, b)
print(out2.data, vjp_a, vjp_b)  # 11.0 [3. 4.] [1. 2.]
```

## Running

Run the full test suite from the project root:

```bash
python3 -m unittest -v
```

Run only the value-based tests:

```bash
python3 -m unittest -v value.test_value
```

Run only the function-based tests:

```bash
python3 -m unittest -v function.test_function
```

Run the interactive value demo:

```bash
python3 value/test.py --demo
python3 -i value/test.py
```

## Comparison

- **Value-based**: Simpler, easier to extend operations. Suitable for learning.
- **Function-based**: More modular, like PyTorch. Easier to manage complex graphs.
