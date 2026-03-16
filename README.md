# nn_backward

A minimal autograd library exploring two design paradigms for automatic differentiation.

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
from value.value import Value
import numpy as np

a = Value(np.array([1.0, 2.0]))
b = Value(np.array([3.0, 4.0]))
out = a.dot(b)
out.backward()
print(a.grad)  # [3. 4.]
```

### Function-based (Modular)
```python
from function.function import Value, dot

a = Value([1.0, 2.0])
b = Value([3.0, 4.0])
out = dot(a, b)
out.backward()
print(a.grad)  # [3. 4.]
```

## Comparison

- **Value-based**: Simpler, easier to extend operations. Suitable for learning.
- **Function-based**: More modular, like PyTorch. Easier to manage complex graphs.