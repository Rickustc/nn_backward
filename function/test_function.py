import unittest

import numpy as np

from function.function import Value, add, dot, grad, matmul, mul, vjp


class FunctionBackpropTests(unittest.TestCase):
    def test_add(self):
        a = Value([1.0, 2.0])
        b = Value([3.0, 4.0])
        out = add(a, b)
        out.backward()
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_almost_equal(b.grad, np.ones_like(b.data))

    def test_mul(self):
        a = Value([2.0, 3.0])
        b = Value([4.0, 5.0])
        out = mul(a, b)
        out.backward()
        expected = np.array([8.0, 15.0])
        np.testing.assert_array_almost_equal(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, b.data)
        np.testing.assert_array_almost_equal(b.grad, a.data)

    def test_dot(self):
        a = Value([1.0, 2.0])
        b = Value([3.0, 4.0])
        out = dot(a, b)
        out.backward()
        expected = 11.0
        self.assertAlmostEqual(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, b.data)
        np.testing.assert_array_almost_equal(b.grad, a.data)

    def test_matmul(self):
        a = Value([[1.0, 2.0], [3.0, 4.0]])
        b = Value([[5.0, 6.0], [7.0, 8.0]])
        out = matmul(a, b)
        out.backward(np.ones_like(out.data))

        np.testing.assert_array_almost_equal(out.data, np.array([[19.0, 22.0], [43.0, 50.0]]))
        np.testing.assert_array_almost_equal(a.grad, np.array([[11.0, 15.0], [11.0, 15.0]]))
        np.testing.assert_array_almost_equal(b.grad, np.array([[4.0, 4.0], [6.0, 6.0]]))

    def test_composed_graph_uses_upstream_gradient(self):
        a = Value([2.0, 3.0])
        b = Value([4.0, 5.0])
        c = Value([6.0, 7.0])
        out = add(mul(a, b), c)
        out.backward()

        np.testing.assert_array_almost_equal(a.grad, b.data)
        np.testing.assert_array_almost_equal(b.grad, a.data)
        np.testing.assert_array_almost_equal(c.grad, np.ones_like(c.data))


class FunctionHigherOrderTests(unittest.TestCase):
    def test_grad_scalar_single_input(self):
        x = Value(3.0)
        out = mul(x, x)
        dx = grad(out, x)
        self.assertAlmostEqual(dx, 6.0)

    def test_grad_multiple_inputs(self):
        a = Value([1.0, 2.0])
        b = Value([3.0, 4.0])
        da, db = grad(dot(a, b), (a, b))
        np.testing.assert_array_almost_equal(da, b.data)
        np.testing.assert_array_almost_equal(db, a.data)

    def test_grad_requires_explicit_seed_for_vector_output(self):
        x = Value([2.0, 3.0])
        out = mul(x, x)
        with self.assertRaises(ValueError):
            grad(out, x)

    def test_vjp_scalar_output_default_seed(self):
        a = Value([1.0, 2.0])
        b = Value([3.0, 4.0])
        out, grads = vjp(lambda left, right: dot(left, right), a, b)
        self.assertAlmostEqual(out.data, 11.0)
        np.testing.assert_array_almost_equal(grads[0], b.data)
        np.testing.assert_array_almost_equal(grads[1], a.data)

    def test_vjp_vector_output_with_explicit_seed(self):
        a = Value([2.0, 3.0])
        b = Value([4.0, 5.0])
        out, grads = vjp(lambda left, right: add(mul(left, right), right), a, b, v=[1.0, 1.0])
        np.testing.assert_array_almost_equal(out.data, np.array([12.0, 20.0]))
        np.testing.assert_array_almost_equal(grads[0], b.data)
        np.testing.assert_array_almost_equal(grads[1], np.array([3.0, 4.0]))

    def test_allow_unused_false_raises(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        out = add(a, b)
        with self.assertRaises(ValueError):
            grad(out, (a, c))

    def test_allow_unused_true_returns_none(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(4.0)
        out = add(a, b)
        da, dc = grad(out, (a, c), allow_unused=True)
        self.assertAlmostEqual(da, 1.0)
        self.assertIsNone(dc)

    def test_second_derivative_with_create_graph(self):
        x = Value(2.0)
        first = grad(mul(mul(x, x), x), x, create_graph=True)
        self.assertIsInstance(first, Value)
        self.assertAlmostEqual(first.data, 12.0)

        second = grad(first, x)
        self.assertAlmostEqual(second, 12.0)

    def test_backward_create_graph_allows_second_derivative(self):
        x = Value(2.0)
        out = mul(mul(x, x), x)
        out.backward(create_graph=True)

        self.assertIsInstance(x.grad, Value)
        self.assertAlmostEqual(x.grad.data, 12.0)

        second = grad(x.grad, x)
        self.assertAlmostEqual(second, 12.0)


if __name__ == "__main__":
    unittest.main()
