import math
import unittest

import numpy as np

from value import Value


class ValueBackpropTests(unittest.TestCase):
    def test_simple_graph(self):
        """Forward/backward for (a * b + 4)^exp."""
        a = Value(2.0)
        b = Value(3.0)
        inter = a * b
        add_node = inter + 4.0
        out = add_node.exp()
        out.backward()

        expected_out = math.exp(a.data * b.data + 4.0)
        self.assertAlmostEqual(out.data, expected_out)

        # d/d(a) = b * exp(a*b + 4)
        self.assertAlmostEqual(a.grad, expected_out * b.data)
        # d/d(b) = a * exp(a*b + 4)
        self.assertAlmostEqual(b.grad, expected_out * a.data)
        # upstream gradient at intermediate add node should match exp result
        self.assertAlmostEqual(add_node.grad, expected_out)

    def test_accumulates_gradients(self):
        """Verify gradients accumulate over multiple paths."""
        x = Value(0.5)
        y = Value(0.75)
        shared = x * y
        out = shared * shared
        out.backward()
        # d/d(x) = 2 * shared * y
        expected_grad = 2 * shared.data * y.data
        self.assertAlmostEqual(x.grad, expected_grad)
        self.assertAlmostEqual(y.grad, expected_grad * (x.data / y.data))

    def test_sub_div_gradients(self):
        """Ensure subtraction/division gradients follow the chain rule."""
        a = Value(5.0)
        b = Value(2.0)
        out = (a - b) * (a / b)
        out.backward()

        expected_grad_a = (2 * a.data - b.data) / b.data
        expected_grad_b = -a.data / b.data - (a.data - b.data) * a.data / (b.data * b.data)

        self.assertAlmostEqual(a.grad, expected_grad_a)
        self.assertAlmostEqual(b.grad, expected_grad_b)

    def test_tanh_gradient(self):
        """tanh should match the analytical derivative 1 - tanh(x)^2."""
        x = Value(0.25)
        out = x.tanh()
        out.backward()

        expected_out = math.tanh(x.data)
        self.assertAlmostEqual(out.data, expected_out)
        self.assertAlmostEqual(x.grad, 1 - expected_out * expected_out)

    def test_zero_grad_clears_entire_graph(self):
        """zero_grad should reset gradients on the output and its parents."""
        a = Value(1.5)
        b = Value(-2.0)
        out = (-(a * b)).tanh()
        out.backward()

        self.assertNotEqual(out.grad, 0.0)
        self.assertNotEqual(a.grad, 0.0)
        self.assertNotEqual(b.grad, 0.0)

        out.zero_grad()

        self.assertEqual(out.grad, 0.0)
        self.assertEqual(a.grad, 0.0)
        self.assertEqual(b.grad, 0.0)


class ValueVectorTests(unittest.TestCase):
    def test_vector_add(self):
        a = Value(np.array([1.0, 2.0]))
        b = Value(np.array([3.0, 4.0]))
        out = a + b
        out.backward()
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_almost_equal(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, np.ones_like(a.data))
        np.testing.assert_array_almost_equal(b.grad, np.ones_like(b.data))

    def test_dot_product(self):
        a = Value(np.array([1.0, 2.0]))
        b = Value(np.array([3.0, 4.0]))
        out = a.dot(b)
        out.backward()
        expected = 1*3 + 2*4  # 11.0
        self.assertAlmostEqual(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, b.data)
        np.testing.assert_array_almost_equal(b.grad, a.data)

    def test_sum(self):
        a = Value(np.array([1.0, 2.0, 3.0]))
        out = a.sum()
        out.backward()
        expected = 6.0
        self.assertAlmostEqual(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, np.ones_like(a.data))

    def test_reshape(self):
        a = Value(np.array([1.0, 2.0, 3.0, 4.0]))
        out = a.reshape((2, 2))
        out.backward()
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(out.data, expected)
        np.testing.assert_array_almost_equal(a.grad, np.ones_like(a.data))


if __name__ == "__main__":
    unittest.main()
