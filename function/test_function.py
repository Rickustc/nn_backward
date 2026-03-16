import unittest
import numpy as np

from function import Value, add, mul, dot


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


if __name__ == "__main__":
    unittest.main()