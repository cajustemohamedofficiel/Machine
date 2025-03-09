import unittest
import numpy as np
from core.activation import sigmoid, tanh, relu, leaky_relu, elu, softmax, swish, gelu

class TestActivationFunctions(unittest.TestCase):

    def setUp(self):
        # Set up some test data for each function
        self.x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

    def test_sigmoid(self):
        result = sigmoid(self.x)
        expected = 1 / (1 + np.exp(-self.x))
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="Sigmoid function failed")

    def test_tanh(self):
        result = tanh(self.x)
        expected = np.tanh(self.x)
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="Tanh function failed")

    def test_relu(self):
        result = relu(self.x)
        expected = np.maximum(0, self.x)
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="ReLU function failed")

    def test_leaky_relu(self):
        result = leaky_relu(self.x, alpha=0.01)
        expected = np.where(self.x > 0, self.x, 0.01 * self.x)
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="Leaky ReLU function failed")

    def test_elu(self):
        result = elu(self.x, alpha=1.0)
        expected = np.where(self.x > 0, self.x, 1.0 * (np.exp(self.x) - 1))
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="ELU function failed")

    def test_softmax(self):
        result = softmax(self.x.reshape(1, -1))
        expected = np.exp(self.x - np.max(self.x)) / np.sum(np.exp(self.x - np.max(self.x)), axis=0)
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="Softmax function failed")

    def test_swish(self):
        result = swish(self.x)
        expected = self.x * sigmoid(self.x)
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="Swish function failed")

    def test_gelu(self):
        result = gelu(self.x)
        expected = 0.5 * self.x * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * self.x**3)))
        np.testing.assert_almost_equal(result, expected, decimal=5, err_msg="GELU function failed")

if __name__ == '__main__':
    unittest.main()
