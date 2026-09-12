import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.activations import ReLU, Sigmoid, Softmax, Tanh, get_activation


class TestActivations(unittest.TestCase):

    def setUp(self):  
        self.x = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)  

    def test_relu_forward(self):
        relu = ReLU()
        output = relu.forward(self.x)
        expected = np.array([[0, 0, 0, 1, 2]], dtype=np.float32)  
        np.testing.assert_array_equal(output, expected)

    def test_relu_backward(self):
        relu = ReLU()
        output = relu.backward(self.x)
        expected = np.array([[0, 0, 0, 1, 1]], dtype=np.float32)  
        np.testing.assert_array_equal(output, expected)

    def test_sigmoid_forward(self):
        sigmoid = Sigmoid()
        output = sigmoid.forward(self.x)
        expected = 1 / (1 + np.exp(-self.x))
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_sigmoid_backward(self):
        sigmoid = Sigmoid()
        output = sigmoid.backward(self.x)
        s = sigmoid.forward(self.x)
        expected = s * (1 - s)
        np.testing.assert_array_almost_equal(output, expected, decimal=5)  

    def test_tanh_forward(self):
        tanh = Tanh()
        output = tanh.forward(self.x)
        expected = np.tanh(self.x)
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_tanh_backward(self):
        tanh = Tanh()
        output = tanh.backward(self.x)
        t = np.tanh(self.x)
        expected = 1 - t ** 2  
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_softmax_forward(self):
        softmax = Softmax()
        output = softmax.forward(self.x)
        self.assertAlmostEqual(np.sum(output), 1.0, places=5)

    def test_softmax_backward(self):
        softmax = Softmax()
        output = softmax.backward(self.x)
        expected = np.ones_like(self.x)
        np.testing.assert_array_equal(output, expected)

    def test_get_activation(self):
        for name in ["relu", "sigmoid", "tanh", "softmax"]:
            activation = get_activation(name)
            self.assertIsNotNone(activation)

        with self.assertRaises(ValueError):
            get_activation("invalid_name")


if __name__ == '__main__':
    unittest.main()