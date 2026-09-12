import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.layers import BatchNormalization, Dense, Dropout


class TestDense(unittest.TestCase):
    def setUp(self):
        self.dense = Dense(4, 3, activation="relu", init_method="he") 
        self.X = np.random.randn(2, 4)

    def test_forward_shape(self):
        output = self.dense.forward(self.X)
        self.assertEqual(output.shape, (2, 3))

    def test_backward_shape(self):
        self.dense.forward(self.X)
        grad_output = np.random.randn(2, 3)
        grad_input = self.dense.backward(grad_output, learning_rate=0.01)
        self.assertEqual(grad_input.shape, (2, 4))

    def test_get_params(self):
        params = self.dense.get_params()
        self.assertIn("weights", params)
        self.assertIn("bias", params)



class TestDropout(unittest.TestCase):
    def setUp(self):
        self.dropout = Dropout(rate=0.5)
        self.X = np.ones((10, 10))

    def test_forward_training(self):
        self.dropout.train_mode()
        output = self.dropout.forward(self.X)
        self.assertTrue(np.any(output == 0))

    def test_forward_testing(self):
        self.dropout.test_mode()
        output = self.dropout.forward(self.X)

        np.testing.assert_array_equal(output, self.X)

    def test_backward_training(self):
        self.dropout.train_mode()
        self.dropout.forward(self.X)
        grad_output = np.ones((10, 10)) 
        grad_input = self.dropout.backward(grad_output, learning_rate=0.01)
        self.assertEqual(grad_input.shape, (10, 10)) 



class TestBatchNormalization(unittest.TestCase):
    def setUp(self):
        self.bn = BatchNormalization(size=5)
        self.X = np.random.randn(10, 5)

    def test_forward_shape(self):
        output = self.bn.forward(self.X)
        self.assertEqual(output.shape, (10, 5))

    def test_forward_training_mean(self):
        self.bn.train_mode()
        output = self.bn.forward(self.X)

        self.assertAlmostEqual(np.mean(output), 0, places=5)

    def test_moving_average_update(self):
        self.bn.train_mode()
        initial_mean = self.bn.moving_mean.copy()
        self.bn.forward(self.X)

        self.assertFalse(np.array_equal(initial_mean, self.bn.moving_mean))

    def test_backward_shape(self):
        self.bn.train_mode()
        self.bn.forward(self.X)
        grad_output = np.random.randn(10, 5)
        grad_input = self.bn.backward(grad_output, learning_rate=0.01)
        self.assertEqual(grad_input.shape, (10, 5))


    def test_get_params(self):
        params = self.bn.get_params()
        self.assertIn("gamma", params)
        self.assertIn("beta", params)
        self.assertIn("moving_mean", params)
        self.assertIn("moving_var", params)

if __name__ == "__main__":
    unittest.main()
