import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.optimizers import (
    SGD,
    AdaGrad,
    Adam,
    Momentum,
    Nesterov,
    RMSProp,
    get_optimizer,
)


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.params = {"w": np.array([1.0])}
        self.grads = {"w": np.array([0.5])}
        self.lr = 0.01


    def test_sgd(self):
        sgd = SGD(learning_rate=self.lr)
        updated = sgd.update(self.params, self.grads)

        self.assertAlmostEqual(updated["w"][0], 0.995, places=5)

    def test_momentum(self):
        momentum = Momentum(learning_rate=self.lr, alpha=0.9)
        updated = momentum.update(self.params, self.grads)

        self.assertAlmostEqual(updated["w"][0], 0.995, places=5)

    def test_nesterov(self):
        nesterov = Nesterov(learning_rate=self.lr, alpha=0.9)
        updated = nesterov.update(self.params, self.grads)

        self.assertAlmostEqual(updated["w"][0], 0.995,places=5)

    def test_adagrad(self):
        adagrad = AdaGrad(learning_rate=self.lr)
        updated = adagrad.update(self.params, self.grads)

        self.assertAlmostEqual(updated["w"][0], 0.99, places=3)

    def test_rmsprop(self):
        rmsprop = RMSProp(learning_rate=self.lr, rho=0.9)
        updated = rmsprop.update(self.params, self.grads)

        self.assertAlmostEqual(updated["w"][0], 0.9684, places=3)

    def test_adam(self):
        adam = Adam(learning_rate=0.001)
        updated = adam.update(self.params, self.grads)

        self.assertAlmostEqual(updated["w"][0], 0.999, places=3)

    def test_get_optimizer(self):
        for name in ['sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam']:
            optimizer = get_optimizer(name)
            self.assertIsNotNone(optimizer)

        with self.assertRaises(ValueError):
            get_optimizer('invalid_name')




if __name__ == '__main__':
    unittest.main()