import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import NeuralNetwork
from src.utils import one_hot_encode


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.model = NeuralNetwork(learning_rate=0.01, seed=42)
        self.model.add_dense(4, 8, activation="relu", init_method="he")
        self.model.add_dense(8, 3, activation="softmax", init_method="he")

        self.X = np.random.randn(20, 4)
        self.y = one_hot_encode(np.random.randint(0, 3, 20), num_classes=3)

    def test_forward_shape(self):
        output = self.model.forward(self.X, training=True)
        self.assertEqual(output.shape, (20, 3))

    def test_predict_shape(self):
        output = self.model.predict(self.X)
        self.assertEqual(output.shape, (20, 3))

    def test_predict_classes(self):
        classes = self.model.predict_classes(self.X)
        self.assertEqual(classes.shape ,(20,))
        self.assertTrue(np.all(classes >= 0))
        self.assertTrue(np.all(classes <= 2))

    def test_compute_loss(self):
        y_pred = self.model.predict(self.X)
        loss = self.model.compute_loss(y_pred, self.y)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)

    def test_compute_accuracy(self):
        y_pred = self.model.predict(self.X)
        acc = self.model.compute_accuracy(y_pred, self.y)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_train(self):
        history = self.model.train(
            self.X, self.y,
            epochs=10,
            batch_size=4,
            verbose=False
        )

        self.assertIn("loss", history)
        self.assertIn("accuracy", history)
        self.assertEqual(len(history["loss"]), 10)
        self.assertEqual(len(history["accuracy"]), 10)

    def test_evaluate(self):
        loss, acc = self.model.evaluate(self.X, self.y)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0)
        self.assertLessEqual(acc, 1)

    def test_save_load_weights(self):
        os.makedirs("outputs/models", exist_ok=True)
        filepath = "outputs/models/test_model.npz"

        self.model.save_weights(filepath)
        self.assertTrue(os.path.exists(filepath))

        self.model.load_weights(filepath)

        if os.path.exists(filepath):
            os.remove(filepath)



if __name__ == "__main__":
    unittest.main()