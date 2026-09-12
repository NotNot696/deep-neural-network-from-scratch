from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.layers import BatchNormalization, Dense, Dropout
from src.utils import accuracy, cross_entropy_loss, mse_loss, plot_combined_history



class NeuralNetwork:
    def __init__(self, learning_rate: float = 0.01,
                 loss_function: str = "cross_entropy",
                 seed: int = None):
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.seed = seed

        self.layers: List = []

        self.loss_history: List[float] = []
        self.accuracy_history: List[float] = []

        if seed is not None:
            np.random.seed(seed)

    def add_layer(self, layer) -> None:
        self.layers.append(layer)

    def add_dense(self, input_size: int, output_size: int,
                  activation: str = "relu",
                  init_method: str = "he") -> None:
        layer = Dense(input_size, output_size, activation, init_method)
        self.layers.append(layer)

    def add_dropout(self, rate: float = 0.2) -> None:
        layer = Dropout(rate)
        self.layers.append(layer)

    def add_batch_norm(self, size: int, momentum: float = 0.9) -> None:
        layer = BatchNormalization(size, momentum)
        self.layers.append(layer)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = training

            X = layer.forward(X)

        return X

    def backward(self, grad_output: np.ndarray) -> None:
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad_output = layer.backward(grad_output, self.learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.predict(X)
        return np.argmax(y_pred, axis=1)

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        if self.loss_function == "cross_entropy":
            return cross_entropy_loss(y_pred, y_true)
        elif self.loss_function == "mse":
            return mse_loss(y_pred, y_true)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

    def compute_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pred_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(pred_classes == true_classes)

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        y_pred = self.forward(X_batch, training=True)

        loss = self.compute_loss(y_pred, y_batch)

        grad_output = y_pred - y_batch

        self.backward(grad_output)

        return loss

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              verbose : bool = True) -> Dict[str, List[float]]:

        n_samples = X.shape[0]

        self.loss_history = []
        self.accuracy_history = []
        val_loss_history = []
        val_accuracy_history = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0


            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i: i+batch_size]
                y_batch = y_shuffled[i: i+batch_size]

                loss = self.train_step(X_batch, y_batch)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            y_pred_train = self.predict(X)
            train_acc = self.compute_accuracy(y_pred_train, y)
            self.accuracy_history.append(train_acc)


            if X_val is not None and y_val is not None:
                y_pred_val = self.predict(X_val)
                val_loss = self.compute_loss(y_pred_val, y_val)
                val_acc = self.compute_accuracy(y_pred_val, y_val)
                val_loss_history.append(val_loss)
                val_accuracy_history.append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                val_msg = ""
                if X_val is not None:
                    val_msg = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss: {avg_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}{val_msg}")

        if verbose:
            plot_combined_history(
                self.loss_history,
                self.accuracy_history,
                title="Training Progress"
            )

        history = {
            "loss": self.loss_history,
            "accuracy": self.accuracy_history,
        }

        if X_val is not None and y_val is not None:
            history["val_loss"] = val_loss_history
            history["val_accuracy"] = val_accuracy_history

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        y_pred = self.predict(X)
        loss = self.compute_loss(y_pred, y)
        acc = self.compute_accuracy(y_pred, y)
        return loss, acc

    def save_weights(self, filepath: str) -> None:
        weights = []
        for layer in self.layers:
            if hasattr(layer, "get_params"):
                weights.append(layer.get_params())
            else:
                weights.append(None)

        np.savez(filepath, weights=weights)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath: str) -> None:
    
        data = np.load(filepath, allow_pickle=True)
        weights = data["weights"]

        for layer, weight in zip(self.layers, weights):
            if weight is None:
                continue
        
            if hasattr(layer, "set_params"):
                if "gamma" in weight and "beta" in weight:
                    layer.gamma = weight["gamma"]
                    layer.beta = weight["beta"]
                    layer.moving_mean = weight["moving_mean"]
                    layer.moving_var = weight["moving_var"]
                elif "weights" in weight and "bias" in weight:
                    layer.weights = weight["weights"]
                    layer.bias = weight["bias"]

        print(f"Weights loaded from {filepath}")