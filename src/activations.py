import numpy as np


class Activation:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)


class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        t = self.forward(x)
        return 1 - t ** 2


def get_activation(name: str) -> Activation:
    activations = {
        "relu": ReLU(),
        "sigmoid": Sigmoid(),
        "softmax": Softmax(),
        "tanh": Tanh()
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]