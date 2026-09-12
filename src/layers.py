import numpy as np

from src.activations import get_activation
from src.initializers import get_initializer


class Dense:
        
    def __init__(self, input_size: int, output_size: int,
                 activation: str = "relu", init_method: str = "he"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.activation = get_activation(activation)

        init_func = get_initializer(init_method)
        self.weights = init_func(input_size, output_size)
        self.bias = np.zeros((1, output_size))

        self.input = None
        self.output = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        z = np.dot(X, self.weights) + self.bias
        self.output = self.activation.forward(z)
        return self.output

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:  
        grad_z = grad_output * self.activation.backward(self.output)

        grad_input = np.dot(grad_z, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_z)
        grad_bias = np.sum(grad_z, axis=0, keepdims=True)

        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input

    def get_params(self) -> dict:
        return {
            'weights': self.weights,
            'bias': self.bias,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'activation': self.activation_name
        }

    def set_params(self, weights: np.ndarray, bias: np.ndarray):
        self.weights = weights
        self.bias = bias


class Dropout:

    def __init__(self, rate: float = 0.2):
        self.rate = rate
        self.mask = None
        self.training = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        if not self.training:
            return X

        self.mask = np.random.rand(*X.shape) > self.rate
        return X * self.mask / (1 - self.rate)

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if not self.training:
            return grad_output

        return grad_output * self.mask / (1 - self.rate)

    def train_mode(self):
        self.training = True

    def test_mode(self):
        self.training = False


class BatchNormalization:
    def __init__(self, size: int, momentum: float = 0.9, epsilon: float = 1e-8):
        self.size = size
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, size))
        self.beta = np.zeros((1, size))

        self.moving_mean = np.zeros((1, size))
        self.moving_var = np.ones((1, size))

        self.x_normalized = None
        self.mean = None
        self.var = None
        self.input = None

        self.training = True

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        if self.training:
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.var = np.var(X, axis=0, keepdims=True)
            self.x_normalized = (X - self.mean) / np.sqrt(self.var + self.epsilon)

            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        else:
            self.x_normalized = (X - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)

        return self.gamma * self.x_normalized + self.beta

    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
    
        if not self.training:
            return grad_output

        N = self.input.shape[0]

        grad_x_norm = grad_output * self.gamma

        grad_var = np.sum(
            grad_x_norm * (self.input - self.mean) * (-0.5) * (self.var + self.epsilon) ** (-1.5),
            axis=0, keepdims=True
        )

        grad_mean = np.sum(
            grad_x_norm * (-1 / np.sqrt(self.var + self.epsilon)),
            axis=0, keepdims=True
        )
        grad_mean += grad_var * np.mean(-2 * (self.input - self.mean), axis=0, keepdims=True)

        grad_input = grad_x_norm / np.sqrt(self.var + self.epsilon)
        grad_input += grad_var * 2 * (self.input - self.mean) / N
        grad_input += grad_mean / N

        grad_gamma = np.sum(grad_output * self.x_normalized, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta

        return grad_input

    def train_mode(self):
        
        self.training = True

    def test_mode(self):
        
        self.training = False

    def get_params(self) -> dict:
        return {
            'gamma': self.gamma,
            'beta': self.beta,
            'moving_mean': self.moving_mean,
            'moving_var': self.moving_var
        }
    