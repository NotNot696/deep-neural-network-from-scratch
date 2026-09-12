import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.name = "BaseOptimizer"

    def update(self, params: dict, grads: dict) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}(lr={self.lr})"



class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.name = "SGD"

    def update(self, params: dict, grads: dict) -> dict:
        updated = {}
        for key in params:
            updated[key] = params[key] - self.lr * grads[key]
        return updated
    


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.9):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.name = "Momentum"
        self.velocities = {}

    def update(self, params: dict, grads: dict) -> dict:
        updated = {}
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])

            self.velocities[key] = self.alpha * self.velocities[key] - self.lr * grads[key]

            updated[key] = params[key] + self.velocities[key]

        return updated



class Nesterov(Optimizer):
    def __init__(self, learning_rate: float = 0.01, alpha: float = 0.9):
        super().__init__(learning_rate)
        self.alpha = alpha
        self.name = "Nesterov"
        self.velocities = {}

    def update(self, params: dict, grads: dict) -> dict:
        updated = {}
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])

            self.velocities[key] = self.alpha * self.velocities[key] - self.lr * grads[key]

            updated[key] = params[key] + self.velocities[key]

        return updated



class AdaGrad(Optimizer):
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.name = "AdaGrad"
        self.cache = {}

    def update(self, params: dict, grads: dict) -> dict:
        updated = {}
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])

            self.cache[key] += grads[key] ** 2

            adaptive_lr = self.lr / (np.sqrt(self.cache[key]) + self.epsilon)

            updated[key] = params[key] - adaptive_lr * grads[key]

        return updated



class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.01, rho: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.name = "RMSProp"
        self.cache = {}

    def update(self, params: dict, grads: dict) -> dict:
        updated = {}
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])

            self.cache[key] = self.rho * self.cache[key] + (1 - self.rho) * (grads[key] ** 2)

            adaptive_lr = self.lr / (np.sqrt(self.cache[key]) + self.epsilon)

            updated[key] = params[key] - adaptive_lr * grads[key]

        return updated



class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.name = "Adam"
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params: dict, grads: dict) -> dict:
        self.t += 1
        updated = {}

        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]

            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated


def get_optimizer(name: str, **kwargs) -> Optimizer:
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'nesterov': Nesterov,
        'adagrad': AdaGrad,
        'rmsprop': RMSProp,
        'adam': Adam,
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")

    return optimizers[name](**kwargs)