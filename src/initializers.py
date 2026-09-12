import numpy as np


def he_initialization(input_size: int, output_size: int) -> np.ndarray:
    std = np.sqrt(2.0 / input_size)
    return np.random.randn(input_size, output_size) * std

def xavier_initialization(input_size: int, output_size: int) -> np.ndarray:
    fan_avg = (input_size + output_size) / 2.0
    std = np.sqrt(1.0 / fan_avg)
    return np.random.randn(input_size, output_size) * std

def lecun_initialization(input_size: int, output_size: int) -> np.ndarray:
    std = np.sqrt(1.0 / input_size)
    return np.random.randn(input_size, output_size) * std

def zero_initialization(input_size: int, output_size: int) -> np.ndarray:
    return np.zeros((input_size, output_size))

def get_initializer(method: str):
    initializers = {
        "he": he_initialization,
        "xavier": xavier_initialization,
        "lecun": lecun_initialization,
        "zero": zero_initialization
    }
    if method not in initializers:
        raise ValueError(f"Unknown initializer: {method}. Available: {list(initializers.keys())}")

    return initializers[method]