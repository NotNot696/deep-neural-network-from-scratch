import numpy as np

def l1_regularization(weights: np.ndarray, lambda_l1: float = 0.01) -> float:
    return lambda_l1 * np.sum(np.abs(weights))

def l2_regularization(weights: np.ndarray, lambda_l2: float = 0.01) -> float:
    return lambda_l2 * np.sum(weights ** 2)

def l1_l2_regularization(weights: np.ndarray,
                         lambda_l1: float = 0.01,
                         lambda_l2: float = 0.01) -> float:

    return l1_regularization(weights, lambda_l1) + l2_regularization(weights, lambda_l2)

def compute_regularization_loss(model, lambda_l1: float = 0.0, lambda_l2: float = 0.0)-> float:
    total_reg = 0.0

    for layer in model.layers:
        if hasattr(layer, "weights"):
            if lambda_l1 > 0:
                total_reg += l1_regularization(layer.weights, lambda_l1)

            if lambda_l2 > 0:
                total_reg += l2_regularization(layer.weights, lambda_l2)
    return total_reg

def add_regularization_to_loss(original_loss: float, reg_loss: float) -> float:
    return original_loss + reg_loss

def get_regularizer(name: str):
    regularizers = {
        "l1": l1_regularization,
        "l2": l2_regularization,
        "l1_l2": l1_l2_regularization
    }

    if name not in regularizers:
        raise ValueError(f"Unknown regularizer: {name}. Available: {list(regularizers.keys())}")

    return regularizers[name]
