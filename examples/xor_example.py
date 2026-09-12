import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import NeuralNetwork
from src.utils import one_hot_encode, plot_combined_history


def run_xor_example():
    print("=" * 60)
    print("XOR Problem Test")
    print("=" * 60)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    y_onehot = one_hot_encode(y.flatten(), num_classes=2)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y_onehot.shape}")
    print(f"X:\n{X}")
    print(f"y:\n{y_onehot}")

    model = NeuralNetwork(
        learning_rate=0.1,
        loss_function= "cross_entropy",
        seed=42
    )

    model.add_dense(2, 4, activation="relu", init_method="he")
    model.add_dense(4, 2, activation="softmax", init_method="he")

    print("\n" + "-" * 40)
    print("Model Architecture:")
    print("  Input: 2 neurons")
    print("  Hidden: 4 neurons (ReLU)")
    print("  Output: 2 neurons (Softmax)")
    print("-" * 40)
    
    print("\nTraining...")
    history = model.train(
        X, y_onehot,
        epochs=2000,
        batch_size=4,
        verbose=True
    )

    y_pred = model.predict(X)
    loss, acc = model.evaluate(X, y_onehot)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Final Loss: {loss:.6f}")
    print(f"Accuracy: {acc:.2%}")
    
    print("\nPredictions:")
    for i in range(len(X)):
        pred_class = np.argmax(y_pred[i])
        true_class = np.argmax(y_onehot[i])
        print(f"  Input: {X[i]} → Predicted: {pred_class}, True: {true_class}")


    return model, history

if __name__ == "__main__":
    os.makedirs("outputs/plots", exist_ok=True)
    
    model, history = run_xor_example()
    
    from src.utils import plot_combined_history
    plot_combined_history(
        history['loss'], 
        history['accuracy'],
        title="XOR Training Progress",
        save_path="outputs/plots/XOR_training.png"
    )