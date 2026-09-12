import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import get_dataset_info, load_fashion_mnist_data, visualize_sample
from src.model import NeuralNetwork
from src.schedulers import ReduceOnPlateauScheduler
from src.utils import (
    EarlyStopping,
    ExperimentLogger,
    ModelCheckpoint,
    plot_combined_history,
)

def run_fashion_mnist_advanced(max_samples: int= 2000):
    print("=" * 60)
    print("Fashion-MNIST Dataset Test (Advanced)")
    print("=" * 60)
    print("Fashion-MNIST: 10 clothing classes")
    print("  Classes: T-shirt/top, Trouser, Pullover, Dress, Coat,")
    print("           Sandal, Shirt, Sneaker, Bag, Ankle boot")
    print("-" * 60)

    X_train, X_test, y_train, y_test, class_names = load_fashion_mnist_data(
        test_size=0.2,
        shuffle=True,
        seed=42,
        max_samples=max_samples
    )

    get_dataset_info("Fashion-MNIST", X_train, y_train, X_test, y_test, class_names)

    visualize_sample(X_train, y_train, index=6, class_names=class_names, title="Fashion-MNIST Sample")

    model = NeuralNetwork(
        learning_rate=0.005,
        loss_function="cross_entropy",
        seed=42

    )

    model.add_dense(784, 512, activation="relu", init_method="he")
    model.add_batch_norm(512)
    model.add_dropout(0.4)

    model.add_dense(512, 256, activation="relu", init_method="he")
    model.add_batch_norm(256)
    model.add_dropout(0.3)

    model.add_dense(256, 128, activation="relu", init_method="he")
    model.add_dropout(0.2)

    model.add_dense(128, 64, activation="relu", init_method="he")
    model.add_dropout(0.1)

    model.add_dense(64, 10, activation="softmax", init_method="he")

    print("\n" + "-" * 40)
    print("Model Architecture:")
    print("  Input: 784 pixels (28x28)")
    print("  Hidden 1: 512 neurons (ReLU) + BatchNorm + Dropout 0.4")
    print("  Hidden 2: 256 neurons (ReLU) + BatchNorm + Dropout 0.3")
    print("  Hidden 3: 128 neurons (ReLU) + Dropout 0.2")
    print("  Hidden 4: 64 neurons (ReLU) + Dropout 0.1")
    print("  Output: 10 neurons (Softmax)")
    print(f"  Total parameters: ~{(784*512) + (512*256) + (256*128) + (128*64) + (64*10):,}")
    print("-" * 40)


    print("\nAdvanced Techniques:")
    print("  - EarlyStopping (patience=20)")
    print("  - ModelCheckpoint (saving best model)")
    print("  - ReduceOnPlateau (patience=5, factor=0.5)")
    print("  - ExperimentLogger")
    print("-" * 40)

    early_stopping = EarlyStopping(
        patience=20,
        min_delta=1e-4,
        verbose=True
    )

    os.makedirs("outputs/models", exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath="outputs/models/fashion_mnist_best.npz",
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    logger = ExperimentLogger(
        log_dir="outputs/logs",
        experiment_name="fashion_mnist_advanced"
    )

    logger.log_config({
        'dataset': 'Fashion-MNIST',
        'learning_rate': 0.005,
        'epochs': 80,
        'batch_size': 128,
        'architecture': '784-512-256-128-64-10',
        'dropout': [0.4, 0.3, 0.2, 0.1],
        'batch_norm': True,
        'optimizer': 'SGD',
        'scheduler': 'ReduceOnPlateau',
        'early_stopping': True,
        'patience': 20,
        'max_samples': max_samples
    })

    epochs = 80
    scheduler = ReduceOnPlateauScheduler(
        initial_lr=0.005,
        factor=0.5,
        patience=5,
        min_lr = 1e-6
    )

    print("\nTraining...")

    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': []
    }

    batch_size = 128

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            loss = model.train_step(X_batch, y_batch)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)

        y_pred_train = model.predict(X_train)
        train_acc = model.compute_accuracy(y_pred_train, y_train)
        history["accuracy"].append(train_acc)

        y_pred_val = model.predict(X_test)
        val_loss = model.compute_loss(y_pred_val, y_test)
        val_acc = model.compute_accuracy(y_pred_val, y_test)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        current_lr = scheduler.get_lr(epoch, loss=val_loss)
        model.learning_rate = current_lr
        history["lr"].append(current_lr)

        if early_stopping(val_loss, epoch):
            print(f"\n Training stopped at epoch {epoch+1}")
            break


        checkpoint(epoch, model, val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")

    print("\n" + "-" * 40)
    print("Loading best model...")
    model.load_weights("outputs/models/fashion_mnist_best.npz")
    print("-" * 40)


    loss, acc = model.evaluate(X_test, y_test)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Best Epoch: {checkpoint.best_epoch + 1}")
    print(f"Best Val Loss: {checkpoint.best:.6f}")
    print(f"Test Loss: {loss:.6f}")
    print(f"Test Accuracy: {acc:.2%}")

    
    print("\nSample Predictions:")
    y_pred = model.predict(X_test[:10])
    for i in range(min(10, len(X_test))):
        pred_class = np.argmax(y_pred[i])
        true_class = np.argmax(y_test[i])
        print(f"  Sample {i+1}: Predicted: {class_names[pred_class]}, "
              f"True: {class_names[true_class]}")

    
    
    logger.log_results({
        'test_accuracy': acc,
        'test_loss': loss,
        'best_epoch': checkpoint.best_epoch + 1,
        'best_val_loss': checkpoint.best,
        'total_epochs': epoch + 1,
        'early_stopped': early_stopping.early_stop
    })

    logger.log_history(history)
    logger.save()
    logger.print_summary()

    
    os.makedirs("outputs/plots", exist_ok=True)
    plot_combined_history(
        history['loss'],
        history['accuracy'],
        title="Fashion-MNIST Training Progress (Advanced)",
        save_path="outputs/plots/fashion_mnist_training_advanced.png"
    )

    return model, history


if __name__ == "__main__":
    model, history = run_fashion_mnist_advanced(max_samples=None)
        