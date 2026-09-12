import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import get_dataset_info, load_iris_data
from src.model import NeuralNetwork
from src.schedulers import ReduceOnPlateauScheduler
from src.utils import (
    EarlyStopping,
    ExperimentLogger,
    ModelCheckpoint,
    plot_combined_history,
)


def run_iris_advanced():
    print("=" * 60)
    print("Iris Dataset Test (Advanced)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, class_names = load_iris_data(
        test_size=0.2,
        shuffle=True,
        seed=42
    )

    get_dataset_info("Iris", X_train, y_train, X_test, y_test, class_names)
    model = NeuralNetwork(
        learning_rate=0.01,
        loss_function="cross_entropy",
        seed=42
    )

    model.add_dense(4, 16, activation="relu", init_method="he")
    model.add_dropout(0.2)
    model.add_dense(16, 8, activation="relu", init_method="he")
    model.add_dense(8, 3, activation="softmax", init_method="he")

    print("\n" + "-" * 40)
    print("Model Architecture:")
    print("  Input: 4 features")
    print("  Hidden 1: 16 neurons (ReLU) + Dropout 0.2")
    print("  Hidden 2: 8 neurons (ReLU)")
    print("  Output: 3 neurons (Softmax)")
    print("-" * 40)

    print("\nAdvanced Techniques:")
    print("  - EarlyStopping (patience=50)")
    print("  - ModelCheckpoint (saving best model)")
    print("  - ReduceOnPlateau (patience=20, factor=0.5)")
    print("  - ExperimentLogger")
    print("-" * 40)

    early_stopping = EarlyStopping(
        patience=50,
        min_delta=1e-6,
        verbose=True
    )

    os.makedirs("outputs/models", exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath="outputs/models/iris_best.npz",
        monitor="val_loss",
        mode="min",
        verbose=True
    )

    logger = ExperimentLogger(
        log_dir="outputs/logs",
        experiment_name="iris_advanced"
    )
    logger.log_config({
        'dataset': 'Iris',
        'learning_rate': 0.01,
        'epochs': 1000,
        'batch_size': 16,
        'architecture': '4-16-8-3',
        'dropout': [0.2, 0.0],
        'optimizer': 'SGD',
        'scheduler': 'ReduceOnPlateau',
        'early_stopping': True,
        'patience': 50
    })

    scheduler = ReduceOnPlateauScheduler(
        initial_lr=0.01,
        factor=0.5,
        patience=20,
        min_lr=1e-6
    )




    print("\nTraining...")

    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'lr': []
    }

    epochs = 1000
    batch_size = 16

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

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.6f}")

    print("\n" + "-" * 40) 
    print("Loading best model...")
    model.load_weights("outputs/models/iris_best.npz") 
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


    os.makedirs("outputs/plots", exist_ok=True)

    logger.log_history(history)
    logger.save()
    logger.print_summary()

    plot_combined_history(
        history['loss'],
        history['accuracy'],
        title="Iris Training Progress (Advanced)",
        save_path="outputs/plots/iris_training_advanced.png"
    )

    return model, history


if __name__ == "__main__":
    model, history = run_iris_advanced()
