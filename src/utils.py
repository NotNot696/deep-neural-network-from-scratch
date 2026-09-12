import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int = None) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if epoch is not None:
                self.best_epoch = epoch
        else:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                    print(f"  EarlyStopping: {self.counter}/{self.patience} "
                      f"(best: {self.best_loss:.6f})")                

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                     print(f"\n Early stopping triggered! "
                          f"No improvement for {self.patience} epochs. "
                          f"Best loss: {self.best_loss:.6f}")
        
        return self.early_stop

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0



class ModelCheckpoint:
    def __init__(self, filepath: str, monitor: str = "val_loss",
                 mode: str = "min", verbose: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose

        self.best = float('inf') if mode == "min" else -float("inf")
        self.best_epoch = 0
        self.history = []

    def __call__(self, epoch: int, model, value: float) -> bool:
        self.history.append(value)

        improved = False

        if self.mode == "min":
            if value < self.best:
                self.best = value
                self.best_epoch = epoch
                improved = True
        else:
            if value > self.best:
                self.best = value
                self.best_epoch = epoch
                improved = True

        if improved:
            model.save_weights(self.filepath)
            if self.verbose:
                print(f"  Best model saved at epoch {epoch+1} "
                      f"({self.monitor}: {value:.6f})")
        

        return improved

    def reset(self):
        self.best = float('inf') if self.mode == "min" else -float('inf')
        self.best_epoch = 0
        self.history = []



class ExperimentLogger:
    def __init__(self, log_dir: str = "outputs/logs",
                 experiment_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        self.experiment_name = experiment_name

        self.data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {},
            'results': {},
            'history': {},
            'notes': ''
        }

    def log_config(self, config: dict) -> None:
        self.data["config"].update(config)

    def log_results(self, results: dict) -> None:
        self.data["results"].update(results)

    def log_history(self, history: dict) -> None:
        for key, values in history.items():
            if isinstance(values, list) and len(values) > 100:
                step = max(1, len(values) // 100)
                self.data["history"][key] = values[::step]

            else:
                self.data["history"][key] = values

    def log_notes(self, notes: str) -> None:
        self.data["notes"] = notes

    def save(self) -> str:
        filepath = os.path.join(self.log_dir, f"{self.experiment_name}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str, ensure_ascii=False)

        print(f"Experiment log saved to {filepath}")
        return filepath

    def load(self, filepath: str) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"Experiment Summary: {self.experiment_name}")
        print("=" * 60)
        
        print("\nConfig:")
        for key, value in self.data['config'].items():
            print(f"  {key}: {value}")
        
        print("\nResults:")
        for key, value in self.data['results'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nTimestamp: {self.data['timestamp']}")
        print("=" * 60)




def compare_experiments(log_dir: str = "outputs/logs") -> None:
        if not os.path.exists(log_dir):
            print(f"Log directory not found: {log_dir}")
            return 

        experiments = []
        for filename in os.listdir(log_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(log_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        experiments.append(data)


                except Exception as e:
                    print(f"Error reading {filename}: {e}")

        if not experiments:
            print("No experiments found")
            return 

        experiments.sort(
        key=lambda x: x.get('results', {}).get('test_accuracy', 0),
        reverse=True
    )
    
        print("\n" + "=" * 80)
        print("Experiment Comparison")
        print("=" * 80)
        print(f"{'Experiment':<30} {'Test Acc':<12} {'Test Loss':<12} {'Best Epoch':<12}")
        print("-" * 80)
    
        for exp in experiments:
            name = exp.get('experiment_name', 'Unknown')[:28]
            results = exp.get('results', {})
            test_acc = results.get('test_accuracy', 0)
            test_loss = results.get('test_loss', 0)
            best_epoch = results.get('best_epoch', '-')
        
            print(f"{name:<30} {test_acc:<12.4f} {test_loss:<12.6f} {best_epoch:<12}")
    
        print("=" * 80)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    eps = 1e-9
    return -np.mean(y_true * np.log(y_pred + eps))


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    return np.mean(pred_classes == true_classes)


def plot_loss_history(
    loss_history: list,
    title: str = "Training Loss",
    save_path: str = None,
    figsize: tuple = (10, 6),
):

    plt.figure(figsize=figsize)
    plt.plot(loss_history, linewidth=2, color="blue")
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_accuracy_history(
    accuracy_history: list,
    title: str = "Training Accuracy",
    save_path: str = None,
    figsize: tuple = (10, 6),
):
    plt.figure(figsize=figsize)
    plt.plot(accuracy_history, linewidth=2, color="green")
    plt.title(title, fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_combined_history(
    loss_history: list,
    accuracy_history: list,
    title: str = "Training Progress",
    save_path: str = None,
    figsize: tuple = (14, 6),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(loss_history, linewidth=2, color="blue")
    ax1.set_title("Loss", fontsize=14)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, alpha=0.3)

    ax2.plot(accuracy_history, linewidth=2, color="green")
    ax2.set_title("Accuracy", fontsize=14)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y.astype(int)] = 1.0    
    return one_hot


def shuffle_data(X: np.ndarray, y: np.ndarray, seed: int = None) -> tuple:
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def normalize_data(X: np.ndarray, mean: float = None, std: float = None) -> tuple:
    if mean is None:
        mean = np.mean(X)
    if std is None:
        std = np.std(X)

    X_norm = (X - mean) / (std + 1e-8)
    return X_norm, mean, std


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = True,
    seed: int = None,
) -> tuple:
    if seed is not None:
        np.random.seed(seed)

    if shuffle:
        X, y = shuffle_data(X, y, seed)

    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test
