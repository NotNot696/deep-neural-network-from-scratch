import numpy as np
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

from src.utils import normalize_data, one_hot_encode
from src.utils import train_test_split as split_data


def load_iris_data(test_size: float = 0.2,
                   shuffle: bool = True,
                   seed: int = 42) -> tuple:
    print("Loading Iris dataset...")

    data = load_iris()
    X = data.data
    y = data.target
    class_names = data.target_names

    print(f"Iris dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"Data Normalized (mean=0, std=1)")

    y_onehot = one_hot_encode(y, num_classes=len(class_names))

    X_train, X_test, y_train, y_test = split_data(X, y_onehot, test_size=test_size, shuffle=shuffle, seed=seed)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f" Class names: {', '.join(class_names)}")

    return X_train, X_test, y_train, y_test, class_names

def load_mnist_data(test_size: float = 0.2,
                    shuffle: bool = True,
                    seed: int = 42,
                    max_samples: int = None) -> tuple:
    print(f"Loading MNIST dataset (this may take a moment)...")

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    X = X / 255.0

    class_names = [str(i) for i in range(10)]

    print(f"MNIST dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} calsses")

    y_onehot = one_hot_encode(y, num_classes=len(class_names))

    X_train, X_test, y_train, y_test = split_data(X, y_onehot, test_size=test_size, shuffle=shuffle, seed=seed)

    print(f" Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, class_names

def load_fashion_mnist_data(test_size: float = 0.2,
                            shuffle: bool = True,
                            seed: int = 42,
                            max_samples: int = None) -> tuple:
    print("Loading Fashion-MNIST dataset (this may take a moment)...")

    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False, parser="auto")

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    if max_samples is not None:
        X = X[:max_samples]
        y = y[:max_samples]

    X = X / 255.0

    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    print(f"Fashion-MNIST dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")

    y_onehot = one_hot_encode(y, num_classes=len(class_names))

    X_train, X_test, y_train, y_test = split_data(X, y_onehot, test_size=test_size, shuffle=shuffle, seed=seed)

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Class names: {', '.join(class_names)}")

    return X_train, X_test, y_train, y_test, class_names

def get_dataset_info(dataset_name: str, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray, class_names: list) -> None:
    print("\n" + "=" * 60)
    print(f"{dataset_name} Dataset Information")
    print("=" * 60)
    print(f"Total samples: {X_train.shape[0] + X_test.shape[0]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(class_names)}")
    print(f"Class names: {', '.join(class_names)}")

    y_train_classes = np.argmax(y_train, axis=1)
    unique, counts = np.unique(y_train_classes, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} samples ({count/len(y_train)*100:.1f}%)")

def visualize_sample(X: np.ndarray, y: np.ndarray, index: int,
                     class_names: list, title: str = "Sample Image") -> None:

    image = X[index]
    label = np.argmax(y[index])

    size = int(np.sqrt(image.shape[0]))
    if size * size == image.shape[0]:
        image_2d = image.reshape(size, size)
        plt.figure(figsize=(4,4))
        plt.imshow(image_2d, cmap="gray")
        plt.title(f"{title}\nLabel: {class_names[label]}")
        plt.axis("off")
        plt.show()
    else:
        print(f"Not an image dataset (feature size: {image.shape[0]})")