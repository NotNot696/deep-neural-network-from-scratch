# 🧠 Deep Neural Network from Scratch

### A complete, transparent, and extensible implementation of a deep neural network — built entirely from scratch with pure NumPy.

> **No TensorFlow. No PyTorch. No Keras. Just math, code, and a deep understanding of how neural networks really work.**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557C?style=for-the-badge\&logo=matplotlib\&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)

![Tests](https://img.shields.io/badge/Tests-35%20passed-2ea44f?style=for-the-badge\&logo=pytest\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/NotNot696/deep-neural-network-from-scratch?style=for-the-badge\&color=ffd700)

</p>

---

## 📑 Table of Contents

* [📖 Introduction](#-introduction)
* [✨ Key Features](#-key-features)
* [📊 Results](#-results)
* [📁 Project Structure](#-project-structure)
* [🛠️ Installation](#️-installation)
* [🚀 Usage](#-usage)
* [⚙️ Advanced Techniques](#️-advanced-techniques)
* [🧪 Testing](#-testing)
* [📈 Outputs & Visualizations](#-outputs--visualizations)
* [🗺️ Roadmap](#️-roadmap)
* [🤝 Contributing](#-contributing)
* [📄 License](#-license)
* [👤 Author](#-author)
* [🙏 Acknowledgements](#-acknowledgements)

---

## 📖 Introduction

> *"What I cannot create, I do not understand."*
> — **Richard Feynman**

This project is a **complete, ground-up implementation of a deep neural network** written entirely in **NumPy**.

Every forward pass, every backward gradient, every optimizer update — all coded from scratch, with no high-level frameworks hiding the magic.

The goal is simple:

> **Understand deep learning by building it.**

Whether you're a student learning the fundamentals, a researcher validating an idea, or a practitioner who wants to peek under the hood, this project is designed to be **transparent, modular, and ready to extend**.

### 🎯 What Makes This Project Special?

* 🧩 **Every component is modular** — swap optimizers, activations, or schedulers with a single line.
* 🔬 **Fully testable** — 35 unit tests cover every critical path.
* 📊 **Production-grade training utilities** — EarlyStopping, ModelCheckpoint, and ExperimentLogger.
* 📈 **Visual results** — automatically generated plots for every experiment.
* 🎓 **Educational** — comments, docstrings, and clean code throughout.

---

## ✨ Key Features

### 🧠 Core Components

| Component                | Implementations                                       |
| :----------------------- | :---------------------------------------------------- |
| **Activation Functions** | ReLU, Sigmoid, Tanh, Softmax, ELU, SELU               |
| **Weight Initializers**  | He (Kaiming), Xavier (Glorot), LeCun, Zero            |
| **Layers**               | Dense (Fully Connected), Dropout, Batch Normalization |
| **Optimizers**           | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam       |

### ⚙️ Training Utilities

| Utility           | Options                                                |
| :---------------- | :----------------------------------------------------- |
| **Regularizers**  | L1, L2, Elastic Net                                    |
| **LR Schedulers** | Power, Exponential, Piecewise, 1Cycle, ReduceOnPlateau |
| **Callbacks**     | EarlyStopping, ModelCheckpoint                         |
| **Logging**       | ExperimentLogger (JSON)                                |

---

## 📊 Results

The network was trained and evaluated on four datasets — from toy problems to real-world image classification.

| Dataset              | Task                   | Test Accuracy | Test Loss | Best Epoch | Early Stopped |
| :------------------- | :--------------------- | ------------: | --------: | ---------: | :-----------: |
| 🔷 **XOR**           | Binary classification  |   **100.00%** |    0.0007 |          — |       —       |
| 🌸 **Iris**          | 3-class classification |    **96.67%** |    0.0394 |         25 |       ✅       |
| 🔢 **MNIST**         | 10-class digits        |    **96.79%** |    0.0177 |         86 |       ❌       |
| 👕 **Fashion-MNIST** | 10-class clothing      |    **86.81%** |    0.0444 |         51 |       ✅       |

> 💡 **Note:** All results were achieved using advanced techniques — EarlyStopping, ModelCheckpoint, 1Cycle / ReduceOnPlateau scheduling, Batch Normalization, and Dropout — implemented entirely from scratch.

---

## 📁 Project Structure

```text
deep-neural-network-from-scratch/
│
├── src/                         # Core library
│   ├── activations.py           # 6 activation functions
│   ├── initializers.py          # 4 weight initialization strategies
│   ├── layers.py                # Dense, Dropout, BatchNormalization
│   ├── model.py                 # NeuralNetwork class
│   ├── optimizers.py            # 6 gradient-based optimizers
│   ├── regularizers.py          # L1, L2, Elastic Net
│   ├── schedulers.py            # 5 learning rate schedulers
│   ├── utils.py                 # Losses, metrics, callbacks, logger
│   └── data_loader.py            # Iris, MNIST, Fashion-MNIST
│
├── examples/                    # Ready-to-run scripts
│   ├── xor_example.py
│   ├── iris_example.py
│   ├── mnist_example.py
│   └── fashion_mnist_example.py
│
├── tests/                       # 35 unit tests
│   ├── test_activation.py       # 9 tests
│   ├── test_layers.py           # 11 tests
│   ├── test_model.py            # 8 tests
│   └── test_optimizer.py        # 7 tests
│
├── outputs/                     # Generated artifacts
│   ├── models/                  # Saved weights (.npz)
│   ├── logs/                    # Experiment logs (.json)
│   └── plots/                   # Training plots (.png)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Installation

### Prerequisites

* Python **3.8+**
* pip (or conda)

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/NotNot696/deep-neural-network-from-scratch.git

cd deep-neural-network-from-scratch

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Examples

Each example is fully self-contained and showcases different aspects of the library:

```bash
# XOR — the classic non-linear problem
python examples/xor_example.py

# Iris — multi-class classification (3 classes, 4 features)
python examples/iris_example.py

# MNIST — handwritten digit recognition (10 classes, 784 features)
python examples/mnist_example.py

# Fashion-MNIST — clothing classification (10 classes)
python examples/fashion_mnist_example.py
```

### Build Your Own Model

```python
from src.model import NeuralNetwork

# 1. Create the model
model = NeuralNetwork(
    learning_rate=0.01,
    loss_function='cross_entropy',
    seed=42
)

# 2. Design the architecture
model.add_dense(784, 256, activation='relu', init_method='he')
model.add_batch_norm(256)
model.add_dropout(0.3)

model.add_dense(256, 128, activation='relu', init_method='he')
model.add_dropout(0.2)

model.add_dense(128, 64, activation='relu', init_method='he')
model.add_dense(64, 10, activation='softmax', init_method='he')

# 3. Train
history = model.train(
    X_train,
    y_train,
    epochs=100,
    batch_size=128,
    X_val=X_test,
    y_val=y_test,
    verbose=True
)

# 4. Evaluate
loss, acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {acc:.2%}")
```

---

## ⚙️ Advanced Techniques

This project goes beyond the basics.

Every modern training technique you'd expect from a production framework is implemented from scratch.

### 🎯 Training Callbacks

| Callback             | Description                                     |
| :------------------- | :---------------------------------------------- |
| **EarlyStopping**    | Halts training when validation loss plateaus    |
| **ModelCheckpoint**  | Saves the best model during training            |
| **ExperimentLogger** | Records every hyperparameter and result to JSON |

### 📉 Learning Rate Control

| Scheduler           | Strategy                      |
| :------------------ | :---------------------------- |
| **1Cycle**          | Warm-up followed by cool-down |
| **ReduceOnPlateau** | Reduces LR when loss stalls   |
| **Power**           | Gradual polynomial decay      |
| **Exponential**     | Fast exponential decay        |
| **Piecewise**       | Step-wise constant decay      |

### 🧱 Network Regularization

| Technique                 | Purpose                             |
| :------------------------ | :---------------------------------- |
| **Batch Normalization**   | Stabilizes and accelerates training |
| **Dropout**               | Prevents overfitting                |
| **L1 / L2 / Elastic Net** | Penalizes large weights             |

### 🎲 Smart Initialization

| Initializer         | Best For                |
| :------------------ | :---------------------- |
| **He (Kaiming)**    | ReLU networks           |
| **Xavier (Glorot)** | Tanh / Sigmoid networks |
| **LeCun**           | SELU networks           |

---

## 🧪 Testing

The project is covered by **35 comprehensive unit tests** — ensuring every component behaves as expected.

### Run the Test Suite

```bash
python -m pytest tests/ -v
```

### Expected Output

```text
========================================================== test session starts ===========================================================
collected 35 items

tests/test_activation.py .........                         [ 25%]
tests/test_layers.py ...........                           [ 57%]
tests/test_model.py ........                               [ 80%]
tests/test_optimizer.py .......                            [100%]

========================================================== 35 passed in 2.64s ===========================================================
```

### Test Coverage

| Test File            |  Tests | Coverage                                                  |
| :------------------- | :----: | :-------------------------------------------------------- |
| `test_activation.py` |    9   | ReLU, Sigmoid, Tanh, Softmax, ELU, SELU + factory         |
| `test_layers.py`     |   11   | Dense, Dropout, BatchNormalization (forward + backward)   |
| `test_model.py`      |    8   | Forward, predict, train, evaluate, save/load              |
| `test_optimizer.py`  |    7   | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam + factory |
| **Total**            | **35** | ✅ **100% passing**                                        |

---

## 📈 Outputs & Visualizations

Every experiment produces artifacts that make it easy to track, reproduce, and compare results.

```text
outputs/
├── plots/              # Loss curves, accuracy curves, weight histograms
├── models/             # Best model weights (compressed .npz format)
└── logs/               # Full experiment configuration and results (JSON)
```

### What You Get

* 📉 **Training Progress** — loss and accuracy curves, automatically generated
* 💾 **Best Model Checkpoint** — the model with lowest validation loss, saved as `.npz`
* 📝 **Experiment Log** — full config, results, and history, saved as `.json`

### Load the Best Model

```python
model.load_weights("outputs/models/mnist_best.npz")
```

### Sample Experiment Log

```json
{
  "experiment_name": "mnist_advanced",
  "config": {
    "learning_rate": 0.01,
    "epochs": 100
  },
  "results": {
    "test_accuracy": 0.9679,
    "test_loss": 0.017746
  },
  "history": {
    "loss": [],
    "accuracy": []
  }
}
```

---

## 🗺️ Roadmap

* [x] Activation functions (6)
* [x] Weight initializers (4)
* [x] Layers (Dense, Dropout, BatchNorm)
* [x] Optimizers (6)
* [x] Regularizers (L1, L2, Elastic Net)
* [x] LR Schedulers (5)
* [x] Training callbacks (EarlyStopping, ModelCheckpoint, ExperimentLogger)
* [x] Full test suite (35 tests)
* [x] 4 example datasets
* [ ] Convolutional Neural Networks (CNN)
* [ ] Recurrent Neural Networks (RNN / LSTM)
* [ ] GPU acceleration with CuPy
* [ ] Additional optimizers (Nadam, AdaMax)

---

## 🤝 Contributing

Contributions are always welcome!

Whether it's a bug fix, a new feature, or a documentation improvement:

1. **Fork** the repository.
2. **Create** a feature branch:

   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes:

   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push** to the branch:

   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open** a Pull Request.

Please make sure your code passes all tests and follows the existing style.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

### **Amir Arsalan Farahmand**

* GitHub: [@NotNot696](https://github.com/NotNot696)

---

## 🙏 Acknowledgements

This project stands on the shoulders of giants.

Special thanks to the researchers whose foundational work inspired every line of code:

* **Glorot & Bengio (2010)** — *Understanding the difficulty of training deep feedforward neural networks*
* **He et al. (2015)** — *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet*
* **Kingma & Ba (2014)** — *Adam: A Method for Stochastic Optimization*
* **Ioffe & Szegedy (2015)** — *Batch Normalization: Accelerating Deep Network Training*
* **Klambauer et al. (2017)** — *Self-Normalizing Neural Networks (SELU)*
* **Smith (2018)** — *A Disciplined Approach to Neural Network Hyper-Parameters (1Cycle)*

And of course, to the **open-source community** — for making deep learning accessible to everyone.

---

<p align="center">

### ⭐ If this project helped you, please consider giving it a star!

*It means the world and helps others discover the project.*

**Built with ❤️ and pure NumPy**

</p>
