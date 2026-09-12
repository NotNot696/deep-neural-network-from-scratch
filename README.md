```markdown
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Deep%20Neural%20Network&fontSize=70&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=From%20Scratch%20with%20NumPy&descAlignY=58&descSize=20" width="100%"/>

**A complete, transparent, and extensible implementation of a deep neural network — built entirely from scratch with pure NumPy.**

*No TensorFlow. No PyTorch. No Keras. Just math, code, and a deep understanding of how neural networks really work.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[![Tests](https://img.shields.io/badge/Tests-35%20passed-2ea44f?style=for-the-badge&logo=pytest&logoColor=white)](https://github.com/NotNot696/deep-neural-network-from-scratch)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/NotNot696/deep-neural-network-from-scratch?style=for-the-badge&color=ffd700)](https://github.com/NotNot696/deep-neural-network-from-scratch/stargazers)

[**📖 Documentation**](#-introduction) · [**🚀 Quick Start**](#-installation) · [**📊 Results**](#-results) · [**🧪 Tests**](#-testing)

</div>

---

## 📖 Introduction

> *"What I cannot create, I do not understand."* — **Richard Feynman**

This project is a **complete, ground-up implementation of a deep neural network** written entirely in **NumPy**. Every forward pass, every backward gradient, every optimizer update — all coded from scratch, with no high-level frameworks hiding the magic.

The goal is simple: **understand deep learning by building it.** Whether you're a student learning the fundamentals, a researcher validating an idea, or a practitioner who wants to peek under the hood, this project is designed to be **transparent, modular, and ready to extend**.

### 🎯 What makes this project special?

- 🧩 **Every component is modular** — swap optimizers, activations, or schedulers with a single line.
- 🔬 **Fully testable** — 35 unit tests cover every critical path.
- 📊 **Production-grade training utilities** — EarlyStopping, ModelCheckpoint, and ExperimentLogger.
- 📈 **Visual results** — automatically generated plots for every experiment.
- 🎓 **Educational** — comments, docstrings, and clean code throughout.

---

## ✨ Key Features

<table>
<tr>
<td valign="top" width="50%">

### 🧠 Core Components

| Component | Implementations |
|:---|:---|
| **Activation Functions** | ReLU, Sigmoid, Tanh, Softmax, ELU, SELU |
| **Weight Initializers** | He (Kaiming), Xavier (Glorot), LeCun, Zero |
| **Layers** | Dense, Dropout, BatchNormalization |
| **Optimizers** | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam |

</td>
<td valign="top" width="50%">

### ⚙️ Training Utilities

| Utility | Options |
|:---|:---|
| **Regularizers** | L1, L2, Elastic Net |
| **LR Schedulers** | Power, Exponential, Piecewise, 1Cycle, ReduceOnPlateau |
| **Callbacks** | EarlyStopping, ModelCheckpoint |
| **Logging** | ExperimentLogger (JSON) |

</td>
</tr>
</table>

---

## 📊 Results

The network was trained and evaluated on four datasets spanning from toy problems to real-world image classification:

<table align="center">
<thead>
<tr>
<th>Dataset</th>
<th>Task</th>
<th align="center">Test Accuracy</th>
<th align="center">Test Loss</th>
<th align="center">Best Epoch</th>
<th align="center">Early Stopped</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">🔷 <b>XOR</b></td>
<td align="center">Binary classification</td>
<td align="center"><b>100.00%</b></td>
<td align="center">0.0007</td>
<td align="center">—</td>
<td align="center">—</td>
</tr>
<tr>
<td align="center">🌸 <b>Iris</b></td>
<td align="center">3-class classification</td>
<td align="center"><b>96.67%</b></td>
<td align="center">0.0394</td>
<td align="center">25</td>
<td align="center">✅</td>
</tr>
<tr>
<td align="center">🔢 <b>MNIST</b></td>
<td align="center">10-class digits</td>
<td align="center"><b>96.79%</b></td>
<td align="center">0.0177</td>
<td align="center">86</td>
<td align="center">❌</td>
</tr>
<tr>
<td align="center">👕 <b>Fashion-MNIST</b></td>
<td align="center">10-class clothing</td>
<td align="center"><b>86.81%</b></td>
<td align="center">0.0444</td>
<td align="center">51</td>
<td align="center">✅</td>
</tr>
</tbody>
</table>

> 💡 **Note:** All results were achieved using advanced techniques such as **EarlyStopping**, **ModelCheckpoint**, **1Cycle / ReduceOnPlateau scheduling**, **Batch Normalization**, and **Dropout** — implemented entirely from scratch.

---

## 📁 Project Structure

```
deep-neural-network-from-scratch/
│
├── 📂 src/                       # Core library
│   ├── 🧠 activations.py         # 6 activation functions
│   ├── 🎲 initializers.py        # 4 weight initialization strategies
│   ├── 🧱 layers.py              # Dense, Dropout, BatchNormalization
│   ├── 🏗️  model.py              # NeuralNetwork class
│   ├── ⚡ optimizers.py          # 6 gradient-based optimizers
│   ├── 🛡️  regularizers.py       # L1, L2, Elastic Net
│   ├── 📉 schedulers.py          # 5 learning rate schedulers
│   ├── 🔧 utils.py               # Losses, metrics, callbacks, logger
│   └── 📥 data_loader.py         # Iris, MNIST, Fashion-MNIST
│
├── 📂 examples/                  # Ready-to-run scripts
│   ├── xor_example.py
│   ├── iris_example.py
│   ├── mnist_example.py
│   └── fashion_mnist_example.py
│
├── 📂 tests/                     # 35 unit tests
│   ├── test_activation.py        # 9 tests
│   ├── test_layers.py            # 11 tests
│   ├── test_model.py             # 8 tests
│   └── test_optimizer.py         # 7 tests
│
├── 📂 outputs/                   # Generated artifacts
│   ├── models/                   # Saved weights (.npz)
│   ├── logs/                     # Experiment logs (.json)
│   └── plots/                    # Training plots (.png)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Installation

### Prerequisites

- Python **3.8+**
- pip (or conda)

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/NotNot696/deep-neural-network-from-scratch.git
cd deep-neural-network-from-scratch

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the examples

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

### Build your own model

```python
from src.model import NeuralNetwork
from src.utils import one_hot_encode

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
    X_train, y_train,
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

This project doesn't stop at the basics. Every modern training technique you'd expect from a production framework is implemented from scratch:

<table>
<tr>
<td valign="top" width="50%">

### 🎯 Training Callbacks
- **EarlyStopping** — halts training when validation loss plateaus
- **ModelCheckpoint** — saves the best model during training
- **ExperimentLogger** — records every hyperparameter and result to JSON

</td>
<td valign="top" width="50%">

### 📉 Learning Rate Control
- **1Cycle** — warm-up followed by cool-down
- **ReduceOnPlateau** — reduces LR when loss stalls
- **Power / Exponential / Piecewise** — classic decay strategies

</td>
</tr>
<tr>
<td valign="top" width="50%">

### 🧱 Network Regularization
- **Batch Normalization** — stabilizes & accelerates training
- **Dropout** — prevents overfitting
- **L1 / L2 / Elastic Net** — penalizes large weights

</td>
<td valign="top" width="50%">

### 🎲 Smart Initialization
- **He (Kaiming)** — optimal for ReLU networks
- **Xavier (Glorot)** — optimal for Tanh / Sigmoid
- **LeCun** — optimal for SELU

</td>
</tr>
</table>

---

## 🧪 Testing

The project is covered by **35 comprehensive unit tests** — ensuring every component behaves as expected.

```bash
python -m pytest tests/ -v
```

Expected output:

```
========================================================== test session starts ===========================================================
collected 35 items

tests/test_activation.py .........                         [ 25%]
tests/test_layers.py ...........                           [ 57%]
tests/test_model.py ........                               [ 80%]
tests/test_optimizer.py .......                            [100%]

========================================================== 35 passed in 2.64s ===========================================================
```

### Test Coverage

| Test File | Tests | Coverage |
|:---|:---:|:---|
| `test_activation.py` | 9 | ReLU, Sigmoid, Tanh, Softmax, ELU, SELU + factory |
| `test_layers.py` | 11 | Dense, Dropout, BatchNormalization (forward + backward) |
| `test_model.py` | 8 | Forward, predict, train, evaluate, save/load |
| `test_optimizer.py` | 7 | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam + factory |
| **Total** | **35** | ✅ **100% passing** |

---

## 📈 Outputs & Visualizations

Every experiment produces artifacts that make it easy to track, reproduce, and compare results:

```
outputs/
├── 📊 plots/          # Loss curves, accuracy curves, weight histograms
├── 💾 models/         # Best model weights (compressed .npz format)
└── 📝 logs/           # Full experiment configuration and results (JSON)
```

### Sample Outputs

<details>
<summary><b>📉 Training Progress (Loss & Accuracy)</b></summary>

> Automatically generated for every experiment — includes training loss, validation loss, training accuracy, and validation accuracy.

</details>

<details>
<summary><b>💾 Best Model Checkpoint (.npz)</b></summary>

> The best model — chosen by lowest validation loss — is saved automatically. Load it back with a single call:
> ```python
> model.load_weights("outputs/models/mnist_best.npz")
> ```

</details>

<details>
<summary><b>📝 Experiment Log (.json)</b></summary>

> Each experiment is fully logged, including config, results, and history:
> ```json
> {
>   "experiment_name": "mnist_advanced",
>   "config": { "learning_rate": 0.01, "epochs": 100, ... },
>   "results": { "test_accuracy": 0.9679, "test_loss": 0.017746 },
>   "history": { "loss": [...], "accuracy": [...] }
> }
> ```

</details>

---

## 🗺️ Roadmap

- [x] Activation functions (6)
- [x] Weight initializers (4)
- [x] Layers (Dense, Dropout, BatchNorm)
- [x] Optimizers (6)
- [x] Regularizers (L1, L2, Elastic Net)
- [x] LR Schedulers (5)
- [x] Training callbacks (EarlyStopping, ModelCheckpoint, ExperimentLogger)
- [x] Full test suite (35 tests)
- [x] 4 example datasets
- [ ] Convolutional Neural Networks (CNN)
- [ ] Recurrent Neural Networks (RNN / LSTM)
- [ ] GPU acceleration with CuPy
- [ ] Additional optimizers (Nadam, AdaMax)

---

## 🤝 Contributing

Contributions are always welcome! Whether it's a bug fix, a new feature, or a documentation improvement:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

Please make sure your code passes all tests and follows the existing style.

---

## 📄 License

This project is licensed under the **MIT License**

---

## 👤 Author

<div align="center">

**Amir Arsalan Farahmand**

[![GitHub](https://img.shields.io/badge/GitHub-NotNot696-181717?style=for-the-badge&logo=github)](https://github.com/NotNot696)

</div>

---

## 🙏 Acknowledgements

This project stands on the shoulders of giants. Special thanks to the researchers whose foundational work inspired every line of code:

- **Glorot & Bengio (2010)** — *Understanding the difficulty of training deep feedforward neural networks*
- **He et al. (2015)** — *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet*
- **Kingma & Ba (2014)** — *Adam: A Method for Stochastic Optimization*
- **Ioffe & Szegedy (2015)** — *Batch Normalization: Accelerating Deep Network Training*
- **Klambauer et al. (2017)** — *Self-Normalizing Neural Networks (SELU)*
- **Smith (2018)** — *A Disciplined Approach to Neural Network Hyper-Parameters (1Cycle)*

And of course, to the **open-source community** — for making deep learning accessible to everyone.

---

<div align="center">

### ⭐ If this project helped you, please consider giving it a star! ⭐

*It means the world and helps others discover the project.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=fadeIn" width="100%"/>

**Built with ❤️ and pure NumPy**

</div>
```