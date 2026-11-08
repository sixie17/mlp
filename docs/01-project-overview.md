# 01 — Project overview

## The goal

The subject asks for a **multilayer perceptron** (a basic neural network), written from scratch,
that classifies breast tumors as malignant or benign using the Wisconsin breast cancer dataset.

The rules that shape the whole project:

- **No neural-network libraries.** No TensorFlow, Keras, PyTorch or scikit-learn models.
  Everything that "learns" — feedforward, backpropagation, gradient descent — is written by hand.
  Allowed: a library for **linear algebra** (NumPy), for **reading data** (pandas) and for
  **plotting** (matplotlib).
- **At least two hidden layers by default**, and the network must be configurable
  (we use command-line options such as `--layer 24 24 24`).
- **A softmax output layer**, so the output is a probability distribution.
- **Three programs** (or one program with three modes): **split**, **train**, **predict**.
- **Training** prints the training and validation loss at every epoch, saves the model
  (topology + weights), and shows **two learning-curve graphs** (loss and accuracy).
- **Prediction** loads the model and evaluates it with **binary cross-entropy**.
- During the defense you must explain **feedforward**, **backpropagation** and **gradient descent**.

## The workflow

| Step | Command | Input | Output |
|------|---------|-------|--------|
| Explore | `describe` | `data/data.csv` | dimensions + first rows in the terminal |
| Split | `split` | `data/data.csv` | `data/data_training.csv`, `data/data_validation.csv` |
| Train | `train` | both split files | `saved_model.npz`, `learning_curves.png` |
| Predict | `predict` | a dataset + `saved_model.npz` | loss, accuracy, confusion counts |
| Compare (bonus) | `compare` | several `history.json` | `comparison.png` |

## Folder layout

```
mlp/
├── data/
│   └── data.csv                 the raw dataset (569 rows, no header)
├── docs/                        this documentation
├── src/
│   ├── lib/                     the library: all the logic lives here
│   │   ├── preprocessing.py     loading, cleaning, splitting, scaling, one-hot
│   │   ├── activations.py       sigmoid, relu, tanh, identity, softmax + derivatives
│   │   ├── initializers.py      heUniform, heNormal, xavierUniform, xavierNormal
│   │   ├── layers.py            DenseLayer: forward and backward pass of one layer
│   │   ├── network.py           Network: stack of layers, backprop, save/load
│   │   ├── losses.py            binary and categorical cross-entropy
│   │   ├── optimizers.py        SGD, Momentum, Nesterov, RMSprop, Adam
│   │   ├── training.py          fit loop, evaluate, EarlyStopping
│   │   ├── metrics.py           accuracy, precision, recall, F1, ROC AUC
│   │   ├── plots.py             learning curves, model comparison
│   │   ├── pipeline.py          the commands: describe, split, train, predict, compare
│   │   └── cli.py               command-line options and help, shared by both programs
│   ├── mandatory/main.py        mandatory program: plain gradient descent
│   └── bonus/main.py            bonus program: optimizers, early stopping, metrics, compare
├── requirements.txt
└── en.subject.pdf
```

### Why this split between `lib/`, `mandatory/` and `bonus/`?

- `lib/` contains **every** function and class. It does not know about the command line.
- `mandatory/main.py` and `bonus/main.py` are **thin**: they read the command-line arguments and
  call functions from `lib/pipeline.py`. The only differences are the options they offer:
  - mandatory always trains with plain gradient descent (SGD);
  - bonus adds `--optimizer`, `--early_stopping`, extra metrics, a JSON history and `compare`.

This keeps the mandatory part simple to read during the defense, while the bonus reuses the
exact same, already-tested core.

## How the modules depend on each other

```
mandatory/main.py ─┐
bonus/main.py ─────┼──► cli.py (options)      pipeline.py (commands)
                   │                               │
                   │        ┌──────────┬───────────┼────────────┬─────────┐
                   │        ▼          ▼           ▼            ▼         ▼
                   │  preprocessing  network    training     metrics    plots
                   │                 │    │        │   │
                   │                 ▼    │        ▼   ▼
                   │              layers  │   losses  optimizers (used through pipeline)
                   │              │    │  │
                   │              ▼    ▼  ▼
                   │     activations  initializers  preprocessing.one_hot
```

## Results with the default settings

| Program | Settings | Validation accuracy | Validation BCE |
|---------|----------|--------------------:|---------------:|
| mandatory | 24-24 sigmoid, SGD learning rate 0.05, batch 8, 70 epochs | 95.6% | 0.093 |
| bonus | same network, Adam learning rate 0.001 + early stopping | 97.3% | 0.078 |

Across 5 different random splits the default mandatory model averages **97.5%** validation
accuracy (see [12 — Learning curves and overfitting](12-learning-curves-and-overfitting.md)).
