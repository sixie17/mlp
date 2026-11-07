# Multilayer Perceptron — Documentation

A neural network written from scratch in NumPy that predicts whether a breast tumor is
**malignant (M)** or **benign (B)** from 30 measurements of its cell nuclei.

Read the files in order: each one builds on the previous ones.

| # | File | What you learn | Code |
|---|------|----------------|------|
| 01 | [Project overview](01-project-overview.md) | The subject, the workflow, the folder layout | everything |
| 02 | [Dataset and preprocessing](02-dataset-and-preprocessing.md) | The data, cleaning, stratified split, standardization | `lib/preprocessing.py` |
| 03 | [Neural network concepts](03-neural-network-concepts.md) | Perceptron, layers, shapes, what "learning" means | — |
| 04 | [Activation functions](04-activations.md) | sigmoid, ReLU, tanh, softmax and their derivatives | `lib/activations.py` |
| 05 | [Weight initialization](05-weight-initialization.md) | Why random weights, He vs Xavier | `lib/initializers.py` |
| 06 | [Dense layer and network](06-dense-layer-and-network.md) | Feedforward, the `DenseLayer` and `Network` classes, save/load | `lib/layers.py`, `lib/network.py` |
| 07 | [Loss functions](07-loss-functions.md) | Binary and categorical cross-entropy | `lib/losses.py` |
| 08 | [Backpropagation](08-backpropagation.md) | The chain rule, the gradient formulas, step by step | `lib/network.py`, `lib/layers.py` |
| 09 | [Gradient descent and optimizers](09-gradient-descent-and-optimizers.md) | SGD, Momentum, Nesterov, RMSprop, Adam | `lib/optimizers.py` |
| 10 | [Training loop](10-training-loop.md) | Epochs, mini-batches, evaluation, early stopping | `lib/training.py` |
| 11 | [Metrics](11-metrics.md) | Accuracy, precision, recall, F1, ROC AUC | `lib/metrics.py` |
| 12 | [Learning curves and overfitting](12-learning-curves-and-overfitting.md) | Reading the plots, overfitting, choosing layers | `lib/plots.py` |
| 13 | [Command line and pipeline](13-command-line-and-pipeline.md) | Every command and option, what each program does | `lib/cli.py`, `lib/pipeline.py`, `mandatory/`, `bonus/` |
| 14 | [Evaluation guide](14-evaluation-guide.md) | What to run and what to say during the defense | — |

## Quick start

```bash
cd ~/my_projects/mlp
source bin/activate

python src/mandatory/main.py describe
python src/mandatory/main.py split
python src/mandatory/main.py train
python src/mandatory/main.py predict --dataset data/data_validation.csv
```

## The whole project in one picture

```
data/data.csv
   │  describe  → dimensions + first rows
   │  split     → data/data_training.csv (80%) + data/data_validation.csv (20%), stratified
   ▼
 train
   │  load + clean → standardize (mean/std of training set)
   │  build network 30 → 24 → 24 → 2 (softmax)
   │  for each epoch, for each mini-batch:
   │      feedforward → loss → backpropagation → gradient descent
   │  print loss / val_loss each epoch
   ▼
saved_model.npz (layers + weights + scaler)   learning_curves.png
   │
 predict → binary cross-entropy, accuracy, confusion counts
```
