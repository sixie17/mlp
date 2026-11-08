# 14 — Evaluation guide

A checklist for the defense: what to run, what to show, what to explain.

## Before the evaluation

```bash
cd ~/my_projects/mlp
source bin/activate
python src/mandatory/main.py -h          # check everything starts
```

- Dependencies: `numpy`, `pandas`, `matplotlib` only for the code. The NN is written by hand.
- Remove any debug `print` you added (e.g. `print(args)` in `src/mandatory/main.py`).

## Demo — mandatory

```bash
python src/mandatory/main.py describe                      # the data
python src/mandatory/main.py split                         # 456 / 113 rows, stratified
python src/mandatory/main.py train                         # epochs, save, 2 learning curves
python src/mandatory/main.py predict --dataset data/data_validation.csv
python src/mandatory/main.py train --layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314   # modularity
```

Expected: ≈ 95.6% validation accuracy, binary cross-entropy ≈ 0.093 (seed 42).

## Demo — bonus

```bash
python src/bonus/main.py train --optimizer adam --early_stopping --model adam.npz --history adam.json --plot adam.png
python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --model sgd.npz --history sgd.json --plot sgd.png
python src/bonus/main.py compare --histories adam.json sgd.json
python src/bonus/main.py predict --dataset data/data_validation.csv --model adam.npz
```

| Bonus item from the subject | Where |
|-----------------------------|-------|
| more complex optimizer | `--optimizer momentum / nesterov / rmsprop / adam` |
| several learning curves on one graph | `compare` |
| history of the metrics | `--history` JSON file |
| early stopping | `--early_stopping --patience` |
| multiple metrics | precision, recall, F1, AUC in training and `predict` |

## Subject requirements → implementation

| Requirement | Implementation |
|-------------|----------------|
| no NN library, from scratch | `src/lib/` uses only NumPy for math |
| ≥ 2 hidden layers by default | `--layer 24 24` |
| modular network | `--layer`, `--activation`, `--weights_initializer`, `--loss`, `--epochs`, `--batch_size`, `--learning_rate` |
| softmax output | `activations.softmax`, enforced by `Network` |
| split program (seed allowed) | `split`, stratified, `--seed` |
| training with backpropagation + gradient descent | `Network.backward`, `SGD`, `fit` |
| metrics at every epoch | `epoch xx/70 - loss: … - val_loss: …` |
| save topology + weights | `saved_model.npz` (+ scaler) |
| two learning curves | `learning_curves.png` (loss, accuracy) |
| prediction evaluated with binary cross-entropy | `predict` |

## Questions you will be asked — short answers

**Feedforward?**
The input passes through each layer: `Z = X·W + b`, then the activation `A = f(Z)`; the output of a
layer is the input of the next. The last layer applies softmax and gives P(B), P(M).
→ [06](06-dense-layer-and-network.md)

**Backpropagation?**
The chain rule applied from the loss back to the first layer, to get the derivative of the loss for
every weight. At the output, softmax + cross-entropy gives `dZ = (P − Y)/m`. For each layer:
`dW = Xᵀ·dZ`, `db = Σ dZ`, `dX = dZ·Wᵀ`, then `dZ_prev = dX * f'(Z_prev)`.
→ [08](08-backpropagation.md)

**Gradient descent?**
Move every parameter a small step against its gradient: `w ← w − η·∂L/∂w`. The learning rate `η`
sets the step size. We do it on mini-batches of 8 rows: 57 updates per epoch.
→ [09](09-gradient-descent-and-optimizers.md)

**Why softmax?**
It turns the outputs into probabilities that sum to 1 (subject requirement), and with
cross-entropy the output gradient simplifies to `P − Y`.

**What is the loss?**
Cross-entropy: `−log(probability of the correct class)`, averaged. With 2 softmax outputs,
binary and categorical cross-entropy are identical.
→ [07](07-loss-functions.md)

**Why preprocess / standardize?**
Features have scales differing by ~140 000× (`area_worst` vs `fractal_dimension_se`). Without
scaling, big features dominate and gradient descent is slow and unstable. Mean/standard deviation come from the
training set only (no leakage) and are saved in the model for prediction.
→ [02](02-dataset-and-preprocessing.md)

**Missing data?**
None in this file. Strategy: drop rows with no label, missing or non-numeric features, and exact
duplicates; never invent values; zeros are real and kept.

**Why a validation set? Why stratified?**
To measure performance on tumors never used for learning, and to detect overfitting. Stratified so
both sets keep 37% malignant.

**Why random initialization? Why He / Xavier?**
Equal weights → all neurons identical forever. The scale keeps the signal from vanishing or
exploding: He for ReLU (`√(6/input_size)`), Xavier for sigmoid/tanh (`√(6/(input_size+output_size))`).
→ [05](05-weight-initialization.md)

**Why is validation worse than training? Is it overfitting?**
A small gap is normal. Overfitting = validation loss rising while training loss falls. SGD overfits
only slightly; Adam overfits clearly without early stopping. Across 5 splits the default model
averages 97.5% validation accuracy; the split itself changes validation loss from 0.03 to 0.17.
→ [12](12-learning-curves-and-overfitting.md)

**How did you choose the layers?**
Subject default of 2 hidden layers; sizes between 2 and ~64; compared 8-8, 24-24, 64-64 —
all ≈ 97.5% on validation, so a small network is enough.

**What is an epoch / a batch / the learning rate?**
Epoch = one pass over the training set; batch = rows per update; learning rate = step size.

**How do you know backprop is correct?**
Gradient checking: numeric derivative `(L(w+ε) − L(w−ε)) / 2ε` vs backprop → difference ≈ 1e-10.

**What does Adam do?**
Momentum (average of gradients) + RMSprop (per-parameter step size from the average squared
gradient), with bias correction for the first steps.

**What do precision / recall / AUC mean?**
Precision: flagged malignant and really malignant. Recall: malignant tumors that were found (most
important medically). AUC: probability a random malignant tumor gets a higher P(M) than a random
benign one.
→ [11](11-metrics.md)
