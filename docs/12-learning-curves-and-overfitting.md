# 12 — Learning curves and overfitting

Code: `src/lib/plots.py`

## The plots

### `plot_learning_curves(history, out_path, show)` — mandatory

One figure, two graphs (as the subject asks), x-axis = epochs:

- **Loss**: training loss (solid) and validation loss (dashed);
- **Accuracy**: training accuracy (solid) and validation accuracy (dashed).

The figure is saved as a PNG (`--plot`, default `learning_curves.png`) and shown in a window unless
`--no_show`.

### `plot_comparison(histories, out_path, show)` — bonus

Several training runs on the same two graphs: one color per run, solid = training,
dashed = validation. Used by the `compare` command with the JSON files written by `train --history`.

`_finish` is the shared helper that lays out, saves, shows and closes a figure.

## How to read learning curves

| What you see | Meaning |
|--------------|---------|
| both losses go down together | the network is learning patterns that generalize |
| both losses stay high | **underfitting**: model too small, learning rate too low, or too few epochs |
| training loss goes down, validation loss flattens | the model has learned what it can |
| training loss goes down, **validation loss goes up** | **overfitting**: memorizing the training set |
| losses jump around | learning rate too high, or batches too small |

A **small gap** (validation slightly worse than training) is **normal**: the weights were tuned on
the training rows, and validation rows are new. Overfitting is when the gap keeps **growing** and
validation loss rises.

## What our curves show

Default mandatory run (SGD): both losses drop fast during the first ~10 epochs; validation loss
then flattens around 0.085–0.09 while training loss keeps decreasing slowly to 0.041. Validation
accuracy peaks early (≈ 97–98%) and ends at 95.6%. That is **mild overfitting** in the late epochs.

## Is the default configuration overfitting? (measured)

Same model, 5 different random splits:

| Split seed | final train loss / acc | final val loss / acc | best val_loss (epoch) |
|-----------:|-----------------------:|---------------------:|----------------------:|
| 0 | 0.030 / 99.1% | 0.166 / 95.6% | 0.157 (24) |
| 1 | 0.046 / 98.9% | 0.064 / 98.2% | 0.062 (43) |
| 2 | 0.051 / 98.7% | 0.048 / 97.3% | 0.048 (68) |
| 3 | 0.031 / 99.1% | 0.129 / 97.3% | 0.116 (32) |
| 4 | 0.056 / 98.5% | 0.032 / 99.1% | 0.032 (69) |

Conclusions:

1. **The split matters more than the settings.** Validation loss ranges from 0.032 to 0.166 just by
   changing which tumors are in the 113-row validation set. On splits 2 and 4, validation is as
   good as or better than training. One misclassified tumor = 0.9% accuracy.
2. **SGD overfits only slightly**: on some splits validation loss bottoms out around epochs 24–32
   and then rises a little.

Variants, averaged over the same 5 splits:

| Settings | train loss | val loss | val acc |
|----------|-----------:|---------:|--------:|
| default (24-24, SGD 0.05, 70 epochs) | 0.043 | 0.088 | 97.5% |
| 30 epochs | 0.056 | 0.089 | 97.7% |
| smaller 8-8 | 0.042 | 0.091 | 97.5% |
| bigger 64-64 | 0.044 | 0.086 | 97.3% |
| learning rate 0.01 | 0.079 | 0.103 | 97.3% |
| Adam 0.001, no early stopping | 0.029 | **0.117** | 96.6% |

- Fewer epochs / smaller learning rate shrink the gap only by making **training** worse.
- Network size barely matters on this dataset.
- **Adam without early stopping clearly overfits** — it learns fast and reaches the memorizing
  phase early. Early stopping fixes it (stopped at 34, restored 24 → 97.3%, BCE 0.078).

## Ways to fight overfitting

| Technique | In this project |
|-----------|-----------------|
| early stopping | yes, bonus `--early_stopping` |
| smaller network | `--layer 8 8` etc. |
| fewer epochs | `--epochs` |
| more data | not possible here |
| L2 regularization (weight decay): add `λ·W` to `dW` | not implemented (possible extra bonus) |
| dropout: randomly switch neurons off during training | not implemented |

## Choosing the dense layers

`--layer` takes one number per hidden layer = its number of neurons. Input (30) and output
(2, softmax) are added automatically.

```bash
--layer 24 24        # 30 → 24 → 24 → 2  (default)
--layer 32 16 8      # funnel
--layer 64 64 --activation relu --weights_initializer heUniform
```

Guidelines:

1. **Depth:** at least 2 hidden layers (subject default). This dataset is nearly linearly
   separable; more layers add training time and overfitting risk, not accuracy.
2. **Width:** between the output size and a bit above the input size (2 … ~64). Equal widths
   (`24 24`) or a funnel (`32 16`) are both standard.
3. **Parameters vs data:** we have 456 training rows. `8 8` = 338 parameters, `24 24` = 1 394,
   `64 64` = 6 274 (13× the number of rows — more room to memorize).
4. **Match activation and initializer:** relu ↔ He, sigmoid/tanh ↔ Xavier.
5. **Decide on validation loss, never training loss.** Start small; grow only while validation
   loss improves. If two configurations are equal, keep the smaller one.

Compare layouts with the bonus program:

```bash
python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --layer 8 8      --model s.npz --history small.json  --no_show
python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --layer 24 24    --model m.npz --history medium.json --no_show
python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --layer 64 64 64 --model b.npz --history big.json    --no_show
python src/bonus/main.py compare --histories small.json medium.json big.json --labels "8-8" "24-24" "64-64-64"
```
