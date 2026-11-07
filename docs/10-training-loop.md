# 10 — Training loop

Code: `src/lib/training.py`

This is where feedforward, backpropagation and gradient descent come together.

## `fit(...)`

```python
fit(network, x_train, y_train, x_valid, y_valid, optimizer, loss="categoricalCrossentropy",
    epochs=70, batch_size=8, seed=None, early_stopping=None, extra_metrics=False, verbose=True)
```

### Overview

```
for each of 70 epochs:
    shuffle the 456 training rows
    for each of 57 mini-batches of 8 rows:
        probs = forward(batch)                 feedforward
        grads = backward(probs, labels)        backpropagation
        optimizer.apply(params, grads)         gradient descent
    evaluate on the FULL training set and the FULL validation set
    store in history, print the epoch line
    (bonus) stop if validation loss has not improved for `patience` epochs
return history
```

### Line by line

```python
get_loss(loss)                                  # unknown loss name → error now, not after training
if epochs < 1 or batch_size < 1: raise ValueError(...)
rng = np.random.default_rng(seed)               # reproducible shuffling
n = x_train.shape[0]                            # 456
history = {}
width = len(str(epochs))                        # 70 → 2 → prints "05/70"
```

```python
for epoch in range(1, epochs + 1):
    order = rng.permutation(n)                  # new random order every epoch
```

Shuffling every epoch means the mini-batches are different each time; the network can't learn
anything from the order of the rows.

```python
    for start in range(0, n, batch_size):       # 0, 8, 16, …, 448
        batch = order[start:start + batch_size] # 8 row indices (the last batch may be smaller)
        probs = network.forward(x_train[batch])
        grads = network.backward(probs, y_train[batch])
        optimizer.apply(network.parameters(), grads)
```

The three core steps. 57 updates per epoch × 70 epochs = 3 990 weight updates.

```python
    train_res = evaluate(network, x_train, y_train, loss)
    valid_res = evaluate(network, x_valid, y_valid, loss, extra_metrics)
```

After the epoch, measure on the **complete** sets with the **current** weights. (Averaging the 57
batch losses would mix values from 57 different versions of the weights.) Evaluation never changes
weights; validation data is never used for learning.

```python
    record = {"loss": train_res["loss"], "acc": train_res["acc"]}
    record.update({f"val_{k}": v for k, v in valid_res.items()})
    for key, value in record.items():
        history.setdefault(key, []).append(value)
```

`record` for one epoch, e.g. `{"loss": 0.17, "acc": 0.958, "val_loss": 0.19, "val_acc": 0.965}`.
`history` accumulates one list per key; it feeds the learning curves and the bonus JSON file.

```python
    if verbose:
        print(f"epoch {epoch:0{width}d}/{epochs} - loss: ... - val_loss: ... - acc: ... - val_acc: ...")
```

The format required by the subject (`epoch 39/70 - loss: 0.0750 - val_loss: 0.0406`), plus
accuracy, plus `val_f1` and `val_auc` in the bonus.

## `evaluate(network, x, y, loss, extra_metrics=False)`

```python
probs = network.predict_proba(x)
preds = np.argmax(probs, axis=1)          # class with the highest probability
results = {"loss": get_loss(loss)(probs, y), "acc": accuracy(y, preds)}
if extra_metrics:
    results.update({"precision": ..., "recall": ..., "f1": ..., "auc": roc_auc(y, probs[:, 1])})
```

## `EarlyStopping` (bonus)

```python
EarlyStopping(monitor="val_loss", patience=10, min_delta=0.0, restore_best=True)
```

**Idea:** training loss keeps decreasing forever, but at some point validation loss stops improving
and starts rising — the network is memorizing the training set (overfitting). Stop at that moment
and keep the best weights.

| Attribute | Meaning |
|-----------|---------|
| `monitor` | which history key to watch (`val_loss`) |
| `mode` | `"min"` for losses (lower is better), `"max"` otherwise (e.g. `val_acc`) |
| `patience` | how many epochs without improvement are tolerated |
| `min_delta` | minimum change that counts as an improvement |
| `best`, `best_epoch`, `best_weights` | the best value so far, when, and a copy of the weights |
| `wait` | epochs since the last improvement |

`should_stop(epoch, value, network)` is called after each epoch:

- improved → remember value, epoch and a **copy** of the weights, reset `wait`, continue;
- not improved → `wait += 1`; stop when `wait >= patience`.

After the loop, if `restore_best`, the network gets the best weights back:

```
> early stopping at epoch 34: no val_loss improvement for 10 epochs (best: epoch 24)
> restored weights from epoch 24
```

So the saved model is the epoch-24 model, not the epoch-34 one.

## Where `fit` is called

`pipeline.run_train` prepares the data (load, clean, standardize), builds the network and the
optimizer, calls `fit`, then saves the model, the optional history and the plot.
