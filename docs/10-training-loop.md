# 10 — Training loop

Code: `src/utils/training.py`

This is where feedforward, backpropagation and gradient descent come together.

## `fit(...)`

```python
fit(network, training_features, training_labels, validation_features, validation_labels, optimizer, loss="categoricalCrossentropy",
    epochs=70, batch_size=8, seed=None, early_stopping=None, extra_metrics=False, verbose=True)
```

### Overview

```
for each of 70 epochs:
    shuffle the 456 training rows
    for each of 57 mini-batches of 8 rows:
        probabilities = forward(batch)                 feedforward
        gradients = backward(probabilities, labels)        backpropagation
        optimizer.apply(parameters, gradients)         gradient descent
    evaluate on the FULL training set and the FULL validation set
    store in history, print the epoch line
    (bonus) stop if validation loss has not improved for `patience` epochs
return history
```

### Line by line

```python
get_loss(loss)                                  # unknown loss name → error now, not after training
if epochs < 1 or batch_size < 1: raise ValueError(...)
random_generator = np.random.default_rng(seed)               # reproducible shuffling
n = training_features.shape[0]                            # 456
history = {}
width = len(str(epochs))                        # 70 → 2 → prints "05/70"
```

```python
for epoch in range(1, epochs + 1):
    order = random_generator.permutation(n)                  # new random order every epoch
```

Shuffling every epoch means the mini-batches are different each time; the network can't learn
anything from the order of the rows.

```python
    for start in range(0, n, batch_size):       # 0, 8, 16, …, 448
        batch = order[start:start + batch_size] # 8 row indices (the last batch may be smaller)
        probabilities = network.forward(training_features[batch])
        gradients = network.backward(probabilities, training_labels[batch])
        optimizer.apply(network.parameters(), gradients)
```

The three core steps. 57 updates per epoch × 70 epochs = 3 990 weight updates.

```python
    training_results = evaluate(network, training_features, training_labels, loss)
    validation_results = evaluate(network, validation_features, validation_labels, loss, extra_metrics)
```

After the epoch, measure on the **complete** sets with the **current** weights. (Averaging the 57
batch losses would mix values from 57 different versions of the weights.) Evaluation never changes
weights; validation data is never used for learning.

```python
    record = {"loss": training_results["loss"], "accuracy": training_results["accuracy"]}
    record.update({f"val_{k}": v for k, v in validation_results.items()})
    for key, value in record.items():
        history.setdefault(key, []).append(value)
```

`record` for one epoch, e.g. `{"loss": 0.17, "accuracy": 0.958, "val_loss": 0.19, "validation_accuracy": 0.965}`.
`history` accumulates one list per key; it feeds the learning curves and the bonus JSON file.

```python
    if verbose:
        print(f"epoch {epoch:0{width}d}/{epochs} - loss: ... - val_loss: ... - accuracy: ... - validation_accuracy: ...")
```

The format required by the subject (`epoch 39/70 - loss: 0.0750 - val_loss: 0.0406`), plus
accuracy, plus `validation_f1` and `validation_auc` in the bonus.

## `evaluate(network, x, y, loss, extra_metrics=False)`

```python
probabilities = network.predict_probabilities(x)
predictions = np.argmax(probabilities, axis=1)          # class with the highest probability
results = {"loss": get_loss(loss)(probabilities, y), "accuracy": accuracy(y, predictions)}
if extra_metrics:
    results.update({"precision": ..., "recall": ..., "f1": ..., "auc": roc_auc(y, probabilities[:, 1])})
```

## `EarlyStopping` (bonus)

```python
EarlyStopping(monitor="val_loss", patience=10, minimum_delta=0.0, restore_best=True)
```

**Idea:** training loss keeps decreasing forever, but at some point validation loss stops improving
and starts rising — the network is memorizing the training set (overfitting). Stop at that moment
and keep the best weights.

| Attribute | Meaning |
|-----------|---------|
| `monitor` | which history key to watch (`val_loss`) |
| `mode` | `"min"` for losses (lower is better), `"max"` otherwise (e.g. `validation_accuracy`) |
| `patience` | how many epochs without improvement are tolerated |
| `minimum_delta` | minimum change that counts as an improvement |
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
