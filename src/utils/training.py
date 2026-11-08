"""
Training loop (mini-batch gradient descent), evaluation and early stopping.
"""
import numpy as np

from .losses import get_loss
from .metrics import accuracy, f1_score, precision, recall, roc_auc


def evaluate(network, features, labels, loss="categoricalCrossentropy", extra_metrics=False):
    probabilities = network.predict_probabilities(features)
    predictions = np.argmax(probabilities, axis=1)
    results = {"loss": get_loss(loss)(probabilities, labels), "accuracy": accuracy(labels, predictions)}
    if extra_metrics:
        results.update({
            "precision": precision(labels, predictions),
            "recall": recall(labels, predictions),
            "f1": f1_score(labels, predictions),
            "auc": roc_auc(labels, probabilities[:, 1]),
        })
    return results


class EarlyStopping:
    """Stops when `monitor` has not improved by `minimum_delta` for `patience` epochs."""

    def __init__(self, monitor="validation_loss", patience=10, minimum_delta=0.0, restore_best=True):
        self.monitor = monitor
        self.patience = patience
        self.minimum_delta = minimum_delta
        self.restore_best = restore_best
        self.mode = "minimum" if "loss" in monitor else "maximum"
        self.best = None
        self.best_epoch = 0
        self.best_weights = None
        self.epochs_without_improvement = 0

    def _improved(self, value):
        if self.best is None:
            return True
        if self.mode == "minimum":
            return value < self.best - self.minimum_delta
        return value > self.best + self.minimum_delta

    def should_stop(self, epoch, value, network):
        if self._improved(value):
            self.best = value
            self.best_epoch = epoch
            self.best_weights = network.get_weights()
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def fit(network, training_features, training_labels, validation_features, validation_labels, optimizer,
        loss="categoricalCrossentropy", epochs=70, batch_size=8, seed=None, early_stopping=None,
        extra_metrics=False, verbose=True):
    """
    For each epoch: shuffle, then for each mini-batch
        1. feedforward  -> probabilities
        2. backpropagation -> gradients of the loss with respect to every W and b
        3. optimizer step (gradient descent) -> update W and b
    then measure loss / accuracy on the full training and validation sets.
    Returns the history: one list per metric, one value per epoch.
    """
    get_loss(loss)  # validate the name before training
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be >= 1")
    random_generator = np.random.default_rng(seed)
    sample_count = training_features.shape[0]
    history = {}
    digits = len(str(epochs))

    for epoch in range(1, epochs + 1):
        order = random_generator.permutation(sample_count)
        for start in range(0, sample_count, batch_size):
            batch = order[start:start + batch_size]
            probabilities = network.forward(training_features[batch])
            gradients = network.backward(probabilities, training_labels[batch])
            optimizer.apply(network.parameters(), gradients)

        training_results = evaluate(network, training_features, training_labels, loss)
        validation_results = evaluate(network, validation_features, validation_labels, loss, extra_metrics)
        record = {"loss": training_results["loss"], "accuracy": training_results["accuracy"]}
        record.update({f"validation_{key}": value for key, value in validation_results.items()})
        for key, value in record.items():
            history.setdefault(key, []).append(value)

        if verbose:
            # "loss" / "val_loss" is the exact format shown in the subject.
            line = (f"epoch {epoch:0{digits}d}/{epochs} - loss: {record['loss']:.4f}"
                    f" - val_loss: {record['validation_loss']:.4f}"
                    f" - accuracy: {record['accuracy']:.4f}"
                    f" - validation_accuracy: {record['validation_accuracy']:.4f}")
            if extra_metrics:
                line += (f" - validation_f1: {record['validation_f1']:.4f}"
                         f" - validation_auc: {record['validation_auc']:.4f}")
            print(line)

        if early_stopping and early_stopping.should_stop(epoch, record[early_stopping.monitor], network):
            if verbose:
                print(f"> early stopping at epoch {epoch}: no {early_stopping.monitor} improvement "
                      f"for {early_stopping.patience} epochs (best: epoch {early_stopping.best_epoch})")
            break

    if early_stopping and early_stopping.restore_best and early_stopping.best_weights is not None:
        network.set_weights(early_stopping.best_weights)
        if verbose:
            print(f"> restored weights from epoch {early_stopping.best_epoch}")
    return history
