"""
High-level commands used by src/mandatory/main.py and src/bonus/main.py.
"""
import json
import os

import numpy as np

from .losses import binary_cross_entropy
from .metrics import accuracy, confusion_counts, f1_score, precision, recall, roc_auc
from .network import create_network, load_model, save_model
from .optimizers import get_optimizer
from .plots import plot_comparison, plot_learning_curves
from .preprocessing import (CLASS_NAMES, StandardScaler, describe_dataset, get_features_and_labels,
                            load_dataset, save_dataset, split_dataset)
from .training import fit


def run_describe(dataset, row_count=5):
    describe_dataset(dataset, row_count)


def run_split(dataset, output_directory="data", validation_ratio=0.2, seed=42):
    data = load_dataset(dataset)
    training_data, validation_data = split_dataset(data, validation_ratio, seed)
    os.makedirs(output_directory, exist_ok=True)
    training_path = os.path.join(output_directory, "data_training.csv")
    validation_path = os.path.join(output_directory, "data_validation.csv")
    save_dataset(training_data, training_path)
    save_dataset(validation_data, validation_path)

    for name, part, path in (("training", training_data, training_path),
                             ("validation", validation_data, validation_path)):
        counts = part["diagnosis"].value_counts()
        print(f"{name:<10} : {len(part):>3} rows (M: {counts.get('M', 0)}, B: {counts.get('B', 0)}) -> {path}")
    return training_path, validation_path


def load_training_data(training_path, validation_path):
    """Loads both sets and standardizes them with statistics from the training set only."""
    training_features, training_labels = get_features_and_labels(load_dataset(training_path))
    validation_features, validation_labels = get_features_and_labels(load_dataset(validation_path))
    scaler = StandardScaler().fit(training_features)
    return (scaler.transform(training_features), training_labels,
            scaler.transform(validation_features), validation_labels, scaler)


def run_train(training_path, validation_path, hidden_layers=(24, 24), activation="sigmoid",
              weights_initializer="heUniform", loss="categoricalCrossentropy", optimizer="sgd",
              learning_rate=0.05, epochs=70, batch_size=8, seed=42, model_path="saved_model.npz",
              plot_path="learning_curves.png", show_plot=True, early_stopping=None,
              extra_metrics=False, history_path=None):
    (training_features, training_labels,
     validation_features, validation_labels, scaler) = load_training_data(training_path, validation_path)
    print(f"training set shape   : {training_features.shape}")
    print(f"validation set shape : {validation_features.shape}")

    network = create_network(training_features.shape[1], hidden_layers, activation=activation,
                             weights_initializer=weights_initializer, seed=seed)
    print(network.summary())
    print(f"optimizer: {optimizer} (learning rate {learning_rate}), loss: {loss}, "
          f"batch size: {batch_size}, epochs: {epochs}")

    history = fit(network, training_features, training_labels, validation_features, validation_labels,
                  get_optimizer(optimizer, learning_rate), loss=loss, epochs=epochs,
                  batch_size=batch_size, seed=seed, early_stopping=early_stopping,
                  extra_metrics=extra_metrics)

    print(f"> saving model '{model_path}' to disk...")
    save_model(model_path, network, scaler)

    if history_path:
        config = {"layers": list(hidden_layers), "activation": activation, "optimizer": optimizer,
                  "learning_rate": learning_rate, "batch_size": batch_size, "loss": loss}
        with open(history_path, "w") as history_file:
            json.dump({"config": config, "history": history}, history_file, indent=2)
        print(f"> metrics history saved to '{history_path}'")

    plot_learning_curves(history, plot_path, show_plot)
    return history


def run_predict(dataset, model_path="saved_model.npz", extra_metrics=False, output_path=None):
    network, scaler_values = load_model(model_path)
    data = load_dataset(dataset)
    features, labels = get_features_and_labels(data)
    if features.shape[1] != network.input_size:
        raise ValueError(f"model expects {network.input_size} features, dataset has {features.shape[1]}")
    features = StandardScaler.from_dict(scaler_values).transform(features)

    probabilities = network.predict_probabilities(features)
    predictions = np.argmax(probabilities, axis=1)
    true_positives, true_negatives, false_positives, false_negatives = confusion_counts(labels, predictions)
    results = {"binary_cross_entropy": binary_cross_entropy(probabilities, labels),
               "accuracy": accuracy(labels, predictions)}

    print(f"samples              : {len(labels)}")
    print(f"binary cross-entropy : {results['binary_cross_entropy']:.4f}")
    print(f"accuracy             : {results['accuracy']:.4f} "
          f"({true_positives + true_negatives}/{len(labels)} correct)")
    print(f"true positives       : {true_positives}   (malignant, predicted malignant)")
    print(f"true negatives       : {true_negatives}   (benign, predicted benign)")
    print(f"false positives      : {false_positives}   (benign, predicted malignant)")
    print(f"false negatives      : {false_negatives}   (malignant, predicted benign)")
    if extra_metrics:
        results.update({"precision": precision(labels, predictions), "recall": recall(labels, predictions),
                        "f1": f1_score(labels, predictions), "auc": roc_auc(labels, probabilities[:, 1])})
        for key in ("precision", "recall", "f1", "auc"):
            print(f"{key:<21}: {results[key]:.4f}")

    if output_path:
        output_table = data[["id", "diagnosis"]].copy()
        output_table["predicted"] = [CLASS_NAMES[prediction] for prediction in predictions]
        output_table["probability_malignant"] = probabilities[:, 1].round(6)
        output_table.to_csv(output_path, index=False)
        print(f"> predictions saved to '{output_path}'")
    return results


def run_compare(history_paths, labels=None, output_path="comparison.png", show=True):
    if labels and len(labels) != len(history_paths):
        raise ValueError("give one label per history file")
    histories = {}
    for index, path in enumerate(history_paths):
        with open(path) as history_file:
            saved = json.load(history_file)
        if labels:
            name = labels[index]
        else:
            config = saved.get("config", {})
            name = f"{config.get('optimizer', '?')} {config.get('layers', '')}"
        if name in histories:
            name = f"{name} ({index + 1})"
        histories[name] = saved["history"]
    plot_comparison(histories, output_path, show)
