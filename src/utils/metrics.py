"""
Classification metrics, implemented from scratch. Class 1 = malignant (positive class).
"""
import numpy as np


def accuracy(true_labels, predicted_labels):
    return float(np.mean(true_labels == predicted_labels))


def confusion_counts(true_labels, predicted_labels):
    true_positives = int(np.sum((predicted_labels == 1) & (true_labels == 1)))
    true_negatives = int(np.sum((predicted_labels == 0) & (true_labels == 0)))
    false_positives = int(np.sum((predicted_labels == 1) & (true_labels == 0)))
    false_negatives = int(np.sum((predicted_labels == 0) & (true_labels == 1)))
    return true_positives, true_negatives, false_positives, false_negatives


def precision(true_labels, predicted_labels):
    true_positives, _, false_positives, _ = confusion_counts(true_labels, predicted_labels)
    flagged = true_positives + false_positives
    return true_positives / flagged if flagged else 0.0


def recall(true_labels, predicted_labels):
    true_positives, _, _, false_negatives = confusion_counts(true_labels, predicted_labels)
    actual_positives = true_positives + false_negatives
    return true_positives / actual_positives if actual_positives else 0.0


def f1_score(true_labels, predicted_labels):
    precision_value = precision(true_labels, predicted_labels)
    recall_value = recall(true_labels, predicted_labels)
    total = precision_value + recall_value
    return 2 * precision_value * recall_value / total if total else 0.0


def roc_auc(true_labels, scores):
    """
    AUC = probability that a random positive gets a higher score than a random negative
    (ties count as half). Compares every positive/negative pair.
    """
    positive_scores = scores[true_labels == 1]
    negative_scores = scores[true_labels == 0]
    if len(positive_scores) == 0 or len(negative_scores) == 0:
        return 0.0
    higher = np.sum(positive_scores[:, None] > negative_scores[None, :])
    ties = np.sum(positive_scores[:, None] == negative_scores[None, :])
    return float((higher + 0.5 * ties) / (len(positive_scores) * len(negative_scores)))
