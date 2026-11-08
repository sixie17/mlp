# 11 — Metrics

Code: `src/utils/metrics.py`

Metrics measure how good the predictions are, in ways people can interpret. Unlike the loss, they
are not used for learning. **Malignant (1) is the positive class.**

## Confusion counts — `confusion_counts(y_true, y_pred)`

| | predicted M | predicted B |
|---|---|---|
| **really M** | TP (true positive) | FN (false negative) — a missed cancer |
| **really B** | FP (false positive) — a false alarm | TN (true negative) |

Returns `(tp, tn, fp, fn)`. Example from the mandatory validation run:
`TP=41 TN=67 FP=4 FN=1` (113 tumors).

## Accuracy — `accuracy(y_true, y_pred)`

```
accuracy = (TP + TN) / total            = fraction of correct predictions
```

Mandatory example: (41 + 67) / 113 = 95.6%.

Limitation: with unbalanced classes accuracy can be misleading. A model that always says "B" gets
62.7% here while detecting no cancer at all.

## Precision — `precision(y_true, y_pred)`

```
precision = TP / (TP + FP)
```

"When the model says malignant, how often is it right?" Low precision = many false alarms.

## Recall (sensitivity) — `recall(y_true, y_pred)`

```
recall = TP / (TP + FN)
```

"Of all malignant tumors, how many did the model find?" Low recall = missed cancers. **For a
medical screening this is usually the most important metric**: a false alarm leads to more tests,
a missed cancer can be fatal.

## F1 score — `f1_score(y_true, y_pred)`

```
F1 = 2 · precision · recall / (precision + recall)
```

The harmonic mean of the two: high only if **both** are high. One number to compare models when
both kinds of error matter.

All three return 0 instead of dividing by zero when there is nothing to count.

## ROC AUC — `roc_auc(y_true, scores)`

Uses the predicted **probabilities** P(M), not the 0/1 predictions.

**Meaning:** pick a random malignant tumor and a random benign tumor. AUC is the probability that
the model gives the malignant one the higher P(M).

```python
pos = scores[y_true == 1]                       # P(M) of malignant tumors
neg = scores[y_true == 0]                       # P(M) of benign tumors
greater = np.sum(pos[:, None] > neg[None, :])   # every (malignant, benign) pair, by broadcasting
ties    = np.sum(pos[:, None] == neg[None, :])
auc = (greater + 0.5 * ties) / (len(pos) * len(neg))
```

`pos[:, None] > neg[None, :]` builds a (42 × 71) table comparing every pair at once.

| AUC | Meaning |
|-----|---------|
| 1.0 | perfect ranking |
| 0.5 | random guessing |
| < 0.5 | worse than random |

AUC does not depend on the 0.5 decision threshold: it measures how well the model **ranks** tumors.
Ours is ≈ 0.996–0.997, so almost every malignant tumor gets a higher P(M) than every benign one;
the few errors come from where the 0.5 threshold falls.

## Where they appear

| Place | Metrics |
|-------|---------|
| every epoch (mandatory) | loss, accuracy (train + validation) |
| every epoch (bonus) | + validation precision, recall, F1, AUC (F1 and AUC printed) |
| `predict` (mandatory) | binary cross-entropy, accuracy, TP/TN/FP/FN |
| `predict` (bonus) | + precision, recall, F1, AUC |

Bonus `predict` example (Adam + early stopping):

```
binary cross-entropy : 0.0783
accuracy             : 0.9735 (110/113 correct)
confusion            : TP=41 TN=69 FP=2 FN=1  (positive = M)
precision            : 0.9535
recall               : 0.9762
f1                   : 0.9647
auc                  : 0.9970
```
