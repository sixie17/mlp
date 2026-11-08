# 02 — Dataset and preprocessing

Code: `src/lib/preprocessing.py`

## The dataset

`data/data.csv` is the **Wisconsin Diagnostic Breast Cancer** dataset: 569 tumors, each described
by measurements of cell nuclei seen under a microscope (taken with a fine-needle aspiration).

- **No header line.** We name the 32 columns ourselves.
- Column 0: `id` — a sample number. It identifies the row but says nothing about the tumor,
  so it is **not** used as a feature.
- Column 1: `diagnosis` — the label to predict: `M` (malignant) or `B` (benign).
- Columns 2–31: **30 features** = 10 measurements × 3 statistics.

The 10 measurements of each nucleus:

| Measurement | Meaning |
|-------------|---------|
| radius | mean distance from the center to the edge |
| texture | variation of gray levels |
| perimeter | length of the outline |
| area | surface |
| smoothness | local variation of the radius |
| compactness | perimeter² / area − 1 |
| concavity | how deep the concave parts of the outline are |
| concave_points | how many concave parts the outline has |
| symmetry | symmetry of the nucleus |
| fractal_dimension | "coastline" complexity of the outline |

Each measurement appears three times:

| Suffix | Meaning |
|--------|---------|
| `_mean` | mean over all the nuclei in the image |
| `_se` | standard error (how much it varies) |
| `_worst` | mean of the three largest values |

So the columns are `radius_mean … fractal_dimension_mean, radius_se … fractal_dimension_se,
radius_worst … fractal_dimension_worst`.

In the code:

```python
MEASUREMENTS = ["radius", "texture", ..., "fractal_dimension"]
FEATURE_COLUMNS = [f"{m}_{stat}" for stat in ("mean", "se", "worst") for m in MEASUREMENTS]
COLUMNS = ["id", "diagnosis"] + FEATURE_COLUMNS
LABELS = {"B": 0, "M": 1}         # text label -> number
CLASS_NAMES = {0: "B", 1: "M"}    # number -> text label (used when writing predictions)
```

Facts about this file: 569 rows, **357 B (62.7%)** and **212 M (37.3%)**, no missing values,
no duplicates. 13 tumors have `concavity = 0` and `concave_points = 0` — those are real
measurements (a perfectly convex outline), not missing data.

## Step 1 — Reading: `read_raw_dataset(path)`

Reads the CSV with `header=None`, checks there are exactly 32 columns, and gives them the names
above. No cleaning yet. Used by `describe` (which shows the raw data) and by `load_dataset`.

## Step 2 — Cleaning: `clean_dataset(df)`

Real data can be broken. The strategy is **listwise deletion**: a bad row is removed, never
"repaired" with an invented value.

| Problem | What we do | Why |
|---------|------------|-----|
| No diagnosis | drop the row | nothing to learn from or evaluate |
| Diagnosis not `M`/`B` | stop with an error | the file is not the expected dataset |
| Text in a feature cell | turn it into "missing" | it can't be used as a number |
| Missing feature | drop the row | they are medical measurements: filling them with a mean would invent data. This dataset has none, so dropping costs nothing |
| Exact duplicate row | drop it | the same tumor could land in both training and validation and make validation look better than it is |
| Value 0 | **keep** | a real measurement |

The function returns `(clean_df, report)` where `report` counts what each step removed.

**Alternative you can mention:** *imputation* — replacing missing values with the median of the
training set (and saving that median in the model like the scaler). It is useful when many values
are missing; here it is not needed.

## Step 3 — Loading: `load_dataset(path)`

`read_raw_dataset` + `clean_dataset`, plus a warning if rows were removed, e.g.

```
data/broken.csv: warning: removed 4/570 rows (no diagnosis: 1, missing/non-numeric features: 2, duplicates: 1)
```

Every command that uses data (`split`, `train`, `predict`) goes through this function, so the same
rules apply everywhere.

## Step 4 — Splitting: `split_dataset(df, validation_ratio=0.2, seed=42)`

We need two sets:

- **training set** — the network learns from it (its weights are adjusted on it);
- **validation set** — never used for learning, only to measure how well the network does on
  tumors it has never seen.

The split is **stratified**: each class is shuffled and split separately, so both sets keep the
same proportion of M and B as the full dataset.

```python
for label in LABELS:                                   # "B", then "M"
    class_indices = random_generator.permutation(df.index[df["diagnosis"] == label].to_numpy())
    validation_count = int(round(len(class_indices) * validation_ratio))
    validation_indices.extend(class_indices[:validation_count])                    # 20% of this class
    training_indices.extend(class_indices[validation_count:])                    # 80% of this class
training_indices = random_generator.permutation(training_indices)                 # mix the classes again
validation_indices = random_generator.permutation(validation_indices)
```

Result:

| Set | Rows | M | B | % M |
|-----|-----:|--:|--:|----:|
| full | 569 | 212 | 357 | 37.3% |
| training | 456 | 170 | 286 | 37.3% |
| validation | 113 | 42 | 71 | 37.2% |

**Why stratify?** With a purely random split, the small validation set could by chance get, say,
30% or 45% malignant tumors, and its scores would not represent the real problem.

**Why a seed?** `np.random.default_rng(seed)` produces the same "random" order every time, so the
split is reproducible (the subject explicitly allows a seed).

`save_dataset` writes both parts back in the raw format (no header), so they can be loaded with
the same `load_dataset`.

## Step 5 — Features and labels: `get_features_and_labels(df)`

Turns a table into the two arrays the network needs:

- `X`: shape `(n, 30)`, float — the 30 feature columns (the `id` is dropped);
- `y`: shape `(n,)`, int — `M → 1`, `B → 0`.

## Step 6 — Standardization: `StandardScaler`

For each feature:

```
x' = (x − mean) / standard_deviation
```

After this every feature has mean 0 and standard deviation 1.

**Why it is necessary:** the raw features have very different scales — `area_worst` goes up to
~4254 while `fractal_dimension_se` is ~0.003 (a factor of ~140 000). In `Z = X @ W + b` the large
features would dominate, the gradients would be badly balanced, and gradient descent would zigzag
and need a tiny learning rate. With standardized inputs all features start on equal footing.

**Fit on training data only:**

```python
scaler = StandardScaler().fit(training_features)       # mean and standard deviation from the training set
training_features = scaler.transform(training_features)
validation_features = scaler.transform(validation_features)          # validation uses the TRAINING mean / standard deviation
```

Computing the mean / standard deviation on the validation set would let information about it leak into training
(*data leakage*). In real life, new patients arrive one by one: you must scale them with
statistics you already have.

**Saved with the model:** `to_dict()` / `from_dict()` let `save_model` store the mean and standard deviation in
the model file, so `predict` applies exactly the same transformation.

**Edge case:** if a feature has standard deviation = 0 (constant column), we divide by 1 instead of 0.

## Step 7 — One-hot encoding: `one_hot(y, class_count=2)`

The network has 2 output neurons (P(B), P(M)), so labels become 2-element target vectors:

```
y = [1, 0, 0]   →   [[0, 1],
                     [1, 0],
                     [1, 0]]
```

Used by the loss and by backpropagation (`probabilities − one_hot(y)`).

## Exploration: `describe_dataset(path, row_count=5)`

Prints the dimensions and the first rows of the raw file, rows labelled by ID, one column per
feature:

```
dimensions: 569 rows x 32 columns (id, diagnosis, 30 features)

         diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean ...
ID
842302           M        17.99         10.38           122.8     1001.0 ...
```
