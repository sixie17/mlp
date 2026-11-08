"""
Dataset loading and preprocessing for the Wisconsin breast cancer dataset.

Raw file: no header, 32 columns -> id, diagnosis (M/B), then 30 features
(10 measurements x mean / standard error / worst).
"""
import shutil

import numpy as np
import pandas as pd

MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURE_COLUMNS = [f"{measurement}_{statistic}"
                   for statistic in ("mean", "se", "worst") for measurement in MEASUREMENTS]
COLUMNS = ["id", "diagnosis"] + FEATURE_COLUMNS
LABELS = {"B": 0, "M": 1}
CLASS_NAMES = {0: "B", 1: "M"}


def read_raw_dataset(path):
    """Reads a raw (header-less) CSV and names the columns. No cleaning."""
    data = pd.read_csv(path, header=None)
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {data.shape[1]}")
    data.columns = COLUMNS
    return data


def clean_dataset(data):
    """
    Cleaning strategy (listwise deletion: a bad row is removed, never guessed):
      1. Rows without a diagnosis are dropped: there is nothing to learn from or evaluate.
      2. A diagnosis other than M / B is an error: the file is not the expected dataset.
      3. Non-numeric feature values are turned into missing values.
      4. Rows with any missing feature are dropped. We do not impute (fill with the mean /
         median) because the values are medical measurements, and the dataset has no missing
         values anyway, so dropping costs nothing while imputing would invent data.
      5. Exact duplicate rows are dropped, so the same tumor cannot end up in both the
         training and the validation set.
    Zeros are kept: concavity = 0 is a real measurement (no concave parts on the outline).

    Returns (clean_data, report) where report counts what each step removed.
    """
    data = data.copy()
    report = {"input_rows": len(data)}

    data["diagnosis"] = data["diagnosis"].astype("string").str.strip()
    missing_label = data["diagnosis"].isna()
    report["missing_diagnosis"] = int(missing_label.sum())
    data = data[~missing_label]

    unknown_labels = set(data["diagnosis"].unique()) - set(LABELS)
    if unknown_labels:
        raise ValueError(f"unknown diagnosis values {sorted(map(str, unknown_labels))} (expected M or B)")

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    incomplete = data[FEATURE_COLUMNS].isna().any(axis=1)
    report["incomplete_rows"] = int(incomplete.sum())
    data = data[~incomplete]

    duplicated = data.duplicated()
    report["duplicate_rows"] = int(duplicated.sum())
    data = data[~duplicated]

    report["duplicate_identifiers"] = int(data["id"].duplicated().sum())
    report["output_rows"] = len(data)
    if data.empty:
        raise ValueError("no usable rows left after cleaning")
    return data.reset_index(drop=True), report


def print_cleaning_report(report, prefix=""):
    removed = report["input_rows"] - report["output_rows"]
    if removed:
        print(f"{prefix}warning: removed {removed}/{report['input_rows']} rows "
              f"(no diagnosis: {report['missing_diagnosis']}, missing/non-numeric features: "
              f"{report['incomplete_rows']}, duplicates: {report['duplicate_rows']})")
    if report["duplicate_identifiers"]:
        print(f"{prefix}warning: {report['duplicate_identifiers']} ids appear more than once "
              f"with different values")


def load_dataset(path, verbose=True):
    """Reads, names, validates and cleans a dataset. Prints a warning if rows were removed."""
    try:
        data, report = clean_dataset(read_raw_dataset(path))
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    if verbose:
        print_cleaning_report(report, prefix=f"{path}: ")
    return data


def save_dataset(data, path):
    """Writes the dataset back in the raw format (no header) so it can be reloaded with load_dataset."""
    data[COLUMNS].to_csv(path, header=False, index=False)


def split_dataset(data, validation_ratio=0.2, seed=42):
    """
    Stratified random split: each class is split with the same ratio,
    so the training and validation sets keep the same proportion of M / B.
    validation_ratio is the share of rows that goes to the validation set (0.2 = 20%).
    """
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    random_generator = np.random.default_rng(seed)
    training_indices, validation_indices = [], []
    for label in LABELS:
        class_indices = random_generator.permutation(data.index[data["diagnosis"] == label].to_numpy())
        validation_count = int(round(len(class_indices) * validation_ratio))
        validation_indices.extend(class_indices[:validation_count])
        training_indices.extend(class_indices[validation_count:])
    training_indices = random_generator.permutation(training_indices)
    validation_indices = random_generator.permutation(validation_indices)
    return (data.loc[training_indices].reset_index(drop=True),
            data.loc[validation_indices].reset_index(drop=True))


def get_features_and_labels(data):
    """Drops the id, returns the features (n, 30) as floats and the labels (n,) as integers (M=1, B=0)."""
    features = data[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    labels = data["diagnosis"].map(LABELS).to_numpy(dtype=np.int64)
    return features, labels


def describe_dataset(path, row_count=5):
    """Prints the dataset dimensions and its first rows (rows labelled by ID, one column per feature)."""
    raw = read_raw_dataset(path)
    print(f"dimensions: {raw.shape[0]} rows x {raw.shape[1]} columns "
          f"(id, diagnosis, {len(FEATURE_COLUMNS)} features)\n")

    view = raw.head(row_count).set_index("id")
    view.index.name = "ID"
    width = shutil.get_terminal_size((120, 20)).columns
    with pd.option_context("display.max_columns", None, "display.width", width):
        print(view)


def one_hot(labels, class_count=2):
    """Turns integer labels into target rows: 1 -> [0, 1], 0 -> [1, 0]."""
    encoded = np.zeros((len(labels), class_count))
    encoded[np.arange(len(labels)), labels] = 1.0
    return encoded


class StandardScaler:
    """
    Standardization: x' = (x - mean) / standard_deviation, per feature.
    Fit on the training set only, then reuse the same mean / standard deviation for the
    validation set and for prediction.
    The features have very different ranges (area ~1000, smoothness ~0.1), which would
    otherwise make gradient descent slow and unstable.
    """

    def __init__(self, mean=None, standard_deviation=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.standard_deviation = (None if standard_deviation is None
                                   else np.asarray(standard_deviation, dtype=np.float64))

    def fit(self, features):
        self.mean = features.mean(axis=0)
        standard_deviation = features.std(axis=0)
        self.standard_deviation = np.where(standard_deviation == 0, 1.0, standard_deviation)
        return self

    def transform(self, features):
        if self.mean is None:
            raise RuntimeError("StandardScaler must be fitted before transform")
        return (features - self.mean) / self.standard_deviation

    def to_dict(self):
        return {"mean": self.mean.tolist(), "standard_deviation": self.standard_deviation.tolist()}

    @classmethod
    def from_dict(cls, values):
        return cls(values["mean"], values["standard_deviation"])
