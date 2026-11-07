# 13 — Command line and pipeline

Code: `src/lib/cli.py`, `src/lib/pipeline.py`, `src/mandatory/main.py`, `src/bonus/main.py`

## The two programs

```bash
python src/mandatory/main.py <command> [options]
python src/bonus/main.py     <command> [options]
```

| Command | mandatory | bonus |
|---------|:---------:|:-----:|
| `describe` | ✓ | ✓ |
| `split` | ✓ | ✓ |
| `train` | ✓ (always SGD) | ✓ (+ `--optimizer`, `--early_stopping`, `--patience`, `--history`) |
| `predict` | ✓ | ✓ (+ precision, recall, F1, AUC) |
| `compare` | | ✓ |

Help: run a program with no command, or `-h` / `--help`, or `<command> -h` for its options with
their defaults.

### How a `main.py` works

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # make src/ importable → "lib"
from lib.cli import ...
from lib.pipeline import ...

def build_parser(): ...        # declare the commands (from cli.py)
def main():
    args = parse_args(build_parser())
    try:
        if args.command == "train":
            run_train(...)     # call the pipeline with the parsed options
        ...
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)    # clean message instead of a traceback
        return 1
    return 0
```

The `sys.path` line lets you run the script from anywhere: it adds `src/` to Python's import path,
so `import lib` finds `src/lib/`.

The mandatory program passes `optimizer="sgd"`; the bonus program passes the user's choice,
creates an `EarlyStopping` when `--early_stopping` is set, and turns `extra_metrics` on.

## `cli.py` — options and help

| Name | Role |
|------|------|
| `HelpFormatter` | shows each option's default and keeps the examples' line breaks |
| `Parser` | on a wrong argument, prints the full help and then the error |
| `create_parser(description, examples)` | the main parser + the "commands" group |
| `parse_args(parser)` | no arguments at all → print help and exit |
| `add_describe_command`, `add_split_command`, `add_train_command`, `add_predict_command`, `add_compare_command` | declare each command and its options |

The choices for `--activation`, `--weights_initializer`, `--loss` and `--optimizer` are read from
the registries (`ACTIVATIONS`, `INITIALIZERS`, `LOSSES`, `OPTIMIZERS`), so adding a new function
there makes it available on the command line automatically.

## All options

### `describe`

| Option | Default | Meaning |
|--------|---------|---------|
| `--dataset` | `data/data.csv` | file to show |
| `--rows` | 5 | number of rows |

### `split`

| Option | Default | Meaning |
|--------|---------|---------|
| `--dataset` | `data/data.csv` | raw dataset |
| `--out_dir` | `data` | where to write `data_training.csv` and `data_validation.csv` |
| `--valid_ratio` | 0.2 | share of rows for validation |
| `--seed` | 42 | makes the split reproducible |

### `train`

| Option | Default | Meaning |
|--------|---------|---------|
| `--train` / `--valid` | `data/data_training.csv` / `data/data_validation.csv` | the two sets |
| `--layer` | `24 24` | hidden layer sizes |
| `--activation` | `sigmoid` | hidden activation: identity, relu, sigmoid, tanh |
| `--weights_initializer` | `heUniform` | heUniform, heNormal, xavierUniform, xavierNormal, glorot… |
| `--loss` | `categoricalCrossentropy` | or `binaryCrossentropy` (same values here) |
| `--epochs` | 70 | passes over the training set |
| `--batch_size` | 8 | rows per update |
| `--learning_rate` | 0.05 (mandatory) / 0.001 (bonus) | step size |
| `--seed` | 42 | weight init + shuffling |
| `--model` | `saved_model.npz` | output model |
| `--plot` | `learning_curves.png` | output plot |
| `--no_show` | off | don't open the plot window |
| `--optimizer` (bonus) | `adam` | sgd, momentum, nesterov, rmsprop, adam |
| `--early_stopping` (bonus) | off | stop when val_loss stops improving |
| `--patience` (bonus) | 10 | epochs tolerated without improvement |
| `--history` (bonus) | `history.json` | metrics history file |

### `predict`

| Option | Default | Meaning |
|--------|---------|---------|
| `--dataset` | `data/data_validation.csv` | data to predict (must contain the diagnosis to evaluate) |
| `--model` | `saved_model.npz` | model to load |
| `--output` | none | optional CSV: `id, diagnosis, predicted, probability_M` |

### `compare` (bonus)

| Option | Default | Meaning |
|--------|---------|---------|
| `--histories` | required | JSON files written by `train --history` |
| `--labels` | from each file's config | one name per file |
| `--plot` | `comparison.png` | output plot |
| `--no_show` | off | don't open the window |

## `pipeline.py` — what each command does

### `run_describe(dataset, n_rows)`
Calls `describe_dataset`.

### `run_split(dataset, out_dir, valid_ratio, seed)`
`load_dataset` → `split_dataset` → `save_dataset` ×2 → prints row counts and class balance.

### `load_training_data(train_path, valid_path)`
Loads both sets, extracts `X`/`y`, fits a `StandardScaler` **on the training set only** and applies
it to both.

### `run_train(...)`
1. `load_training_data` → prints `x_train shape : (456, 30)` and `x_valid shape : (113, 30)`;
2. `create_network` → prints the summary;
3. `get_optimizer` and `fit` → epoch lines;
4. `save_model` → `> saving model 'saved_model.npz' to disk...`;
5. bonus: writes the JSON history `{"config": {...}, "history": {...}}`;
6. `plot_learning_curves`.

### `run_predict(dataset, model_path, extra_metrics, output_path)`
1. `load_model` → network + scaler;
2. `load_dataset`, `get_features_and_labels`, check the number of features;
3. standardize with the **saved** mean/std;
4. forward → probabilities → `argmax` → predicted classes;
5. print binary cross-entropy, accuracy, confusion counts (+ bonus metrics);
6. optional predictions CSV.

### `run_compare(history_paths, labels, out_path, show)`
Reads each JSON history, names each run (label or `"<optimizer> <layers>"`), calls
`plot_comparison`.

## Typical sessions

Mandatory:

```bash
python src/mandatory/main.py describe
python src/mandatory/main.py split
python src/mandatory/main.py train
python src/mandatory/main.py predict --dataset data/data_validation.csv
```

Bonus:

```bash
python src/bonus/main.py train --optimizer adam --early_stopping --model adam.npz --history adam.json --plot adam.png
python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --model sgd.npz --history sgd.json --plot sgd.png
python src/bonus/main.py compare --histories adam.json sgd.json
python src/bonus/main.py predict --dataset data/data_validation.csv --model adam.npz
```

Outputs (`*.npz`, `*.png`, `history*.json`, the split CSVs) are listed in `.gitignore`: they are
regenerated by the commands and are not part of the submission.
