#!/usr/bin/env python3
"""
Bonus part: everything from the mandatory program, plus
  - optimizers: momentum, nesterov, rmsprop, adam        (--optimizer)
  - early stopping                                       (--early_stopping --patience)
  - extra metrics: precision, recall, f1, ROC AUC         (always on)
  - metrics history saved to JSON                        (--history)
  - several models' learning curves on the same graph    (compare command)

    python src/bonus/main.py split
    python src/bonus/main.py train --optimizer adam --learning_rate 0.001 --early_stopping --history adam.json
    python src/bonus/main.py train --optimizer sgd  --learning_rate 0.05 --history sgd.json --model sgd.npz
    python src/bonus/main.py compare --histories adam.json sgd.json
    python src/bonus/main.py predict --dataset data/data_validation.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.cli import (add_compare_command, add_describe_command, add_predict_command,  # noqa: E402
                     add_split_command, add_train_command, create_parser, parse_arguments)
from utils.optimizers import OPTIMIZERS  # noqa: E402
from utils.pipeline import run_compare, run_describe, run_predict, run_split, run_train  # noqa: E402
from utils.training import EarlyStopping  # noqa: E402

EXAMPLES = """\
  python src/bonus/main.py split
  python src/bonus/main.py train --optimizer adam --early_stopping --model adam.npz --history adam.json
  python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --model sgd.npz --history sgd.json
  python src/bonus/main.py compare --histories adam.json sgd.json
  python src/bonus/main.py predict --dataset data/data_validation.csv --model adam.npz"""


def build_parser():
    parser, subparsers = create_parser(
        "Multilayer perceptron - bonus part (optimizers, early stopping, extra metrics, "
        "metrics history, model comparison).", EXAMPLES)
    add_describe_command(subparsers)
    add_split_command(subparsers)

    train = add_train_command(subparsers, default_learning_rate=0.001)
    train.add_argument("--optimizer", choices=list(OPTIMIZERS), default="adam", help="optimization algorithm")
    train.add_argument("--early_stopping", action="store_true",
                       help="stop when the validation loss stops improving")
    train.add_argument("--patience", type=int, default=10, help="epochs without improvement before stopping")
    train.add_argument("--history", default="history.json", help="where to save the metrics history")

    add_predict_command(subparsers)
    add_compare_command(subparsers)
    return parser


def main():
    arguments = parse_arguments(build_parser())
    try:
        if arguments.command == "describe":
            run_describe(arguments.dataset, arguments.rows)
        elif arguments.command == "split":
            run_split(arguments.dataset, arguments.output_directory, arguments.validation_ratio, arguments.seed)
        elif arguments.command == "train":
            early_stopping = EarlyStopping(patience=arguments.patience) if arguments.early_stopping else None
            run_train(arguments.training, arguments.validation, hidden_layers=arguments.layer,
                      activation=arguments.activation, weights_initializer=arguments.weights_initializer,
                      loss=arguments.loss, optimizer=arguments.optimizer,
                      learning_rate=arguments.learning_rate, epochs=arguments.epochs,
                      batch_size=arguments.batch_size, seed=arguments.seed, model_path=arguments.model,
                      plot_path=arguments.plot, show_plot=not arguments.no_show,
                      early_stopping=early_stopping, extra_metrics=True, history_path=arguments.history)
        elif arguments.command == "predict":
            run_predict(arguments.dataset, arguments.model, extra_metrics=True, output_path=arguments.output)
        elif arguments.command == "compare":
            run_compare(arguments.histories, arguments.labels, arguments.plot, show=not arguments.no_show)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
