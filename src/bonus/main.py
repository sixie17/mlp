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

from lib.cli import (add_compare_command, add_describe_command, add_predict_command,  # noqa: E402
                     add_split_command, add_train_command, create_parser, parse_args)
from lib.optimizers import OPTIMIZERS  # noqa: E402
from lib.pipeline import run_compare, run_describe, run_predict, run_split, run_train  # noqa: E402
from lib.training import EarlyStopping  # noqa: E402

EXAMPLES = """\
  python src/bonus/main.py split
  python src/bonus/main.py train --optimizer adam --early_stopping --model adam.npz --history adam.json
  python src/bonus/main.py train --optimizer sgd --learning_rate 0.05 --model sgd.npz --history sgd.json
  python src/bonus/main.py compare --histories adam.json sgd.json
  python src/bonus/main.py predict --dataset data/data_validation.csv --model adam.npz"""


def build_parser():
    parser, sub = create_parser(
        "Multilayer perceptron - bonus part (optimizers, early stopping, extra metrics, "
        "metrics history, model comparison).", EXAMPLES)
    add_describe_command(sub)
    add_split_command(sub)

    train = add_train_command(sub, default_lr=0.001)
    train.add_argument("--optimizer", choices=list(OPTIMIZERS), default="adam", help="optimization algorithm")
    train.add_argument("--early_stopping", action="store_true", help="stop when val_loss stops improving")
    train.add_argument("--patience", type=int, default=10, help="epochs without improvement before stopping")
    train.add_argument("--history", default="history.json", help="where to save the metrics history")

    add_predict_command(sub)
    add_compare_command(sub)
    return parser


def main():
    args = parse_args(build_parser())
    try:
        if args.command == "describe":
            run_describe(args.dataset, args.rows)
        elif args.command == "split":
            run_split(args.dataset, args.out_dir, args.valid_ratio, args.seed)
        elif args.command == "train":
            early = EarlyStopping(patience=args.patience) if args.early_stopping else None
            run_train(args.train, args.valid, hidden_layers=args.layer, activation=args.activation,
                      weights_initializer=args.weights_initializer, loss=args.loss,
                      optimizer=args.optimizer, learning_rate=args.learning_rate, epochs=args.epochs,
                      batch_size=args.batch_size, seed=args.seed, model_path=args.model,
                      plot_path=args.plot, show_plot=not args.no_show, early_stopping=early,
                      extra_metrics=True, history_path=args.history)
        elif args.command == "predict":
            run_predict(args.dataset, args.model, extra_metrics=True, output_path=args.output)
        elif args.command == "compare":
            run_compare(args.histories, args.labels, args.plot, show=not args.no_show)
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
