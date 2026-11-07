#!/usr/bin/env python3
"""
Mandatory part: one program with three phases (plus a data exploration command).

    python src/mandatory/main.py describe --dataset data/data.csv
    python src/mandatory/main.py split   --dataset data/data.csv
    python src/mandatory/main.py train   --layer 24 24 --epochs 70 --batch_size 8 --learning_rate 0.05
    python src/mandatory/main.py predict --dataset data/data_validation.csv

Training uses plain mini-batch gradient descent (SGD).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.cli import (add_describe_command, add_predict_command, add_split_command,
                     add_train_command, create_parser, parse_args)
from lib.pipeline import run_describe, run_predict, run_split, run_train

EXAMPLES = """\
  python src/mandatory/main.py describe
  python src/mandatory/main.py split
  python src/mandatory/main.py train
  python src/mandatory/main.py train --layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
  python src/mandatory/main.py predict --dataset data/data_validation.csv"""


def build_parser():
    parser, sub = create_parser(
        "Multilayer perceptron - mandatory part (training with plain gradient descent).", EXAMPLES)
    add_describe_command(sub)
    add_split_command(sub)
    add_train_command(sub, default_lr=0.05)
    add_predict_command(sub)
    return parser


def main():
    args = parse_args(build_parser())
    print(args)
    try:
        if args.command == "describe":
            run_describe(args.dataset, args.rows)
        elif args.command == "split":
            run_split(args.dataset, args.out_dir, args.valid_ratio, args.seed)
        elif args.command == "train":
            run_train(args.train, args.valid, hidden_layers=args.layer, activation=args.activation,
                      weights_initializer=args.weights_initializer, loss=args.loss, optimizer="sgd",
                      learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size,
                      seed=args.seed, model_path=args.model, plot_path=args.plot,
                      show_plot=not args.no_show)
        elif args.command == "predict":
            run_predict(args.dataset, args.model, output_path=args.output)
    except (ValueError, OSError, KeyError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
