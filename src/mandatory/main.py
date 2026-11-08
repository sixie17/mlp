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
                     add_train_command, create_parser, parse_arguments)
from lib.pipeline import run_describe, run_predict, run_split, run_train

EXAMPLES = """\
  python src/mandatory/main.py describe
  python src/mandatory/main.py split
  python src/mandatory/main.py train
  python src/mandatory/main.py train --layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314
  python src/mandatory/main.py predict --dataset data/data_validation.csv"""


def build_parser():
    parser, subparsers = create_parser(
        "Multilayer perceptron - mandatory part (training with plain gradient descent).", EXAMPLES)
    add_describe_command(subparsers)
    add_split_command(subparsers)
    add_train_command(subparsers, default_learning_rate=0.05)
    add_predict_command(subparsers)
    return parser


def main():
    arguments = parse_arguments(build_parser())
    print(arguments)
    try:
        if arguments.command == "describe":
            run_describe(arguments.dataset, arguments.rows)
        elif arguments.command == "split":
            run_split(arguments.dataset, arguments.output_directory, arguments.validation_ratio, arguments.seed)
        elif arguments.command == "train":
            run_train(arguments.training, arguments.validation, hidden_layers=arguments.layer,
                      activation=arguments.activation, weights_initializer=arguments.weights_initializer,
                      loss=arguments.loss, optimizer="sgd", learning_rate=arguments.learning_rate,
                      epochs=arguments.epochs, batch_size=arguments.batch_size, seed=arguments.seed,
                      model_path=arguments.model, plot_path=arguments.plot, show_plot=not arguments.no_show)
        elif arguments.command == "predict":
            run_predict(arguments.dataset, arguments.model, output_path=arguments.output)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
