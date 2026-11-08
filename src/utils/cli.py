"""
Command-line argument definitions shared by the mandatory and bonus programs.
Every program and command supports -h / --help; running a program without a command prints the help.
"""
import argparse
import sys

from .activations import ACTIVATIONS
from .initializers import INITIALIZERS
from .losses import LOSSES


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Shows each option's default value and keeps the line breaks of the examples."""


class Parser(argparse.ArgumentParser):
    """Prints the full help (not just a one-line usage) when the arguments are wrong."""

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\nerror: {message}\n")


def create_parser(description, examples):
    parser = Parser(description=description, epilog=f"examples:\n{examples}", formatter_class=HelpFormatter)
    subparsers = parser.add_subparsers(dest="command", metavar="command",
                                       title="commands (run '<command> -h' for its options)")
    return parser, subparsers


def parse_arguments(parser):
    """Parses the command line; with no command at all, prints the help and exits."""
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def _add_command(subparsers, name, help_text):
    return subparsers.add_parser(name, help=help_text, description=help_text, formatter_class=HelpFormatter)


def add_describe_command(subparsers):
    command = _add_command(subparsers, "describe", "show the dataset dimensions and first rows")
    command.add_argument("--dataset", default="data/data.csv", help="dataset to describe")
    command.add_argument("--rows", type=int, default=5, help="number of rows to show")
    return command


def add_split_command(subparsers):
    command = _add_command(subparsers, "split",
                           "split the raw dataset into training and validation files (stratified)")
    command.add_argument("--dataset", default="data/data.csv", help="raw dataset")
    command.add_argument("--output_directory", default="data",
                         help="where to write data_training.csv and data_validation.csv")
    command.add_argument("--validation_ratio", type=float, default=0.2,
                         help="share of rows used for validation (0.2 = 20%%)")
    command.add_argument("--seed", type=int, default=42, help="random seed for a repeatable split")
    return command


def add_train_command(subparsers, default_learning_rate):
    hidden_activations = [name for name in ACTIVATIONS if name != "softmax"]
    command = _add_command(subparsers, "train",
                           "train the network, save the model and show the learning curves")
    command.add_argument("--training", default="data/data_training.csv", help="training set")
    command.add_argument("--validation", default="data/data_validation.csv", help="validation set")
    command.add_argument("--layer", type=int, nargs="+", default=[24, 24], metavar="UNITS",
                         help="size of each hidden layer, one number per layer")
    command.add_argument("--activation", choices=hidden_activations, default="sigmoid",
                         help="hidden layer activation")
    command.add_argument("--weights_initializer", choices=list(INITIALIZERS), default="heUniform",
                         help="weight initialization")
    command.add_argument("--loss", choices=list(LOSSES), default="categoricalCrossentropy",
                         help="loss reported during training")
    command.add_argument("--epochs", type=int, default=70, help="passes over the training set")
    command.add_argument("--batch_size", type=int, default=8, help="rows per gradient descent step")
    command.add_argument("--learning_rate", type=float, default=default_learning_rate,
                         help="gradient descent step size")
    command.add_argument("--seed", type=int, default=42, help="seed for weight initialization and shuffling")
    command.add_argument("--model", default="saved_model.npz", help="where to save the model")
    command.add_argument("--plot", default="learning_curves.png", help="where to save the learning curves")
    command.add_argument("--no_show", action="store_true", help="save the plot without opening a window")
    return command


def add_predict_command(subparsers):
    command = _add_command(subparsers, "predict",
                           "load a model, predict a dataset and evaluate it (binary cross-entropy)")
    command.add_argument("--dataset", default="data/data_validation.csv", help="dataset to predict")
    command.add_argument("--model", default="saved_model.npz", help="saved model")
    command.add_argument("--output", help="optional CSV file for the predictions")
    return command


def add_compare_command(subparsers):
    command = _add_command(subparsers, "compare", "plot several training histories on the same graph")
    command.add_argument("--histories", nargs="+", required=True, metavar="FILE",
                         help="history JSON files from train")
    command.add_argument("--labels", nargs="+", metavar="NAME", help="one name per history file")
    command.add_argument("--plot", default="comparison.png", help="where to save the plot")
    command.add_argument("--no_show", action="store_true", help="save the plot without opening a window")
    return command
