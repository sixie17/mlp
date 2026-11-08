"""
Learning curves (mandatory) and multi-model comparison (bonus).
"""
import matplotlib.pyplot as plt


def _finish(figure, output_path, show):
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=120)
        print(f"> plot saved to '{output_path}'")
    if show:
        plt.show()
    plt.close(figure)


def plot_learning_curves(history, output_path=None, show=True, title="Learning curves"):
    """Two graphs: loss and accuracy, training versus validation, per epoch."""
    epochs = range(1, len(history["loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.suptitle(title)

    loss_axis.plot(epochs, history["loss"], label="training loss")
    loss_axis.plot(epochs, history["validation_loss"], "--", label="validation loss")
    loss_axis.set_xlabel("epochs")
    loss_axis.set_ylabel("loss")
    loss_axis.set_title("Loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend()

    accuracy_axis.plot(epochs, history["accuracy"], label="training accuracy")
    accuracy_axis.plot(epochs, history["validation_accuracy"], "--", label="validation accuracy")
    accuracy_axis.set_xlabel("epochs")
    accuracy_axis.set_ylabel("accuracy")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.grid(True, alpha=0.3)
    accuracy_axis.legend()

    _finish(figure, output_path, show)


def plot_comparison(histories, output_path=None, show=True):
    """
    histories: {model_name: history}. One color per model,
    solid = training, dashed = validation, on shared loss / accuracy graphs.
    """
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.suptitle("Model comparison")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for index, (name, history) in enumerate(histories.items()):
        color = colors[index % len(colors)]
        epochs = range(1, len(history["loss"]) + 1)
        loss_axis.plot(epochs, history["loss"], color=color, label=f"{name} training")
        loss_axis.plot(epochs, history["validation_loss"], "--", color=color, label=f"{name} validation")
        accuracy_axis.plot(epochs, history["accuracy"], color=color, label=f"{name} training")
        accuracy_axis.plot(epochs, history["validation_accuracy"], "--", color=color,
                           label=f"{name} validation")

    for axis, name in ((loss_axis, "loss"), (accuracy_axis, "accuracy")):
        axis.set_xlabel("epochs")
        axis.set_ylabel(name)
        axis.set_title(name.capitalize())
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")

    _finish(figure, output_path, show)
