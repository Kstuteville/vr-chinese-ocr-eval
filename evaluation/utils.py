import matplotlib.pyplot as plt

from evaluation.metrics import evaluate, robustness_gap, macro_mean_cer, cer, mean_cer, exact_match


def plot_loss(history):
    """
    Plot the training and validation loss and accuracy.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
