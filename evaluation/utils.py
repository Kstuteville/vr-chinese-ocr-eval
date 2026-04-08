import numpy as np
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on a test set and return accuracy as a percentage.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    accuracy = (y_pred_classes == y_test).sum() / len(y_test)
    return accuracy * 100


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
