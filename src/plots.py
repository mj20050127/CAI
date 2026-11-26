"""使用无窗口绘图后端，输出 PNG，适合 Windows 终端与报告插图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(rows, path):
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for field, label in [("train_objective", "Training objective"),
                          ("train_clean_loss", "Clean train subset CE"), ("val_loss", "Validation CE")]:
        axes[0].plot(epochs, [float(row[field]) for row in rows], label=label)
    for field, label in [("train_clean_accuracy", "Clean train subset"), ("val_accuracy", "Validation")]:
        axes[1].plot(epochs, [100 * float(row[field]) for row in rows], label=label)
    axes[2].plot(epochs, [float(row["lr"]) for row in rows])
    for ax, title, ylabel in zip(axes, ["Loss", "Accuracy (clean inputs)", "Learning rate"], ["Cross entropy", "Accuracy (%)", "LR used"]):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion(matrix, names, path, normalized=False):
    matrix = np.asarray(matrix)
    values = matrix.astype(float)
    if normalized:
        support = values.sum(axis=1, keepdims=True)
        values = np.divide(values, support, out=np.zeros_like(values), where=support != 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    for i in range(len(names)):
        for j in range(len(names)):
            label = f"{values[i, j]:.2f}" if normalized else str(matrix[i, j])
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                    color="white" if values[i, j] > values.max() * 0.55 else "black")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
           yticklabels=names, xlabel="Predicted class", ylabel="True class",
           title="Row-normalized confusion matrix" if normalized else "Confusion matrix (counts)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_errors(images, records, names, path):
    wrong = sorted((r for r in records if r["true_label"] != r["predicted_label"]),
                   key=lambda r: r["confidence"], reverse=True)[:16]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax in axes.flat:
        ax.axis("off")
    if not wrong:
        fig.text(0.5, 0.5, "No misclassified samples", ha="center")
    for ax, record in zip(axes.flat, wrong):
        ax.imshow(images[record["index"]], interpolation="nearest")
        ax.set_title(f"#{record['index']} True: {names[record['true_label']]}\n"
                     f"Pred: {names[record['predicted_label']]} ({record['confidence']:.1%})", fontsize=9)
    fig.suptitle("Up to 16 highest-confidence errors")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)
