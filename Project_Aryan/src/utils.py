"""Small shared helpers: seeding, metrics, plotting."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, f1_score

THEME = {
    "primary": "#536475",
    "secondary": "#8496a8",
    "light": "#bfc8d1",
    "dark": "#212529",
    "accent": "#495057",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_classification_metrics(y_true, y_pred, class_names) -> dict:
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    return {
        "per_class_f1": {class_names[i]: float(per_class_f1[i]) for i in range(len(class_names))},
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def save_metrics(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)


def plot_confusion_matrix(cm, class_names, out_path: Path, title: str = "Confusion matrix", normalize: bool = True):
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.55), max(5, len(class_names) * 0.5)))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
        square=True, linewidths=0.3,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(history: dict, out_path: Path, title: str = "Loss curves"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train", color="#536475")
    axes[0].plot(history["val_loss"], label="val", color="#bfc8d1")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train", color="#536475")
    axes[1].plot(history["val_acc"], label="val", color="#bfc8d1")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
