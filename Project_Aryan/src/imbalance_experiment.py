"""Bonus D -- class imbalance experiment.

Downsamples two EuroSAT classes (Pasture, PermanentCrop -- the two smallest
natural choices that aren't already minority classes, so the imbalance is
clearly induced rather than incidental) to 20% of their training-split size,
retrains the baseline CNN, and compares per-class F1 with and without one
mitigation: class-weighted cross-entropy loss (inverse training frequency).

Run: python -m src.imbalance_experiment   (run AFTER prepare_data.py)
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config, utils
from .baseline_cnn import BaselineCNN
from .datasets import TileDataset
from .train_baseline import run_epoch
from .transforms import baseline_transforms

DOWNSAMPLE_CLASSES = ["Pasture", "PermanentCrop"]
DOWNSAMPLE_FRACTION = 0.20
EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3


def make_imbalanced_train_df(seed=config.SEED):
    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    train_df = df[df["split"] == "train"].copy()
    rng = np.random.RandomState(seed)

    keep_parts = []
    for class_name in config.EUROSAT_CLASSES:
        sub = train_df[train_df["class_name"] == class_name]
        if class_name in DOWNSAMPLE_CLASSES:
            n_keep = max(1, int(len(sub) * DOWNSAMPLE_FRACTION))
            sub = sub.sample(n_keep, random_state=rng.randint(1_000_000))
        keep_parts.append(sub)
    return pd.concat(keep_parts).reset_index(drop=True)


def class_weights_from(train_df):
    counts = train_df["class_idx"].value_counts().sort_index()
    counts = counts.reindex(range(config.NUM_EUROSAT_CLASSES), fill_value=1)
    weights = 1.0 / counts.to_numpy(dtype=np.float32)
    weights = weights / weights.sum() * config.NUM_EUROSAT_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def train_and_eval(train_df, val_df, device, class_weight=None):
    train_ds = TileDataset(train_df, transform=baseline_transforms(train=True))
    val_ds = TileDataset(val_df, transform=baseline_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = BaselineCNN().to(device)
    weight = class_weight.to(device) if class_weight is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"  epoch {epoch+1}/{EPOCHS} val_acc={val_acc:.4f}")

    metrics = utils.compute_classification_metrics(val_labels, val_preds, config.EUROSAT_CLASSES)
    return metrics


def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()

    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    val_df = df[df["split"] == "val"]
    imbalanced_train_df = make_imbalanced_train_df()
    print("Imbalanced train counts:\n", imbalanced_train_df["class_name"].value_counts())

    print("\n=== Training on imbalanced data, NO mitigation ===")
    no_mitigation_metrics = train_and_eval(imbalanced_train_df, val_df, device, class_weight=None)

    print("\n=== Training on imbalanced data, WITH class-weighted loss ===")
    weights = class_weights_from(imbalanced_train_df)
    print("Class weights:", dict(zip(config.EUROSAT_CLASSES, weights.tolist())))
    weighted_metrics = train_and_eval(imbalanced_train_df, val_df, device, class_weight=weights)

    comparison = {
        "downsampled_classes": DOWNSAMPLE_CLASSES,
        "downsample_fraction": DOWNSAMPLE_FRACTION,
        "no_mitigation": {
            "macro_f1": no_mitigation_metrics["macro_f1"],
            "per_class_f1": {c: no_mitigation_metrics["per_class_f1"][c] for c in DOWNSAMPLE_CLASSES},
        },
        "weighted_loss_mitigation": {
            "macro_f1": weighted_metrics["macro_f1"],
            "per_class_f1": {c: weighted_metrics["per_class_f1"][c] for c in DOWNSAMPLE_CLASSES},
        },
    }
    utils.save_metrics(comparison, config.METRICS_DIR / "imbalance_experiment.json")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
