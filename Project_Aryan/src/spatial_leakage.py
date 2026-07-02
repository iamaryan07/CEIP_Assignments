"""Deliverable 6 -- spatial leakage write-up.

Trains the same scratch 3-layer CNN architecture (fresh weights, same epoch
budget) on the naive random per-image split instead of the spatial-block
split, then compares validation accuracy against the block-split baseline
already trained by train_baseline.py. The random split lets near-duplicate,
spatially-adjacent tiles leak across train/val, which is expected to inflate
the random-split accuracy relative to the honest block-split number.

Run: python -m src.spatial_leakage   (run AFTER train_baseline.py)
"""
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config, utils
from .baseline_cnn import BaselineCNN
from .datasets import TileDataset
from .train_baseline import run_epoch
from .transforms import baseline_transforms

EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3


def train_on_split(df, device):
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    train_ds = TileDataset(train_df, transform=baseline_transforms(train=True))
    val_ds = TileDataset(val_df, transform=baseline_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        best_val_acc = max(best_val_acc, val_acc)
        print(f"[random-split] Epoch {epoch+1}/{EPOCHS} val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
    return best_val_acc


def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()

    block_metrics_path = config.METRICS_DIR / "baseline_eurosat_val.json"
    if not block_metrics_path.exists():
        raise RuntimeError("Run train_baseline.py first to produce the block-split baseline metrics.")
    with open(block_metrics_path) as f:
        block_metrics = json.load(f)
    block_val_acc = sum(block_metrics["per_class_f1"].values()) / len(block_metrics["per_class_f1"])
    # per-class F1 isn't accuracy; recompute accuracy directly from the confusion matrix instead.
    import numpy as np
    cm = np.array(block_metrics["confusion_matrix"])
    block_val_acc = float(np.trace(cm) / cm.sum())

    print("Training fresh baseline CNN on the RANDOM (leaky) per-image split...")
    df_random = pd.read_csv(config.DATA_DIR / "eurosat_index_random.csv")
    random_val_acc = train_on_split(df_random, device)

    gap = random_val_acc - block_val_acc
    result = {
        "block_split_val_accuracy": block_val_acc,
        "random_split_val_accuracy": random_val_acc,
        "leakage_gap": gap,
        "explanation": (
            "EuroSAT tiles were extracted by walking each Sentinel-2 scene "
            "raster-by-raster, so tiles with adjacent file indices are also "
            "adjacent on the ground and visually near-duplicate (same field, "
            "same cloud shadow, same crop row pattern). A random per-image "
            "split scatters these near-duplicates across train AND val, so "
            "the model partly memorizes local texture instead of learning a "
            "generalizable class boundary -- inflating val accuracy. The "
            "spatial-block split keeps every tile from one ~25-tile "
            "contiguous block on the same side of the split, removing that "
            "leakage channel. The gap above is the cost of that leakage."
        ),
    }
    utils.save_metrics(result, config.METRICS_DIR / "spatial_leakage_experiment.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
