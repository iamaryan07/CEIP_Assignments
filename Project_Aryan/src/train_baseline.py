"""Deliverable 2 -- train the scratch 3-layer CNN baseline on the
spatial-block EuroSAT split. Logs loss/accuracy curves and per-class F1.

Run: python -m src.train_baseline
"""
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config, utils
from .baseline_cnn import BaselineCNN
from .datasets import TileDataset
from .transforms import baseline_transforms

EPOCHS = 12
BATCH_SIZE = 128
LR = 1e-3


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            preds = logits.argmax(1)
            total_loss += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            n += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print("Device:", device)

    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_ds = TileDataset(train_df, transform=baseline_transforms(train=True))
    val_ds = TileDataset(val_df, transform=baseline_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = BaselineCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_val_preds, best_val_labels = None, None
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {time.time()-t0:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_preds, best_val_labels = val_preds, val_labels
            torch.save(model.state_dict(), config.CHECKPOINT_DIR / "baseline_cnn.pt")

    utils.plot_loss_curves(history, config.FIGURES_DIR / "baseline_loss_curves.png", title="Baseline scratch CNN")

    metrics = utils.compute_classification_metrics(best_val_labels, best_val_preds, config.EUROSAT_CLASSES)
    utils.save_metrics(metrics, config.METRICS_DIR / "baseline_eurosat_val.json")
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], config.EUROSAT_CLASSES,
        config.FIGURES_DIR / "baseline_confusion_matrix.png",
        title="Baseline CNN -- EuroSAT validation confusion matrix",
    )
    print("Macro-F1 (val):", metrics["macro_f1"])
    print("Saved checkpoint + figures + metrics.")


if __name__ == "__main__":
    main()
