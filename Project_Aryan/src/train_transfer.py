"""Deliverable 3 -- ResNet-18 transfer learning with the required two-phase
fine-tuning strategy, plus the frozen-vs-unfrozen ablation table and the
UC Merced holdout evaluation (with class re-mapping, zero-shot, no UCM
fine-tuning -- this is a domain-generalization test).

Run: python -m src.train_transfer
"""
import copy
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config, utils
from .datasets import TileDataset
from .transfer_model import build_resnet18, freeze_backbone, unfreeze_last_blocks
from .transforms import resnet_transforms

PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_LR = PHASE1_LR / 10  # "reduce learning rate by 10x"
BATCH_SIZE = 64
IMAGE_SIZE = 224


def make_loaders():
    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    train_ds = TileDataset(train_df, transform=resnet_transforms(train=True, image_size=IMAGE_SIZE))
    val_ds = TileDataset(val_df, transform=resnet_transforms(train=False, image_size=IMAGE_SIZE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


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


def train_two_phase(device, train_loader, val_loader, log_prefix="two_phase"):
    model = build_resnet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Phase 1: freeze backbone, train head only.
    freeze_backbone(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PHASE1_LR)
    for epoch in range(PHASE1_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        for k, v in zip(history, (tr_loss, val_loss, tr_acc, val_acc)):
            history[k].append(v)
        print(f"[{log_prefix}] Phase1 Epoch {epoch+1}/{PHASE1_EPOCHS} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"| {time.time()-t0:.1f}s")

    phase1_val_acc = history["val_acc"][-1]

    # Phase 2: unfreeze last 2 conv blocks, LR / 10.
    unfreeze_last_blocks(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PHASE2_LR)
    for epoch in range(PHASE2_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        for k, v in zip(history, (tr_loss, val_loss, tr_acc, val_acc)):
            history[k].append(v)
        print(f"[{log_prefix}] Phase2 Epoch {epoch+1}/{PHASE2_EPOCHS} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"| {time.time()-t0:.1f}s")

    return model, history, phase1_val_acc, val_preds, val_labels


def train_frozen_only(device, train_loader, val_loader, total_epochs):
    """Ablation: keep the backbone frozen for the *entire* training budget
    (no phase 2 unfreezing) at PHASE1_LR, for a fair frozen-vs-unfrozen
    comparison at equal epoch count."""
    model = build_resnet18(pretrained=True).to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PHASE1_LR)
    val_preds, val_labels = [], []
    for epoch in range(total_epochs):
        t0 = time.time()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"[frozen_only] Epoch {epoch+1}/{total_epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"| {time.time()-t0:.1f}s")
    return model, val_acc, val_preds, val_labels


def evaluate_ucmerced_holdout(model, device):
    """Zero-shot domain-generalization evaluation: relabel UC Merced images
    into the EuroSAT 10-class space via config.UCMERCED_TO_EUROSAT and run
    the EuroSAT-trained classifier on them directly -- no fine-tuning."""
    rows = []
    for ucm_class, eurosat_class in config.UCMERCED_TO_EUROSAT.items():
        for f in sorted((config.UCMERCED_DIR / ucm_class).glob("*.jpg")):
            rows.append({"path": str(f), "class_name": eurosat_class,
                         "class_idx": config.EUROSAT_CLASSES.index(eurosat_class)})
    df = pd.DataFrame(rows)
    ds = TileDataset(df, transform=resnet_transforms(train=False, image_size=IMAGE_SIZE))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    return all_preds, all_labels, len(df)


def main():
    utils.set_seed(config.SEED)
    device = utils.get_device()
    print("Device:", device)

    train_loader, val_loader = make_loaders()

    print("\n=== Training two-phase fine-tuned model (the model we keep) ===")
    model, history, phase1_val_acc, val_preds, val_labels = train_two_phase(device, train_loader, val_loader)
    torch.save(model.state_dict(), config.CHECKPOINT_DIR / "resnet18_finetuned.pt")
    utils.plot_loss_curves(history, config.FIGURES_DIR / "transfer_loss_curves.png",
                            title="ResNet-18 two-phase fine-tuning")

    metrics = utils.compute_classification_metrics(val_labels, val_preds, config.EUROSAT_CLASSES)
    utils.save_metrics(metrics, config.METRICS_DIR / "transfer_eurosat_val.json")
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], config.EUROSAT_CLASSES,
        config.FIGURES_DIR / "transfer_eurosat_confusion_matrix.png",
        title="ResNet-18 (two-phase) -- EuroSAT validation confusion matrix",
    )
    print("Two-phase EuroSAT val macro-F1:", metrics["macro_f1"])

    print("\n=== Ablation: frozen-only backbone, same total epoch budget ===")
    frozen_model, frozen_val_acc, frozen_preds, frozen_labels = train_frozen_only(
        device, train_loader, val_loader, total_epochs=PHASE1_EPOCHS + PHASE2_EPOCHS
    )
    frozen_metrics = utils.compute_classification_metrics(frozen_labels, frozen_preds, config.EUROSAT_CLASSES)
    utils.save_metrics(frozen_metrics, config.METRICS_DIR / "frozen_only_eurosat_val.json")

    ablation = {
        "frozen_only": {"val_acc": frozen_val_acc, "macro_f1": frozen_metrics["macro_f1"]},
        "two_phase_after_phase1": {"val_acc": phase1_val_acc},
        "two_phase_final": {"val_acc": history["val_acc"][-1], "macro_f1": metrics["macro_f1"]},
    }
    utils.save_metrics(ablation, config.METRICS_DIR / "frozen_vs_unfrozen_ablation.json")
    print("Ablation:", ablation)

    print("\n=== UC Merced holdout (zero-shot, class-mapped) ===")
    ucm_preds, ucm_labels, n = evaluate_ucmerced_holdout(model, device)
    ucm_metrics = utils.compute_classification_metrics(ucm_labels, ucm_preds, config.EUROSAT_CLASSES)
    utils.save_metrics(ucm_metrics, config.METRICS_DIR / "transfer_ucmerced_holdout.json")
    utils.plot_confusion_matrix(
        ucm_metrics["confusion_matrix"], config.EUROSAT_CLASSES,
        config.FIGURES_DIR / "ucmerced_holdout_confusion_matrix.png",
        title=f"ResNet-18 -- UC Merced holdout confusion matrix (n={n}, zero-shot)",
    )
    print(f"UC Merced holdout macro-F1 (n={n}):", ucm_metrics["macro_f1"])
    print("\nDone. Checkpoint saved to", config.CHECKPOINT_DIR / "resnet18_finetuned.pt")


if __name__ == "__main__":
    main()
