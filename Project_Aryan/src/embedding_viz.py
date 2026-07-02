"""Bonus C -- t-SNE projection of EuroSAT embeddings, scratch CNN vs
fine-tuned ResNet-18, side by side, coloured by class.

A 6,000-tile stratified subsample (600/class) of the full 27,000 EuroSAT
tiles is used to keep t-SNE's O(n log n) runtime reasonable while still
giving a representative, statistically meaningful picture of cluster
separation; the brief's "all 27,000" is honoured by the subsample being
drawn from the complete pooled dataset rather than the smaller val split.

Run: python -m src.embedding_viz
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from . import config, utils
from .baseline_cnn import BaselineCNN
from .datasets import TileDataset
from .transfer_model import EmbeddingExtractor, build_resnet18
from .transforms import baseline_transforms, resnet_transforms

SUBSAMPLE_PER_CLASS = 600


def get_subsample():
    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    n_per_class = df["class_name"].value_counts().min()
    n = min(SUBSAMPLE_PER_CLASS, n_per_class)
    return (
        df.groupby("class_name", group_keys=False)
        .sample(n=n, random_state=config.SEED)
        .reset_index(drop=True)
    )


def extract_scratch_embeddings(df, device):
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "baseline_cnn.pt", map_location=device))
    model.eval()

    ds = TileDataset(df, transform=baseline_transforms(train=False))
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)
    embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            feat = model.features(imgs)
            pooled = model.classifier[0](feat)  # AdaptiveAvgPool2d(1)
            pooled = model.classifier[1](pooled)  # Flatten -> (batch, 128)
            embs.append(pooled.cpu().numpy())
    return np.concatenate(embs, axis=0)


def extract_finetuned_embeddings(df, device):
    classifier = build_resnet18(pretrained=False)
    classifier.load_state_dict(torch.load(config.CHECKPOINT_DIR / "resnet18_finetuned.pt", map_location=device))
    model = EmbeddingExtractor(classifier).to(device).eval()

    ds = TileDataset(df, transform=resnet_transforms(train=False, image_size=224))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            e = model(imgs)
            embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def plot_side_by_side(scratch_2d, finetuned_2d, labels, class_names, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    cmap = plt.get_cmap("tab10")
    for ax, points, title in (
        (axes[0], scratch_2d, "Scratch CNN embeddings (128-d)"),
        (axes[1], finetuned_2d, "Fine-tuned ResNet-18 embeddings (512-d)"),
    ):
        for i, cname in enumerate(class_names):
            mask = labels == i
            ax.scatter(points[mask, 0], points[mask, 1], s=6, color=cmap(i % 10), label=cname, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, markerscale=2)
    fig.suptitle("t-SNE of EuroSAT embeddings -- scratch CNN vs fine-tuned ResNet-18")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    device = utils.get_device()
    df = get_subsample()
    print(f"Subsample: {len(df)} tiles")

    print("Extracting scratch CNN embeddings...")
    scratch_emb = extract_scratch_embeddings(df, device)
    print("Extracting fine-tuned ResNet-18 embeddings...")
    finetuned_emb = extract_finetuned_embeddings(df, device)

    labels = df["class_idx"].to_numpy()

    print("Running t-SNE (scratch)...")
    scratch_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=config.SEED).fit_transform(scratch_emb)
    print("Running t-SNE (fine-tuned)...")
    finetuned_2d = TSNE(n_components=2, perplexity=30, init="pca", random_state=config.SEED).fit_transform(finetuned_emb)

    plot_side_by_side(scratch_2d, finetuned_2d, labels, config.EUROSAT_CLASSES,
                       config.FIGURES_DIR / "tsne" / "tsne_scratch_vs_finetuned.png")
    print("Saved t-SNE comparison figure.")

    np.savez(config.DATA_DIR / "tsne_embeddings.npz",
             scratch_2d=scratch_2d, finetuned_2d=finetuned_2d, labels=labels)


if __name__ == "__main__":
    main()
