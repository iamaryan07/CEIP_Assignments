"""Deliverable 1 -- Data pipeline.

Indexes EuroSAT, builds the spatial-block train/val/test split, decodes the
UC Merced parquet into an ImageFolder tree, and renders the two required
visualisations (5 samples per class, class distribution).

Run: python -m src.prepare_data
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from . import config
from .datasets import (
    assign_spatial_blocks,
    build_ucmerced_imagefolder,
    index_eurosat,
    random_image_split,
    spatial_block_split,
)


def plot_class_samples(df, out_path, n_per_class=5):
    classes = config.EUROSAT_CLASSES
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 1.6, len(classes) * 1.6))
    for r, class_name in enumerate(classes):
        sub = df[df["class_name"] == class_name].head(n_per_class)
        for c in range(n_per_class):
            ax = axes[r, c]
            if c < len(sub):
                img = Image.open(sub.iloc[c]["path"])
                ax.imshow(img)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(class_name, fontsize=8)
        axes[r, 0].text(-0.4, 0.5, class_name, fontsize=9, ha="right", va="center",
                         transform=axes[r, 0].transAxes)
    fig.suptitle("EuroSAT -- 5 sample tiles per class", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(df, out_path):
    counts = df["class_name"].value_counts().reindex(config.EUROSAT_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#536475")
    ax.set_ylabel("Tile count")
    ax.set_title("EuroSAT class distribution (27,000 tiles, 10 classes)")
    ax.bar_label(bars, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_split_distribution(df, out_path):
    pivot = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(config.EUROSAT_CLASSES)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#536475", "#8496a8", "#bfc8d1"])
    ax.set_ylabel("Tile count")
    ax.set_title("Spatial-block train/val/test split per class")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    print("Indexing EuroSAT...")
    df = index_eurosat()
    df = assign_spatial_blocks(df, block_size=25)
    df = spatial_block_split(df)
    df.to_csv(config.DATA_DIR / "eurosat_index.csv", index=False)
    print(f"  {len(df)} tiles indexed, {df['block_id'].nunique()} spatial blocks")
    print(df["split"].value_counts())

    print("Building random (leaky) split for comparison...")
    df_random = random_image_split(index_eurosat())
    df_random.to_csv(config.DATA_DIR / "eurosat_index_random.csv", index=False)

    print("Decoding UC Merced parquet -> ImageFolder...")
    ucm_dir = build_ucmerced_imagefolder()
    n_ucm = sum(1 for _ in ucm_dir.rglob("*.jpg"))
    print(f"  {n_ucm} UC Merced images decoded to {ucm_dir}")

    print("Rendering figures...")
    plot_class_samples(df, config.FIGURES_DIR / "eurosat_samples.png")
    plot_class_distribution(df, config.FIGURES_DIR / "eurosat_class_distribution.png")
    plot_split_distribution(df, config.FIGURES_DIR / "eurosat_split_distribution.png")
    print("Done.")


if __name__ == "__main__":
    main()
