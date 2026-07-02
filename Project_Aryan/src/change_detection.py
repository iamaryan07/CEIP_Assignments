"""Deliverable 4 -- Temporal change detector.

Module 2 of the brief: reuse the fine-tuned ResNet-18 backbone as a feature
extractor (classifier head stripped), extract 512-dim embeddings, and detect
"change" via cosine similarity between a before/after (T1/T2) tile pair.

EuroSAT has no real before/after acquisitions, so a synthetic T1/T2 series is
built by treating each spatial block (see datasets.py) as one "geographic
region" with a known land-use class:
  - "unchanged" pairs draw T1 and T2 from the SAME region (same class) --
    ground truth label = 0.
  - "changed" pairs draw T1 from one region and T2 from a *different* region
    of a *different* class -- ground truth label = 1, simulating the region's
    land cover actually changing between the two time periods.
Because the ground-truth label is known by construction, we can compute a
real ROC curve for the cosine-similarity change score and justify a threshold,
exactly as the brief asks for.

Run: python -m src.change_detection
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from . import config, utils
from .datasets import TileDataset
from .transfer_model import EmbeddingExtractor, build_resnet18
from .transforms import resnet_transforms

IMAGE_SIZE = 224
BATCH_SIZE = 64
N_PAIRS_PER_CLASS_SIDE = 250  # -> ~2*10*250 = 5000 pairs total (balanced)


def load_embedding_model(device):
    classifier = build_resnet18(pretrained=False)
    classifier.load_state_dict(torch.load(config.CHECKPOINT_DIR / "resnet18_finetuned.pt", map_location=device))
    classifier.eval()
    model = EmbeddingExtractor(classifier).to(device)
    model.eval()
    return model


def build_change_pairs(seed: int = config.SEED) -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    df = df[df["split"] == "test"].reset_index(drop=True)
    rng = np.random.RandomState(seed)

    blocks = df.groupby("block_id")
    block_ids = [b for b, g in blocks if len(g) >= 2]
    pairs = []

    # Unchanged pairs: two tiles from the same region/block (same class).
    for _ in range(N_PAIRS_PER_CLASS_SIDE * config.NUM_EUROSAT_CLASSES):
        bid = block_ids[rng.randint(len(block_ids))]
        g = df[df["block_id"] == bid]
        pair = g.sample(2, random_state=rng.randint(1_000_000))
        t1, t2 = pair.iloc[0], pair.iloc[1]
        pairs.append({
            "t1_path": t1["path"], "t2_path": t2["path"],
            "t1_class": t1["class_name"], "t2_class": t2["class_name"],
            "region_id": bid, "changed": 0,
        })

    # Changed pairs: T1 from one region, T2 from a DIFFERENT region of a
    # different class (simulated land-cover change).
    class_to_blocks = df.groupby("class_name")["block_id"].unique().to_dict()
    classes = config.EUROSAT_CLASSES
    for _ in range(N_PAIRS_PER_CLASS_SIDE * config.NUM_EUROSAT_CLASSES):
        c1 = classes[rng.randint(len(classes))]
        c2 = classes[rng.randint(len(classes))]
        while c2 == c1:
            c2 = classes[rng.randint(len(classes))]
        b1 = class_to_blocks[c1][rng.randint(len(class_to_blocks[c1]))]
        b2 = class_to_blocks[c2][rng.randint(len(class_to_blocks[c2]))]
        t1 = df[df["block_id"] == b1].sample(1, random_state=rng.randint(1_000_000)).iloc[0]
        t2 = df[df["block_id"] == b2].sample(1, random_state=rng.randint(1_000_000)).iloc[0]
        pairs.append({
            "t1_path": t1["path"], "t2_path": t2["path"],
            "t1_class": t1["class_name"], "t2_class": t2["class_name"],
            "region_id": f"{b1}__{b2}", "changed": 1,
        })

    pairs_df = pd.DataFrame(pairs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pairs_df


def embed_paths(model, paths, device):
    ds = TileDataset(
        pd.DataFrame({"path": paths, "class_idx": 0, "class_name": "x"}),
        transform=resnet_transforms(train=False, image_size=IMAGE_SIZE),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    embs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            e = model(imgs)
            embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(a_n * b_n, axis=1)


def select_operating_points(y_true, change_score, fpr, tpr, thresholds):
    """Three bonus-B operating points derived from the ROC curve:
    high recall (catch almost all real changes, tolerate false alarms),
    balanced (Youden's J), and high precision (few false alarms)."""
    j = tpr - fpr
    balanced_idx = int(np.argmax(j))

    high_recall_idx = next((i for i in range(len(thresholds)) if tpr[i] >= 0.95), balanced_idx)
    high_precision_idx = next(
        (i for i in reversed(range(len(thresholds))) if fpr[i] <= 0.05), balanced_idx
    )

    def point(idx):
        return {
            "threshold_on_change_score": float(thresholds[idx]),
            "similarity_threshold": float(1 - thresholds[idx]),
            "tpr": float(tpr[idx]), "fpr": float(fpr[idx]),
        }

    return {
        "high_recall": point(high_recall_idx),
        "balanced": point(balanced_idx),
        "high_precision": point(high_precision_idx),
    }


def roc_points_for_frontend(fpr, tpr, n=60):
    """Downsample the ROC curve to ~n points for a lightweight interactive
    chart on the Next.js analytics page."""
    idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in idx]


def plot_roc(y_true, change_score, out_path):
    fpr, tpr, thresholds = roc_curve(y_true, change_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#536475", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#bfc8d1", lw=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Change detector ROC curve\n(change score = 1 - cosine similarity)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return fpr, tpr, thresholds, roc_auc


def make_change_heatmap(model, device, t1_path, t2_path, t1_class, t2_class, changed_gt, similarity, threshold, out_path):
    transform = resnet_transforms(train=False, image_size=IMAGE_SIZE)
    img1 = Image.open(t1_path).convert("RGB")
    img2 = Image.open(t2_path).convert("RGB")
    x1 = transform(img1).unsqueeze(0).to(device)
    x2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        fmap1 = model.forward_feature_map(x1)[0]  # (512, 7, 7)
        fmap2 = model.forward_feature_map(x2)[0]

    f1 = fmap1.permute(1, 2, 0).reshape(-1, fmap1.shape[0])  # (49, 512)
    f2 = fmap2.permute(1, 2, 0).reshape(-1, fmap2.shape[0])
    patch_sim = F.cosine_similarity(f1, f2, dim=1).reshape(7, 7).cpu().numpy()
    change_map = 1 - patch_sim

    change_map_up = np.array(Image.fromarray((change_map * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BICUBIC
    )) / 255.0

    flagged = similarity < threshold
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(img1.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[0].set_title(f"T1 (before)\n{t1_class}")
    axes[0].axis("off")
    axes[1].imshow(img2.resize((IMAGE_SIZE, IMAGE_SIZE)))
    axes[1].set_title(f"T2 (after)\n{t2_class}")
    axes[1].axis("off")
    im = axes[2].imshow(change_map_up, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title("Change heatmap")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    flag_text = "CHANGED" if flagged else "unchanged"
    gt_text = "changed" if changed_gt else "unchanged"
    fig.suptitle(
        f"cosine similarity = {similarity:.3f} | predicted: {flag_text} | ground truth: {gt_text}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    device = utils.get_device()
    model = load_embedding_model(device)
    print("Embedding model loaded.")

    print("Building synthetic T1/T2 region pairs...")
    pairs_df = build_change_pairs()
    pairs_df.to_csv(config.DATA_DIR / "change_pairs.csv", index=False)
    print(f"  {len(pairs_df)} pairs, {pairs_df['changed'].mean():.2%} changed")

    print("Extracting embeddings...")
    t1_emb = embed_paths(model, pairs_df["t1_path"].tolist(), device)
    t2_emb = embed_paths(model, pairs_df["t2_path"].tolist(), device)
    similarity = cosine_sim(t1_emb, t2_emb)
    change_score = 1 - similarity
    pairs_df["cosine_similarity"] = similarity

    y_true = pairs_df["changed"].to_numpy()
    fpr, tpr, thresholds, roc_auc = plot_roc(y_true, change_score, config.FIGURES_DIR / "change_detector_roc.png")
    operating_points = select_operating_points(y_true, change_score, fpr, tpr, thresholds)
    operating_points["auc"] = float(roc_auc)
    operating_points["roc_points"] = roc_points_for_frontend(fpr, tpr)
    operating_points["n_pairs"] = int(len(pairs_df))
    operating_points["pct_changed"] = float(pairs_df["changed"].mean())
    print("Operating points:", json.dumps({k: v for k, v in operating_points.items() if k != "roc_points"}, indent=2))
    utils.save_metrics(operating_points, config.METRICS_DIR / "change_detector_operating_points.json")

    balanced_threshold = operating_points["balanced"]["similarity_threshold"]
    pred = (similarity < balanced_threshold).astype(int)
    accuracy = (pred == y_true).mean()
    print(f"Balanced operating point similarity threshold = {balanced_threshold:.3f} | accuracy = {accuracy:.3f}")

    print("Generating change heatmaps for 5 sample pairs...")
    sample_changed = pairs_df[pairs_df["changed"] == 1].sample(3, random_state=config.SEED)
    sample_unchanged = pairs_df[pairs_df["changed"] == 0].sample(2, random_state=config.SEED)
    samples = pd.concat([sample_changed, sample_unchanged])
    for i, row in enumerate(samples.itertuples()):
        out_path = config.FIGURES_DIR / "change_heatmaps" / f"pair_{i+1}.png"
        make_change_heatmap(
            model, device, row.t1_path, row.t2_path, row.t1_class, row.t2_class,
            row.changed, row.cosine_similarity, balanced_threshold, out_path,
        )
    print("Saved 5 change heatmaps to", config.FIGURES_DIR / "change_heatmaps")

    pairs_df.to_csv(config.DATA_DIR / "change_pairs_scored.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
