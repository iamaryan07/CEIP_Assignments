"""Deliverable 7 -- error analysis: top-5 misclassified examples (ranked by
prediction confidence, i.e. the model's most *confidently wrong* calls),
shown visually with a hypothesis for each failure.

Run: python -m src.error_analysis   (run AFTER train_transfer.py)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from . import config, utils
from .datasets import TileDataset
from .transfer_model import build_resnet18
from .transforms import resnet_transforms

IMAGE_SIZE = 224

# Hand-authored hypotheses for visually-plausible confusions between EuroSAT
# classes, used when a misclassified pair matches one of these known
# look-alike categories; otherwise a generic fallback is used.
HYPOTHESES = {
    frozenset({"AnnualCrop", "PermanentCrop"}):
        "Both are flat, regularly-furrowed agricultural fields at 10m/px -- "
        "crop type (annual vs permanent/orchard) is mostly inferable from "
        "row spacing and texture that's hard to resolve at this resolution.",
    frozenset({"Highway", "River"}):
        "Both render as long, thin, gently-curving linear features bisecting "
        "the tile; asphalt-grey and water-grey are easy to confuse under "
        "haze or shadow.",
    frozenset({"Pasture", "HerbaceousVegetation"}):
        "Both are green, low-texture, non-forested ground cover with no "
        "strong distinguishing edges at 64x64 resolution.",
    frozenset({"Residential", "Industrial"}):
        "Both show dense rectilinear rooftops/structures; the residential "
        "vs industrial distinction depends on building size/spacing cues "
        "that are subtle at this scale.",
    frozenset({"SeaLake", "River"}):
        "Both are water; the model may be keying on hue/turbidity rather "
        "than the shape cues (linear vs basin) a human would use.",
}


def hypothesis_for(true_class, pred_class):
    key = frozenset({true_class, pred_class})
    if key in HYPOTHESES:
        return HYPOTHESES[key]
    return (
        f"No documented look-alike pattern for {true_class}/{pred_class}; "
        "likely an outlier tile (unusual lighting, mixed land cover at the "
        "tile boundary, or partial cloud/shadow contamination)."
    )


def main():
    device = utils.get_device()
    model = build_resnet18(pretrained=False)
    model.load_state_dict(torch.load(config.CHECKPOINT_DIR / "resnet18_finetuned.pt", map_location=device))
    model.to(device).eval()

    df = pd.read_csv(config.DATA_DIR / "eurosat_index.csv")
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    ds = TileDataset(val_df, transform=resnet_transforms(train=False, image_size=IMAGE_SIZE))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    records = []
    with torch.no_grad():
        idx = 0
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            for j in range(imgs.size(0)):
                records.append({
                    "path": val_df.iloc[idx]["path"],
                    "true_idx": int(labels[j]),
                    "pred_idx": int(preds[j]),
                    "confidence": float(confs[j]),
                })
                idx += 1

    results_df = pd.DataFrame(records)
    misclassified = results_df[results_df["true_idx"] != results_df["pred_idx"]]
    top5 = misclassified.sort_values("confidence", ascending=False).head(5)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    entries = []
    for i, row in enumerate(top5.itertuples()):
        true_class = config.EUROSAT_CLASSES[row.true_idx]
        pred_class = config.EUROSAT_CLASSES[row.pred_idx]
        img = Image.open(row.path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class} ({row.confidence:.2f})", fontsize=9)
        entries.append({
            "path": row.path,
            "true_class": true_class,
            "pred_class": pred_class,
            "confidence": row.confidence,
            "hypothesis": hypothesis_for(true_class, pred_class),
        })

    fig.suptitle("Top-5 most confidently WRONG predictions (EuroSAT validation split)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "error_analysis" / "top5_misclassified.png", dpi=150)

    with open(config.METRICS_DIR / "error_analysis.json", "w") as f:
        json.dump(entries, f, indent=2)

    for e in entries:
        print(f"- {e['true_class']} -> {e['pred_class']} (conf={e['confidence']:.2f}): {e['hypothesis']}")
    print("\nSaved figure + hypotheses.")


if __name__ == "__main__":
    main()
