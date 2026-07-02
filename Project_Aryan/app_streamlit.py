"""Module 3 -- Geo-Dashboard (Streamlit).

Two tabs:
  - Change Detector: upload a before/after tile pair, see predicted land-use
    class + confidence, cosine-similarity change flag, and a heatmap.
  - Model Analytics: full project results -- all metrics, figures, and
    experiment outputs for grading.

Run: streamlit run app_streamlit.py --server.fileWatcherType none
"""
import json

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.transfer_model import EmbeddingExtractor, build_resnet18
from src.transforms import resnet_transforms

st.set_page_config(page_title="TerraScope -- Geo-Dashboard", layout="wide")

IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ─────────────────────────────────────────────────────────────────

def md_table(headers, rows):
    """Render a markdown table -- avoids PyArrow entirely."""
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return f"{head}\n{sep}\n{body}"


def show_fig(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.caption(f"_(figure not found: {path.name})_")


# ── loaders ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    classifier = build_resnet18(pretrained=False)
    classifier.load_state_dict(
        torch.load(config.CHECKPOINT_DIR / "resnet18_finetuned.pt",
                   map_location=DEVICE, weights_only=False))
    classifier.to(DEVICE).eval()
    embedder = EmbeddingExtractor(classifier).to(DEVICE).eval()
    return classifier, embedder


@st.cache_data
def load_json(name):
    path = config.METRICS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def get_classifier_samples():
    """One representative image per EuroSAT class."""
    samples = {}
    base = config.DATA_DIR / "eurosat" / "2750"
    for cls in config.EUROSAT_CLASSES:
        folder = base / cls
        if folder.exists():
            imgs = sorted(folder.glob("*.jpg"))
            if imgs:
                samples[cls] = imgs[0]
    return samples


@st.cache_data
def get_change_samples():
    """A handful of pre-scored T1/T2 pairs from change_pairs_scored.csv."""
    import pandas as pd
    csv = config.DATA_DIR / "change_pairs_scored.csv"
    if not csv.exists():
        return []
    df = pd.read_csv(csv)
    changed   = df[df["changed"] == 1].head(3)
    unchanged = df[df["changed"] == 0].head(2)
    rows = []
    for _, r in pd.concat([changed, unchanged]).iterrows():
        label = "Changed" if r["changed"] else "Unchanged"
        rows.append({
            "label":   f"{label} -- {r['t1_class']} / {r['t2_class']}",
            "t1_path": r["t1_path"],
            "t2_path": r["t2_path"],
        })
    return rows


@st.cache_data
def load_operating_points():
    data = load_json("change_detector_operating_points.json")
    if data:
        return data
    return {
        "high_recall":    {"similarity_threshold": 0.75},
        "balanced":       {"similarity_threshold": 0.6},
        "high_precision": {"similarity_threshold": 0.45},
    }


# ── inference helpers ────────────────────────────────────────────────────────

def predict(classifier, img: Image.Image):
    transform = resnet_transforms(train=False, image_size=IMAGE_SIZE)
    x = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(classifier(x), dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    return config.EUROSAT_CLASSES[idx], float(probs[idx]), probs


def embed(embedder, img: Image.Image):
    transform = resnet_transforms(train=False, image_size=IMAGE_SIZE)
    x = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return embedder(x)[0].cpu().numpy()


def feature_map(embedder, img: Image.Image):
    transform = resnet_transforms(train=False, image_size=IMAGE_SIZE)
    x = transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return embedder.forward_feature_map(x)[0]


def change_heatmap(embedder, img1, img2):
    fmap1, fmap2 = feature_map(embedder, img1), feature_map(embedder, img2)
    f1 = fmap1.permute(1, 2, 0).reshape(-1, fmap1.shape[0])
    f2 = fmap2.permute(1, 2, 0).reshape(-1, fmap2.shape[0])
    patch_sim = F.cosine_similarity(f1, f2, dim=1).reshape(7, 7).cpu().numpy()
    change_map = np.clip(1 - patch_sim, 0, 1)
    big = Image.fromarray((change_map * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return big


# ── tab 1: classifier ────────────────────────────────────────────────────────

def render_classifier(classifier):
    st.header("Single-image land-use classifier")
    st.caption(
        "The fine-tuned ResNet-18 returns the predicted land-use class, confidence score, "
        "and full probability distribution over all 10 EuroSAT classes."
    )

    clf_samples = get_classifier_samples()
    sample_options = ["-- upload your own --"] + list(clf_samples.keys())
    chosen = st.selectbox("Select a sample image", sample_options, key="clf_sample")

    img = None
    if chosen != "-- upload your own --":
        path = clf_samples.get(chosen)
        if path and path.exists():
            img = Image.open(path).convert("RGB")
            st.caption(f"Sample: {path.name}")
        else:
            st.warning(f"Sample not found for {chosen}.")

    if img is None:
        uploaded = st.file_uploader("Or upload your own tile", type=["jpg", "jpeg", "png"], key="clf")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")

    if img is None:
        st.info("Select a sample above or upload a tile to classify it.")
        return
    cls, conf, probs = predict(classifier, img)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(img, caption="Input tile", width="stretch")
        st.metric("Predicted class", cls)
        st.metric("Confidence", f"{conf:.1%}")
    with c2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#2e6fac" if c == cls else "#536475" for c in config.EUROSAT_CLASSES]
        ax.bar(config.EUROSAT_CLASSES, probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Class probabilities")
        ax.set_xticks(range(len(config.EUROSAT_CLASSES)))
        ax.set_xticklabels(config.EUROSAT_CLASSES, rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("**All class scores**")
    ranked = sorted(zip(config.EUROSAT_CLASSES, probs), key=lambda x: x[1], reverse=True)
    st.markdown(md_table(
        ["Rank", "Class", "Probability"],
        [[i + 1, c, f"{p:.1%}"] for i, (c, p) in enumerate(ranked)]
    ))


# ── tab 2: change detector ───────────────────────────────────────────────────

def render_change_detector(classifier, embedder, operating_points):
    st.header("Before / after satellite tile pair")
    st.caption(
        "The fine-tuned ResNet-18 backbone classifies each tile and its 512-dim "
        "embedding drives a cosine-similarity change flag."
    )

    change_samples = get_change_samples()
    if change_samples:
        sample_labels = ["-- upload your own --"] + [s["label"] for s in change_samples]
        chosen_pair = st.selectbox("Select a sample pair", sample_labels, key="cd_sample")
    else:
        chosen_pair = "-- upload your own --"

    img1 = img2 = None

    if chosen_pair != "-- upload your own --":
        pair = next(s for s in change_samples if s["label"] == chosen_pair)
        from pathlib import Path
        p1, p2 = Path(pair["t1_path"]), Path(pair["t2_path"])
        if p1.exists() and p2.exists():
            img1 = Image.open(p1).convert("RGB")
            img2 = Image.open(p2).convert("RGB")
            st.caption(f"T1: {p1.name}    T2: {p2.name}")
        else:
            st.warning("Sample image files not found on disk.")

    if img1 is None or img2 is None:
        col1, col2 = st.columns(2)
        with col1:
            file1 = st.file_uploader("Before (T1)", type=["jpg", "jpeg", "png"], key="t1")
        with col2:
            file2 = st.file_uploader("After (T2)", type=["jpg", "jpeg", "png"], key="t2")
        if not (file1 and file2):
            st.info("Select a sample above or upload both tiles to run change detection.")
            return
        img1 = Image.open(file1).convert("RGB")
        img2 = Image.open(file2).convert("RGB")

    cls1, conf1, probs1 = predict(classifier, img1)
    cls2, conf2, probs2 = predict(classifier, img2)
    emb1, emb2 = embed(embedder, img1), embed(embedder, img2)
    similarity = float(
        np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

    op_name = st.session_state.get("op_name", "balanced")
    threshold = operating_points[op_name]["similarity_threshold"]
    changed = similarity < threshold

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img1, caption=f"T1 -- {cls1} ({conf1:.1%})", width="stretch")
    with c2:
        heat = change_heatmap(embedder, img1, img2)
        st.image(heat, caption="Change heatmap", width="stretch")
    with c3:
        st.image(img2, caption=f"T2 -- {cls2} ({conf2:.1%})", width="stretch")

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Cosine similarity", f"{similarity:.3f}")
    m2.metric("Threshold used", f"{threshold:.3f}", help=f"Operating point: {op_name}")
    m3.metric("Change flag", "CHANGED" if changed else "unchanged",
              delta=None if not changed else "below threshold")

    if changed:
        st.warning(f"Similarity {similarity:.3f} < threshold {threshold:.3f} -- flagged as **changed**.")
    else:
        st.success(f"Similarity {similarity:.3f} >= threshold {threshold:.3f} -- **no significant change**.")

    with st.expander("Full class probability distributions"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
        for ax, probs, title in (
            (ax1, probs1, "T1 probabilities"),
            (ax2, probs2, "T2 probabilities"),
        ):
            ax.bar(config.EUROSAT_CLASSES, probs, color="#536475")
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title(title)
            ax.set_xticks(range(len(config.EUROSAT_CLASSES)))
            ax.set_xticklabels(config.EUROSAT_CLASSES, rotation=45, ha="right", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ── tab 2: model analytics ───────────────────────────────────────────────────

def render_analytics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline   = load_json("baseline_eurosat_val.json")
    transfer   = load_json("transfer_eurosat_val.json")
    ablation   = load_json("frozen_vs_unfrozen_ablation.json")
    ucm        = load_json("transfer_ucmerced_holdout.json")
    leakage    = load_json("spatial_leakage_experiment.json")
    change_ops = load_json("change_detector_operating_points.json")
    imbalance  = load_json("imbalance_experiment.json")
    errors     = load_json("error_analysis.json")

    FIGS = config.FIGURES_DIR

    # ── 1. Dataset ──────────────────────────────────────────────────────────
    st.subheader("1. Dataset -- EuroSAT")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(FIGS / "eurosat_class_distribution.png", "Class distribution (27,000 tiles)")
    with c2:
        show_fig(FIGS / "eurosat_samples.png", "Sample tiles per class")
    show_fig(FIGS / "eurosat_split_distribution.png", "Train / val / test split distribution")

    st.divider()

    # ── 2. Module 1 -- Classifier ───────────────────────────────────────────
    st.subheader("2. Module 1 -- Land-Use Classifier")

    if baseline and transfer:
        model_rows = [
            ["Baseline scratch CNN",             f"{baseline['macro_f1']:.3f}"],
            ["ResNet-18 (two-phase fine-tuning)", f"{transfer['macro_f1']:.3f}"],
        ]
        if ucm:
            model_rows.append(["ResNet-18 -> UC Merced holdout (zero-shot)", f"{ucm['macro_f1']:.3f}"])
        st.markdown(md_table(["Model", "Macro-F1 (EuroSAT val)"], model_rows))

    if ablation:
        st.markdown("**Frozen vs. two-phase ablation** (equal 8-epoch budget)")
        st.markdown(md_table(
            ["Configuration", "Val accuracy"],
            [
                ["Frozen backbone -- all 8 epochs",          f"{ablation['frozen_only']['val_acc']:.3f}"],
                ["Two-phase -- after phase 1 (3 epochs)",    f"{ablation['two_phase_after_phase1']['val_acc']:.3f}"],
                ["Two-phase -- final (+5 unfrozen epochs)",  f"{ablation['two_phase_final']['val_acc']:.3f}"],
            ]
        ))

    c1, c2 = st.columns(2)
    with c1:
        show_fig(FIGS / "baseline_loss_curves.png", "Baseline CNN -- loss & accuracy curves")
    with c2:
        show_fig(FIGS / "transfer_loss_curves.png", "ResNet-18 two-phase -- loss & accuracy curves")

    st.markdown("**Confusion matrices**")
    c1, c2 = st.columns(2)
    with c1:
        show_fig(FIGS / "baseline_confusion_matrix.png", "Baseline CNN (EuroSAT val)")
    with c2:
        show_fig(FIGS / "transfer_eurosat_confusion_matrix.png", "ResNet-18 (EuroSAT val)")

    st.divider()

    # ── 3. Domain generalization -- UC Merced ───────────────────────────────
    st.subheader("3. Domain Generalization -- UC Merced Zero-Shot Holdout")
    st.markdown(
        "ResNet-18 trained only on EuroSAT, evaluated on UC Merced via a "
        "21->10 semantic class mapping. No UC Merced images used during training."
    )
    if ucm:
        st.markdown(f"**Overall macro-F1: {ucm['macro_f1']:.3f}**")
        st.markdown(md_table(
            ["Class", "F1"],
            [[cls, f"{f1:.3f}"] for cls, f1 in ucm["per_class_f1"].items()]
        ))
    show_fig(FIGS / "ucmerced_holdout_confusion_matrix.png",
             "ResNet-18 on UC Merced holdout (zero-shot, class-mapped)")

    st.divider()

    # ── 4. Module 2 -- Change Detector ──────────────────────────────────────
    st.subheader("4. Module 2 -- Temporal Change Detector")
    st.markdown(
        "512-dim embeddings from the ResNet-18 backbone (head stripped). "
        "Synthetic T1/T2 pairs from EuroSAT spatial blocks. "
        "Change score = 1 - cosine similarity."
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        show_fig(FIGS / "change_detector_roc.png", "ROC curve")
    with c2:
        if change_ops:
            st.markdown(f"**ROC AUC: {change_ops.get('auc', 0):.3f}**")
            st.markdown(md_table(
                ["Operating point", "Sim. threshold", "TPR", "FPR"],
                [
                    ["High recall",          f"{change_ops['high_recall']['similarity_threshold']:.3f}",    f"{change_ops['high_recall']['tpr']:.3f}",    f"{change_ops['high_recall']['fpr']:.3f}"],
                    ["Balanced (Youden's J)", f"{change_ops['balanced']['similarity_threshold']:.3f}",      f"{change_ops['balanced']['tpr']:.3f}",      f"{change_ops['balanced']['fpr']:.3f}"],
                    ["High precision",       f"{change_ops['high_precision']['similarity_threshold']:.3f}", f"{change_ops['high_precision']['tpr']:.3f}", f"{change_ops['high_precision']['fpr']:.3f}"],
                ]
            ))

    st.markdown("**Sample change heatmaps** (3 changed, 2 unchanged pairs from test set)")
    heatmap_dir = FIGS / "change_heatmaps"
    pairs = sorted(heatmap_dir.glob("pair_*.png")) if heatmap_dir.exists() else []
    if pairs:
        cols = st.columns(min(len(pairs), 3))
        for i, p in enumerate(pairs):
            with cols[i % 3]:
                st.image(str(p), caption=p.stem, width="stretch")
    else:
        st.caption("_(heatmap figures not found -- run `python -m src.change_detection`)_")

    st.divider()

    # ── 5. Spatial Leakage Experiment ───────────────────────────────────────
    st.subheader("5. Spatial Leakage Experiment")
    if leakage:
        st.markdown(md_table(
            ["Split strategy", "Val accuracy"],
            [
                ["Spatial block split (honest)",      f"{leakage['block_split_val_accuracy']:.3f}"],
                ["Random per-image split (leaky)",    f"{leakage['random_split_val_accuracy']:.3f}"],
                ["Leakage gap",                       f"+{leakage['leakage_gap']:.3f}"],
            ]
        ))
        st.info(leakage["explanation"])
    else:
        st.caption("_(not yet run -- `python -m src.spatial_leakage`)_")

    st.divider()

    # ── 6. Bonus D -- Class Imbalance Experiment ────────────────────────────
    st.subheader("6. Bonus D -- Class Imbalance Experiment")
    if imbalance:
        st.markdown(
            f"Downsampled **{', '.join(imbalance['downsampled_classes'])}** to "
            f"{imbalance['downsample_fraction']*100:.0f}% of training size. "
            "Mitigation: class-weighted cross-entropy loss."
        )
        imbalance_rows = [
            [c,
             f"{imbalance['no_mitigation']['per_class_f1'][c]:.3f}",
             f"{imbalance['weighted_loss_mitigation']['per_class_f1'][c]:.3f}"]
            for c in imbalance["downsampled_classes"]
        ]
        st.markdown(md_table(["Class", "No mitigation F1", "Weighted-loss F1"], imbalance_rows))
        c1, c2 = st.columns(2)
        c1.metric("Overall F1 -- no mitigation", f"{imbalance['no_mitigation']['macro_f1']:.3f}")
        c2.metric("Overall F1 -- weighted loss",  f"{imbalance['weighted_loss_mitigation']['macro_f1']:.3f}")
    else:
        st.caption("_(not yet run -- `python -m src.imbalance_experiment`)_")

    st.divider()

    # ── 7. Error Analysis ───────────────────────────────────────────────────
    st.subheader("7. Error Analysis -- Top-5 Most Confidently Wrong Predictions")
    show_fig(FIGS / "error_analysis" / "top5_misclassified.png",
             "Top-5 highest-confidence misclassifications (EuroSAT val)")
    if errors:
        for e in errors:
            st.markdown(
                f"**{e['true_class']} -> predicted {e['pred_class']}** "
                f"(confidence {e['confidence']:.1%})  \n{e['hypothesis']}"
            )
            st.divider()

    # ── 8. Bonus A -- GradCAM ───────────────────────────────────────────────
    st.subheader("8. Bonus A -- GradCAM Visualizations")
    st.markdown("Gradient-weighted class activation maps -- which pixels drove each prediction.")
    show_fig(FIGS / "gradcam" / "gradcam_examples.png", "GradCAM overlays on EuroSAT val samples")

    st.divider()

    # ── 9. Bonus C -- t-SNE Embedding Visualization ─────────────────────────
    st.subheader("9. Bonus C -- t-SNE Embedding Visualization")
    st.markdown(
        "t-SNE of 6,000 EuroSAT embeddings: scratch CNN vs. fine-tuned ResNet-18 feature space."
    )
    show_fig(FIGS / "tsne" / "tsne_scratch_vs_finetuned.png",
             "t-SNE: scratch CNN (left) vs. fine-tuned ResNet-18 (right)")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    st.title("TerraScope -- Geo-Dashboard")

    if not (config.CHECKPOINT_DIR / "resnet18_finetuned.pt").exists():
        st.error(
            "No trained checkpoint found at ml/checkpoints/resnet18_finetuned.pt. "
            "Run `python -m src.train_transfer` first."
        )
        return

    classifier, embedder = load_models()
    operating_points = load_operating_points()

    with st.sidebar:
        st.header("Settings")
        op_name = st.radio(
            "Operating point (Bonus B)",
            options=["high_recall", "balanced", "high_precision"],
            index=1,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        st.session_state["op_name"] = op_name
        threshold = operating_points[op_name]["similarity_threshold"]
        st.metric("Similarity threshold", f"{threshold:.3f}")
        if isinstance(operating_points.get("auc"), float):
            st.metric("Change-detector ROC AUC", f"{operating_points['auc']:.3f}")

    tab1, tab2, tab3 = st.tabs(["Classifier", "Change Detector", "Model Analytics"])

    with tab1:
        render_classifier(classifier)

    with tab2:
        render_change_detector(classifier, embedder, operating_points)

    with tab3:
        render_analytics()


if __name__ == "__main__":
    main()
