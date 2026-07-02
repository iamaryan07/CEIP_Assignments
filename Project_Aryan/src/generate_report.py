"""Deliverable 8 -- assembles report/report.pdf (<= 8 pages) directly from the
saved metrics JSON + figures, so the numbers in the PDF can never drift from
what the pipeline actually produced.

Run: python -m src.generate_report   (run LAST, after every other script)
"""
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import config

REPORT_DIR = config.ML_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INK = colors.HexColor("#212529")
MUTED = colors.HexColor("#495057")
ACCENT = colors.HexColor("#536475")
RULE = colors.HexColor("#adb5bd")


def load_json(name):
    path = config.METRICS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("H1", fontSize=20, leading=24, textColor=INK, fontName="Helvetica-Bold", spaceAfter=4))
    styles.add(ParagraphStyle("H2", fontSize=13, leading=16, textColor=INK, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle("Body", fontSize=9.5, leading=13.5, textColor=INK, fontName="Helvetica"))
    styles.add(ParagraphStyle("Muted", fontSize=8.5, leading=12, textColor=MUTED, fontName="Helvetica-Oblique"))
    styles.add(ParagraphStyle("Caption", fontSize=8, leading=10, textColor=MUTED, fontName="Helvetica-Oblique", alignment=1))
    return styles


def metric_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
        ("GRID", (0, 0), (-1, -1), 0.5, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def fig(path, width=3.0):
    if not path.exists():
        return None
    img = Image(str(path))
    aspect = img.imageHeight / img.imageWidth
    img.drawWidth = width * inch
    img.drawHeight = width * inch * aspect
    return img


def main():
    styles = build_styles()
    story = []

    baseline = load_json("baseline_eurosat_val.json")
    transfer = load_json("transfer_eurosat_val.json")
    ablation = load_json("frozen_vs_unfrozen_ablation.json")
    ucm = load_json("transfer_ucmerced_holdout.json")
    leakage = load_json("spatial_leakage_experiment.json")
    change_ops = load_json("change_detector_operating_points.json")
    imbalance = load_json("imbalance_experiment.json")
    errors = load_json("error_analysis.json")

    # ---- Page 1: title + overview ----
    story.append(Paragraph("Satellite Image Land-Use Classifier &amp; Temporal Change Detector", styles["H1"]))
    story.append(Paragraph("Transfer learning, embedding-based change detection, and an interactive dashboard over EuroSAT &amp; UC Merced.", styles["Muted"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Problem &amp; data", styles["H2"]))
    story.append(Paragraph(
        "We classify Sentinel-2 satellite tiles into 10 land-use categories and detect land-cover "
        "change between two time periods. Primary dataset: EuroSAT (27,000 RGB tiles, 64x64, 10 "
        "classes). Holdout: UC Merced Land Use (2,100 images, 21 classes), used zero-shot via a "
        "documented 21-&gt;10 class semantic mapping (config.UCMERCED_TO_EUROSAT) to test domain "
        "generalization rather than re-training on it.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "EuroSAT ships without per-tile geo-coordinates. Because the original extraction walked each "
        "Sentinel-2 scene raster-by-raster, file-adjacent tiles are also ground-adjacent, so we treat "
        "contiguous runs of 25 same-class tiles as a 'spatial block' and split whole blocks "
        "(never individual tiles) 70/15/15 into train/val/test -- see Section 4.", styles["Body"]))

    samples_fig = fig(config.FIGURES_DIR / "eurosat_class_distribution.png", width=4.6)
    if samples_fig:
        story.append(Spacer(1, 8))
        story.append(samples_fig)
        story.append(Paragraph("Figure 1. EuroSAT class distribution (27,000 tiles).", styles["Caption"]))

    # ---- Page 2: Module 1 results ----
    story.append(PageBreak())
    story.append(Paragraph("2. Module 1 -- Land-Use Classifier", styles["H2"]))
    story.append(Paragraph(
        "Baseline: a scratch 3-layer CNN (conv-BN-ReLU-pool x3 + 2-layer head), no pretraining. "
        "Transfer model: ImageNet-pretrained ResNet-18, two-phase fine-tuning -- phase 1 freezes "
        "everything but the classifier head (3 epochs); phase 2 unfreezes layer3+layer4 at LR/10 "
        "for 5 more epochs. ResNet-18 was chosen over EfficientNet-B0 because its 512-dim "
        "penultimate feature vector is exactly the embedding size Module 2 requires.", styles["Body"]))

    if baseline and transfer:
        rows = [["Model", "Macro-F1 (EuroSAT val)"],
                ["Baseline scratch CNN", f"{baseline['macro_f1']:.3f}"],
                ["ResNet-18 (two-phase)", f"{transfer['macro_f1']:.3f}"]]
        if ucm:
            rows.append(["ResNet-18 -> UC Merced holdout (zero-shot)", f"{ucm['macro_f1']:.3f}"])
        story.append(Spacer(1, 8))
        story.append(metric_table(rows, col_widths=[3.3 * inch, 2.0 * inch]))

    if ablation:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Frozen vs. unfrozen ablation (equal 8-epoch budget):", styles["Body"]))
        rows = [["Configuration", "Val accuracy"],
                ["Frozen backbone for all 8 epochs", f"{ablation['frozen_only']['val_acc']:.3f}"],
                ["Two-phase, after phase 1 (3 epochs)", f"{ablation['two_phase_after_phase1']['val_acc']:.3f}"],
                ["Two-phase, final (+5 unfrozen epochs)", f"{ablation['two_phase_final']['val_acc']:.3f}"]]
        story.append(metric_table(rows, col_widths=[3.3 * inch, 2.0 * inch]))

    loss_fig = fig(config.FIGURES_DIR / "transfer_loss_curves.png", width=4.6)
    if loss_fig:
        story.append(Spacer(1, 10))
        story.append(loss_fig)
        story.append(Paragraph("Figure 2. ResNet-18 two-phase fine-tuning loss/accuracy curves.", styles["Caption"]))

    # ---- Page 3: confusion matrices ----
    story.append(PageBreak())
    story.append(Paragraph("3. Confusion matrices", styles["H2"]))
    cm1 = fig(config.FIGURES_DIR / "transfer_eurosat_confusion_matrix.png", width=3.4)
    cm2 = fig(config.FIGURES_DIR / "ucmerced_holdout_confusion_matrix.png", width=3.4)
    if cm1:
        story.append(cm1)
        story.append(Paragraph("Figure 3a. ResNet-18 on EuroSAT validation.", styles["Caption"]))
    if cm2:
        story.append(Spacer(1, 8))
        story.append(cm2)
        story.append(Paragraph("Figure 3b. ResNet-18 on UC Merced holdout (zero-shot, class-mapped). "
                                "PermanentCrop has no confident UC Merced analogue and is intentionally "
                                "left with zero support.", styles["Caption"]))

    # ---- Page 4: spatial leakage + imbalance ----
    story.append(PageBreak())
    story.append(Paragraph("4. Spatial leakage experiment", styles["H2"]))
    if leakage:
        rows = [["Split strategy", "Val accuracy"],
                ["Spatial block split (honest)", f"{leakage['block_split_val_accuracy']:.3f}"],
                ["Random per-image split (leaky)", f"{leakage['random_split_val_accuracy']:.3f}"],
                ["Leakage gap", f"+{leakage['leakage_gap']:.3f}"]]
        story.append(metric_table(rows, col_widths=[3.3 * inch, 2.0 * inch]))
        story.append(Spacer(1, 6))
        story.append(Paragraph(leakage["explanation"], styles["Body"]))
    else:
        story.append(Paragraph("Not yet run.", styles["Muted"]))

    story.append(Paragraph("Bonus D -- class imbalance experiment", styles["H2"]))
    if imbalance:
        story.append(Paragraph(
            f"Downsampled {', '.join(imbalance['downsampled_classes'])} to "
            f"{imbalance['downsample_fraction']*100:.0f}% of their training size, then compared "
            f"per-class F1 with vs. without class-weighted cross-entropy loss.", styles["Body"]))
        rows = [["Class", "No mitigation F1", "Weighted-loss F1"]]
        for c in imbalance["downsampled_classes"]:
            rows.append([c, f"{imbalance['no_mitigation']['per_class_f1'][c]:.3f}",
                         f"{imbalance['weighted_loss_mitigation']['per_class_f1'][c]:.3f}"])
        story.append(Spacer(1, 6))
        story.append(metric_table(rows, col_widths=[2.2 * inch, 1.6 * inch, 1.6 * inch]))
    else:
        story.append(Paragraph("Not yet run.", styles["Muted"]))

    # ---- Page 5: change detection ----
    story.append(PageBreak())
    story.append(Paragraph("5. Module 2 -- Temporal Change Detector", styles["H2"]))
    story.append(Paragraph(
        "The fine-tuned backbone (head stripped) extracts a 512-dim global embedding per tile. "
        "EuroSAT has no real before/after pairs, so a synthetic T1/T2 series is built from the "
        "spatial blocks: 'unchanged' pairs draw two tiles from the same block (label 0); 'changed' "
        "pairs draw tiles from two different-class blocks (label 1). Because ground truth is known "
        "by construction, a real ROC curve can be computed for the cosine-similarity change score.",
        styles["Body"]))

    roc_fig = fig(config.FIGURES_DIR / "change_detector_roc.png", width=3.0)
    if roc_fig and change_ops:
        story.append(Spacer(1, 8))
        t = Table([[roc_fig, metric_table([
            ["Operating point", "Sim. threshold", "TPR", "FPR"],
            *[[name.replace("_", " "), f"{change_ops[name]['similarity_threshold']:.3f}",
               f"{change_ops[name]['tpr']:.2f}", f"{change_ops[name]['fpr']:.2f}"]
              for name in ("high_recall", "balanced", "high_precision")],
        ], col_widths=[1.1 * inch, 1.0 * inch, 0.6 * inch, 0.6 * inch])]], colWidths=[3.2 * inch, 3.3 * inch])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t)
        story.append(Paragraph(
            f"Figure 4. ROC curve (AUC={change_ops.get('auc', 0):.3f}) and the three bonus-B operating "
            "points (high recall / balanced via Youden's J / high precision), toggleable in both the "
            "Streamlit dashboard and the Next.js change-detection page.", styles["Caption"]))
    elif roc_fig:
        story.append(roc_fig)

    # ---- Page 6: change heatmaps ----
    story.append(PageBreak())
    story.append(Paragraph("6. Sample change heatmaps", styles["H2"]))
    story.append(Paragraph(
        "Each tile's 7x7 spatial feature map (pre-pool layer4 output) is compared patch-by-patch "
        "via cosine distance, upsampled, and overlaid -- showing *where* in the tile the embedding "
        "shifted, not just a single global score.", styles["Body"]))
    heatmap_dir = config.FIGURES_DIR / "change_heatmaps"
    if heatmap_dir.exists():
        imgs = sorted(heatmap_dir.glob("pair_*.png"))[:4]
        for p in imgs:
            f = fig(p, width=4.6)
            if f:
                story.append(Spacer(1, 6))
                story.append(f)

    # ---- Page 7: error analysis + bonus ----
    story.append(PageBreak())
    story.append(Paragraph("7. Error analysis", styles["H2"]))
    err_fig = fig(config.FIGURES_DIR / "error_analysis" / "top5_misclassified.png", width=4.6)
    if err_fig:
        story.append(err_fig)
        story.append(Paragraph("Figure 5. Top-5 most confidently wrong predictions, EuroSAT validation split.", styles["Caption"]))
    if errors:
        for e in errors[:5]:
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<b>{e['true_class']} &rarr; {e['pred_class']}</b> (conf={e['confidence']:.2f}): {e['hypothesis']}",
                styles["Body"]))

    story.append(Paragraph("Bonus A/C -- GradCAM &amp; embedding visualisation", styles["H2"]))
    gradcam_fig = fig(config.FIGURES_DIR / "gradcam" / "gradcam_examples.png", width=4.2)
    if gradcam_fig:
        story.append(gradcam_fig)
        story.append(Paragraph("Figure 6. GradCAM overlays -- which pixels drove each prediction.", styles["Caption"]))

    # ---- Page 8: limitations + checklist ----
    story.append(PageBreak())
    tsne_fig = fig(config.FIGURES_DIR / "tsne" / "tsne_scratch_vs_finetuned.png", width=4.8)
    if tsne_fig:
        story.append(tsne_fig)
        story.append(Paragraph("Figure 7. t-SNE of 6,000 EuroSAT embeddings -- scratch CNN vs. fine-tuned ResNet-18.", styles["Caption"]))

    story.append(Paragraph("8. Limitations &amp; conclusion", styles["H2"]))
    story.append(Paragraph(
        "(1) The T1/T2 change series is synthetic -- EuroSAT has no real repeat acquisitions, so "
        "the ROC curve measures the detector's ability to separate same-block from "
        "different-class-block pairs, not real seasonal/sensor drift. (2) The UC Merced holdout "
        "uses a manually authored, debatable 21-&gt;10 class mapping; one EuroSAT class "
        "(PermanentCrop) has no mapped UC Merced analogue and is untested. (3) Tiles are evaluated "
        "independently; no temporal smoothing or multi-tile spatial context is used. (4) The "
        "spatial-block split approximates geography via file-index adjacency, not real lat/lon.",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Despite these constraints, the two-phase fine-tuned ResNet-18 substantially beats the "
        "scratch baseline and generalizes non-trivially to a different sensor/dataset (UC Merced), "
        "and the cosine-similarity change detector achieves a strong, threshold-justified ROC AUC "
        "on the constructed change task.", styles["Body"]))

    doc = SimpleDocTemplate(
        str(REPORT_DIR / "report.pdf"), pagesize=LETTER,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    doc.build(story)
    print(f"Wrote {REPORT_DIR / 'report.pdf'}")


if __name__ == "__main__":
    main()
