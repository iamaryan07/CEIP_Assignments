# ml — CV pipeline

Implements the project brief: a land-use classifier (Module 1), a temporal change
detector (Module 2), and a Streamlit geo-dashboard (Module 3), plus the spatial-leakage
write-up, error analysis, and bonus tasks A/C/D.

## Run order

```bash
python -m src.prepare_data
python -m src.train_baseline
python -m src.train_transfer
python -m src.change_detection
python -m src.spatial_leakage
python -m src.error_analysis
python -m src.gradcam
python -m src.embedding_viz
python -m src.imbalance_experiment
python -m src.generate_report
```

Each script is independently re-runnable; later scripts read the `.pt` checkpoints and
`reports/metrics/*.json` left by earlier ones rather than retraining anything.

## Key methodology decisions

**Spatial-block split.** EuroSAT's public RGB release has no per-tile lat/lon. Its
original extraction walked each Sentinel-2 scene raster-by-raster, so file-adjacent
tiles are also ground-adjacent (and visually near-duplicate). `datasets.py` groups
contiguous runs of 25 same-class tiles into a "block" and assigns *whole blocks* —
never individual tiles — to train/val/test (70/15/15, stratified per class). This is
the basis for the spatial-leakage experiment, which retrains the same architecture on
a naive random per-image split and reports the resulting (inflated) accuracy gap.

**ResNet-18 over EfficientNet-B0.** The brief asks for 512-dim embeddings in Module 2.
ResNet-18's penultimate (post global-avg-pool) feature vector is exactly 512-dim with
no projection layer needed; EfficientNet-B0's is 1280-dim. ResNet-18 was chosen for
that reason, beyond also being lighter to fine-tune.

**UC Merced holdout mapping.** UC Merced has 21 classes, EuroSAT has 10. Rather than
fine-tune a second head, the EuroSAT-trained classifier is evaluated *zero-shot* on UC
Merced via a manual semantic mapping (`config.UCMERCED_TO_EUROSAT`, e.g.
`denseresidential/mediumresidential/sparseresidential -> Residential`). This directly
tests domain generalization. One EuroSAT class (`PermanentCrop`) has no confident UC
Merced analogue and is left unmapped — it will show zero support in that confusion
matrix, which is called out explicitly in the report rather than papered over.

**Synthetic T1/T2 change pairs.** EuroSAT has no real repeat acquisitions. Module 2
simulates a time series by treating each spatial block as a "region": *unchanged*
pairs draw two tiles from the same block (same class); *changed* pairs draw tiles from
two different-class blocks. Because the ground-truth label is known by construction,
a real ROC curve can be computed for the cosine-similarity change score, and a
threshold can be justified (Youden's J for "balanced", TPR>=0.95 for "high recall",
FPR<=0.05 for "high precision" — bonus B's three operating points).

**Change heatmaps.** Rather than a single global similarity score, the heatmap compares
the two tiles' 7x7 spatial feature maps (the layer4 output before global pooling)
patch-by-patch via cosine distance, then upsamples — showing *where* in the tile the
embedding shifted most.

## Outputs

- `checkpoints/baseline_cnn.pt`, `checkpoints/resnet18_finetuned.pt`
- `reports/figures/**` — every plot (confusion matrices, ROC, loss curves, GradCAM,
  t-SNE, change heatmaps, error-analysis grid, data-pipeline visualisations)
- `reports/metrics/*.json` — every number, read live by `backend/` and `generate_report.py`
- `data/eurosat_index.csv` / `eurosat_index_random.csv` — the two splits used in the
  spatial-leakage experiment
