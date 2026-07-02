"""Dataset indexing, spatial-block splitting, and PyTorch Dataset wrappers.

EuroSAT's public RGB release ships only class-labelled tiles, no per-tile
lat/lon metadata. The original Sentinel-2 extraction walked each scene
raster-by-raster, so tiles that are *adjacent in file index* are also
adjacent on the ground. We exploit that ordering to build "spatial blocks":
contiguous runs of N images per class are treated as one geographic block,
and whole blocks (never individual images) are assigned to a split. This
prevents the train/val/test leakage that a naive random per-image split
would cause, and is the basis for the spatial-leakage experiment in
spatial_leakage.py.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from . import config


def index_eurosat(eurosat_dir: Path = config.EUROSAT_DIR) -> pd.DataFrame:
    """Build a dataframe of every EuroSAT tile with a deterministic block id."""
    rows = []
    for class_name in config.EUROSAT_CLASSES:
        class_dir = eurosat_dir / class_name
        files = sorted(class_dir.glob("*.jpg"))
        for i, f in enumerate(files):
            rows.append(
                {
                    "path": str(f),
                    "class_name": class_name,
                    "class_idx": config.EUROSAT_CLASSES.index(class_name),
                    "file_order": i,
                }
            )
    return pd.DataFrame(rows)


def assign_spatial_blocks(df: pd.DataFrame, block_size: int = 25) -> pd.DataFrame:
    """Assign a `block_id` to contiguous runs of `block_size` images per class."""
    df = df.copy()
    df["block_id"] = (
        df["class_name"] + "_" + (df["file_order"] // block_size).astype(str)
    )
    return df


def spatial_block_split(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Split whole spatial blocks (never individual tiles) into train/val/test,
    stratified per class so every class keeps the same approximate ratios."""
    assert abs(train + val + test - 1.0) < 1e-6
    df = df.copy()
    rng = random.Random(seed)
    split_assignment: dict[str, str] = {}

    for class_name in df["class_name"].unique():
        blocks = sorted(df.loc[df["class_name"] == class_name, "block_id"].unique())
        rng.shuffle(blocks)
        n = len(blocks)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        for b in blocks[:n_train]:
            split_assignment[b] = "train"
        for b in blocks[n_train : n_train + n_val]:
            split_assignment[b] = "val"
        for b in blocks[n_train + n_val :]:
            split_assignment[b] = "test"

    df["split"] = df["block_id"].map(split_assignment)
    return df


def random_image_split(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Naive random per-image split (ignores spatial blocks). Used ONLY as the
    'leaky' comparison point in the spatial-leakage experiment -- never for
    reporting headline metrics."""
    assert abs(train + val + test - 1.0) < 1e-6
    df = df.copy()
    rng = np.random.RandomState(seed)
    splits = []
    for class_name in df["class_name"].unique():
        idx = df.index[df["class_name"] == class_name].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        for i in idx[:n_train]:
            splits.append((i, "train"))
        for i in idx[n_train : n_train + n_val]:
            splits.append((i, "val"))
        for i in idx[n_train + n_val :]:
            splits.append((i, "test"))
    split_map = dict(splits)
    df["split"] = df.index.map(split_map)
    return df


def build_ucmerced_imagefolder() -> Path:
    """Decode the UC Merced parquet (downloaded from HF blanchon/UC_Merced) into
    an ImageFolder-style directory tree under config.UCMERCED_DIR. Idempotent."""
    import io

    marker = config.UCMERCED_DIR / ".done"
    if marker.exists():
        return config.UCMERCED_DIR

    parquet_path = config.UCMERCED_RAW_DIR / "data" / "train-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)

    # Confirmed via `datasets.load_dataset(...).features["label"]` ClassLabel
    # names, which are alphabetically ordered and match config.UCMERCED_CLASSES.
    class_names = config.UCMERCED_CLASSES

    config.UCMERCED_DIR.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    for _, row in df.iterrows():
        class_name = class_names[int(row["label"])]
        img_bytes = row["image"]
        if isinstance(img_bytes, dict):
            img_bytes = img_bytes.get("bytes")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        class_dir = config.UCMERCED_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        counters[class_name] = counters.get(class_name, 0) + 1
        img.save(class_dir / f"{counters[class_name]:04d}.jpg", quality=95)

    marker.write_text("done")
    return config.UCMERCED_DIR


@dataclass
class TileSample:
    path: str
    class_idx: int
    class_name: str


class TileDataset(Dataset):
    """Generic image-classification dataset built from a dataframe with
    `path` and `class_idx` columns."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.samples = [
            TileSample(r.path, int(r.class_idx), r.class_name) for r in df.itertuples()
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, s.class_idx

    @property
    def class_names(self):
        return [s.class_name for s in self.samples]
