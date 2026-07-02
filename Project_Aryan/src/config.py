"""Shared paths, constants, and the UC Merced -> EuroSAT class mapping."""
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT / "data"
EUROSAT_DIR = DATA_DIR / "eurosat" / "2750"
UCMERCED_RAW_DIR = DATA_DIR / "ucmerced_raw"
UCMERCED_DIR = DATA_DIR / "ucmerced"
CHECKPOINT_DIR = ML_ROOT / "checkpoints"
FIGURES_DIR = ML_ROOT / "reports" / "figures"
METRICS_DIR = ML_ROOT / "reports" / "metrics"

for d in (CHECKPOINT_DIR, FIGURES_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64  # native EuroSAT RGB tile size
SEED = 42

UCMERCED_CLASSES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt",
]

EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
NUM_EUROSAT_CLASSES = len(EUROSAT_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# UC Merced (21 classes) -> EuroSAT (10 classes) semantic mapping used to evaluate
# the EuroSAT-trained model on UC Merced as a true zero-shot domain-generalization
# holdout (no fine-tuning on UC Merced labels). This is a manual, documented
# judgment call -- see report.md section "UC Merced holdout mapping".
UCMERCED_TO_EUROSAT = {
    "agricultural": "AnnualCrop",
    "forest": "Forest",
    "chaparral": "HerbaceousVegetation",
    "golfcourse": "HerbaceousVegetation",
    "tenniscourt": "HerbaceousVegetation",
    "freeway": "Highway",
    "overpass": "Highway",
    "intersection": "Highway",
    "buildings": "Industrial",
    "storagetanks": "Industrial",
    "mobilehomepark": "Industrial",
    "airplane": "Industrial",
    "runway": "Industrial",
    "parkinglot": "Industrial",
    "baseballdiamond": "Pasture",
    "denseresidential": "Residential",
    "mediumresidential": "Residential",
    "sparseresidential": "Residential",
    "river": "River",
    "beach": "SeaLake",
    "harbor": "SeaLake",
    # "PermanentCrop" has no confident UC Merced analogue and is intentionally
    # left without any source class -- it will have zero support in the
    # UC Merced holdout confusion matrix. This is noted in error_analysis.
}
