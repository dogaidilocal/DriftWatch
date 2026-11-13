"""Project-wide paths and constants."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CIFAR10C_DIR = DATA_DIR / "CIFAR-10-C"
# Clean CIFAR-10, python version, as extracted from cifar-10-python.tar.gz.
# Read from disk rather than downloaded: cs.toronto.edu is not reachable from
# the environments this project runs in.
CIFAR10_PY_DIR = DATA_DIR / "cifar-10-batches-py"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# CIFAR-10-C layout: each .npy is (50000, 32, 32, 3) uint8.
# Rows are stacked by severity: 1..5, 10000 test images each.
#
# DW_IMAGES_PER_SEVERITY overrides this for a reduced run on a subsample of each
# severity block (the first K images, the same K across every corruption so the
# comparison stays like-for-like). Used when the full 2.9 GB set cannot be moved
# to the machine doing inference. Any result produced with an override must say so.
IMAGES_PER_SEVERITY = int(os.environ.get("DW_IMAGES_PER_SEVERITY", 10_000))
SEVERITIES = (1, 2, 3, 4, 5)

# The 15 corruptions of the main benchmark, grouped as in Hendrycks & Dietterich (2019).
CORRUPTIONS = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}
ALL_CORRUPTIONS = [c for group in CORRUPTIONS.values() for c in group]

# Weight derivation and evaluation must use disjoint corruption sets (see report, 2.4).
# Split chosen so each group appears on both sides.
CALIBRATION_CORRUPTIONS = [
    "gaussian_noise", "defocus_blur", "motion_blur",
    "brightness", "contrast", "pixelate", "snow",
]
HELDOUT_CORRUPTIONS = [
    "shot_noise", "impulse_noise", "glass_blur", "zoom_blur",
    "frost", "fog", "elastic_transform", "jpeg_compression",
]

# CIFAR-10 normalisation used by the chenyaofo/pytorch-cifar-models checkpoints.
# If clean accuracy comes out far below the published figure, this is the first
# thing to check: a wrong normalisation is the usual cause.
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2023, 0.1994, 0.2010)

MODEL_NAME = "cifar10_resnet20"
BATCH_SIZE = 256
SEED = 0
