"""Stage 2, week 8 — image-quality features for the clean and corrupted sets.

Computes the six features of features.py for
  - the full clean CIFAR-10 test set (10,000 images), and
  - every corruption x severity batch on disk (K images each),
and caches them in cache/stage2_features.npz so they are computed once.

Then draws the sanity-check figure: per-feature histograms of clean data against
a few severity-5 corruptions chosen to hit different features. If corrupted
batches did not separate from clean on at least some features, the drift tests
of week 9 would have nothing to detect.

Usage:
    $env:DW_IMAGES_PER_SEVERITY = "2000"
    python src/stage2_features.py            # compute (or reuse cache) + figure
    python src/stage2_features.py --force    # recompute even if cached
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from config import FIGURES_DIR, IMAGES_PER_SEVERITY, ROOT, SEVERITIES
from data import available_corruptions, load_clean, load_corruption
from features import FEATURE_NAMES, batch_features

CACHE_DIR = ROOT / "cache"
CACHE_PATH = CACHE_DIR / "stage2_features.npz"

# Severity-5 corruptions shown in the sanity figure, one per feature they should move.
SANITY = {
    "gaussian_noise": "#2a78d6",  # noise up
    "defocus_blur": "#eb6834",    # sharpness, edge density down
    "brightness": "#1baf7a",      # brightness up
    "contrast": "#eda100",        # contrast down
}
CLEAN_FILL = "#d9d8d3"
INK = "#0b0b0b"
INK_2 = "#52514e"


def batch_key(name, severity):
    return f"{name}__s{severity}"


def load_cache():
    """Return the cached feature dict, or None if absent or built for another K."""
    if not CACHE_PATH.exists():
        return None
    z = np.load(CACHE_PATH)
    if int(z["images_per_severity"]) != IMAGES_PER_SEVERITY:
        print(f"Cache was built with K={int(z['images_per_severity'])}, "
              f"current K={IMAGES_PER_SEVERITY}; recomputing.")
        return None
    if tuple(z["feature_names"]) != FEATURE_NAMES:
        print("Cache feature list differs from features.py; recomputing.")
        return None
    return {k: z[k] for k in z.files}


def compute_all():
    out = {
        "images_per_severity": np.array(IMAGES_PER_SEVERITY),
        "feature_names": np.array(FEATURE_NAMES),
    }

    t0 = time.time()
    clean_imgs, _ = load_clean()
    out["clean"] = batch_features(clean_imgs, progress=True)
    print(f"  clean                {out['clean'].shape}  ({time.time() - t0:.1f}s)")

    names = available_corruptions()
    for name in names:
        t0 = time.time()
        for sev in SEVERITIES:
            imgs, _ = load_corruption(name, sev)
            out[batch_key(name, sev)] = batch_features(imgs)
        print(f"  {name:20s} 5 x {IMAGES_PER_SEVERITY}  ({time.time() - t0:.1f}s)")

    CACHE_DIR.mkdir(exist_ok=True)
    np.savez_compressed(CACHE_PATH, **out)
    print(f"\nCached {len(names)} corruptions + clean to {CACHE_PATH}")
    return out


def sanity_figure(feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = feats["clean"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), facecolor="#fcfcfb")
    for j, (ax, fname) in enumerate(zip(axes.flat, FEATURE_NAMES)):
        ax.set_facecolor("#fcfcfb")
        cols = [clean[:, j]] + [feats[batch_key(n, 5)][:, j] for n in SANITY]
        allv = np.concatenate(cols)
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 50)

        # The outer 0.5% fall outside the bins and are not drawn. Clipping them
        # into the edge bins instead would draw spikes that are not in the data.
        ax.hist(clean[:, j], bins=bins, density=True,
                color=CLEAN_FILL, label="clean (10,000)")
        for name, colour in SANITY.items():
            v = feats[batch_key(name, 5)][:, j]
            ax.hist(v, bins=bins, density=True, histtype="step",
                    linewidth=2, color=colour, label=f"{name} s5")

        ax.set_title(fname, fontsize=11, color=INK, loc="left")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        f"Feature distributions: clean vs. four severity-5 corruptions "
        f"(K={IMAGES_PER_SEVERITY} images per corrupted batch)",
        y=0.985, fontsize=11, color=INK)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out = FIGURES_DIR / "stage2_feature_histograms.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {out}")


def separation_summary(feats):
    """Text version of the sanity check: KS statistic, clean vs each sanity batch."""
    from scipy.stats import ks_2samp

    clean = feats["clean"]
    print(f"\nKS statistic, clean vs severity 5 (0 = identical, 1 = fully separated)")
    print(f"  {'':16s}" + "".join(f"{f:>14s}" for f in FEATURE_NAMES))
    for name in SANITY:
        b = feats[batch_key(name, 5)]
        row = [ks_2samp(clean[:, j], b[:, j]).statistic for j in range(len(FEATURE_NAMES))]
        print(f"  {name:16s}" + "".join(f"{v:14.3f}" for v in row))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    print(f"K = {IMAGES_PER_SEVERITY} images per severity"
          + ("  (DW_IMAGES_PER_SEVERITY override)" if IMAGES_PER_SEVERITY != 10_000 else ""))

    feats = None if args.force else load_cache()
    if feats is None:
        print("Computing features ...")
        feats = compute_all()
    else:
        print(f"Using cached features from {CACHE_PATH}")

    for k, v in feats.items():
        if v.ndim == 2 and not np.isfinite(v).all():
            raise SystemExit(f"non-finite feature values in {k}")

    separation_summary(feats)
    sanity_figure(feats)


if __name__ == "__main__":
    main()
