"""Stage 2, week 9 — drift of every batch against the clean reference.

For each batch (95 corrupted + 1 clean null batch) and each of the six features:
  - KS statistic: largest gap between the two cumulative distributions, in [0, 1].
  - Wasserstein distance, divided by the reference standard deviation so that
    features with different units share one scale.
Across all six features jointly:
  - MMD with an RBF kernel whose width is fixed once, from the reference.
Aggregated:
  - unweighted_ks / unweighted_w: the mean over features — every feature counts
    equally. This is the baseline the impact-weighted score of week 10 must beat.

Joined with the Stage 1 accuracy drop into results/stage2_drift.csv, one row per
batch. That table is what RQ2 asks about: do standard drift metrics track the
accuracy actually lost?

Reference and batches are disjoint images. Every corrupted batch is a corrupted
copy of clean test images [0, K); the reference is clean test images [K, 10000).
Clean images [0, K) form the null batch (corruption = "clean", severity 0): the
drift any metric reports when nothing has changed, i.e. its noise floor.

Usage:
    $env:DW_IMAGES_PER_SEVERITY = "2000"
    python src/stage2_features.py     # once, builds the feature cache
    python src/stage2_drift.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (CALIBRATION_CORRUPTIONS, HELDOUT_CORRUPTIONS, IMAGES_PER_SEVERITY,
                    RESULTS_DIR, SEVERITIES)
from drift import mmd_rbf, per_feature_drift, reference_gamma, unweighted_score
from features import FEATURE_NAMES
from stage2_features import batch_key, load_cache

K = IMAGES_PER_SEVERITY
OUT_CSV = RESULTS_DIR / "stage2_drift.csv"
STAGE1_CSV = RESULTS_DIR / "stage1_accuracy.csv"


def split_of(name):
    if name in CALIBRATION_CORRUPTIONS:
        return "calibration"
    if name in HELDOUT_CORRUPTIONS:
        return "heldout"
    if name == "clean":
        return "null"
    return "extra"  # in CIFAR-10-C but outside the 15-corruption benchmark split


def drift_row(name, severity, reference, batch, gamma):
    ks = per_feature_drift(reference, batch, method="ks")
    w = per_feature_drift(reference, batch, method="wasserstein")
    row = {"corruption": name, "severity": severity, "split": split_of(name),
           "n_reference": len(reference), "n_batch": len(batch)}
    row.update({f"ks_{f}": ks[f] for f in FEATURE_NAMES})
    row.update({f"w_{f}": w[f] for f in FEATURE_NAMES})
    row["unweighted_ks"] = unweighted_score(ks)
    row["unweighted_w"] = unweighted_score(w)
    row["mmd"] = mmd_rbf(reference, batch, gamma=gamma)
    return row


def main():
    feats = load_cache()
    if feats is None:
        raise SystemExit("No usable feature cache. Run src/stage2_features.py first.")

    print(f"K = {K} images per batch"
          + ("  (DW_IMAGES_PER_SEVERITY override)" if K != 10_000 else ""))
    clean = feats["clean"]
    if K >= len(clean):
        raise SystemExit(f"K={K} leaves no clean images for a disjoint reference")
    reference, null_batch = clean[K:], clean[:K]
    gamma = reference_gamma(reference)
    print(f"reference = clean test images [{K}, {len(clean)}): {len(reference)} images")
    print(f"null batch = clean test images [0, {K})")
    print(f"MMD kernel gamma (fixed, from reference) = {gamma:.4f}\n")

    names = sorted({k.split("__")[0] for k in feats if "__s" in k})
    rows = [drift_row("clean", 0, reference, null_batch, gamma)]
    for name in names:
        for sev in SEVERITIES:
            rows.append(drift_row(name, sev, reference, feats[batch_key(name, sev)], gamma))
        print(f"  {name}")
    df = pd.DataFrame(rows)

    s1 = pd.read_csv(STAGE1_CSV)
    if not (s1["n_images"] == K).all():
        raise SystemExit(f"{STAGE1_CSV.name} was measured on n_images != K={K}; "
                         "the drift and accuracy rows would not describe the same images")
    s1 = s1[["corruption", "severity", "accuracy", "accuracy_drop"]]
    df = df.merge(s1, on=["corruption", "severity"], how="left", validate="one_to_one")
    if df["accuracy_drop"].isna().any():
        missing = df.loc[df["accuracy_drop"].isna(), ["corruption", "severity"]]
        raise SystemExit(f"no Stage 1 accuracy for:\n{missing.to_string(index=False)}")

    first = ["corruption", "severity", "split", "unweighted_ks", "unweighted_w", "mmd",
             "accuracy", "accuracy_drop"]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"\nWrote {len(df)} rows to {OUT_CSV}")

    # --- summary -----------------------------------------------------------
    null = df[df.corruption == "clean"].iloc[0]
    ks_crit = 1.358 * np.sqrt((len(reference) + K) / (len(reference) * K))
    print(f"\nNoise floor (clean null batch): unweighted_ks {null.unweighted_ks:.4f}, "
          f"unweighted_w {null.unweighted_w:.4f}, mmd {null.mmd:.5f}")
    print(f"  per-feature KS: " + ", ".join(f"{f} {null[f'ks_{f}']:.3f}" for f in FEATURE_NAMES))
    print(f"  KS 5% critical value for n={len(reference)}, m={K}: {ks_crit:.3f}")

    corrupt = df[df.severity > 0]
    below = (corrupt[[f"ks_{f}" for f in FEATURE_NAMES]].max(axis=1) < ks_crit)
    print(f"  corrupted batches with no feature above the KS critical value: "
          f"{int(below.sum())} / {len(corrupt)}")
    if below.any():
        print("    " + ", ".join(f"{r.corruption} s{r.severity}"
                                 for r in corrupt[below].itertuples()))

    print(f"\nSpearman correlation with accuracy_drop over the {len(corrupt)} corrupted "
          f"batches (preliminary; the full RQ2 analysis is week 12):")
    for col in ("unweighted_ks", "unweighted_w", "mmd"):
        rho, p = spearmanr(corrupt[col], corrupt["accuracy_drop"])
        print(f"  {col:14s} rho = {rho:+.3f}   (p = {p:.1e})")


if __name__ == "__main__":
    main()
