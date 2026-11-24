"""Drift scoring — Stage 2.

Two-sample tests comparing a reference feature matrix against a batch feature
matrix. Per-feature tests are what make attribution possible: the score can
report not only that drift occurred but in which feature.

No labels and no model outputs are used anywhere in this module.
"""
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from features import FEATURE_NAMES


def per_feature_drift(reference, batch, method="ks"):
    """Drift value per feature.

    reference, batch: (N, F) and (M, F) feature matrices
    method: "ks" (Kolmogorov-Smirnov statistic) or "wasserstein"

    Returns a dict {feature_name: value}. KS values are already in [0, 1];
    Wasserstein values are standardised by the reference spread so that
    features on different scales remain comparable.
    """
    if reference.shape[1] != batch.shape[1]:
        raise ValueError("reference and batch have different feature counts")

    out = {}
    for j, name in enumerate(FEATURE_NAMES[:reference.shape[1]]):
        ref_col, bat_col = reference[:, j], batch[:, j]
        if method == "ks":
            out[name] = float(ks_2samp(ref_col, bat_col).statistic)
        elif method == "wasserstein":
            scale = ref_col.std()
            raw = wasserstein_distance(ref_col, bat_col)
            out[name] = float(raw / scale) if scale > 0 else 0.0
        else:
            raise ValueError(f"unknown method: {method}")
    return out


def unweighted_score(drift_dict):
    """Baseline aggregation: every feature counts equally."""
    return float(np.mean(list(drift_dict.values())))


def weighted_score(drift_dict, weights):
    """Impact-weighted aggregation.

    weights: dict {feature_name: weight}, derived from the Stage 1 benchmark on
    a calibration split. Features absent from weights contribute nothing.
    """
    return float(sum(weights.get(k, 0.0) * v for k, v in drift_dict.items()))


def dominant_feature(drift_dict, weights):
    """The feature contributing most to the weighted score.

    This is the attribution output: it names the suspected cause of the
    degradation. Neither the unweighted score nor a scalar accuracy estimator
    produces anything comparable.
    """
    contributions = {k: weights.get(k, 0.0) * v for k, v in drift_dict.items()}
    if not contributions or max(contributions.values()) <= 0:
        return None
    return max(contributions, key=contributions.get)


def reference_gamma(reference, n=500, seed=0):
    """RBF kernel width from the reference alone (median heuristic).

    Pass the result as mmd_rbf(..., gamma=...) whenever MMD values are compared
    across batches. Left at None, mmd_rbf picks gamma from the pooled reference
    and batch, so a strongly shifted batch widens its own kernel and compresses
    its own MMD — the values then sit on different scales and cannot be ranked.
    """
    rng = np.random.default_rng(seed)
    X = _standardise(reference, reference)
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    d2 = _sq_dists(X, X)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    return 1.0 / med if med > 0 else 1.0


def mmd_rbf(reference, batch, gamma=None, max_n=2000, seed=0):
    """Maximum Mean Discrepancy with an RBF kernel, across all features jointly.

    Subsampled to max_n per side: MMD is O(n^2) and the full 10,000x10,000
    kernel is both slow and unnecessary for a stable estimate.

    For comparisons across batches, fix gamma with reference_gamma(); see there.
    """
    rng = np.random.default_rng(seed)
    X = _standardise(reference, reference)
    Y = _standardise(batch, reference)

    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]

    if gamma is None:
        # Median heuristic for the kernel width.
        sample = np.vstack([X, Y])
        idx = rng.choice(len(sample), min(500, len(sample)), replace=False)
        d2 = _sq_dists(sample[idx], sample[idx])
        med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
        gamma = 1.0 / med if med > 0 else 1.0

    kxx = np.exp(-gamma * _sq_dists(X, X)).mean()
    kyy = np.exp(-gamma * _sq_dists(Y, Y)).mean()
    kxy = np.exp(-gamma * _sq_dists(X, Y)).mean()
    return float(max(kxx + kyy - 2 * kxy, 0.0))


def _standardise(M, reference):
    """Standardise by the reference statistics, so both sides share one scale."""
    mu = reference.mean(axis=0)
    sd = reference.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (M - mu) / sd


def _sq_dists(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
