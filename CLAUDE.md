# DriftWatch

Impact-weighted drift detection for label-free prediction of model performance
degradation. CMPE 490 research project, TED University, Fall 2026.
Student: Doğa İdil Öcal. Supervisor: Prof. Dr. Ayşenur Birtürk.

## The idea in one paragraph

A deployed model loses accuracy when incoming data stops resembling its training
data, and it does so silently — it keeps producing confident predictions that are
wrong more often than before. Measuring the loss needs labels, which production data
does not have. The usual workaround, drift detection, asks *did the input
distribution change?* and not *did that change cost anything?* DriftWatch measures
offline how much accuracy each kind of degradation actually costs, then uses those
measured costs to weight a label-free drift score computed at deployment time.

The differentiator against the label-free performance-estimation literature is
diagnostic, not predictive: competing estimators output one number, while an
impact-weighted score also names *which factor* is responsible.

## Where the project stands

**Stage 1 is complete.** 19 CIFAR-10-C corruption types × 5 severities = 95
measurements, in `results/stage1_accuracy.csv`.

- Clean accuracy 0.9260 over the full 10,000 test images, matching the figure
  published for this checkpoint to four decimal places.
- Clean labels agree element by element with the CIFAR-10-C label file. This matters:
  the benchmark assumes every severity block reuses the clean labels in order, and a
  violation would have depressed every accuracy figure while still looking plausible.
- **Headline finding:** at severity 5 the accuracy drop ranges from 5.8 points
  (brightness) to 70.4 points (gaussian noise) — a **12.1× spread** among corruptions
  carrying the same severity label. This is the premise the whole method rests on,
  and it is now measured rather than assumed.
- Four corruption types are genuinely non-monotonic in severity (tested against
  sampling error; five of seven upward transitions survive at z > 2.6).
- Glass blur costs 45.9 points at severity 1 while defocus blur costs 0.9 — same
  family, same label. Weights must be learned from features, not assigned per family.

**Stage 2, weeks 8–9 done (24 Nov, K=2000 override).** `src/stage2_features.py`
caches the six features to `cache/stage2_features.npz` (gitignored) and draws
`figures/stage2_feature_histograms.png`; `src/stage2_drift.py` writes
`results/stage2_drift.csv`, one row per batch (95 corrupted + 1 clean null batch).

- Reference = clean test images [K, 10000); every batch is images [0, K), so the two
  sides are disjoint. The clean [0, K) row is the noise floor (unweighted KS 0.017;
  KS 5% critical value 0.034). Only pixelate s1 stays below it on every feature.
- MMD uses one kernel width fixed from the reference (`drift.reference_gamma`). The
  default per-call median heuristic rescales the kernel per batch and makes MMD
  values incomparable across batches.
- **Preliminary RQ2:** Spearman with accuracy drop over 95 batches is only ~0.5
  (unweighted KS 0.49, unweighted Wasserstein 0.54, MMD 0.54). Within one corruption
  drift ranks severity almost perfectly (ρ ≈ 1 for 15 of 19); the failure is
  *across* corruptions. Contrast and fog move features a lot and cost little;
  pixelate, JPEG and glass blur cost 20–52 points while barely moving any feature.
  Week 10 risk: weighting rescales features but cannot help where no feature moves.

## Current task — weeks 8 and 9 of the plan

Week 8 (feature extraction) slipped and is being done now, together with week 9.

1. Compute the six features for the clean reference set and for all 95 corrupted
   batches; cache to disk so they are computed once.
2. Sanity check: corrupted batches should separate visibly from clean ones on at
   least some features. Plot a few histograms and look.
3. Per-feature KS and Wasserstein drift values for every batch.
4. MMD across all features jointly, for the same batches.
5. One table, one row per batch: corruption, severity, drift score, accuracy drop.

That table answers **RQ2** — whether standard drift metrics correlate with actual
accuracy loss — which the literature survey found unmeasured. Week 10 then fits the
impact weights on top of it.

The user is working through this herself, step by step, in VS Code. Explain what each
step does and why before running it; do not run ahead and complete the task for her.
She is a capable programmer but new to this domain, so define domain terms the first
time they appear and keep English technical terms even when writing Turkish.

## First-time setup on the user's machine

The working copy at `C:\Users\Ismail Ocal\Documents\driftwatch` is a plain folder,
not a git clone, so files arrive there by being written rather than pulled. It
therefore holds the code and the data but not `index.html`, which lives in the
GitHub repository and serves the project site. Nothing depends on that file locally.

Three things are missing until setup is done, and all three are expected:

```powershell
# 1. virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt

# 2. corrupted data, rebuilt from the transfer chunks already in _transfer\
$env:DW_IMAGES_PER_SEVERITY = "2000"
python tools\unpack_subsample.py _transfer

# 3. clean data — the archive is ALREADY in the user's Downloads folder.
#    Do not tell her to download it again.
tar -xzf "$env:USERPROFILE\Downloads\cifar-10-python.tar.gz" -C data
```

After this, `data\CIFAR-10-C\` holds 19 corruptions and `data\cifar-10-batches-py\`
holds the clean test set.

## Running the code

Everything is CPU-only. No training ever happens — the model is used for inference
only.

```powershell
.venv\Scripts\Activate.ps1
$env:DW_IMAGES_PER_SEVERITY = "2000"    # REQUIRED, see below
python src\stage1_sweep.py --limit 2000 --corruptions gaussian_noise brightness
```

**`DW_IMAGES_PER_SEVERITY` is the single most common source of confusion.** The real
CIFAR-10-C files hold 10,000 images per severity block; the local copy holds 2,000,
rebuilt from transfer chunks by `tools/unpack_subsample.py` because the full 2.9 GB
set transfers at about 2 MB/s over the available link. Without the variable the
loader expects 50,000-row arrays and fails. It is per-terminal-session, so it must be
set again in every new terminal.

Any result produced with the override must say so. Sampling error is roughly one
percentage point, and differences of that order are not read as real.

## Layout

```
src/config.py          paths, corruption lists, normalisation, calibration/held-out split
src/data.py            load_corruption() with severity slicing; load_clean() from local pickle
src/model.py           load_model() from local weights with strict=True; predict(); accuracy()
src/resnet20.py        CIFAR ResNet-20, written to match the published checkpoint
src/features.py        the six image features
src/drift.py           per_feature_drift, unweighted/weighted score, dominant_feature, mmd_rbf
src/stage1_sweep.py    the benchmark sweep
src/stage1_figures.py  accuracy-severity curves and ranked-drop figure
src/stage2_features.py six features for clean + all batches, cached; sanity histograms
src/stage2_drift.py    per-batch KS / Wasserstein / MMD joined with accuracy drop
results/stage2_drift.csv
tests/test_slicing.py  verifies severity slicing against known per-block offsets
tools/unpack_subsample.py
data/CIFAR-10-C/       19 corruptions, 2,000 images per severity block
data/cifar-10-batches-py/   clean CIFAR-10, extracted from cifar-10-python.tar.gz
weights/cifar10_resnet20.pt
results/stage1_accuracy.csv
index.html             the project site, published at dogaidilocal.github.io/DriftWatch
```

**Data layout.** Each CIFAR-10-C `.npy` is `(5*K, 32, 32, 3)` uint8 with severity
blocks stacked consecutively, K images each. Severity *s* is rows
`[(s-1)*K : s*K]`. `labels.npy` holds K labels, reused for every block.

**Features (6):** sharpness (Laplacian variance), noise (residual after light blur),
brightness, contrast, edge density (Canny), colourfulness.

**Model:** `chenyaofo/pytorch-cifar-models`, id `cifar10_resnet20`, weight file
`cifar10_resnet20-4118986f.pt`, 272,474 parameters. Normalisation mean
(0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010).

## Conventions and decisions already made

- **Weights are fitted by non-negative least squares.** Ordinary least squares can
  assign a negative coefficient under collinearity, which would read as "drift in this
  feature predicts higher accuracy" — uninterpretable, and it would break the
  attribution output, since the dominant feature is only well defined when
  contributions cannot cancel.
- **Drift values must share a scale before regression.** KS statistics are bounded in
  [0, 1] and dimensionless; Wasserstein distances carry the feature's units and are
  divided by the reference standard deviation. Always the *statistic*, never the
  p-value.
- **Calibration and held-out corruption types are split in advance** (see
  `config.py`). Weights are fitted on one subset and evaluated on the disjoint one.
  Justified by Mintun et al. (NeurIPS 2021), who showed methods tuned on the original
  15 corruptions transfer poorly to a perceptually dissimilar set.
- **Weights load with `strict=True`** so a mismatch fails loudly instead of leaving
  part of the network randomly initialised.
- Baselines to beat: an unweighted drift score, and ATC (Garg et al., ICLR 2022).
  Metrics: Spearman correlation, AUROC, detection delay, attribution accuracy.

## Environment gotchas

These cost real time to discover. Do not re-litigate them.

- **`torch.hub.load` fails with HTTP 400** from every environment tried. This is why
  `src/resnet20.py` defines the architecture locally and `load_model()` reads a local
  weight file. Do not restore the `torch.hub` path.
- **`cs.toronto.edu`, Zenodo, HuggingFace, `download.pytorch.org` and the arXiv API
  are not reachable** from the cloud sandbox. PyPI, GitHub raw and GitHub releases
  are. Datasets have to be downloaded in a browser. On the user's own machine with a
  normal terminal there is no such restriction.
- Severity slicing and strict weight loading are the two steps that can be silently
  wrong, which is why both have explicit checks. Keep them.

## Deliverables

| Item | Weight | Status |
|---|---|---|
| Research Statement | 4% | Submitted |
| Progress Report 1 — literature survey | 8% | Submitted 13 Nov |
| Progress Report 2 — proposed method in detail | 8% | Due Fri 11 Dec, 13:00 |
| Final Report — conference-paper level | 50% | Due Wed 31 Dec |
| Online Presentation | 30% | January |

The published site lives in `index.html` at the repository root and is served from
the default branch by GitHub Pages. It carries the real Stage 1 figures; keep it in
step with `results/stage1_accuracy.csv` when numbers change.
