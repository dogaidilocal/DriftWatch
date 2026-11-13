# DriftWatch

CMPE 490 — impact-weighted drift detection for label-free prediction of model
performance degradation.

## Setup

```bash
cd driftwatch
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download `CIFAR-10-C.tar` (2.9 GB) from https://zenodo.org/records/2535967 and
extract it so the `.npy` files land here:

```
driftwatch/data/CIFAR-10-C/
    labels.npy
    gaussian_noise.npy
    defocus_blur.npy
    ...
```

```bash
tar -xf CIFAR-10-C.tar -C data/
```

Clean CIFAR-10 is downloaded automatically by torchvision on the first run.

### File layout

Each corruption `.npy` is `(50000, 32, 32, 3)` uint8: the 10,000 test images at
severity 1, then severity 2, and so on to severity 5. `labels.npy` holds 10,000
labels, reused for every severity block.

## Running Stage 1

```bash
# quick check first — one corruption, 2000 images per severity
python src/stage1_sweep.py --corruptions gaussian_noise --limit 2000

# full sweep
python src/stage1_sweep.py
```

Writes `results/stage1_accuracy.csv` with one row per (corruption, severity).

### The check that matters

The first thing `stage1_sweep.py` prints is clean accuracy. With the pretrained
ResNet-20 this should land close to the figure published in the model
repository. If it is far below, the normalisation constants `NORM_MEAN` and
`NORM_STD` in `src/config.py` are wrong — fix that before running anything else,
or every number afterwards is wrong in the same way.

## Testing without the dataset

```bash
python tests/make_fake_data.py       # writes fake .npy files with real shapes
python tests/test_slicing.py         # verifies severity slicing
python src/stage1_sweep.py --no-pretrained --skip-clean --limit 2000
```

Delete `data/CIFAR-10-C/` afterwards so the fake files do not get mistaken for
the real ones.

## Layout

```
src/config.py          paths, corruption lists, normalisation, splits
src/data.py            CIFAR-10-C loading and severity slicing
src/model.py           pretrained classifier, inference only
src/stage1_sweep.py    Stage 1 degradation sweep
tests/                 fake data generator and slicing tests
results/               CSV output
figures/               plots
```

The model is never trained or modified. All weighting happens in the drift score
computation, outside the model.
