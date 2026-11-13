"""Stage 1 — controlled degradation benchmark.

Runs the pretrained classifier over every available corruption at every severity
and records accuracy. The resulting accuracy-severity curves are the impact map
that Stage 2 weights its drift signals with.

Usage:
    python src/stage1_sweep.py                      # every corruption found on disk
    python src/stage1_sweep.py --corruptions gaussian_noise brightness
    python src/stage1_sweep.py --limit 2000         # quick run on a subset of images
    python src/stage1_sweep.py --no-pretrained      # pipeline smoke test, meaningless numbers
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import ALL_CORRUPTIONS, RESULTS_DIR, SEVERITIES
from data import available_corruptions, load_clean, load_corruption
from model import accuracy, load_model

RESULTS_CSV = RESULTS_DIR / "stage1_accuracy.csv"


def parse_args():
    ap = argparse.ArgumentParser(description="Stage 1 degradation sweep")
    ap.add_argument("--corruptions", nargs="*", default=None,
                    help="corruption names; default is every .npy found on disk")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images per severity (quick runs)")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random weights; for testing the pipeline only")
    ap.add_argument("--skip-clean", action="store_true",
                    help="skip the clean-accuracy reference measurement")
    ap.add_argument("--clean-ref", type=float, default=None,
                    help="clean accuracy measured elsewhere, used as the drop reference "
                         "when clean CIFAR-10 is not available locally; recorded as "
                         "corruption='clean_published' so its provenance stays visible")
    return ap.parse_args()


def main():
    args = parse_args()
    pretrained = not args.no_pretrained

    if not pretrained:
        print("!! --no-pretrained: random weights. Accuracy figures are meaningless.\n")

    print("Loading model ...")
    model = load_model(pretrained=pretrained)

    rows = []

    # --- clean reference ------------------------------------------------
    # Week-6 check: with the pretrained ResNet-20 this should land near the
    # figure published in the model repository. A large gap almost always
    # means the normalisation constants in config.py are wrong.
    clean_acc = None
    if not args.skip_clean:
        print("Measuring clean accuracy ...")
        imgs, labels = load_clean()
        if args.limit:
            imgs, labels = imgs[:args.limit], labels[:args.limit]
        t0 = time.time()
        clean_acc = accuracy(model, imgs, labels)
        print(f"  clean accuracy = {clean_acc:.4f}   ({len(imgs)} images, {time.time() - t0:.1f}s)")
        rows.append({"corruption": "clean", "severity": 0,
                     "accuracy": clean_acc, "n_images": len(imgs)})
    elif args.clean_ref is not None:
        clean_acc = args.clean_ref
        print(f"Clean reference taken from the model source: {clean_acc:.4f} "
              "(not measured here)")
        rows.append({"corruption": "clean_published", "severity": 0,
                     "accuracy": clean_acc, "n_images": 0})

    # --- corruption sweep -----------------------------------------------
    present = available_corruptions()
    if not present:
        print(f"\nNo corruption files found. Extract CIFAR-10-C.tar first.")
        if rows:
            _write(rows)
        return

    wanted = args.corruptions or [c for c in ALL_CORRUPTIONS if c in present]
    missing = [c for c in wanted if c not in present]
    if missing:
        print(f"\nNot on disk, skipping: {', '.join(missing)}")
    wanted = [c for c in wanted if c in present]

    print(f"\nSweeping {len(wanted)} corruptions x {len(SEVERITIES)} severities\n")

    for name in wanted:
        for sev in SEVERITIES:
            t0 = time.time()
            imgs, labels = load_corruption(name, sev)
            if args.limit:
                imgs, labels = imgs[:args.limit], labels[:args.limit]
            acc = accuracy(model, imgs, labels)
            drop = None if clean_acc is None else clean_acc - acc
            rows.append({"corruption": name, "severity": sev,
                         "accuracy": acc, "n_images": len(imgs)})
            drop_str = "" if drop is None else f"   drop {drop:+.4f}"
            print(f"  {name:<20} sev {sev}   acc {acc:.4f}{drop_str}   ({time.time() - t0:.1f}s)")

    _write(rows)


def _write(rows):
    df = pd.DataFrame(rows)
    if "accuracy" in df and (df["severity"] == 0).any():
        clean = float(df.loc[df["severity"] == 0, "accuracy"].iloc[0])
        df["accuracy_drop"] = clean - df["accuracy"]
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
