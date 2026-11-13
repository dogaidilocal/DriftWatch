"""Verify that severity slicing picks the right block.

This is the one piece of data handling that can be silently wrong: if the
severity offset is off, every accuracy figure in the study is attached to the
wrong severity and nothing downstream reveals it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from config import IMAGES_PER_SEVERITY
from data import load_corruption, load_labels

FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}{'  ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main():
    print("Slicing tests (fake data: block s has mean shifted by +10*s)\n")

    labels = load_labels()
    check("labels length is 10000", labels.shape == (IMAGES_PER_SEVERITY,), str(labels.shape))

    means = []
    for sev in (1, 2, 3, 4, 5):
        imgs, lab = load_corruption("gaussian_noise", sev)
        check(f"severity {sev} shape", imgs.shape == (IMAGES_PER_SEVERITY, 32, 32, 3), str(imgs.shape))
        check(f"severity {sev} labels align", lab.shape == (IMAGES_PER_SEVERITY,))
        means.append(float(imgs.mean()))

    print(f"\n  block means: {[round(m, 2) for m in means]}")
    check("block means increase with severity", all(b > a for a, b in zip(means, means[1:])))

    diffs = [round(b - a, 2) for a, b in zip(means, means[1:])]
    check("consecutive blocks differ by ~10", all(abs(d - 10) < 1.5 for d in diffs), str(diffs))

    try:
        load_corruption("gaussian_noise", 6)
        check("severity 6 rejected", False)
    except ValueError:
        check("severity 6 rejected", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("All slicing tests passed.")


if __name__ == "__main__":
    main()
