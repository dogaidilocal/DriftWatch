"""Generate fake CIFAR-10-C files with the real shapes, to smoke-test the pipeline.

Not part of the study. Lets the loading, slicing and sweep code be exercised
without the 2.9 GB download.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from config import CIFAR10C_DIR, IMAGES_PER_SEVERITY

N = IMAGES_PER_SEVERITY
RNG = np.random.default_rng(0)
FAKE = ["gaussian_noise", "defocus_blur", "brightness"]


def main():
    CIFAR10C_DIR.mkdir(parents=True, exist_ok=True)

    labels = RNG.integers(0, 10, size=N).astype(np.int64)
    np.save(CIFAR10C_DIR / "labels.npy", labels)

    # Headroom up to 255 so the per-severity +10 shift is not lost to clipping,
    # which would make the slicing test measure saturation instead of offsets.
    base = RNG.integers(0, 180, size=(N, 32, 32, 3), dtype=np.uint8)

    for name in FAKE:
        blocks = []
        for sev in range(1, 6):
            # Tag each severity block with a recognisable mean shift, so the
            # slicing can be verified: block s has mean roughly base_mean + 10*s.
            shifted = np.clip(base.astype(np.int16) + 10 * sev, 0, 255).astype(np.uint8)
            blocks.append(shifted)
        arr = np.concatenate(blocks, axis=0)
        assert arr.shape == (5 * N, 32, 32, 3), arr.shape
        np.save(CIFAR10C_DIR / f"{name}.npy", arr)
        print(f"wrote {name}.npy  {arr.shape}  {arr.nbytes / 1e6:.0f} MB")

    print(f"wrote labels.npy  {labels.shape}")


if __name__ == "__main__":
    main()
