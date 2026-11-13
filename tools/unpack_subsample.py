"""Rebuild the per-corruption CIFAR-10-C layout from transfer chunks.

Not part of the study. The full CIFAR-10-C set is 2.9 GB and the link between the
machine holding it and the machine running inference is slow, so a subsample of
K images per severity was packed into a few chunk files for transfer. This script
unpacks those back into the documented one-file-per-corruption layout, with
K rows per severity block instead of 10,000, so data.py and stage1_sweep.py run
unchanged under DW_IMAGES_PER_SEVERITY=K.

Each chunkN.txt lists "corruption,severity" per block, in the order the blocks
appear in chunkN.npy.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from config import CIFAR10C_DIR, IMAGES_PER_SEVERITY, SEVERITIES

K = IMAGES_PER_SEVERITY


def main(transfer_dir):
    transfer_dir = Path(transfer_dir)
    CIFAR10C_DIR.mkdir(parents=True, exist_ok=True)

    labels = np.load(transfer_dir / "labels_sub.npy")
    if len(labels) != K:
        raise SystemExit(f"labels_sub.npy has {len(labels)} rows, expected K={K}")
    np.save(CIFAR10C_DIR / "labels.npy", labels)

    blocks = {}  # corruption -> {severity: array}
    for chunk_path in sorted(transfer_dir.glob("chunk*.npy")):
        meta = (chunk_path.with_suffix(".txt")).read_text().split()
        arr = np.load(chunk_path, mmap_mode="r")
        if arr.shape[0] != len(meta) * K:
            raise SystemExit(
                f"{chunk_path.name} has {arr.shape[0]} rows, expected {len(meta) * K}"
            )
        for i, entry in enumerate(meta):
            name, sev = entry.split(",")
            blocks.setdefault(name, {})[int(sev)] = np.asarray(arr[i * K:(i + 1) * K])

    for name, by_sev in sorted(blocks.items()):
        missing = [s for s in SEVERITIES if s not in by_sev]
        if missing:
            raise SystemExit(f"{name}: missing severities {missing}")
        # Stack in severity order, so the slicing in data.py addresses the right block.
        out = np.concatenate([by_sev[s] for s in SEVERITIES], axis=0)
        assert out.shape == (5 * K, 32, 32, 3), (name, out.shape)
        np.save(CIFAR10C_DIR / f"{name}.npy", out)
        print(f"{name:20s} {out.shape}  {out.nbytes / 1e6:5.0f} MB")

    print(f"\n{len(blocks)} corruptions written to {CIFAR10C_DIR} at K={K} per severity")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: unpack_subsample.py <transfer_dir>")
    main(sys.argv[1])
