"""Loading clean CIFAR-10 and corrupted CIFAR-10-C batches."""
import numpy as np

from config import CIFAR10C_DIR, DATA_DIR, IMAGES_PER_SEVERITY


def load_labels():
    """Shared label file: one label per test image, reused for every severity block."""
    path = CIFAR10C_DIR / "labels.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"labels.npy not found at {path}. Extract CIFAR-10-C.tar into {CIFAR10C_DIR}."
        )
    labels = np.load(path)
    return labels[:IMAGES_PER_SEVERITY].astype(np.int64)


def load_corruption(name, severity):
    """Return (images, labels) for one corruption at one severity.

    images: uint8, shape (10000, 32, 32, 3)
    """
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be 1..5, got {severity}")

    path = CIFAR10C_DIR / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Is CIFAR-10-C.tar extracted?")

    # mmap avoids pulling the whole 150 MB file into memory for one severity slice.
    arr = np.load(path, mmap_mode="r")
    if arr.shape[0] != 5 * IMAGES_PER_SEVERITY:
        raise ValueError(
            f"{name}.npy has {arr.shape[0]} rows, expected {5 * IMAGES_PER_SEVERITY}. "
            "The file layout may differ from the documented one."
        )

    start = (severity - 1) * IMAGES_PER_SEVERITY
    images = np.array(arr[start:start + IMAGES_PER_SEVERITY])
    return images, load_labels()


def load_clean():
    """Clean CIFAR-10 test set as (images, labels), uint8 (10000, 32, 32, 3).

    Downloads via torchvision on first call.
    """
    from torchvision.datasets import CIFAR10

    ds = CIFAR10(root=str(DATA_DIR), train=False, download=True)
    images = ds.data.astype(np.uint8)
    labels = np.array(ds.targets, dtype=np.int64)
    return images, labels


def available_corruptions():
    """Corruption names actually present on disk."""
    if not CIFAR10C_DIR.exists():
        return []
    names = sorted(p.stem for p in CIFAR10C_DIR.glob("*.npy") if p.stem != "labels")
    return names
