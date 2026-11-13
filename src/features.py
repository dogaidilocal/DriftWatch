"""Image quality features — Stage 2.

Each image is reduced to a small vector of measurements computed from the pixels
alone. The model is not involved: these are properties of the input, which is
what makes the drift score available without labels and without inference.

Features are computed per feature, not as one aggregate, because attribution
depends on knowing which measurement moved.
"""
import cv2
import numpy as np

FEATURE_NAMES = (
    "sharpness",     # variance of the Laplacian; falls under blur
    "noise",         # high-frequency residual energy; rises under noise
    "brightness",    # mean intensity
    "contrast",      # intensity standard deviation
    "edge_density",  # fraction of pixels on a Canny edge
    "colourfulness", # spread across colour channels
)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def image_features(img):
    """Feature vector for a single uint8 RGB image, shape (32, 32, 3)."""
    g = _gray(img).astype(np.float32)

    sharpness = float(cv2.Laplacian(g, cv2.CV_32F).var())

    # Noise estimate: residual after a light blur. Cheap, and it separates
    # additive noise from blur, which the Laplacian variance alone does not.
    blurred = cv2.GaussianBlur(g, (3, 3), 0)
    noise = float(np.abs(g - blurred).mean())

    brightness = float(g.mean())
    contrast = float(g.std())

    edges = cv2.Canny(_gray(img), 100, 200)
    edge_density = float((edges > 0).mean())

    r, gc, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    rg = r - gc
    yb = 0.5 * (r + gc) - b
    colourfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                          + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    return np.array([sharpness, noise, brightness, contrast, edge_density, colourfulness],
                    dtype=np.float64)


def batch_features(images, progress=False):
    """Feature matrix for a batch of images: (N, len(FEATURE_NAMES))."""
    it = range(len(images))
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc="features", leave=False)
    return np.stack([image_features(images[i]) for i in it])
