"""Preprocessing for THEMIS ASI frames.

`preprocess_frame` takes a single 256x256 uint16 frame as it comes off the CDF
and produces the (3, 224, 224) float32 tensor the SimCLR encoder expects.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision
from PIL import Image

IMAGE_SIZE = 224  # from simclr_config.yaml::image_size; do not substitute 256
RADIUS = 0.85     # circular crop radius as fraction of image half-width


def _circular_mask(size: int = 256, radius: float = RADIUS) -> np.ndarray:
    dist = int(radius * 128)
    X, Y = np.ogrid[:size, :size]
    d = np.sqrt((X - 128) ** 2 + (Y - 128) ** 2)
    return d <= dist


_MASK_256 = _circular_mask(256, RADIUS)
_TEST_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=IMAGE_SIZE),
    torchvision.transforms.ToTensor(),
])


def normalize_masked(raw: np.ndarray) -> np.ndarray:
    """p1..p99 contrast-stretch + circular mask -> (256, 256) float32 in [0, 1].

    The single source of truth for the THEMIS frame normalization. Both
    `preprocess_frame` (the embedding input) and `render._render_uncached`
    (the dashboard display) call this so the two can never drift — what you
    see in a result tile is exactly what was embedded.

    Degenerate-frame guard: a flat / dead / mostly-fill frame has its 1st and
    99th percentiles equal, so the original ``arr / np.percentile(arr, 99)``
    (after subtracting p1) divided by **zero** -> an all-NaN array. That NaN
    rendered as a fully black tile *and* — via the identical code in the embed
    path — was stored in the FAISS index as a NaN vector, quietly poisoning
    distance ranking. When the p1..p99 span collapses we fall back to a full
    min..max stretch so any real structure still shows; a genuinely constant
    frame then becomes an honest all-zero (black) frame rather than NaN.
    """
    if raw.shape != (256, 256):
        # A malformed / mis-axised CDF (e.g. an image var shaped
        # (256, 256, 1200) instead of (N, 256, 256)) yields a non-2D frame
        # here, which used to surface as a cryptic numpy broadcast error at
        # `rgb[:, :, 1] = arr` and take down the whole embed worker. Fail
        # loudly and early so the caller can skip the offending CDF.
        raise ValueError(
            f"expected a single (256, 256) frame, got {raw.shape}"
        )
    arr = raw.astype(np.float32)
    arr = arr - np.percentile(arr, 1)
    hi = np.percentile(arr, 99)
    if hi > 0:
        arr = arr / hi
    else:
        span = float(arr.max() - arr.min())
        arr = (arr - float(arr.min())) / span if span > 0 else np.zeros_like(arr)
    arr = np.clip(arr, 0, 1)
    arr[~_MASK_256] = 0
    return arr


def preprocess_frame(raw: np.ndarray) -> torch.Tensor:
    """Preprocess a single THEMIS ASI frame to model input.

    raw : (256, 256) uint16, exactly as read from the CDF
    returns : (3, 224, 224) float32 in [0, 1] with R=B=0
    """
    arr = normalize_masked(raw)
    rgb = np.zeros((256, 256, 3), dtype=np.float32)
    rgb[:, :, 1] = arr
    pil = Image.fromarray((rgb * 255).astype(np.uint8))
    return _TEST_TRANSFORM(pil)
