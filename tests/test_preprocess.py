"""Preprocessing is the first place a frame can be silently corrupted.

``normalize_masked`` is the single source of truth for how a THEMIS frame
becomes model input, and the same function feeds the dashboard display — so a
change here alters both what is indexed and what a user sees, without either
one complaining. The degenerate-frame case is tested hardest because it has
already failed once in production: a flat frame has p1 == p99, the original
code divided by that zero span, and the resulting NaN vector went into the
index and quietly poisoned distance ranking.
"""
from __future__ import annotations

import numpy as np
import pytest

from themisim.preprocess import IMAGE_SIZE, RADIUS, normalize_masked, preprocess_frame


def _frame(value=None, seed=0):
    if value is not None:
        return np.full((256, 256), value, dtype=np.uint16)
    return np.random.default_rng(seed).integers(0, 4000, (256, 256), dtype=np.uint16)


# --------------------------------------------------------------------------- #
# normalize_masked
# --------------------------------------------------------------------------- #
def test_output_is_in_unit_range_and_finite():
    out = normalize_masked(_frame())
    assert out.shape == (256, 256)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_flat_frame_yields_zeros_not_nan():
    """The regression this file exists for.

    A constant frame has p1 == p99. Dividing by that span gives NaN, and a NaN
    embedding is worse than a useless one: it ranks unpredictably against every
    query instead of simply never matching.
    """
    for value in (0, 7, 4095, 65535):
        out = normalize_masked(_frame(value))
        assert np.isfinite(out).all(), f"value={value} produced non-finite output"
        assert np.array_equal(out, np.zeros_like(out)), f"value={value}"


def test_near_flat_frame_still_finite():
    """A frame with almost no dynamic range must not divide by ~0 either."""
    frame = _frame(100)
    frame[0, 0] = 101
    out = normalize_masked(frame)
    assert np.isfinite(out).all()


def test_circular_mask_zeroes_the_corners():
    """Outside the fisheye circle there is no sky, so it must not be embedded."""
    out = normalize_masked(_frame(value=None))
    for corner in ((0, 0), (0, 255), (255, 0), (255, 255)):
        assert out[corner] == 0.0, corner
    assert out[128, 128] >= 0.0
    # The mask should keep roughly pi*r^2/size^2 of the frame.
    kept = np.count_nonzero(out) / out.size
    assert kept < np.pi * RADIUS**2 / 4 + 0.05


def test_contrast_stretch_is_percentile_based_not_min_max():
    """Outliers must not compress the whole frame; p1-p99 is the contract."""
    frame = _frame(seed=3)
    frame[10, 10] = 65535          # a hot pixel
    out = normalize_masked(frame)
    # A min-max stretch would push the bulk of the image toward zero; a
    # percentile stretch leaves the mid-tones spread out.
    inside = out[100:156, 100:156]
    assert inside.mean() > 0.05


def test_rejects_a_frame_of_the_wrong_shape():
    """A mis-axised CDF must fail loudly here, not broadcast-error later.

    Raising at this funnel is what turns a malformed file into a logged skip of
    one CDF rather than a crash that takes down an entire embed worker.
    """
    with pytest.raises(ValueError, match="256, 256"):
        normalize_masked(np.zeros((256, 256, 1), dtype=np.uint16))
    with pytest.raises(ValueError):
        normalize_masked(np.zeros((128, 128), dtype=np.uint16))


def test_is_deterministic():
    frame = _frame(seed=5)
    assert np.array_equal(normalize_masked(frame), normalize_masked(frame))


# --------------------------------------------------------------------------- #
# preprocess_frame
# --------------------------------------------------------------------------- #
def test_model_input_shape_and_range():
    """(3, 224, 224) float32 in [0, 1] is what the encoder was trained on."""
    out = preprocess_frame(_frame())
    assert tuple(out.shape) == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype.is_floating_point
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert bool(out.isfinite().all())


def test_signal_is_in_the_green_channel_only():
    """R and B are zero by construction; the encoder was trained that way."""
    out = preprocess_frame(_frame())
    assert float(out[0].abs().max()) == 0.0
    assert float(out[2].abs().max()) == 0.0
    assert float(out[1].max()) > 0.0


def test_flat_frame_survives_the_whole_preprocess_path():
    out = preprocess_frame(_frame(7))
    assert bool(out.isfinite().all())
    assert float(out.abs().max()) == 0.0


def test_preprocess_is_deterministic():
    """Inference-time preprocessing must carry no augmentation randomness."""
    import torch

    frame = _frame(seed=11)
    assert torch.equal(preprocess_frame(frame), preprocess_frame(frame))
