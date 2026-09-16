"""Build orchestration: the glue between inventory, embed, concat and index.

``auto_nlist`` and ``resolve_device`` are small but consequential -- the first
decides the index's shape, the second decides whether a build takes minutes or
hours -- and neither had any coverage. ``build_index`` itself is exercised for
real here (not monkeypatched) over a tiny synthetic tree, which is the only way
to know the four stages actually hand off to one another.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from themisim.index import DEFAULT_NLIST_FULL, DEFAULT_NLIST_PILOT
from themisim.pipeline import auto_nlist, build_index, resolve_device


# --------------------------------------------------------------------------- #
# auto_nlist
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n, expected",
    [
        (100_000_000, DEFAULT_NLIST_FULL),      # full-archive threshold
        (1_009_488_343, DEFAULT_NLIST_FULL),    # the production index
        (1_000_000, DEFAULT_NLIST_PILOT),       # pilot threshold
        (99_999_999, DEFAULT_NLIST_PILOT),      # just below full
    ],
)
def test_auto_nlist_at_the_scale_boundaries(n, expected):
    assert auto_nlist(n) == expected


def test_auto_nlist_falls_back_to_sqrt_for_small_sets():
    assert auto_nlist(57_000) == int(np.sqrt(57_000))
    assert auto_nlist(11_457) == int(np.sqrt(11_457))


def test_auto_nlist_never_returns_zero():
    """nlist=0 would produce an invalid factory string, not a small index."""
    for n in (0, 1, 2, 10):
        assert auto_nlist(n) >= 1


def test_auto_nlist_is_monotone_non_decreasing():
    sizes = [1, 100, 10_000, 57_000, 999_999, 1_000_000, 100_000_000]
    values = [auto_nlist(n) for n in sizes]
    assert values == sorted(values)


def test_auto_nlist_leaves_enough_training_data_per_centroid():
    """FAISS wants tens of points per centroid; sqrt(n) gives sqrt(n) of them."""
    for n in (5_000, 57_000, 999_999):
        assert n / auto_nlist(n) >= 39


# --------------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------------- #
def test_resolve_device_passes_explicit_values_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_picks_an_available_backend():
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_resolve_device_auto_falls_back_when_torch_is_unusable(monkeypatch):
    """A broken CUDA install must degrade to CPU, not abort the build."""
    import torch

    def boom():
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert resolve_device("auto") == "cpu"


# --------------------------------------------------------------------------- #
# build_index, run for real
# --------------------------------------------------------------------------- #
def test_build_index_refuses_an_empty_data_root(tmp_path, synthetic_checkpoint):
    empty = tmp_path / "cdf"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no THEMIS CDFs"):
        build_index(empty, tmp_path / "art", synthetic_checkpoint, device="cpu")


@pytest.mark.slow
def test_build_index_runs_all_four_stages(synthetic_cdf_tree, tmp_path,
                                          synthetic_checkpoint):
    """inventory -> embed -> concat -> train/add, with nothing stubbed out.

    Small, but it is the difference between believing the stages compose and
    knowing it. Every artifact the query path needs must exist afterwards, and
    the vector count must equal the frames actually present on disk.
    """
    artifacts = tmp_path / "art"
    stages = []
    out = build_index(
        synthetic_cdf_tree, artifacts, synthetic_checkpoint,
        nlist=4, device="cpu", batch_size=8, num_workers=0,
        streaming=False, use_gpu_quantizer=False,
        on_progress=lambda stage, info: stages.append(stage),
    )

    assert out == artifacts / "index.faiss"
    for name in ("index.faiss", "vectors.f16.dat", "manifest.parquet",
                 "vectors.meta.json"):
        assert (artifacts / name).exists(), name
    assert {"inventory", "device", "embed_mode", "embedded", "concat", "index"} <= set(stages)

    meta = json.loads((artifacts / "vectors.meta.json").read_text())
    assert meta["n_shards"] == 4          # one per CDF
    from conftest import SYNTHETIC_TREE_FRAMES
    assert meta["shape"] == [SYNTHETIC_TREE_FRAMES, 512]  # ragged hours preserved


@pytest.mark.slow
def test_build_index_resumes_without_re_embedding(synthetic_cdf_tree, tmp_path,
                                                  synthetic_checkpoint):
    """Re-running must reuse existing shards; embedding is the expensive stage."""
    artifacts = tmp_path / "art"
    kw = dict(nlist=4, device="cpu", batch_size=8, num_workers=0,
              streaming=False, use_gpu_quantizer=False)
    build_index(synthetic_cdf_tree, artifacts, synthetic_checkpoint, **kw)

    shards = sorted((artifacts / "shards").rglob("*.f16.npy"))
    before = {p: p.stat().st_mtime_ns for p in shards}
    assert before

    build_index(synthetic_cdf_tree, artifacts, synthetic_checkpoint, **kw)
    after = {p: p.stat().st_mtime_ns for p in sorted((artifacts / "shards").rglob("*.f16.npy"))}
    assert after == before, "shards were rewritten; resume is not working"
