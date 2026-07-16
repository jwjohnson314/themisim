"""SearchEngine result-limit modes — count (k) vs. similarity threshold.

Built without the heavy ``__init__`` (no real index/manifest): a synthetic,
(site, then time)-sorted engine drives the exact filtered path so result counts
reflect the cutoff alone, with no IVF recall variance.
"""
import numpy as np
import pytest

from themisim.concat import N_FEATURES
from themisim.search import (
    NS_PER_SECOND,
    THRESHOLD_HIT_CAP,
    SearchEngine,
    date_range_to_ns,
)


def _synthetic_engine(frames_per_site, d=N_FEATURES, seed=0):
    """A SearchEngine with synthetic (site, then time)-sorted arrays, built
    WITHOUT the heavy __init__. `frames_per_site` maps site name -> n frames;
    within each site time_ns is ascending (as in the real manifest)."""
    eng = SearchEngine.__new__(SearchEngine)
    rng = np.random.default_rng(seed)
    site_codes, time_ns = [], []
    vocab = list(frames_per_site)
    base = int(date_range_to_ns("2015-03-01", "2015-03-01")[0])
    for code, name in enumerate(vocab):
        n = frames_per_site[name]
        site_codes += [code] * n
        # ascending within the block, all within March 2015
        time_ns += list(base + np.cumsum(rng.integers(1, 5, size=n)) * NS_PER_SECOND)
    total = len(site_codes)
    eng.total_n = total
    eng._site_codes = np.asarray(site_codes, dtype=np.int16)
    eng._time_ns = np.asarray(time_ns, dtype=np.int64)
    eng._frame_idx = np.zeros(total, dtype=np.int32)
    eng._site_vocab = np.asarray(vocab, dtype=object)
    eng._site_to_code = {s: i for i, s in enumerate(vocab)}
    eng._shard_codes = np.zeros(total, dtype=np.int32)
    eng._shard_vocab = np.asarray(["/x/s.f16.npy"], dtype=object)
    eng.memmap = rng.standard_normal((total, d)).astype(np.float16)
    eng.brute_force_max = 10**9
    eng._init_site_blocks()
    return eng


@pytest.fixture
def engine():
    # One site, many frames, so a date filter scores the whole block exactly.
    return _synthetic_engine({"fsmi": 1500}, seed=42)


@pytest.fixture
def march():
    return [date_range_to_ns("2015-03-01", "2015-03-31")]


def test_threshold_returns_only_above_cutoff(engine, march):
    """Every hit in threshold mode scores at or above the cutoff."""
    gid = 700
    q = engine.memmap[gid].astype(np.float32).copy()
    cutoff = 0.5
    hits = engine.search(
        q, min_score=cutoff, time_ns_ranges=march, diversify_seconds=0
    )
    assert all(h.score >= cutoff for h in hits)


def test_threshold_count_monotonic_in_cutoff(engine, march):
    """A lower cutoff returns at least as many hits as a higher one."""
    gid = 700
    q = engine.memmap[gid].astype(np.float32).copy()
    loose = engine.search(
        q, min_score=0.2, time_ns_ranges=march, diversify_seconds=0
    )
    strict = engine.search(
        q, min_score=0.8, time_ns_ranges=march, diversify_seconds=0
    )
    assert len(loose) >= len(strict)
    assert len(strict) >= 1  # the self frame scores ~1.0 and is in-window


def test_threshold_mode_ignores_k(engine, march):
    """Threshold mode is not bounded by `k`: an all-inclusive cutoff returns
    the same (>1) hit set whether k is 1 or 1000."""
    gid = 700
    q = engine.memmap[gid].astype(np.float32).copy()
    small_k = engine.search(
        q, k=1, min_score=-1.0, time_ns_ranges=march, diversify_seconds=0
    )
    large_k = engine.search(
        q, k=1000, min_score=-1.0, time_ns_ranges=march, diversify_seconds=0
    )
    assert len(small_k) > 1  # k=1 would cap count mode at 1
    assert len(small_k) == len(large_k)
    assert [h.global_id for h in small_k] == [h.global_id for h in large_k]
    assert len(small_k) <= THRESHOLD_HIT_CAP


def test_threshold_none_is_count_mode(engine, march):
    """min_score=None (the default) is unchanged count mode: exactly k hits
    when the candidate pool is large enough."""
    gid = 700
    q = engine.memmap[gid].astype(np.float32).copy()
    hits = engine.search(
        q, k=10, min_score=None, time_ns_ranges=march, diversify_seconds=0
    )
    assert len(hits) == 10
