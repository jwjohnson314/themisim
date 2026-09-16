"""The validator must pass a good index and fail every way one can be wrong.

A validation report that only ever says PASS is worth nothing, so each check
here is paired with a deliberately corrupted artifact set that must trip it.
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from themisim import validate
from themisim.search import SearchEngine


@pytest.fixture
def artifacts(built_artifacts, tmp_path):
    """A writable copy of the good index, for corruption tests."""
    out = tmp_path / "artifacts"
    shutil.copytree(built_artifacts, out)
    # The engine caches manifest columns next to the manifest; drop the copy so
    # a corrupted manifest cannot be masked by a stale cache.
    shutil.rmtree(out / "manifest.parquet.arrays.cache", ignore_errors=True)
    return out


def _report(artifacts, **kw):
    kw.setdefault("n_queries", 40)
    kw.setdefault("nprobe", 8)
    kw.setdefault("nprobe_sweep", (1, 8))
    kw.setdefault("prefilter_sweep", (100, 500))
    return validate.validate_index(artifacts, **kw)


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #
def test_good_index_passes_every_check(artifacts):
    report = _report(artifacts)
    failed = {
        name: c for name, c in report["checks"].items() if c.get("passed") is False
    }
    assert not failed, json.dumps(failed, indent=2, default=str)
    assert report["passed"] is True


def test_report_is_json_serializable_and_carries_provenance(artifacts, tmp_path):
    report = _report(artifacts)
    path = validate.write_report(report, tmp_path / "r.json")
    reloaded = json.loads(path.read_text())
    assert reloaded["passed"] is True
    # Without provenance a reviewer's differing number is unadjudicable.
    for key in ("python", "platform", "numpy", "faiss", "build_device"):
        assert key in reloaded["provenance"], key
    assert reloaded["coverage_gaps"], "the report must state what it does not cover"
    assert validate.format_report(report).count("PASS") >= 1


def test_self_retrieval_separates_exact_and_approximate_paths(artifacts):
    """The exact path is deterministic and must be perfect; the IVF path is not.

    Conflating them is how a self-retrieval check becomes flaky: on the
    approximate path a frame can legitimately miss the candidate pool.
    """
    sr = _report(artifacts)["checks"]["self_retrieval"]
    assert sr["exact_path_fraction"] == 1.0
    assert sr["exact_min_score"] >= 0.999
    assert 0.0 <= sr["ivf_path_fraction"] <= 1.0


def test_pipeline_recall_beats_raw_faiss_recall(artifacts):
    """The exact rerank is the whole point of the prefilter, so it must help.

    This also documents why raw FAISS recall is the wrong number to gate on:
    it is measured on the PQ codes alone, which is not what a user ever sees.
    """
    grid = _report(artifacts)["checks"]["retrieval"]["grid"]
    assert grid
    for point in grid:
        assert point["pipeline_recall_at_k"] >= point["faiss_recall_at_k"] - 1e-9
    best = max(g["pipeline_recall_at_k"] for g in grid)
    worst_raw = min(g["faiss_recall_at_k"] for g in grid)
    assert best > worst_raw


def test_recall_is_reported_against_fraction_of_database_scanned(artifacts):
    """nprobe alone is meaningless across index sizes; nprobe/nlist is not."""
    ret = _report(artifacts)["checks"]["retrieval"]
    for point in ret["grid"]:
        assert point["fraction_scanned"] == pytest.approx(point["nprobe"] / ret["nlist"])


# --------------------------------------------------------------------------- #
# each check must actually be able to fail
# --------------------------------------------------------------------------- #
def test_non_finite_vector_is_caught(artifacts):
    meta = json.loads((artifacts / "vectors.meta.json").read_text())
    total_n = meta["shape"][0]
    mm = np.memmap(
        artifacts / "vectors.f16.dat", dtype=np.float16, mode="r+",
        shape=(total_n, validate.N_FEATURES),
    )
    mm[3] = np.nan
    mm.flush()
    del mm
    engine = SearchEngine(artifacts)
    result = validate.check_vector_health(engine)
    assert result["non_finite_rows"] == 1
    assert result["passed"] is False


def test_zero_norm_vector_is_caught(artifacts):
    """A zero row survives normalization and then matches nothing, silently."""
    meta = json.loads((artifacts / "vectors.meta.json").read_text())
    total_n = meta["shape"][0]
    mm = np.memmap(
        artifacts / "vectors.f16.dat", dtype=np.float16, mode="r+",
        shape=(total_n, validate.N_FEATURES),
    )
    mm[5] = 0
    mm.flush()
    del mm
    result = validate.check_vector_health(SearchEngine(artifacts))
    assert result["zero_norm_rows"] == 1
    assert result["passed"] is False


def test_shuffled_manifest_breaks_the_layout_invariants(artifacts):
    """Site blocks must be contiguous and time_ns non-decreasing within them.

    ``SearchEngine._filter_ranges`` binary-searches on both; a shuffled manifest
    is the cheapest way to prove the checks would notice if they stopped holding.
    """
    import pandas as pd

    path = artifacts / "manifest.parquet"
    df = pd.read_parquet(path)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    shuffled["global_id"] = np.arange(len(shuffled), dtype=np.int64)
    shuffled.to_parquet(path, index=False)
    shutil.rmtree(artifacts / "manifest.parquet.arrays.cache", ignore_errors=True)

    result = validate.check_integrity(SearchEngine(artifacts))
    assert result["passed"] is False
    assert not result["contiguous_site_blocks"] or not result["time_monotone_within_sites"]


def test_untrained_index_is_rejected_at_load(artifacts):
    """An untrained index fails obscurely deep inside faiss; catch it early."""
    import faiss

    from themisim.index import _build_factory_string

    empty = faiss.index_factory(
        validate.N_FEATURES, _build_factory_string(4), faiss.METRIC_INNER_PRODUCT
    )
    faiss.write_index(empty, str(artifacts / "index.faiss"))
    with pytest.raises(ValueError, match="untrained"):
        SearchEngine(artifacts)


def test_truncated_memmap_is_rejected(artifacts):
    with open(artifacts / "vectors.f16.dat", "r+b") as fh:
        fh.truncate(1024)
    with pytest.raises(ValueError):
        SearchEngine(artifacts)


def test_index_row_count_mismatch_is_rejected(artifacts):
    meta_path = artifacts / "vectors.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["shape"][0] -= 1
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ntotal"):
        SearchEngine(artifacts)


def test_encoder_reference_detects_wrong_vectors(artifacts, tmp_path):
    """A reference built from different vectors must not pass."""
    engine = SearchEngine(artifacts)
    ref = tmp_path / "ref.npz"
    rng = np.random.default_rng(0)
    gids = np.arange(5)
    np.savez_compressed(
        ref,
        site=np.array([engine.site_of(int(g)) for g in gids]),
        datetime=np.array(
            [validate._dtstr_of(engine, int(g)) for g in gids]
        ),
        frame_idx=np.array([engine.frame_idx_of(int(g)) for g in gids], dtype=np.int32),
        vectors=rng.standard_normal((5, validate.N_FEATURES)).astype(np.float16),
    )
    out = validate.check_encoder_reference(engine, ref)
    assert out["n_reference_frames"] == 5
    assert out["cosine_mean"] < 0.5

    good = validate.check_encoder_reference(
        engine, _write_matching_reference(engine, tmp_path / "good.npz", gids)
    )
    assert good["cosine_mean"] > 0.9999


def _write_matching_reference(engine, path, gids):
    np.savez_compressed(
        path,
        site=np.array([engine.site_of(int(g)) for g in gids]),
        datetime=np.array([validate._dtstr_of(engine, int(g)) for g in gids]),
        frame_idx=np.array([engine.frame_idx_of(int(g)) for g in gids], dtype=np.int32),
        vectors=np.asarray(engine.memmap[gids], dtype=np.float16),
    )
    return path


def test_missing_reference_is_skipped_not_failed(artifacts, tmp_path):
    out = validate.check_encoder_reference(SearchEngine(artifacts), tmp_path / "nope.npz")
    assert out["passed"] is None
    assert "skipped" in out


def test_recall_is_skipped_when_the_database_is_too_large(artifacts, monkeypatch):
    monkeypatch.setattr(validate, "MAX_EXHAUSTIVE_N", 1)
    out = validate.check_retrieval(SearchEngine(artifacts))
    assert out["passed"] is None
    assert "skipped" in out


def test_thresholds_are_honoured(artifacts):
    """An impossible gate must turn a good index into a FAIL."""
    report = _report(artifacts, thresholds={"recall_at_10": 1.1})
    assert report["checks"]["retrieval"]["passed"] is False
    assert report["passed"] is False


# --------------------------------------------------------------------------- #
# filtered-query fallback
# --------------------------------------------------------------------------- #
class TestRestrictedIvfFallback:
    """A filtered query must never return nothing while frames match.

    When a filtered subset exceeds ``brute_force_max`` the search restricts an
    IVF probe with an IDSelector, and that probe can come back empty: the cells
    nearest the query need not intersect a time-disjoint filter. Measured on the
    full archive with a one-month filter, 7% of queries returned zero candidates
    at the default nprobe. The old code returned ``[]`` for those -- a fast,
    silent, wrong answer while millions of frames matched.
    """

    @staticmethod
    def _forced(artifacts):
        """An engine that sends every filtered query down the restricted path."""
        return SearchEngine(artifacts, brute_force_max=0)

    def _all_sites_range(self, engine):
        lo, hi = engine.time_ns_bounds()
        return [(int(lo), int(hi))]

    def test_restricted_path_is_actually_taken(self, artifacts):
        """Guard the premise: brute_force_max=0 must bypass exact scoring."""
        eng = self._forced(artifacts)
        assert eng.brute_force_max == 0
        ranges = eng._filter_ranges(None, self._all_sites_range(eng))
        assert ranges, "filter should match the whole index"
        assert sum(e - s for s, e in ranges) > eng.brute_force_max

    def test_empty_probe_falls_back_to_exact_instead_of_returning_nothing(
        self, artifacts, monkeypatch
    ):
        eng = self._forced(artifacts)
        gid = 11
        vec = np.asarray(eng.memmap[gid], dtype=np.float32)

        # Every restricted probe finds nothing, at any nprobe.
        def empty_search(q, k, params=None):
            return (np.full((1, k), -1.0, np.float32), np.full((1, k), -1, np.int64))

        monkeypatch.setattr(eng.index, "search", empty_search)
        hits = eng.search(vec, k=5, diversify_seconds=0,
                          time_ns_ranges=self._all_sites_range(eng))
        assert hits, "returned no results while the filter matched the whole index"
        assert hits[0].global_id == gid
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_nprobe_is_escalated_before_the_expensive_fallback(
        self, artifacts, monkeypatch
    ):
        """An exact scan of a filtered subset reads gigabytes; try cheap first."""
        from themisim.search import RESTRICTED_NPROBE_ESCALATION

        eng = self._forced(artifacts)
        seen = []
        real = eng.index.search

        def spy(q, k, params=None):
            seen.append(getattr(params, "nprobe", None))
            return (np.full((1, k), -1.0, np.float32), np.full((1, k), -1, np.int64))

        monkeypatch.setattr(eng.index, "search", spy)
        eng.search(np.asarray(eng.memmap[3], dtype=np.float32), k=3, nprobe=1,
                   diversify_seconds=0, time_ns_ranges=self._all_sites_range(eng))

        assert len(seen) == 2, f"expected one retry before falling back, got {seen}"
        assert seen[0] == 1
        assert seen[1] == min(int(eng._ivf.nlist), 1 * RESTRICTED_NPROBE_ESCALATION)

    def test_a_probe_that_finds_candidates_skips_the_fallback(self, artifacts):
        """The normal restricted path must not pay for the fallback."""
        eng = self._forced(artifacts)
        gid = 7
        hits = eng.search(np.asarray(eng.memmap[gid], dtype=np.float32), k=5,
                          diversify_seconds=0,
                          time_ns_ranges=self._all_sites_range(eng))
        assert hits and hits[0].global_id == gid

    def test_an_empty_filter_still_returns_nothing(self, artifacts):
        """No frames match => [] is the correct answer, not a fallback trigger."""
        eng = self._forced(artifacts)
        hits = eng.search(np.asarray(eng.memmap[0], dtype=np.float32), k=5,
                          diversify_seconds=0, time_ns_ranges=[(1, 2)])
        assert hits == []

    def test_unfiltered_queries_are_unaffected(self, artifacts):
        eng = SearchEngine(artifacts)
        gid = 21
        hits = eng.search(np.asarray(eng.memmap[gid], dtype=np.float32), k=5,
                          nprobe=8, prefilter=200, diversify_seconds=0)
        assert hits and hits[0].global_id == gid
