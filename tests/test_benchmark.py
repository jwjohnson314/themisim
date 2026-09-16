"""The benchmark's own arithmetic must be right, or every number it prints lies.

The load-bearing piece is :func:`brute_force_topk`: it produces the ground truth
every recall figure is measured against, so it is checked against a direct
argsort rather than trusted. The rest — percentiles, recall, formatting — is
cheap to verify and easy to get subtly wrong.

Nothing here touches a large index; the synthetic shard fixtures are enough.
"""
from __future__ import annotations

import json

# NB: conftest imports themisim (hence torch) before this module is collected,
# which is what makes the bare `import faiss` below safe. See the comment there.
import faiss
import numpy as np
import pytest

from themisim import benchmark as bm


# --------------------------------------------------------------------------- #
# brute force: the ground truth generator
# --------------------------------------------------------------------------- #
def _reference_topk(mm, queries, k):
    """Straightforward, obviously-correct exact top-k, for cross-checking."""
    pool = np.ascontiguousarray(np.asarray(mm, dtype=np.float32))
    faiss.normalize_L2(pool)
    sims = queries @ pool.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return order.astype(np.int64), np.take_along_axis(sims, order, axis=1)


@pytest.fixture
def memmap_and_queries(concat_artifacts):
    from themisim.concat import open_memmap

    meta = json.loads((concat_artifacts / "vectors.meta.json").read_text())
    mm = open_memmap(concat_artifacts / "vectors.f16.dat", meta["shape"][0])
    q = np.ascontiguousarray(np.asarray(mm[[3, 17, 400, 900]], dtype=np.float32))
    faiss.normalize_L2(q)
    return mm, q


def test_brute_force_matches_a_direct_argsort(memmap_and_queries):
    mm, q = memmap_and_queries
    ids, scores, elapsed = bm.brute_force_topk(mm, q, k=10, chunk_rows=len(mm))
    ref_ids, ref_scores = _reference_topk(mm, q, 10)
    assert np.array_equal(ids, ref_ids)
    np.testing.assert_allclose(scores, ref_scores, atol=1e-6)
    assert elapsed > 0


def test_brute_force_is_invariant_to_chunking(memmap_and_queries):
    """Chunk boundaries must not change the answer.

    The running top-k is merged across chunks; an off-by-one in the merge only
    shows up when a true neighbour sits near a chunk edge, so several odd chunk
    sizes are tried rather than one round number.
    """
    mm, q = memmap_and_queries
    ref_ids, ref_scores = _reference_topk(mm, q, 10)
    for chunk in (7, 101, 512, 1000, len(mm), len(mm) * 2):
        ids, scores, _ = bm.brute_force_topk(mm, q, k=10, chunk_rows=chunk)
        assert np.array_equal(ids, ref_ids), f"chunk_rows={chunk}"
        np.testing.assert_allclose(scores, ref_scores, atol=1e-6)


def test_brute_force_returns_results_best_first(memmap_and_queries):
    mm, q = memmap_and_queries
    _ids, scores, _ = bm.brute_force_topk(mm, q, k=10, chunk_rows=333)
    assert np.all(np.diff(scores, axis=1) <= 1e-7)


def test_brute_force_finds_the_query_itself(memmap_and_queries):
    """A frame's own vector must be its own nearest neighbour at cosine 1."""
    mm, _q = memmap_and_queries
    gids = np.array([3, 17, 400, 900])
    q = np.ascontiguousarray(np.asarray(mm[gids], dtype=np.float32))
    faiss.normalize_L2(q)
    ids, scores, _ = bm.brute_force_topk(mm, q, k=5, chunk_rows=250)
    assert np.array_equal(ids[:, 0], gids)
    np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_recall_is_1_when_the_pipeline_matches_ground_truth():
    gt = np.arange(30).reshape(3, 10)
    out = bm.recall_at_k(gt.copy(), gt)
    assert out["recall_at_k"] == 1.0
    assert out["exact_top1_hit_rate"] == 1.0
    assert out["recall_min"] == 1.0


def test_recall_counts_set_overlap_not_order():
    gt = np.array([[1, 2, 3, 4]])
    shuffled = np.array([[4, 3, 2, 1]])
    assert bm.recall_at_k(shuffled, gt)["recall_at_k"] == 1.0
    # ...but top-1 agreement is order-sensitive, and reported separately.
    assert bm.recall_at_k(shuffled, gt)["exact_top1_hit_rate"] == 0.0


def test_recall_is_fractional_on_partial_overlap():
    gt = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    ann = np.array([[1, 2, 99, 98], [5, 6, 7, 97]])
    out = bm.recall_at_k(ann, gt)
    assert out["recall_at_k"] == pytest.approx((0.5 + 0.75) / 2)
    assert out["recall_min"] == 0.5


def test_percentiles_are_ordered():
    p = bm._percentiles([0.01, 0.02, 0.03, 0.10, 0.50])
    assert p["min_s"] <= p["median_s"] <= p["p95_s"] <= p["p99_s"] <= p["max_s"]
    assert p["n"] == 5
    assert p["median_s"] == pytest.approx(0.03)


# --------------------------------------------------------------------------- #
# environment probing
# --------------------------------------------------------------------------- #
def test_evict_is_a_no_op_on_a_readable_file(tmp_path):
    """Cold-cache measurement must not need root, and must not corrupt data."""
    f = tmp_path / "x.bin"
    payload = b"themis" * 5000
    f.write_bytes(payload)
    bm.evict(f)
    assert f.read_bytes() == payload


def test_hardware_report_has_the_fields_the_numbers_depend_on():
    hw = bm.hardware_report()
    for key in ("cpu_model", "cpu_logical", "faiss", "faiss_omp_threads", "platform"):
        assert key in hw, key
    assert isinstance(hw["faiss_omp_threads"], int)


def test_storage_report_skips_throughput_for_tiny_files(concat_artifacts):
    """A 500 MB probe against a 1 MB file would time scheduling noise."""
    rep = bm.storage_report(concat_artifacts, probe=True)
    assert "vectors.f16.dat" in rep
    for entry in rep.values():
        assert entry["size_bytes"] > 0
        if entry["size_bytes"] < bm.IO_PROBE_BYTES:
            assert "cold_sequential_read_bytes_per_s" not in entry


def test_artifact_paths_covers_index_vectors_and_manifest_cache(built_artifacts):
    names = {p.name for p in bm.artifact_paths(built_artifacts)}
    assert "index.faiss" in names
    assert "vectors.f16.dat" in names


def test_hyperparameters_record_the_operating_point(engine):
    hp = bm.hyperparameters(engine, k=10, nprobe=64, prefilter=500)
    assert hp["index"]["factory"].startswith("OPQ64_64,IVF")
    assert hp["index"]["n_vectors"] == engine.total_n
    assert hp["query"]["k"] == 10
    assert hp["query"]["nprobe"] == 64
    assert hp["query"]["prefilter"] == 500
    # nprobe is clamped against nlist when reporting how much was scanned
    assert 0 < hp["query"]["fraction_of_cells_probed"] <= 1.0
    assert hp["baseline"]["chunk_rows"] > 0


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_speedup_below_1x_stays_legible():
    """An index slower than a linear scan is a finding, not a rounding error."""
    assert "0.29x" in bm._fmt_x(0.29)
    assert "1.00x" in bm._fmt_x(1.0)
    assert "1,234x" in bm._fmt_x(1234.0)
    assert bm._fmt_x(None) == "n/a"


def test_latency_and_throughput_on_a_real_small_index(engine, built_artifacts):
    gids = np.array([0, 5, 11, 23])
    lat = bm.measure_latency(engine, gids, k=5, nprobe=4, prefilter=50, warmup=1)
    assert lat["n"] == 4 and lat["median_s"] > 0
    assert lat["min_s"] <= lat["median_s"] <= lat["max_s"]
    assert lat["cold"] is False

    thr = bm.measure_throughput(engine, gids, k=5, nprobe=4, prefilter=50)
    assert thr["queries_per_s"] > 0
    assert thr["n_queries"] == 4


def test_ann_topk_returns_aligned_ids_and_scores(engine):
    gids = np.array([0, 7, 19])
    ids, scores = bm.ann_topk(engine, gids, k=5, nprobe=8, prefilter=100)
    assert ids.shape == scores.shape == (3, 5)
    # each frame retrieves itself first at cosine ~1
    assert np.array_equal(ids[:, 0], gids)
    np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-3)


def test_report_renders_and_round_trips_through_json(engine, built_artifacts):
    report = bm.run_benchmark(
        built_artifacts, k=5, nprobe=4, prefilter=50,
        n_latency=4, n_cold=2, n_recall=4, chunk_rows=500,
    )
    assert report["accuracy"]["n_queries"] == 4
    assert report["brute_force"]["wall_clock_s_one_pass"] > 0
    assert set(report["speedup_vs_brute_force"]) == {
        "single_query_warm", "single_query_cold", "batched_throughput"
    }
    json.loads(json.dumps(report))  # must be serializable
    text = bm.format_benchmark(report)
    for expected in ("Hardware", "Operating point", "Latency", "Brute-force baseline",
                     "Accuracy vs that exact baseline", "recall@5"):
        assert expected in text, expected
