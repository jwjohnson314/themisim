"""Index-build internals: the training sampler and the recall helper.

Both are exercised here with *shrunken constants*. That is the point of the
file: at pilot scale ``target_n`` exceeds every bucket's population, so
``stratified_sample_from_parquet`` returns the whole database and its reservoir
branch never executes. The reservoir, the bottom-k merge and the chunking
invariance are the parts that replaced a 120 GB OOM on the full archive, and
they are only reachable by forcing small targets and many row groups.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from themisim import index as index_mod
from themisim.concat import open_memmap


def _manifest(path):
    return pd.read_parquet(path)


def test_sampler_is_deterministic_sorted_and_unique(concat_artifacts):
    p = concat_artifacts / "manifest.parquet"
    a = index_mod.stratified_sample_from_parquet(p, target_n=200, seed=0)
    b = index_mod.stratified_sample_from_parquet(p, target_n=200, seed=0)
    assert np.array_equal(a, b)
    assert np.all(np.diff(a) > 0)          # sorted and unique
    assert a.dtype == np.int64


def test_sampler_reservoir_branch_respects_per_bucket_cap(concat_artifacts):
    """With a small target the per-bucket cap binds and the bottom-k runs."""
    p = concat_artifacts / "manifest.parquet"
    manifest = _manifest(p)
    manifest["ym"] = pd.to_datetime(manifest["time_ns"]).dt.strftime("%Y%m")
    populations = manifest.groupby(["site", "ym"]).size()
    n_buckets = len(populations)

    target = 40
    ids = index_mod.stratified_sample_from_parquet(p, target_n=target, seed=0)
    per_bucket = max(1, target // n_buckets)
    assert len(ids) == int(sum(min(per_bucket, pop) for pop in populations))
    assert len(ids) < len(manifest), "cap must actually bind, else this proves nothing"

    # every bucket is represented
    chosen = manifest.set_index("global_id").loc[ids]
    assert set(map(tuple, chosen[["site", "ym"]].values)) == set(populations.index)


def test_sampler_is_invariant_to_row_group_chunking(concat_artifacts, tmp_path):
    """The docstring promises chunking-invariance; nothing verified it.

    The bottom-k-of-iid-uniform-keys scheme is an exact uniform sample only if
    merging across row groups is order-independent. Rewriting the same manifest
    with a different row-group size is the direct test of that claim.
    """
    import pyarrow.parquet as pq

    src = concat_artifacts / "manifest.parquet"
    table = pq.read_table(src)

    one = tmp_path / "one_group.parquet"
    pq.write_table(table, one, row_group_size=len(table))
    many = tmp_path / "many_groups.parquet"
    pq.write_table(table, many, row_group_size=37)

    assert pq.ParquetFile(one).num_row_groups == 1
    assert pq.ParquetFile(many).num_row_groups > 1

    for target in (40, 200, 10_000):
        a = index_mod.stratified_sample_from_parquet(one, target_n=target, seed=0)
        b = index_mod.stratified_sample_from_parquet(many, target_n=target, seed=0)
        assert np.array_equal(a, b), f"chunking changed the sample at target_n={target}"


def test_streaming_sampler_matches_the_in_memory_one(concat_artifacts, tmp_path):
    """Both samplers document the same selection semantics; check they agree.

    They cannot produce identical *ids* (different RNG draw order), but the
    per-bucket counts they select are part of the contract.
    """
    p = concat_artifacts / "manifest.parquet"
    manifest = _manifest(p)
    target = 60

    streamed = index_mod.stratified_sample_from_parquet(p, target_n=target, seed=0)
    in_memory = index_mod.stratified_sample(manifest, target_n=target, seed=0)

    def bucket_counts(ids):
        rows = manifest.set_index("global_id").loc[np.sort(ids)]
        ym = pd.to_datetime(rows["time_ns"]).dt.strftime("%Y%m")
        return rows.groupby([rows["site"], ym]).size().sort_index()

    pd.testing.assert_series_equal(
        bucket_counts(streamed), bucket_counts(in_memory), check_names=False
    )


def test_sampler_validates_row_count(concat_artifacts):
    p = concat_artifacts / "manifest.parquet"
    with pytest.raises(AssertionError):
        index_mod.stratified_sample_from_parquet(p, target_n=50, total_n_check=999_999)


def test_gather_train_rows_is_invariant_to_chunk_size(concat_artifacts):
    """The HDD-friendly block sweep must return the same rows as a plain gather.

    ``TRAIN_GATHER_CHUNK`` is 2M rows, so on any pilot-sized memmap the loop
    body runs exactly once and the block-boundary arithmetic is never tested.
    """
    meta = json.loads((concat_artifacts / "vectors.meta.json").read_text())
    total_n = meta["shape"][0]
    mm = open_memmap(concat_artifacts / "vectors.f16.dat", total_n)
    ids = np.sort(np.random.default_rng(0).choice(total_n, size=64, replace=False))

    reference = np.asarray(mm[ids], dtype=np.float32)
    for chunk in (7, 64, 1000, total_n, total_n * 2):
        got = index_mod._gather_train_rows(mm, ids, chunk_rows=chunk)
        assert np.array_equal(got, reference), f"chunk_rows={chunk}"


def test_evaluate_recall_is_perfect_for_an_exact_index(concat_artifacts):
    """Against a Flat inner-product index, recall@k must be exactly 1.0.

    This pins the helper itself: if it reported less than 1.0 for exhaustive
    search, every recall number it produces elsewhere would be meaningless.
    """
    import faiss

    meta = json.loads((concat_artifacts / "vectors.meta.json").read_text())
    total_n = meta["shape"][0]
    mm = open_memmap(concat_artifacts / "vectors.f16.dat", total_n)

    pool = np.ascontiguousarray(np.asarray(mm, dtype=np.float32))
    faiss.normalize_L2(pool)
    flat = faiss.IndexFlatIP(pool.shape[1])
    flat.add(pool)

    ids = np.arange(total_n, dtype=np.int64)
    queries = pool[:25].copy()
    # evaluate_recall reaches for the inner IVF; a Flat index has none, so wrap
    # it the way the factory would and give it a single cell.
    ivf = faiss.index_factory(pool.shape[1], "IVF1,Flat", faiss.METRIC_INNER_PRODUCT)
    ivf.train(pool)
    ivf.add(pool)
    out = index_mod.evaluate_recall(ivf, queries, pool, ids, k=10, nprobe=1)
    assert out["recall_at_k"] == pytest.approx(1.0)
    assert out["n_queries"] == 25
    assert out["k"] == 10


def test_add_chunking_does_not_change_the_index_contents(concat_artifacts, tmp_path):
    """``ADD_CHUNK_SIZE`` is 100k, so pilot builds add in a single chunk.

    Forcing many small chunks proves the loop's offset arithmetic adds every
    row exactly once.
    """
    meta = json.loads((concat_artifacts / "vectors.meta.json").read_text())
    total_n = meta["shape"][0]
    mm = open_memmap(concat_artifacts / "vectors.f16.dat", total_n)
    ids = np.arange(total_n, dtype=np.int64)

    one = index_mod.train_and_build_index(
        mm, out_path=tmp_path / "one.faiss", nlist=4, train_ids=ids,
        use_gpu_quantizer=False, add_chunk_size=total_n,
    )
    many = index_mod.train_and_build_index(
        mm, out_path=tmp_path / "many.faiss", nlist=4, train_ids=ids,
        use_gpu_quantizer=False, add_chunk_size=13,
    )
    assert one.ntotal == many.ntotal == total_n


def test_factory_string_matches_the_documented_design():
    assert index_mod._build_factory_string(4096) == "OPQ64_64,IVF4096_HNSW32,PQ64x8"
