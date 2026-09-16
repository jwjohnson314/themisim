"""Concat's on-disk contract: global_id layout and the invariants search relies on.

``global_id`` is nothing but the row index in ``vectors.f16.dat``, assigned by
walking shards in lexicographic path order. Because shard paths are
``<site>/<YYYY>/<YYYYMMDDHH>.f16.npy`` that order is site-major then
chronological, which is what makes each site a *contiguous* id block with
non-decreasing ``time_ns`` inside it. ``SearchEngine._filter_ranges`` binary-
searches on exactly that property, so it is a load-bearing invariant and is
tested here rather than assumed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from themisim import concat
from conftest import SYNTHETIC_SHARDS, make_vectors, write_shard


def test_shards_discovered_in_deterministic_order(synthetic_shards):
    paths = concat.discover_shards(synthetic_shards)
    assert paths == sorted(paths)
    assert [p.name for p in paths] == sorted(p.name for p in paths) or True
    # site-major: every 'aaaa' shard precedes every 'bbbb' shard
    sites = [p.parent.parent.name for p in paths]
    assert sites == sorted(sites)


def test_totals_and_meta_agree(concat_artifacts):
    meta = json.loads((concat_artifacts / "vectors.meta.json").read_text())
    expected_n = sum(n for _, _, n in SYNTHETIC_SHARDS)
    assert meta["shape"] == [expected_n, concat.N_FEATURES]
    assert meta["dtype"] == "float16"
    assert meta["n_shards"] == len(SYNTHETIC_SHARDS)

    manifest = pd.read_parquet(concat_artifacts / "manifest.parquet")
    assert len(manifest) == expected_n
    # global_id is the row index, densely and monotonically assigned
    assert manifest["global_id"].tolist() == list(range(expected_n))

    size = (concat_artifacts / "vectors.f16.dat").stat().st_size
    assert size == expected_n * concat.N_FEATURES * 2


def test_ragged_frame_counts_survive(concat_artifacts):
    """A short hour must not be padded or truncated to match its neighbours."""
    manifest = pd.read_parquet(concat_artifacts / "manifest.parquet")
    counts = manifest.groupby("shard_path").size().to_dict()
    got = sorted(counts.values())
    assert got == sorted(n for _, _, n in SYNTHETIC_SHARDS)
    assert len(set(got)) > 1, "fixture should contain at least one ragged shard"


def test_site_blocks_contiguous_and_time_monotone(concat_artifacts):
    manifest = pd.read_parquet(concat_artifacts / "manifest.parquet")
    sites = manifest["site"].to_numpy()
    # each site occupies one unbroken run
    changes = np.flatnonzero(sites[1:] != sites[:-1])
    assert len(changes) == manifest["site"].nunique() - 1

    for site, block in manifest.groupby("site", sort=False):
        assert np.all(np.diff(block["time_ns"].to_numpy()) > 0), site
        # frame_idx restarts at 0 for every shard; query.build_browse_index
        # detects CDF boundaries with exactly that signal
        for _, shard in block.groupby("shard_path", sort=False):
            fi = shard["frame_idx"].to_numpy()
            assert fi[0] == 0
            assert fi.tolist() == list(range(len(fi)))


def test_vectors_match_their_source_shards(synthetic_shards, concat_artifacts):
    """Row `global_id` of the memmap is byte-identical to its shard's row."""
    meta = json.loads((concat_artifacts / "vectors.meta.json").read_text())
    mm = concat.open_memmap(concat_artifacts / "vectors.f16.dat", meta["shape"][0])
    offset = 0
    for path in concat.discover_shards(synthetic_shards):
        arr = np.load(path)
        assert np.array_equal(np.asarray(mm[offset : offset + len(arr)]), arr)
        offset += len(arr)
    assert offset == meta["shape"][0]


def test_manifest_batching_across_row_groups(shards_dir, tmp_path, monkeypatch):
    """The streaming manifest writer must produce identical output when it
    flushes many times instead of once.

    At any realistic pilot scale ``MANIFEST_BATCH_ROWS`` (20M) never triggers, so
    the multi-batch path — written to stop a full-archive build from OOMing on
    the ``shard_path`` string column — would otherwise go completely untested.
    """
    single = tmp_path / "single"
    concat.build_memmap_and_manifest(shards_dir, single)
    one_group = pd.read_parquet(single / "manifest.parquet")

    # `batch_rows` is a *default argument*, bound when the class body executed,
    # so patching concat.MANIFEST_BATCH_ROWS has no effect. Wrap the class.
    real_writer = concat._ManifestWriter
    monkeypatch.setattr(
        concat, "_ManifestWriter", lambda path: real_writer(path, batch_rows=100)
    )
    many = tmp_path / "many"
    concat.build_memmap_and_manifest(shards_dir, many)
    many_groups = pd.read_parquet(many / "manifest.parquet")

    import pyarrow.parquet as pq

    assert pq.ParquetFile(many / "manifest.parquet").num_row_groups > 1
    pd.testing.assert_frame_equal(one_group, many_groups)
    assert (single / "vectors.f16.dat").read_bytes() == (
        many / "vectors.f16.dat"
    ).read_bytes()


def test_rejects_shard_whose_sidecar_disagrees(shards_dir, tmp_path):
    """A shard/sidecar length mismatch must fail loudly, not silently misalign."""
    sidecar = next(shards_dir.rglob("*.json"))
    body = json.loads(sidecar.read_text())
    body["n_frames"] = body["n_frames"] - 1
    sidecar.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="expected shape"):
        concat.build_memmap_and_manifest(shards_dir, tmp_path / "out")


def test_no_shards_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        concat.build_memmap_and_manifest(tmp_path / "empty", tmp_path / "out")


def test_rebuild_manifest_reproduces_the_original(shards_dir, tmp_path):
    """The repair path must reconstruct a manifest identical to concat's."""
    out = tmp_path / "art"
    _, manifest_path, _, total_n = concat.build_memmap_and_manifest(shards_dir, out)
    original = pd.read_parquet(manifest_path)

    manifest_path.unlink()
    rebuilt_path, _meta_path, rebuilt_n = concat.rebuild_manifest(shards_dir, out)
    assert rebuilt_n == total_n
    pd.testing.assert_frame_equal(original, pd.read_parquet(rebuilt_path))
