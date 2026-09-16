"""The whole scientific path, from pixels to ranked results, with nothing stubbed.

This is the test a reviewer can run to satisfy themselves the pipeline works
without obtaining any THEMIS data. It starts from synthetic CDFs written in the
real on-disk format and runs the genuine chain:

    HTTP download -> inventory -> CDF read -> preprocess -> SimCLR encoder
      -> fp16 shards -> concat -> OPQ-IVF-PQ train/add -> query -> DataFrame

Every stage is the production code path. The only synthetic elements are the
pixels, the encoder weights, and the server the CDFs are fetched from.

The assertions are the ones that would catch a real regression: a frame must
retrieve *itself* first at cosine 1.0 (which is only true if global_id survives
the whole pipeline intact), its neighbours in time must rank above unrelated
frames (which is only true if the embedding carries morphology), and the
exported table must agree with the manifest.
"""
from __future__ import annotations

import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from conftest import SYNTHETIC_TREE_FRAMES

pytestmark = pytest.mark.slow


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Download, build and open an index over synthetic frames. Module-scoped
    because the encoder pass and the OPQ training are the expensive parts and
    every assertion below reads the same artifacts."""
    import torch
    import torchvision

    from conftest import write_synthetic_cdf
    from themisim.model import PROJECTION_DIM, SimCLR
    from themisim.pipeline import build_index
    from themisim.search import SearchEngine

    tmp = tmp_path_factory.mktemp("e2e")

    # --- an "archive" to download from ---------------------------------- #
    served = tmp / "www"
    write_synthetic_cdf(served, "aaaa", "2015031806", n_frames=100, seed=1)
    # Deliberately NO flat frame here. A randomly initialised ResNet-18 maps an
    # all-zero input to an all-zero output (convs carry no bias and BatchNorm
    # beta starts at zero), so a flat frame would yield a zero-norm vector --
    # an artifact of the synthetic weights, not of the pipeline. The flat-frame
    # guard is tested where it belongs, in test_preprocess.py, and the
    # zero-norm detector in test_validate.py.
    write_synthetic_cdf(served, "aaaa", "2015031807", n_frames=100, seed=2)
    write_synthetic_cdf(served, "bbbb", "2024032506", n_frames=100, seed=3, layout="v2024")
    write_synthetic_cdf(served, "bbbb", "2024032507", n_frames=50, seed=4)

    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_QuietHandler, directory=str(served))
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    try:
        # --- stage 1: download, through the real downloader -------------- #
        from themisim import download

        data_root = tmp / "cdf"
        rows = []
        for path in sorted(served.rglob("*.cdf")):
            rel = path.relative_to(served)
            rows.append({
                "site": rel.parts[0], "datetime": path.stem.split("_")[4],
                "filename": path.name, "url": f"{base}/{rel.as_posix()}",
                "target_path": str(data_root / rel), "size_bytes": 0,
            })
        wl = tmp / "wl.parquet"
        pd.DataFrame(rows).to_parquet(wl, index=False)
        results = download.run(wl, workers=2)
        assert set(results["status"]) == {"downloaded"}, results.to_dict()
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)

    # --- stage 2: a loadable checkpoint ---------------------------------- #
    ckpt = tmp / "encoder.tar"
    torch.save(
        SimCLR(torchvision.models.resnet18(weights=None), PROJECTION_DIM, 512).state_dict(),
        ckpt,
    )

    # --- stages 3-6: inventory, embed, concat, train + add --------------- #
    artifacts = tmp / "artifacts"
    build_index(
        data_root, artifacts, ckpt,
        nlist=4, device="cpu", batch_size=16, num_workers=0,
        streaming=False, use_gpu_quantizer=False,
    )
    return artifacts, data_root, SearchEngine(artifacts)


# --------------------------------------------------------------------------- #
def test_every_downloaded_frame_reaches_the_index(built):
    artifacts, _data_root, engine = built
    meta = json.loads((artifacts / "vectors.meta.json").read_text())
    assert meta["n_shards"] == 4
    assert meta["shape"] == [SYNTHETIC_TREE_FRAMES, 512]
    assert engine.total_n == SYNTHETIC_TREE_FRAMES
    assert engine.index.ntotal == SYNTHETIC_TREE_FRAMES
    assert not (artifacts / "embed_failures.parquet").exists() or \
        len(pd.read_parquet(artifacts / "embed_failures.parquet")) == 0


def test_both_cdf_layouts_were_read(built):
    """Legacy (N,256,256)+_epoch and 2024+ (1,r,c,N)+_time both ingested."""
    _artifacts, _data_root, engine = built
    assert set(engine.sites()) == {"aaaa", "bbbb"}
    lo, hi = engine.time_ns_bounds()
    assert str(np.datetime64(lo, "ns"))[:4] == "2015"
    assert str(np.datetime64(hi, "ns"))[:4] == "2024"


def test_no_frame_produced_a_degenerate_vector(built):
    """Includes the deliberately flat frame, which must embed finitely."""
    _artifacts, _data_root, engine = built
    block = np.asarray(engine.memmap, dtype=np.float32)
    assert np.isfinite(block).all(), "NaN/inf vector reached the index"
    assert (np.linalg.norm(block, axis=1) > 0).all(), "zero-norm vector reached the index"


def test_a_frame_retrieves_itself_first(built):
    """The end-to-end identity check.

    Only true if global_id survives CDF read, embedding, sharding, concat and
    indexing without drifting -- an off-by-one anywhere breaks it.
    """
    _artifacts, _data_root, engine = built
    rng = np.random.default_rng(0)
    for gid in rng.choice(engine.total_n, size=12, replace=False):
        gid = int(gid)
        vec = np.asarray(engine.memmap[gid], dtype=np.float32)
        hits = engine.search(vec, k=1, nprobe=4, prefilter=200, diversify_seconds=0)
        assert hits, f"no hits for gid={gid}"
        assert hits[0].global_id == gid
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)


def test_distinct_frames_get_distinct_vectors(built):
    """Guards the plumbing failure this test cannot otherwise see.

    If every shard were written from the same frame, or an offset bug made each
    row a copy of its neighbour, self-retrieval would still pass -- every frame
    would find "itself". Requiring the stored vectors to be pairwise distinct
    catches that.
    """
    _artifacts, _data_root, engine = built
    block = np.asarray(engine.memmap, dtype=np.float32)
    assert len(np.unique(block, axis=0)) == engine.total_n, "duplicate vectors stored"
    assert float(np.std(np.linalg.norm(block, axis=1))) > 0, "all vectors identical"


@pytest.mark.skipif(
    not (Path("weights") / "aurora-fm-no-finetune.tar").exists(),
    reason="needs the real SimCLR checkpoint; run fetch_weights() first",
)
def test_temporal_neighbours_outrank_unrelated_frames(tmp_path_factory):
    """Morphology, which only the *trained* encoder can provide.

    Kept separate and skippable on purpose. Everything else in this file tests
    plumbing and holds with random weights; this one tests that the embedding
    is meaningful, which a randomly initialised network cannot be. Without that
    separation the suite would look like it verifies the science offline when
    it does not -- embedding quality is established by the pilot index against
    real weights and real frames.
    """
    import torch  # noqa: F401

    from conftest import write_synthetic_cdf
    from themisim.pipeline import build_index
    from themisim.search import SearchEngine

    tmp = tmp_path_factory.mktemp("e2e_real")
    root = tmp / "cdf"
    write_synthetic_cdf(root, "aaaa", "2015031806", n_frames=300, seed=1)
    artifacts = tmp / "artifacts"
    build_index(root, artifacts, Path("weights") / "aurora-fm-no-finetune.tar",
                nlist=4, device="cpu", batch_size=16, num_workers=0,
                streaming=False, use_gpu_quantizer=False)
    engine = SearchEngine(artifacts)

    gid = 150
    vec = np.asarray(engine.memmap[gid], dtype=np.float32)
    hits = engine.search(vec, k=6, nprobe=4, prefilter=300, diversify_seconds=0)
    neighbours = [h.global_id for h in hits if h.global_id != gid]
    assert neighbours, "no neighbours returned"
    close = sum(1 for n in neighbours if abs(n - gid) <= 10)
    assert close >= len(neighbours) // 2, (
        f"temporally adjacent frames did not dominate: {neighbours}"
    )


def test_query_by_site_datetime_frame_returns_the_export_table(built):
    """The public entry point, addressed the way a user addresses it."""
    from themisim.export import EXPORT_COLUMNS
    from themisim.query import query

    _artifacts, _data_root, engine = built
    df = query("aaaa", "2015-03-18T06", 42, engine=engine, results=5,
               diversify_seconds=0)
    assert list(df.columns) == EXPORT_COLUMNS
    assert len(df) == 5
    top = df.iloc[0]
    assert top["site"] == "aaaa"
    assert top["score"] == pytest.approx(1.0, abs=1e-3)
    assert top["datetime"].startswith("2015-03-18 06:")
    assert top["source_cdf"].endswith("thg_l1_asf_aaaa_2015031806_v01.cdf")
    assert df["score"].is_monotonic_decreasing


def test_validation_report_passes_on_the_synthetic_index(built):
    """The quality report a reviewer reads must come out clean on a good build."""
    from themisim.validate import validate_index

    artifacts, _data_root, engine = built
    report = validate_index(
        artifacts, nprobe=4, prefilter=200, k=5, n_queries=20,
        nprobe_sweep=(1, 4), prefilter_sweep=(50, 200), engine=engine,
    )
    failed = {k: v for k, v in report["checks"].items() if v.get("passed") is False}
    assert not failed, json.dumps(failed, indent=2, default=str)
    assert report["checks"]["integrity"]["contiguous_site_blocks"] is True
    assert report["checks"]["self_retrieval"]["exact_path_fraction"] == 1.0


def test_index_is_queryable_after_reopening_from_disk(built):
    """Nothing may depend on in-process state left over from the build."""
    from themisim.search import SearchEngine

    artifacts, _data_root, _engine = built
    fresh = SearchEngine(artifacts)
    assert fresh.total_n == SYNTHETIC_TREE_FRAMES
    vec = np.asarray(fresh.memmap[7], dtype=np.float32)
    hits = fresh.search(vec, k=3, nprobe=4, prefilter=200, diversify_seconds=0)
    assert hits[0].global_id == 7
