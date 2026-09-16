"""Shared fixtures: synthetic shards, and a real index built from them.

The build path (embed -> concat -> train/add -> query) had no test coverage at
all, because every stage before this file wanted real CDFs, a GPU-sized dataset,
or both. Nothing here touches the network or reads a CDF: shards are written
directly in the on-disk layout ``embed`` produces, which is the narrowest seam
that still exercises ``concat``, ``index`` and ``search`` for real, including a
genuinely trained FAISS index.

Two properties are deliberately built into the synthetic data:

* **ragged frame counts** — a real archive hour is ~1200 frames but night-edge
  hours are short, so nothing downstream may assume a constant. One shard here
  is deliberately a different length.
* **cluster structure** — vectors are drawn around a handful of centres rather
  than iid Gaussian, so nearest-neighbour queries have a meaningful answer and
  recall numbers mean something.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Import themisim before anything in the test session can reach faiss. Its
# __init__ imports torch first, which is load-bearing on machines with a
# CUDA-enabled conda faiss: faiss links the conda CUDA runtime, and if it loads
# first, torch's libc10_cuda.so binds to that runtime and dies with
# "undefined symbol: cudaGetDriverEntryPointByVersion". conftest runs before
# every test module, so doing it here fixes the order for all of them and does
# not depend on which file pytest happens to collect first.
import themisim  # noqa: F401,E402  (side effect: establishes torch-before-faiss)

N_FEATURES = 512
NS_PER_FRAME = 3_000_000_000  # THEMIS runs at a 3-second cadence

#: (site, YYYYMMDDHH, n_frames). Two sites, two year-months, one short hour.
#: Site codes are four lowercase letters so they satisfy the archive's
#: ``CDF_NAME_RE`` and survive a round-trip through ``parse_cdf_filename``.
SYNTHETIC_SHARDS = [
    ("aaaa", "2015031806", 260),
    ("aaaa", "2015031807", 260),
    ("aaaa", "2016011903", 180),   # short hour: ragged n_frames
    ("bbbb", "2015031806", 260),
    ("bbbb", "2015031807", 260),
    ("bbbb", "2016011903", 260),
]


def _hour_start_ns(dtstr: str) -> int:
    import datetime as dt

    t = dt.datetime(
        int(dtstr[:4]), int(dtstr[4:6]), int(dtstr[6:8]), int(dtstr[8:10]),
        tzinfo=dt.timezone.utc,
    )
    return int(t.timestamp()) * 1_000_000_000


def make_vectors(n: int, rng: np.random.Generator, *, n_clusters: int = 8) -> np.ndarray:
    """Clustered unit-ish vectors, so nearest neighbours are not arbitrary."""
    centres = rng.standard_normal((n_clusters, N_FEATURES)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    which = rng.integers(0, n_clusters, size=n)
    out = centres[which] + 0.25 * rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.astype(np.float16)


def write_shard(
    shards_dir: Path, site: str, dtstr: str, vecs: np.ndarray, *, time_ns=None
) -> Path:
    """Write one shard + sidecar exactly where ``embed.shard_paths`` would."""
    out = shards_dir / site / dtstr[:4]
    out.mkdir(parents=True, exist_ok=True)
    shard = out / f"{dtstr}.f16.npy"
    np.save(shard, vecs)
    if time_ns is None:
        start = _hour_start_ns(dtstr)
        time_ns = [start + i * NS_PER_FRAME for i in range(len(vecs))]
    (out / f"{dtstr}.json").write_text(
        json.dumps(
            {
                "site": site,
                "datetime": dtstr,
                "n_frames": int(len(vecs)),
                "time_ns": [int(t) for t in time_ns],
                "source_sha256": "0" * 64,
                "source_path": f"/synthetic/thg_l1_asf_{site}_{dtstr}_v01.cdf",
            }
        )
    )
    return shard


@pytest.fixture(scope="session")
def synthetic_shards(tmp_path_factory) -> Path:
    """A shards directory in the real on-disk layout. Session-scoped: read-only."""
    shards_dir = tmp_path_factory.mktemp("shards_src") / "shards"
    rng = np.random.default_rng(42)
    for site, dtstr, n in SYNTHETIC_SHARDS:
        write_shard(shards_dir, site, dtstr, make_vectors(n, rng))
    return shards_dir


@pytest.fixture
def shards_dir(tmp_path) -> Path:
    """A fresh, writable copy of the synthetic shard set for mutating tests."""
    out = tmp_path / "shards"
    rng = np.random.default_rng(42)
    for site, dtstr, n in SYNTHETIC_SHARDS:
        write_shard(out, site, dtstr, make_vectors(n, rng))
    return out


@pytest.fixture(scope="session")
def concat_artifacts(synthetic_shards, tmp_path_factory) -> Path:
    """``vectors.f16.dat`` + ``manifest.parquet`` + ``vectors.meta.json``."""
    from themisim.concat import build_memmap_and_manifest

    out = tmp_path_factory.mktemp("concat_artifacts")
    build_memmap_and_manifest(synthetic_shards, out)
    return out


@pytest.fixture(scope="session")
def built_artifacts(concat_artifacts, tmp_path_factory) -> Path:
    """A complete, queryable artifacts directory with a real trained index.

    Session-scoped because training the production factory
    (``OPQ64_64,IVF{nlist}_HNSW32,PQ64x8``) costs tens of seconds even at this
    size — OPQ runs 50 iterations regardless of how little data it is given.
    ``nlist`` is pinned rather than left to ``auto_nlist``, whose ``sqrt``
    branch would pick 1 here and build an HNSW graph over a single centroid.
    """
    import shutil

    from themisim.concat import open_memmap
    from themisim.index import stratified_sample_from_parquet, train_and_build_index

    art = tmp_path_factory.mktemp("built_artifacts")
    for name in ("vectors.f16.dat", "manifest.parquet", "vectors.meta.json"):
        shutil.copy(concat_artifacts / name, art / name)

    total_n = json.loads((art / "vectors.meta.json").read_text())["shape"][0]
    train_ids = stratified_sample_from_parquet(
        art / "manifest.parquet", target_n=total_n, seed=0, total_n_check=total_n
    )
    train_and_build_index(
        open_memmap(art / "vectors.f16.dat", total_n),
        out_path=art / "index.faiss",
        nlist=8,
        train_ids=train_ids,
        use_gpu_quantizer=False,
    )
    return art


@pytest.fixture
def engine(built_artifacts):
    """A ``SearchEngine`` over the built synthetic index."""
    from themisim.search import SearchEngine

    return SearchEngine(built_artifacts)


# --------------------------------------------------------------------------- #
# synthetic THEMIS CDFs — the input end of the pipeline
# --------------------------------------------------------------------------- #
#: Frames in a synthetic hour. Far below the real ~1200, but the pipeline must
#: not assume a constant count, and small files keep the suite quick.
SYNTHETIC_FRAMES = 24


def write_synthetic_cdf(
    data_root: Path,
    site: str,
    dtstr: str,
    *,
    n_frames: int = SYNTHETIC_FRAMES,
    seed: int = 0,
    layout: str = "legacy",
    flat_frame: int | None = None,
) -> Path:
    """Write a THEMIS-format CDF that ``embed._read_cdf`` can actually read.

    This is what makes a genuine end-to-end test possible: without it every
    test has to start downstream of the CDF reader, the preprocessing and the
    encoder — which is to say, downstream of the science.

    ``layout`` selects between the two mutually exclusive on-disk shapes the
    reader supports: ``"legacy"`` is ``(N, 256, 256)`` with CDF_EPOCH
    timestamps, ``"v2024"`` is ``(1, row, col, N)`` with Unix-second timestamps
    in a ``_time`` variable. Only real 2024+ archive files exercise the second
    branch otherwise.

    ``flat_frame`` makes one frame constant, which drives ``normalize_masked``
    into its ``p1 == p99`` guard — the path that once produced NaN embeddings
    and silently poisoned the index.
    """
    from cdflib import cdfepoch, cdfwrite

    out = data_root / site / dtstr[:4] / dtstr[4:6]
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"thg_l1_asf_{site}_{dtstr}_v01.cdf"

    rng = np.random.default_rng(seed)
    # A moving bright blob on a noisy background, so consecutive frames are
    # genuinely similar and nearest-neighbour results mean something.
    imgs = rng.integers(200, 600, size=(n_frames, 256, 256)).astype(np.uint16)
    yy, xx = np.ogrid[:256, :256]
    # The blob drifts monotonically rather than orbiting. A periodic path makes
    # frames an exact period apart visually identical, so temporal distance
    # stops being a proxy for visual similarity -- with a sinusoidal path the
    # real encoder correctly returned frames +/-25, +/-50 and +/-125 away,
    # which looks like a retrieval failure and is in fact a data-design one.
    span = max(n_frames - 1, 1)
    for i in range(n_frames):
        cy = 60 + int(130 * i / span)
        cx = 60 + int(130 * i / span)
        radius = 24 + int(12 * i / span)
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
        imgs[i][blob] = 3000
    if flat_frame is not None:
        imgs[flat_frame] = 7  # constant: p1 == p99

    import datetime as _dt

    start = _dt.datetime(
        int(dtstr[:4]), int(dtstr[4:6]), int(dtstr[6:8]), int(dtstr[8:10]),
        tzinfo=_dt.timezone.utc,
    )
    # Derive each timestamp from a real datetime rather than writing `3 * i`
    # into the seconds field: past 20 frames that overflows 59 s, and
    # cdfepoch.compute_epoch normalises the overflow into *extra* records, so
    # the epoch array silently ends up longer than the image array.
    stamps = [start + _dt.timedelta(seconds=3 * i) for i in range(n_frames)]

    writer = cdfwrite.CDF(str(path), cdf_spec={"Compressed": 0}, delete=True)
    try:
        if layout == "legacy":
            writer.write_var(
                {"Variable": f"thg_asf_{site}", "Data_Type": 12,
                 "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": [256, 256]},
                var_data=imgs,
            )
            epochs = cdfepoch.compute_epoch(
                [[t.year, t.month, t.day, t.hour, t.minute, t.second,
                  t.microsecond // 1000] for t in stamps]
            )
            writer.write_var(
                {"Variable": f"thg_asf_{site}_epoch", "Data_Type": 31,
                 "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
                var_data=np.asarray(epochs),
            )
        elif layout == "v2024":
            secs = np.array([t.timestamp() for t in stamps], dtype=np.float64)
            writer.write_var(
                {"Variable": f"thg_asf_{site}", "Data_Type": 12, "Num_Elements": 1,
                 "Rec_Vary": True, "Dim_Sizes": [256, 256, n_frames]},
                var_data=np.moveaxis(imgs, 0, -1)[None, ...],
            )
            writer.write_var(
                {"Variable": f"thg_asf_{site}_time", "Data_Type": 45,
                 "Num_Elements": 1, "Rec_Vary": True, "Dim_Sizes": []},
                var_data=secs,
            )
        else:
            raise ValueError(f"unknown layout {layout!r}")
    finally:
        writer.close()
    return path


@pytest.fixture
def synthetic_cdf_tree(tmp_path) -> Path:
    """A ``--data-root``-shaped tree of real, readable CDFs.

    Two sites and two year-months so the (site, year_month) stratification has
    something to stratify, both on-disk layouts, one short hour, and one flat
    frame to exercise the degenerate-frame guard.
    """
    root = tmp_path / "cdf"
    # Frame counts are sized so the tree can actually train the production
    # factory: PQ64x8 clusters 256 centroids per sub-quantizer, so fewer than
    # 256 vectors makes FAISS refuse outright ("nx >= k failed").
    write_synthetic_cdf(root, "aaaa", "2015031806", n_frames=100, seed=1)
    write_synthetic_cdf(root, "aaaa", "2015031807", n_frames=100, seed=2, flat_frame=3)
    write_synthetic_cdf(root, "bbbb", "2024032506", n_frames=100, seed=3, layout="v2024")
    write_synthetic_cdf(root, "bbbb", "2024032507", n_frames=50, seed=4)
    return root


#: Total frames in ``synthetic_cdf_tree``; the ragged last hour is deliberate.
SYNTHETIC_TREE_FRAMES = 100 + 100 + 100 + 50


@pytest.fixture(scope="session")
def synthetic_checkpoint(tmp_path_factory) -> Path:
    """A checkpoint ``load_simclr`` accepts, without the 44 MB download.

    ``load_simclr`` loads with ``strict=True``, so a state_dict produced by the
    real ``SimCLR`` class is exactly what it expects — which means this both
    stands in for the published weights *and* tests that the loader's key
    parity check is satisfiable by the class it claims to match.
    """
    import torch
    import torchvision

    from themisim.model import PROJECTION_DIM, SimCLR

    encoder = torchvision.models.resnet18(weights=None)
    model = SimCLR(encoder=encoder, projection_dim=PROJECTION_DIM, n_features=512)
    path = tmp_path_factory.mktemp("weights") / "synthetic-simclr.tar"
    torch.save(model.state_dict(), path)
    return path
