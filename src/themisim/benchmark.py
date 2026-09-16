"""Measure what the index actually costs: latency, throughput, and recall.

The claim an ANN index makes is that it trades a little accuracy for a lot of
speed. That trade is only meaningful if both halves are measured against the
thing being replaced, so everything here is reported next to an exact
brute-force scan of the same database.

Three things make these numbers worth reading carefully:

* **The operating point is the pipeline, not the index.**
  :meth:`~themisim.search.SearchEngine.search` pulls the top ``prefilter``
  candidates from the compressed index and then rescores them exactly in fp32
  against the stored vectors. Latency is therefore the sum of an index probe and
  a scattered read of ``prefilter`` rows, and recall is the recall of that whole
  pipeline. Timing the FAISS call alone would flatter the system and describe
  something no user runs.

* **Storage placement dominates.** The rerank reads ``prefilter`` rows from
  arbitrary offsets in a ~1 TB memmap. On a spinning disk that is ``prefilter``
  seeks at several milliseconds each; on SSD it is a fraction of that. So the
  report records which device each artifact lives on and whether it was rotational.

* **Cold and warm are different systems.** With ``IO_FLAG_MMAP`` the index is
  paged in on demand, so a query against a cold cache pays disk latency that a
  warm one does not. Both are measured, using ``posix_fadvise(DONTNEED)`` to
  evict a file's pages without needing root.

Brute force is run as a *single batched pass*: one sequential sweep of the
memmap scores every query at once. That is the honest baseline — it is what you
would actually do for a batch of queries — and the per-query cost of a lone
brute-force query (one whole pass) is reported separately.
"""
from __future__ import annotations

import json
import mmap
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np

from themisim.concat import N_FEATURES
from themisim.search import (
    DEFAULT_BRUTE_FORCE_MAX,
    DEFAULT_DIVERSIFY_SECONDS,
    DEFAULT_NPROBE,
    DEFAULT_PREFILTER,
    SearchEngine,
    _RERANK_CHUNK,
)

#: Rows per chunk in the brute-force sweep. 1M rows is 1 GB of fp16 on disk and
#: 2 GB once cast to fp32 — large enough that the read stays sequential, small
#: enough to keep peak memory bounded well below RAM.
BRUTE_CHUNK_ROWS = 1_000_000

#: Bytes read when measuring a device's cold sequential throughput.
IO_PROBE_BYTES = 512 << 20

#: Grid for the recall/latency trade-off curve. nprobe and prefilter are swept
#: together because they interact: at a fixed candidate budget, probing more
#: cells can lower end-to-end recall by crowding true neighbours out of the
#: prefilter, so neither is meaningful reported alone.
DEFAULT_NPROBE_SWEEP = (1, 8, 32, 64, 256)
DEFAULT_PREFILTER_SWEEP = (100, 500, 1000, 2000)

#: Operating points characterised in full (median/p95, warm and cold, at each
#: diversification setting): the library default, and the higher-recall setting
#: used in the paper's figures.
DEFAULT_OPERATING_POINTS = ((64, 500), (256, 1000))


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _mem_total_bytes() -> Optional[int]:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def hardware_report() -> dict:
    """CPU, memory, thread and library configuration.

    Recorded because every number in this report is hardware-bound: latency is
    dominated by storage, throughput by core count and BLAS threading.
    """
    import numpy as _np

    try:
        logical = os.cpu_count()
        physical = int(
            subprocess.run(
                ["lscpu", "-p=Core,Socket"], capture_output=True, text=True, timeout=10
            ).stdout.count("\n")
        )
    except Exception:
        logical, physical = os.cpu_count(), None

    report = {
        "cpu_model": _cpu_model(),
        "cpu_logical": logical,
        "ram_total_bytes": _mem_total_bytes(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "faiss_omp_threads": int(faiss.omp_get_max_threads()),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    try:
        report["faiss_compile_options"] = faiss.get_compile_options()
    except Exception:
        pass
    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_threads"] = int(torch.get_num_threads())
    except Exception:
        report["torch"] = None
    # The query path is CPU-only by design; note GPUs so the reader knows they
    # were present and deliberately unused.
    try:
        report["faiss_num_gpus"] = int(faiss.get_num_gpus())
    except Exception:
        report["faiss_num_gpus"] = 0
    return report


def _device_of(path: Path) -> dict:
    """Which block device backs ``path``, and is it rotational?

    The single most explanatory fact about query latency here: the exact rerank
    issues ``prefilter`` random reads, and a seek costs ~6 ms on a spinning disk
    versus tens of microseconds on SSD.
    """
    info: Dict[str, object] = {"path": str(path)}
    try:
        src = subprocess.run(
            ["df", "--output=source,target", str(path)],
            capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()[-1].split()
        info["device"], info["mount"] = src[0], src[1]
        base = Path(src[0]).name.rstrip("0123456789")
        rot = Path(f"/sys/block/{base}/queue/rotational")
        if rot.exists():
            info["rotational"] = rot.read_text().strip() == "1"
            info["media"] = "HDD" if info["rotational"] else "SSD"
    except Exception:
        pass
    return info


def evict(path: Union[str, Path]) -> None:
    """Drop ``path``'s clean pages from the page cache. No root required.

    ``posix_fadvise(POSIX_FADV_DONTNEED)`` is the unprivileged equivalent of
    ``echo 3 > /proc/sys/vm/drop_caches``, and it is narrower: it evicts only
    this file rather than the whole system's cache, so a cold-cache measurement
    does not also punish everything else running on the machine.
    """
    fd = os.open(os.path.realpath(str(path)), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def measure_sequential_read(path: Union[str, Path], nbytes: int = IO_PROBE_BYTES) -> float:
    """Cold sequential read throughput in bytes/s, used to bound the scan cost."""
    path = os.path.realpath(str(path))
    evict(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        t0 = time.perf_counter()
        got = 0
        while got < nbytes:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            got += len(block)
        dt = time.perf_counter() - t0
    finally:
        os.close(fd)
    return got / dt if dt > 0 else float("nan")


def storage_report(artifacts: Union[str, Path], *, probe: bool = True) -> dict:
    """Size, backing device and cold read throughput of each artifact."""
    artifacts = Path(artifacts)
    out = {}
    for name in ("index.faiss", "vectors.f16.dat", "manifest.parquet"):
        p = artifacts / name
        if not p.exists():
            continue
        real = Path(os.path.realpath(p))
        entry = _device_of(real)
        entry["size_bytes"] = real.stat().st_size
        # A probe shorter than a few readahead windows measures scheduling
        # noise, not the device, so small files get no throughput figure at all
        # rather than a misleading one.
        if probe and entry["size_bytes"] >= IO_PROBE_BYTES:
            entry["cold_sequential_read_bytes_per_s"] = measure_sequential_read(real)
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# latency / throughput
# --------------------------------------------------------------------------- #
def _percentiles(times: Sequence[float]) -> dict:
    a = np.asarray(times, dtype=np.float64)
    return {
        "n": int(a.size),
        "median_s": float(np.median(a)),
        "p95_s": float(np.percentile(a, 95)),
        "p99_s": float(np.percentile(a, 99)),
        "mean_s": float(a.mean()),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
    }


def measure_latency(
    engine: SearchEngine,
    query_gids: Sequence[int],
    *,
    k: int = 10,
    nprobe: int = DEFAULT_NPROBE,
    prefilter: int = DEFAULT_PREFILTER,
    cold: bool = False,
    evict_paths: Sequence[Union[str, Path]] = (),
    warmup: int = 10,
    diversify_seconds: int = 0,
) -> dict:
    """Per-query wall-clock for the full search pipeline.

    Queries a frame by its own stored vector, which is what
    :func:`themisim.query.query` does. ``diversify_seconds=0`` so the timing
    covers retrieval only and not the temporal de-duplication a caller may or
    may not want.

    With ``cold=True`` every artifact in ``evict_paths`` is evicted before each
    query, so each measurement pays the page-in cost from scratch.
    """
    times: List[float] = []
    if not cold:
        for gid in list(query_gids)[:warmup]:
            engine.search(
                np.asarray(engine.memmap[int(gid)], dtype=np.float32),
                k=k, nprobe=nprobe, prefilter=prefilter,
                diversify_seconds=diversify_seconds,
            )
    for gid in query_gids:
        if cold:
            for p in evict_paths:
                evict(p)
        vec = np.asarray(engine.memmap[int(gid)], dtype=np.float32)
        t0 = time.perf_counter()
        engine.search(vec, k=k, nprobe=nprobe, prefilter=prefilter,
                      diversify_seconds=diversify_seconds)
        times.append(time.perf_counter() - t0)
    out = _percentiles(times)
    out.update({"cold": bool(cold), "k": k, "nprobe": nprobe, "prefilter": prefilter,
                "diversify_seconds": int(diversify_seconds)})
    out["queries_per_s"] = 1.0 / out["median_s"] if out["median_s"] > 0 else float("inf")
    return out


def measure_throughput(
    engine: SearchEngine,
    query_gids: Sequence[int],
    *,
    k: int = 10,
    nprobe: int = DEFAULT_NPROBE,
    prefilter: int = DEFAULT_PREFILTER,
) -> dict:
    """Sustained warm throughput: queries issued back to back, cache hot.

    Single-threaded at the Python level; FAISS still uses its own OMP pool
    inside each call, so this is "one client, all cores", not "one core".
    """
    for gid in list(query_gids)[:3]:
        engine.search(
            np.asarray(engine.memmap[int(gid)], dtype=np.float32),
            k=k, nprobe=nprobe, prefilter=prefilter, diversify_seconds=0,
        )
    t0 = time.perf_counter()
    for gid in query_gids:
        engine.search(
            np.asarray(engine.memmap[int(gid)], dtype=np.float32),
            k=k, nprobe=nprobe, prefilter=prefilter, diversify_seconds=0,
        )
    dt = time.perf_counter() - t0
    return {
        "n_queries": len(query_gids),
        "elapsed_s": dt,
        "queries_per_s": len(query_gids) / dt,
        "k": k, "nprobe": nprobe, "prefilter": prefilter,
    }


def measure_stage_breakdown(
    engine: SearchEngine,
    query_gids: Sequence[int],
    *,
    nprobe: int = DEFAULT_NPROBE,
    prefilter: int = DEFAULT_PREFILTER,
    warmup: int = 5,
) -> dict:
    """Split query latency into the index probe and the exact rerank.

    These two halves scale with completely different things — the probe with
    ``nprobe`` and the compressed index's size, the rerank with ``prefilter``
    and the *storage* holding the raw vectors — so a single latency figure hides
    which one a deployment is actually paying for. Reporting the split is what
    turns "queries take 400 ms" into an actionable statement.

    Also reports how contiguous the rerank candidates are: because ``global_id``
    is assigned in time order and the IVF cells correlate with time, the
    ``prefilter`` candidates tend to fall in a few runs rather than scattering
    across the whole file. That locality is the difference between a handful of
    seeks and ``prefilter`` of them.
    """
    probe_t: List[float] = []
    rerank_t: List[float] = []
    runs: List[int] = []
    span: List[float] = []

    gids = list(query_gids)
    for gid in gids[:warmup]:
        q = np.ascontiguousarray(
            np.asarray(engine.memmap[int(gid)], dtype=np.float32).reshape(1, -1)
        )
        faiss.normalize_L2(q)
        engine._ivf.nprobe = int(nprobe)
        engine.index.search(q, prefilter)

    for gid in gids:
        q = np.ascontiguousarray(
            np.asarray(engine.memmap[int(gid)], dtype=np.float32).reshape(1, -1)
        )
        faiss.normalize_L2(q)
        engine._ivf.nprobe = int(nprobe)

        t0 = time.perf_counter()
        _scores, ids = engine.index.search(q, prefilter)
        t1 = time.perf_counter()

        cand = np.sort(ids[0][ids[0] >= 0].astype(np.int64))
        engine._exact_scores(cand, q)
        t2 = time.perf_counter()

        probe_t.append(t1 - t0)
        rerank_t.append(t2 - t1)
        if cand.size:
            # Number of contiguous id runs, and how much of the file they span.
            runs.append(int(1 + np.count_nonzero(np.diff(cand) > 1)))
            span.append(float(cand[-1] - cand[0]) / max(engine.total_n, 1))

    probe, rerank = _percentiles(probe_t), _percentiles(rerank_t)
    total = probe["median_s"] + rerank["median_s"]
    return {
        "n": len(gids),
        "nprobe": nprobe,
        "prefilter": prefilter,
        "index_probe": probe,
        "exact_rerank": rerank,
        "rerank_share_of_median": rerank["median_s"] / total if total else float("nan"),
        "candidate_contiguous_runs_median": float(np.median(runs)) if runs else float("nan"),
        "candidate_id_span_fraction_median": float(np.median(span)) if span else float("nan"),
    }


def measure_operating_points(
    engine: SearchEngine,
    query_gids: Sequence[int],
    *,
    k: int = 10,
    points: Sequence[Tuple[int, int]] = DEFAULT_OPERATING_POINTS,
    diversify_settings: Sequence[int] = (0, 30),
    n_cold: int = 10,
    evict_paths: Sequence[Union[str, Path]] = (),
) -> List[dict]:
    """Latency at each named (nprobe, prefilter), warm and cold, per diversify setting.

    Diversification is varied because it changes what is being measured, not
    just how long it takes: recall must be taken at ``diversify_seconds=0`` or
    it scores the de-duplicator rather than the retrieval, while a user at
    library defaults gets 30. Reporting both keeps the latency table honest
    about which one the recall table corresponds to.
    """
    rows = []
    for nprobe, prefilter in points:
        for div in diversify_settings:
            entry = {"nprobe": nprobe, "prefilter": prefilter, "diversify_seconds": div}
            entry["warm"] = measure_latency(
                engine, query_gids, k=k, nprobe=nprobe, prefilter=prefilter,
                cold=False, diversify_seconds=div,
            )
            if n_cold > 0:
                entry["cold"] = measure_latency(
                    engine, query_gids[:n_cold], k=k, nprobe=nprobe,
                    prefilter=prefilter, cold=True, evict_paths=evict_paths,
                    diversify_seconds=div,
                )
                # Paired warm run over the identical query set, so the
                # cold/warm ratio is not confounded by query difficulty.
                entry["warm_paired"] = measure_latency(
                    engine, query_gids[:n_cold], k=k, nprobe=nprobe,
                    prefilter=prefilter, cold=False, diversify_seconds=div,
                )
            rows.append(entry)
    return rows


def ann_topk(
    engine: SearchEngine,
    query_gids: Sequence[int],
    *,
    k: int = 10,
    nprobe: int = DEFAULT_NPROBE,
    prefilter: int = DEFAULT_PREFILTER,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k global ids and cosine scores the deployed pipeline returns."""
    ids = np.full((len(query_gids), k), -1, dtype=np.int64)
    scores = np.full((len(query_gids), k), -np.inf, dtype=np.float32)
    for i, gid in enumerate(query_gids):
        hits = engine.search(
            np.asarray(engine.memmap[int(gid)], dtype=np.float32),
            k=k, nprobe=nprobe, prefilter=prefilter, diversify_seconds=0,
        )
        for j, h in enumerate(hits[:k]):
            ids[i, j] = h.global_id
            scores[i, j] = h.score
    return ids, scores


# --------------------------------------------------------------------------- #
# brute-force baseline
# --------------------------------------------------------------------------- #
def brute_force_topk(
    memmap: np.memmap,
    queries: np.ndarray,
    *,
    k: int = 10,
    chunk_rows: int = BRUTE_CHUNK_ROWS,
    on_progress: Optional[Callable[[int, int, float], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact top-k by cosine over every row, in one sequential pass.

    ``queries`` must already be L2-normalized fp32, shape ``(Q, 512)``. Rows are
    cast to fp32 and normalized per chunk, exactly as
    ``SearchEngine._exact_scores`` does, so the scores are directly comparable.

    One pass scores *all* queries, which is both the efficient way to build
    ground truth and the fair baseline for a batch workload. The cost of a
    single isolated brute-force query is this same full pass.

    Returns ``(ids, scores, elapsed_s)`` with ids/scores sorted best-first.
    """
    n = memmap.shape[0]
    # Release each chunk's pages once it has been scored. Without this a 1 TB
    # sweep grows the resident set until it fills RAM, evicting everything else
    # on the machine and leaving `free` at zero -- which naive memory monitors
    # treat as an emergency even though the pages are reclaimable.
    #
    # It has to be madvise, not posix_fadvise. fadvise acts on the page cache
    # and deliberately leaves alone any page that is mapped into a live address
    # space, so against an active np.memmap it is very nearly a no-op (measured:
    # it released ~1 GB of a 116 GB resident set). MADV_DONTNEED drops the pages
    # from the mapping itself; because this is a read-only MAP_SHARED file
    # mapping, a later access simply re-reads from disk. The sweep is a single
    # forward pass, so nothing is ever read twice and this costs nothing.
    # Both calls are required, in this order, and each is insufficient alone.
    # madvise unmaps the pages from this process (resident set stays flat), but
    # leaves them clean in the page cache; fadvise drops them from the page
    # cache, but silently skips any page that is still mapped. Measured on a
    # 1 TB sweep: madvise alone took RSS from 119 GB to 4.5 GB while the page
    # cache still held 109 GiB; adding fadvise afterwards released that too,
    # taking free memory from 6 GiB to 104 GiB.
    _mm = getattr(memmap, "_mmap", None)
    try:
        _fd: Optional[int] = os.open(memmap.filename, os.O_RDONLY)
    except Exception:
        _fd = None
    row_bytes = memmap.shape[1] * memmap.dtype.itemsize

    q = np.ascontiguousarray(queries, dtype=np.float32)
    nq = q.shape[0]

    best_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    best_ids = np.full((nq, k), -1, dtype=np.int64)

    t0 = time.perf_counter()
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        block = np.ascontiguousarray(np.asarray(memmap[start:end], dtype=np.float32))
        faiss.normalize_L2(block)
        # `q @ block.T` rather than `(block @ qT).T`. Both compute the same
        # numbers, but the transposed form yields an F-contiguous (nq, m) array,
        # and np.argpartition along axis=1 over F-contiguous memory strides
        # across the whole buffer on every comparison. Measured at nq=500,
        # m=1e6: 8.49 s/chunk transposed versus 3.79 s contiguous -- 1.5 hours
        # over a full pass, for an arithmetically identical result.
        sims = q @ block.T                         # (nq, m), C-contiguous
        del block
        m = sims.shape[1]
        take = min(k, m)
        # Partition for the k LARGEST directly instead of negating: `-sims`
        # would materialise a second full copy of the similarity matrix (2 GB
        # at this chunk size) purely to flip a comparison.
        part = np.argpartition(sims, m - take, axis=1)[:, m - take:]
        cand_scores = np.take_along_axis(sims, part, axis=1)
        cand_ids = part.astype(np.int64) + start
        del sims

        merged_scores = np.concatenate([best_scores, cand_scores], axis=1)
        merged_ids = np.concatenate([best_ids, cand_ids], axis=1)
        width = merged_scores.shape[1]
        keep = np.argpartition(merged_scores, width - k, axis=1)[:, width - k:]
        best_scores = np.take_along_axis(merged_scores, keep, axis=1)
        best_ids = np.take_along_axis(merged_ids, keep, axis=1)

        if _mm is not None:
            # madvise needs a page-aligned offset; round the start down and
            # extend the length to compensate.
            off = (start * row_bytes) // mmap.PAGESIZE * mmap.PAGESIZE
            try:
                _mm.madvise(mmap.MADV_DONTNEED, off, end * row_bytes - off)
            except (OSError, ValueError):
                pass
            if _fd is not None:
                try:
                    os.posix_fadvise(
                        _fd, off, end * row_bytes - off, os.POSIX_FADV_DONTNEED
                    )
                except OSError:
                    pass

        if on_progress is not None:
            on_progress(end, n, time.perf_counter() - t0)

    if _fd is not None:
        os.close(_fd)
    order = np.argsort(-best_scores, axis=1)
    elapsed = time.perf_counter() - t0
    return (
        np.take_along_axis(best_ids, order, axis=1),
        np.take_along_axis(best_scores, order, axis=1),
        elapsed,
    )


def recall_at_k(ann_ids: np.ndarray, gt_ids: np.ndarray) -> dict:
    """Fraction of the exact top-k that the pipeline also returned."""
    per_query = [
        len(set(a.tolist()) & set(g.tolist())) / len(g)
        for a, g in zip(ann_ids, gt_ids)
    ]
    a = np.asarray(per_query, dtype=np.float64)
    return {
        "k": int(gt_ids.shape[1]),
        "n_queries": int(a.size),
        "recall_at_k": float(a.mean()),
        "recall_p10": float(np.percentile(a, 10)),
        "recall_min": float(a.min()),
        "exact_top1_hit_rate": float(np.mean(ann_ids[:, 0] == gt_ids[:, 0])),
    }


def hyperparameters(engine: SearchEngine, *, k: int, nprobe: int, prefilter: int) -> dict:
    """Every knob that affects the numbers, recorded so they can be reproduced."""
    from themisim.index import (
        ADD_CHUNK_SIZE,
        DEFAULT_TRAIN_TARGET,
        TRAIN_POINTS_PER_CENTROID,
        _build_factory_string,
    )

    ivf = engine._ivf
    nlist = int(ivf.nlist)
    return {
        "index": {
            "factory": _build_factory_string(nlist),
            "metric": "INNER_PRODUCT on L2-normalized vectors (cosine)",
            "nlist": nlist,
            "n_vectors": int(engine.total_n),
            "dim": N_FEATURES,
            "stored_dtype": "float16",
            "pq_code_bytes_per_vector": 64,
            "is_trained": bool(engine.index.is_trained),
            "loaded_with": "faiss.IO_FLAG_MMAP",
        },
        "query": {
            "k": k,
            "nprobe": nprobe,
            "prefilter": prefilter,
            "fraction_of_cells_probed": min(nprobe, nlist) / nlist,
            "diversify_seconds": 0,
            "diversify_seconds_default": DEFAULT_DIVERSIFY_SECONDS,
            "rerank": "exact fp32 cosine over the prefilter candidates",
            "rerank_chunk_rows": _RERANK_CHUNK,
            "brute_force_max": DEFAULT_BRUTE_FORCE_MAX,
            "defaults_in_library": {
                "nprobe": DEFAULT_NPROBE, "prefilter": DEFAULT_PREFILTER,
            },
        },
        "build": {
            "train_target_default": DEFAULT_TRAIN_TARGET,
            "train_points_per_centroid_floor": TRAIN_POINTS_PER_CENTROID,
            "train_target_used": max(DEFAULT_TRAIN_TARGET, TRAIN_POINTS_PER_CENTROID * nlist),
            "add_chunk_size": ADD_CHUNK_SIZE,
            "training_sample": "stratified by (site, year_month), seed=0",
        },
        "baseline": {
            "method": "exact cosine over all rows, one sequential pass",
            "chunk_rows": BRUTE_CHUNK_ROWS,
            "normalization": "fp16 -> fp32, L2-normalized per chunk (matches _exact_scores)",
        },
        "sampling": {"query_selection": "uniform without replacement over global_id", "seed": 0},
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def artifact_paths(artifacts: Union[str, Path]) -> List[Path]:
    """Files whose page cache determines cold-start cost."""
    artifacts = Path(artifacts)
    paths = [artifacts / "index.faiss", artifacts / "vectors.f16.dat"]
    cache = artifacts / "manifest.parquet.arrays.cache"
    if cache.is_dir():
        paths += sorted(cache.glob("*.npy"))
    return [p for p in paths if Path(os.path.realpath(p)).exists()]


def run_benchmark(
    artifacts: Union[str, Path],
    *,
    k: int = 10,
    nprobe: int = DEFAULT_NPROBE,
    prefilter: int = DEFAULT_PREFILTER,
    n_latency: int = 50,
    n_cold: int = 10,
    n_recall: int = 50,
    brute_force: bool = True,
    chunk_rows: int = BRUTE_CHUNK_ROWS,
    sweep: bool = True,
    sweep_nprobe: Sequence[int] = DEFAULT_NPROBE_SWEEP,
    sweep_prefilter: Sequence[int] = DEFAULT_PREFILTER_SWEEP,
    operating_points: Sequence[Tuple[int, int]] = DEFAULT_OPERATING_POINTS,
    diversify_settings: Sequence[int] = (0, 30),
    ground_truth_path: Optional[Union[str, Path]] = None,
    seed: int = 0,
    on_progress: Optional[Callable[[str, object], None]] = None,
) -> dict:
    """Full report: environment, hyperparameters, latency, throughput, recall.

    The brute-force pass is the expensive part — one sequential sweep of the
    whole memmap — so it runs last, after every measurement that would be
    distorted by saturating the same device.
    """
    artifacts = Path(artifacts)

    def say(stage: str, info: object) -> None:
        if on_progress is not None:
            on_progress(stage, info)

    report: Dict[str, object] = {
        "artifacts": str(artifacts.resolve()),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": hardware_report(),
    }
    say("hardware", report["hardware"]["cpu_model"])

    report["storage"] = storage_report(artifacts)
    say("storage", {n: v.get("media") for n, v in report["storage"].items()})

    # Engine load, from a cold cache: the process-start cost a server pays once.
    # It is deceptively small on a large index because the manifest columns are
    # opened with np.load(mmap_mode="r") rather than read (search.py:234), so
    # most of that ~18 GB is paged in lazily *during* later queries. Read this
    # number as "time to first query", not as "everything is now in memory".
    for p in artifact_paths(artifacts):
        evict(p)
    t0 = time.perf_counter()
    engine = SearchEngine(artifacts)
    report["engine_load_s_cold"] = time.perf_counter() - t0
    say("engine_loaded", f"{engine.total_n:,} vectors in {report['engine_load_s_cold']:.1f}s")

    report["hyperparameters"] = hyperparameters(engine, k=k, nprobe=nprobe, prefilter=prefilter)

    rng = np.random.default_rng(seed)
    n_pick = max(n_latency, n_recall, n_cold)
    gids = rng.choice(engine.total_n, size=min(n_pick, engine.total_n), replace=False)
    gids = np.sort(gids)

    # Cold first, deliberately. Running warm first leaves the interpreter,
    # allocator and CPU caches hot, and the "cold" pass that follows then
    # measures a half-warm process -- on the pilot index that inverted the
    # result and made cold look faster than warm.
    if n_cold > 0:
        cold_set = gids[:n_cold]
        report["latency_cold"] = measure_latency(
            engine, cold_set, k=k, nprobe=nprobe, prefilter=prefilter,
            cold=True, evict_paths=artifact_paths(artifacts),
        )
        say("latency_cold", f"median {report['latency_cold']['median_s'] * 1000:.1f} ms")

        # The cold/warm comparison must use the SAME queries. Comparing a
        # 10-query cold sample against a 50-query warm sample compares query
        # sets as much as cache states -- individual queries differ several-fold
        # in cost depending on how their rerank candidates are laid out on disk.
        report["latency_warm_same_queries"] = measure_latency(
            engine, cold_set, k=k, nprobe=nprobe, prefilter=prefilter, cold=False
        )
        w = report["latency_warm_same_queries"]["median_s"]
        say("latency_warm_same", f"median {w * 1000:.1f} ms (same {n_cold} queries as cold)")

    # A wider warm sample, for percentiles the small paired set cannot support.
    report["latency_warm"] = measure_latency(
        engine, gids[:n_latency], k=k, nprobe=nprobe, prefilter=prefilter, cold=False
    )
    say("latency_warm", f"median {report['latency_warm']['median_s'] * 1000:.1f} ms")

    if operating_points:
        report["operating_points"] = measure_operating_points(
            engine, gids[:n_latency], k=k, points=operating_points,
            diversify_settings=diversify_settings, n_cold=n_cold,
            evict_paths=artifact_paths(artifacts),
        )
        for row in report["operating_points"]:
            say("operating_point",
                f"nprobe={row['nprobe']} prefilter={row['prefilter']} "
                f"div={row['diversify_seconds']}: warm median "
                f"{row['warm']['median_s'] * 1000:.0f} ms"
                + (f", cold {row['cold']['median_s'] * 1000:.0f} ms" if "cold" in row else ""))

    report["stage_breakdown"] = measure_stage_breakdown(
        engine, gids[:n_latency], nprobe=nprobe, prefilter=prefilter
    )
    sb = report["stage_breakdown"]
    say("breakdown", f"probe {sb['index_probe']['median_s'] * 1000:.1f} ms + "
                     f"rerank {sb['exact_rerank']['median_s'] * 1000:.1f} ms "
                     f"({sb['rerank_share_of_median']:.0%} rerank)")

    report["throughput_warm"] = measure_throughput(
        engine, gids[:n_latency], k=k, nprobe=nprobe, prefilter=prefilter
    )
    say("throughput", f"{report['throughput_warm']['queries_per_s']:.2f} q/s")

    if not brute_force:
        return report

    recall_gids = gids[:n_recall]
    ann, ann_scores = ann_topk(engine, recall_gids, k=k, nprobe=nprobe, prefilter=prefilter)
    queries = np.ascontiguousarray(
        np.asarray(engine.memmap[recall_gids], dtype=np.float32)
    )
    faiss.normalize_L2(queries)

    last = [0.0]

    def progress(done: int, total: int, elapsed: float) -> None:
        if elapsed - last[0] >= 120 or done == total:
            last[0] = elapsed
            rate = done / elapsed if elapsed else 0
            eta = (total - done) / rate if rate else float("nan")
            say("brute_force", f"{done / total:.1%} ({elapsed / 60:.1f} min elapsed, "
                              f"{eta / 60:.1f} min left)")

    gt_ids, gt_scores, elapsed = brute_force_topk(
        engine.memmap, queries, k=k, chunk_rows=chunk_rows, on_progress=progress
    )
    report["brute_force"] = {
        "n_queries_in_pass": int(len(recall_gids)),
        "n_vectors_scanned": int(engine.total_n),
        "bytes_read": int(engine.total_n) * N_FEATURES * 2,
        "wall_clock_s_one_pass": elapsed,
        "seconds_per_query_batched": elapsed / len(recall_gids),
        "seconds_per_query_single": elapsed,
        "queries_per_s_batched": len(recall_gids) / elapsed,
        "effective_scan_bytes_per_s": int(engine.total_n) * N_FEATURES * 2 / elapsed,
    }
    say("brute_force_done", f"{elapsed / 60:.1f} min for one pass")

    # Ground truth costs a full scan of the database, so it is kept rather than
    # discarded: it can score any number of operating points afterwards for the
    # price of the ANN queries alone, and a later run can reuse it outright.
    if ground_truth_path is not None:
        np.savez_compressed(
            str(ground_truth_path),
            query_gids=np.asarray(recall_gids, dtype=np.int64),
            gt_ids=gt_ids, gt_scores=gt_scores,
        )
        report["ground_truth_path"] = str(ground_truth_path)
        say("ground_truth", f"saved to {ground_truth_path}")

    report["accuracy"] = recall_at_k(ann, gt_ids)
    # How much similarity is actually lost when the pipeline misses: the gap
    # between the exact best score and the best score it did return. Recall
    # alone cannot say whether a miss cost 0.0001 or 0.3 of cosine similarity.
    report["accuracy"]["mean_top1_score_gap"] = float(
        np.mean(gt_scores[:, 0] - ann_scores[:, 0])
    )
    report["accuracy"]["max_top1_score_gap"] = float(
        np.max(gt_scores[:, 0] - ann_scores[:, 0])
    )

    # The recall/latency trade-off curve. Each extra point costs only its ANN
    # queries -- the expensive exact ground truth is already paid for -- so the
    # whole curve is nearly free once the scan is done.
    if sweep:
        nlist = int(engine._ivf.nlist)
        grid = []
        for probe_n in sweep_nprobe:
            for pf in sweep_prefilter:
                t0 = time.perf_counter()
                ids, _sc = ann_topk(engine, recall_gids, k=k, nprobe=probe_n, prefilter=pf)
                dt = (time.perf_counter() - t0) / len(recall_gids)
                r = recall_at_k(ids, gt_ids)
                grid.append({
                    "nprobe": probe_n,
                    "prefilter": pf,
                    "fraction_of_cells_probed": min(probe_n, nlist) / nlist,
                    "recall_at_k": r["recall_at_k"],
                    "exact_top1_hit_rate": r["exact_top1_hit_rate"],
                    "mean_latency_s": dt,
                })
                say("sweep", f"nprobe={probe_n} prefilter={pf} -> "
                             f"recall@{k}={r['recall_at_k']:.4f} at {dt * 1000:.0f} ms")
        report["operating_point_sweep"] = grid

    warm = report["latency_warm"]["median_s"]
    report["speedup_vs_brute_force"] = {
        "single_query_warm": elapsed / warm,
        "single_query_cold": (
            elapsed / report["latency_cold"]["median_s"] if n_cold > 0 else None
        ),
        "batched_throughput": (
            report["throughput_warm"]["queries_per_s"]
            / report["brute_force"]["queries_per_s_batched"]
        ),
    }
    return report


def _fmt_x(factor: Optional[float]) -> str:
    """Speedup factor. Keeps decimals below 10x so a sub-1x result stays legible
    -- an index that is *slower* than a linear scan is a finding, not a rounding
    error, and ">12,.0f" renders it as a confusing "0x"."""
    if factor is None:
        return "n/a"
    if factor < 10:
        return f"{factor:>12,.2f}x"
    return f"{factor:>12,.0f}x"


def _fmt_ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def format_benchmark(report: dict) -> str:
    """Human-readable rendering of :func:`run_benchmark`'s output."""
    hw, hp = report["hardware"], report["hyperparameters"]
    idx, qry = hp["index"], hp["query"]
    lines = [
        "=" * 78,
        f"THEMISim index benchmark — {report['generated_utc']}",
        "=" * 78,
        "",
        "Hardware",
        f"  CPU            {hw['cpu_model']} ({hw['cpu_logical']} logical)",
        f"  RAM            {(hw['ram_total_bytes'] or 0) / 2**30:.0f} GiB",
        f"  FAISS          {hw['faiss']}, {hw['faiss_omp_threads']} OMP threads"
        f" ({hw.get('faiss_num_gpus', 0)} GPUs present, unused — query path is CPU-only)",
        "",
        "Storage",
    ]
    for name, s in report["storage"].items():
        rate = s.get("cold_sequential_read_bytes_per_s")
        lines.append(
            f"  {name:<18} {s['size_bytes'] / 1e9:8.2f} GB  {s.get('media', '?'):<3}"
            f"  {s.get('device', '?'):<12}"
            + (f"  {rate / 1e6:6.0f} MB/s cold seq" if rate else "")
        )
    lines += [
        "",
        "Index",
        f"  factory        {idx['factory']}",
        f"  metric         {idx['metric']}",
        f"  vectors        {idx['n_vectors']:,} x {idx['dim']}-D {idx['stored_dtype']}"
        f"  ({idx['pq_code_bytes_per_vector']} B/vector PQ codes)",
        f"  nlist          {idx['nlist']:,}",
        "",
        "Operating point",
        f"  k={qry['k']}  nprobe={qry['nprobe']}  prefilter={qry['prefilter']}"
        f"  diversify_seconds={qry['diversify_seconds']}",
        f"  cells probed   {qry['fraction_of_cells_probed']:.4%} of {idx['nlist']:,}",
        f"  rerank         {qry['rerank']}",
        "",
        f"Engine load (cold)  {report['engine_load_s_cold']:.1f} s  — one-time, not per query",
        "",
        "Latency (full pipeline: index probe + exact fp32 rerank)",
        f"  {'':<8} {'n':>4} {'median':>10} {'p95':>10} {'p99':>10} {'max':>10}",
    ]
    for label, key in (
        ("cold", "latency_cold"),
        ("warm*", "latency_warm_same_queries"),
        ("warm", "latency_warm"),
    ):
        if key not in report:
            continue
        m = report[key]
        lines.append(
            f"  {label:<8} {m['n']:>4} {_fmt_ms(m['median_s']):>10} {_fmt_ms(m['p95_s']):>10}"
            f" {_fmt_ms(m['p99_s']):>10} {_fmt_ms(m['max_s']):>10}"
        )
    if "latency_warm_same_queries" in report:
        lines.append("  * warm* is the same query set as cold — the paired comparison.")
    if "stage_breakdown" in report:
        sb = report["stage_breakdown"]
        lines += [
            "",
            "Where the time goes (warm, median)",
            f"  index probe    {_fmt_ms(sb['index_probe']['median_s']):>10}"
            f"   (p95 {_fmt_ms(sb['index_probe']['p95_s'])})",
            f"  exact rerank   {_fmt_ms(sb['exact_rerank']['median_s']):>10}"
            f"   (p95 {_fmt_ms(sb['exact_rerank']['p95_s'])})"
            f"  = {sb['rerank_share_of_median']:.0%} of the query",
            f"  rerank candidates land in {sb['candidate_contiguous_runs_median']:.0f} contiguous"
            f" id runs spanning {sb['candidate_id_span_fraction_median']:.1%} of the database",
        ]

    t = report["throughput_warm"]
    lines += [
        "",
        f"Throughput (warm, sequential)  {t['queries_per_s']:.2f} queries/s"
        f"  ({t['n_queries']} queries in {t['elapsed_s']:.1f} s)",
    ]

    if "brute_force" in report:
        bf, acc, sp = report["brute_force"], report["accuracy"], report["speedup_vs_brute_force"]
        lines += [
            "",
            "Brute-force baseline (exact cosine over every vector)",
            f"  scanned        {bf['n_vectors_scanned']:,} vectors"
            f"  = {bf['bytes_read'] / 1e12:.3f} TB read",
            f"  one full pass  {bf['wall_clock_s_one_pass'] / 60:.2f} min"
            f"  ({bf['effective_scan_bytes_per_s'] / 1e6:.0f} MB/s effective)",
            f"  single query   {_fmt_ms(bf['seconds_per_query_single'])}"
            f"   (a lone query needs the whole pass)",
            f"  batched        {_fmt_ms(bf['seconds_per_query_batched'])}/query"
            f"  over {bf['n_queries_in_pass']} queries sharing one pass"
            f"  = {bf['queries_per_s_batched']:.3f} q/s",
            "",
            "Speedup of the index over brute force",
            f"  warm single query   {_fmt_x(sp['single_query_warm'])}",
            f"  cold single query   {_fmt_x(sp['single_query_cold'])}",
            f"  batched throughput  {_fmt_x(sp['batched_throughput'])}",
            "",
            f"Accuracy vs that exact baseline (n={acc['n_queries']})",
            f"  recall@{acc['k']}           {acc['recall_at_k']:.4f}",
            f"  recall p10          {acc['recall_p10']:.4f}",
            f"  worst query         {acc['recall_min']:.4f}",
            f"  exact top-1 match   {acc['exact_top1_hit_rate']:.4f}",
            f"  top-1 cosine gap    mean {acc['mean_top1_score_gap']:.6f}"
            f"  worst {acc['max_top1_score_gap']:.6f}",
        ]
    if report.get("operating_point_sweep"):
        k_lbl = "recall@" + str(report["accuracy"]["k"])
        lines += [
            "",
            "Operating-point sweep (same exact ground truth, ANN settings varied)",
            f"  {'nprobe':>7} {'scanned':>9} {'prefilter':>10} {k_lbl:>10}"
            f" {'top-1':>8} {'latency':>10}",
        ]
        for g in report["operating_point_sweep"]:
            lines.append(
                f"  {g['nprobe']:>7} {g['fraction_of_cells_probed']:>8.2%} {g['prefilter']:>10}"
                f" {g['recall_at_k']:>10.4f} {g['exact_top1_hit_rate']:>8.4f}"
                f" {_fmt_ms(g['mean_latency_s']):>10}"
            )
    lines.append("")
    return "\n".join(lines)
