"""Train and build the FAISS index over the memmap of vectors.

Stage 7. Inputs are the artifacts from Stage 6 (`vectors.f16.dat`,
`manifest.parquet`, `vectors.meta.json`). Output is `index.faiss`.

Index design (per plan.md):
    factory: OPQ64_64,IVF{nlist}_HNSW32,PQ64x8
    metric:  inner product on L2-normalized vectors (== cosine)
    nlist:   4096 for the pilot (~6M vectors expected; ~3M actual)
             65536 override for the full archive (~756M)
    storage: ~64 bytes/vector PQ codes + IVF + HNSW coarse quantizer

Training set: stratified sample by (site, year_month), capped at
~500K vectors. Cast fp16 -> fp32 for training and adding.

Public surface:
    stratified_sample(manifest_df, target_n, seed) -> np.ndarray of global_ids
    train_and_build_index(memmap, manifest_df, out_path, nlist) -> faiss.Index
    evaluate_recall(index, queries, gt_pool, gt_pool_ids, k, nprobe) -> dict

CLI:
    python -m themisim.index --nlist 4096
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Union

import faiss
import numpy as np
import pandas as pd

from themisim.concat import N_FEATURES, open_memmap

DEFAULT_NLIST_PILOT = 4096
DEFAULT_NLIST_FULL = 65536
DEFAULT_TRAIN_TARGET = 500_000
DEFAULT_NPROBE = 64
ADD_CHUNK_SIZE = 100_000
# Train the IVF coarse quantizer on at least this many points per centroid.
# FAISS warns below 39x nlist ("please provide at least N training points") and
# an undertrained quantizer gives lopsided cells and poor recall; 64x clears the
# warning with margin (nlist=65536 -> ~4.2M points). The CLI floors train_target
# at TRAIN_POINTS_PER_CENTROID * nlist so the full-archive build can't silently
# undertrain the way the 500K pilot default did.
TRAIN_POINTS_PER_CENTROID = 64
# Sequential block size for the HDD-friendly training gather (rows). 2M rows of
# 512 fp16 == 2 GB per read; the whole 1 TB memmap is swept once in ~500 blocks.
TRAIN_GATHER_CHUNK = 2_000_000


def stratified_sample(
    manifest_df: pd.DataFrame,
    target_n: int = DEFAULT_TRAIN_TARGET,
    seed: int = 0,
) -> np.ndarray:
    """Sample global_ids stratified by (site, year_month).

    Each (site, year_month) bucket contributes proportional to its size up
    to `per_bucket = target_n / n_buckets`, capped at the bucket's actual
    population. Returns a 1-D np.int64 array of selected global_ids.
    """
    rng = np.random.default_rng(seed)
    gid = np.asarray(manifest_df["global_id"], dtype=np.int64)
    time_ns = np.asarray(manifest_df["time_ns"], dtype=np.int64)
    # Vectorized (site, year-month) bucket key WITHOUT a per-row strftime — at
    # full-archive scale (~1B rows) strftime materializes a billion Python
    # strings and OOMs. "Months since epoch" is a compact monotone int64, and
    # site collapses to a small int code, so the bucket key is pure numpy.
    ym = time_ns.astype("datetime64[ns]").astype("datetime64[M]").view(np.int64)
    site_codes = pd.factorize(manifest_df["site"], sort=True)[0].astype(np.int32)
    key = pd.DataFrame({"gid": gid, "site": site_codes, "ym": ym.astype(np.int32)})

    grouped = key.groupby(["site", "ym"], sort=True)
    per_bucket = max(1, target_n // grouped.ngroups)
    sampled: list[np.ndarray] = []
    for _key, group in grouped:
        ids = group["gid"].to_numpy()
        n_take = min(per_bucket, len(ids))
        sampled.append(rng.choice(ids, size=n_take, replace=False))
    return np.concatenate(sampled).astype(np.int64)


# Bucket key = site_code * _BUCKET_MULT + months-since-epoch. THEMIS timestamps
# are all post-2004 so months-since-epoch is a small positive int (< ~700);
# _BUCKET_MULT just has to exceed it to keep (site, month) pairs collision-free.
_BUCKET_MULT = 1_000_000


def _group_bucket_keys(tbl, site_to_code: dict) -> np.ndarray:
    """Compute the int64 (site, year_month) bucket key for one parquet row group.

    ``site`` is read as an Arrow ``string`` chunk and dictionary-encoded to a
    small int code against the running ``site_to_code`` table (so the same site
    string maps to the same code across every row group). ``time_ns`` collapses
    to months-since-epoch with pure numpy. Crucially this never materializes the
    ~1B Python ``str`` objects that an honest ``read_parquet`` of the ``site``
    column would, and never holds more than one row group at a time.

    NB: ``tbl.column(...)`` returns a pyarrow ``ChunkedArray``, whose
    ``to_numpy()`` takes NO keyword arguments (a 2026-06-13 crash was exactly
    ``ChunkedArray.to_numpy(zero_copy_only=False)`` raising ``TypeError``).
    ``combine_chunks()`` first yields a contiguous ``Array`` that does.
    """
    tn = tbl.column("time_ns").combine_chunks().to_numpy(zero_copy_only=False)
    ym = tn.astype("datetime64[ns]").astype("datetime64[M]").view(np.int64)
    enc = tbl.column("site").combine_chunks().dictionary_encode()
    local_dict = enc.dictionary.to_pylist()
    remap = np.array(
        [site_to_code.setdefault(s, len(site_to_code)) for s in local_dict],
        dtype=np.int64,
    )
    site_codes = remap[enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)]
    return site_codes * _BUCKET_MULT + ym


def stratified_sample_from_parquet(
    manifest_path: Union[str, Path],
    target_n: int = DEFAULT_TRAIN_TARGET,
    seed: int = 0,
    total_n_check: "int | None" = None,
) -> np.ndarray:
    """Streaming equivalent of :func:`stratified_sample` for a huge manifest.

    Same selection semantics — each (site, year_month) bucket contributes
    ``min(per_bucket, bucket_population)`` ids where ``per_bucket = target_n //
    n_buckets`` — but it never builds the 1.009B-row DataFrame, never runs a
    pandas ``groupby`` over a billion rows, and peaks at a few GB instead of the
    ~120 GB that OOM-killed the whole user session. Returns a sorted int64 array.

    Two passes over the parquet (cheap relative to the multi-hour build):

    1. Count rows per bucket. Bucket cardinality is tiny (~sites x months,
       a few thousand) so the counts live in a plain dict.
    2. Bottom-k reservoir per bucket: give every row an iid uniform key and keep
       the ``cap`` rows with the smallest keys per bucket. Bottom-k of iid keys
       is an exact uniform sample without replacement, and it merges across row
       groups order-independently, so the result is chunking-invariant.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(manifest_path))
    site_to_code: dict[str, int] = {}

    # Pass 1: bucket populations.
    counts: dict[int, int] = {}
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["site", "time_ns"])
        buckets, pops = np.unique(_group_bucket_keys(tbl, site_to_code), return_counts=True)
        for b, c in zip(buckets.tolist(), pops.tolist()):
            counts[b] = counts.get(b, 0) + c
        del tbl
    if total_n_check is not None:
        seen = sum(counts.values())
        assert seen == total_n_check, f"manifest rows {seen} != memmap rows {total_n_check}"
    per_bucket = max(1, target_n // len(counts))

    # Pass 2: per-bucket bottom-k reservoir over the random keys.
    rng = np.random.default_rng(seed)
    res_keys: dict[int, np.ndarray] = {}
    res_gids: dict[int, np.ndarray] = {}
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["global_id", "site", "time_ns"])
        gid = tbl.column("global_id").combine_chunks().to_numpy(zero_copy_only=False)
        gid = gid.astype(np.int64, copy=False)
        bucket = _group_bucket_keys(tbl, site_to_code)
        keys = rng.random(len(gid))
        # Sort this group's rows by bucket so each bucket is a contiguous slice.
        order = np.argsort(bucket, kind="stable")
        bsorted = bucket[order]
        edges = np.flatnonzero(np.r_[True, bsorted[1:] != bsorted[:-1], True])
        for i in range(len(edges) - 1):
            s, e = int(edges[i]), int(edges[i + 1])
            b = int(bsorted[s])
            idx = order[s:e]
            cap = min(per_bucket, counts[b])
            nk, ng = keys[idx], gid[idx]
            if b in res_keys:
                nk = np.concatenate([res_keys[b], nk])
                ng = np.concatenate([res_gids[b], ng])
            if len(ng) > cap:
                keep = np.argpartition(nk, cap)[:cap]
                nk, ng = nk[keep], ng[keep]
            res_keys[b], res_gids[b] = nk, ng
        del tbl
    return np.sort(np.concatenate(list(res_gids.values())).astype(np.int64))


def _build_factory_string(nlist: int) -> str:
    return f"OPQ64_64,IVF{nlist}_HNSW32,PQ64x8"


def _normalize_inplace(arr: np.ndarray) -> None:
    """L2-normalize rows in place. faiss.normalize_L2 requires fp32 contiguous."""
    faiss.normalize_L2(arr)


def _gather_train_rows(
    memmap: np.memmap,
    ids: np.ndarray,
    *,
    chunk_rows: int = TRAIN_GATHER_CHUNK,
) -> np.ndarray:
    """Gather ``memmap[ids]`` as fp32 via sequential block sweeps (HDD-friendly).

    A fancy-index gather (``memmap[ids]``) on the 1 TB HDD-backed memmap issues
    one seek per id — ~4 ms each on the spinning archive drive, i.e. ~5 hours for
    a multi-million-row training sample. Since the training ids span the whole
    file, we instead sweep it in large contiguous blocks and pluck the wanted
    rows out of each block, so the disk reads stay sequential and the file is
    touched exactly once (~1-1.5 h). Returns rows in ascending-id order, which is
    all that matters: training (k-means / OPQ / PQ) is order-agnostic.
    """
    ids_sorted = np.sort(np.asarray(ids, dtype=np.int64))
    out = np.empty((len(ids_sorted), memmap.shape[1]), dtype=np.float32)
    n = memmap.shape[0]
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        lo = int(np.searchsorted(ids_sorted, start, side="left"))
        hi = int(np.searchsorted(ids_sorted, end, side="left"))
        if hi <= lo:
            continue
        block = np.asarray(memmap[start:end])
        out[lo:hi] = block[ids_sorted[lo:hi] - start].astype(np.float32)
    return out


def train_and_build_index(
    memmap: np.memmap,
    manifest_df: "pd.DataFrame | None" = None,
    out_path: Union[str, Path, None] = None,
    *,
    nlist: int = DEFAULT_NLIST_PILOT,
    train_target: int = DEFAULT_TRAIN_TARGET,
    add_chunk_size: int = ADD_CHUNK_SIZE,
    seed: int = 0,
    train_ids: "np.ndarray | None" = None,
    use_gpu_quantizer: bool = True,
    on_progress=None,
) -> faiss.Index:
    """Train OPQ-IVF-PQ on a stratified sample and add ALL memmap rows to the index.

    The index is L2-normalized inner product (== cosine). Vectors are cast
    fp16 -> fp32 only at training and at chunked add time; no fp32 copy of
    the whole memmap is held.

    ``train_ids`` may be supplied precomputed (so the caller can free the
    multi-GB manifest before the long add phase); otherwise it is derived from
    ``manifest_df``. When ``use_gpu_quantizer`` and GPUs are present, the IVF
    coarse-quantizer k-means runs on GPU (the OPQ/PQ training and the add stay
    on CPU) — at nlist=65536 that turns a multi-hour CPU clustering into minutes.
    """
    if out_path is None:
        raise ValueError("out_path is required")
    out_path = Path(out_path)
    n = memmap.shape[0]

    # Training set. _gather_train_rows reads the sample with sequential block
    # sweeps over the memmap (the HDD-friendly alternative to per-id seeks).
    if train_ids is None:
        if manifest_df is None:
            raise ValueError("provide either manifest_df or train_ids")
        train_ids = stratified_sample(manifest_df, target_n=train_target, seed=seed)
    train_ids = np.sort(np.asarray(train_ids, dtype=np.int64))
    train_vecs = _gather_train_rows(memmap, train_ids)
    _normalize_inplace(train_vecs)
    if on_progress:
        on_progress("train_sample", len(train_ids))

    factory = _build_factory_string(nlist)
    index = faiss.index_factory(N_FEATURES, factory, faiss.METRIC_INNER_PRODUCT)

    if use_gpu_quantizer and faiss.get_num_gpus() > 0:
        # Offload only the coarse-quantizer k-means assignment to GPU(s). The
        # IVF operates in the OPQ-transformed space, so the clustering index
        # dimension is the inner IVF's d (post-OPQ), not N_FEATURES.
        try:
            ivf = faiss.extract_index_ivf(index)
            ivf.clustering_index = faiss.index_cpu_to_all_gpus(
                faiss.IndexFlatL2(ivf.d)
            )
            if on_progress:
                on_progress("gpu_quantizer", faiss.get_num_gpus())
        except Exception as exc:  # fall back to CPU clustering
            if on_progress:
                on_progress("gpu_quantizer_skip", repr(exc))

    index.train(train_vecs)
    if on_progress:
        on_progress("trained", factory)
    del train_vecs

    # Add all rows in fp32 chunks. Report roughly every 5M vectors so the log
    # stays readable across the ~10K chunks of a full-archive build.
    report_every = max(1, (5_000_000 // add_chunk_size))
    for i, start in enumerate(range(0, n, add_chunk_size)):
        end = min(start + add_chunk_size, n)
        chunk = np.ascontiguousarray(memmap[start:end].astype(np.float32))
        _normalize_inplace(chunk)
        index.add(chunk)
        if on_progress and (i % report_every == 0 or end == n):
            on_progress("added", f"{end:,}/{n:,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    return index


def evaluate_recall(
    index: faiss.Index,
    query_vecs_fp32: np.ndarray,
    gt_pool_vecs_fp32: np.ndarray,
    gt_pool_global_ids: np.ndarray,
    *,
    k: int = 10,
    nprobe: int = DEFAULT_NPROBE,
) -> dict:
    """Recall@k of FAISS top-k vs exhaustive top-k over a held-out pool.

    Both `query_vecs_fp32` and `gt_pool_vecs_fp32` must already be L2-
    normalized (caller's responsibility). `gt_pool_global_ids[i]` is the
    global_id corresponding to row `gt_pool_vecs_fp32[i]` — used so we can
    compare FAISS-returned global_ids to exhaustive ground-truth global_ids.

    The pool is the universe over which both engines must agree on top-k.
    The function therefore restricts FAISS results to those that fall in
    the pool (FAISS may return ids outside it; those are dropped before
    intersection). For pilot scale we pass the full database as the pool,
    so this restriction is a no-op.

    Note on nprobe: assigning `index.nprobe = N` works on a bare
    IndexIVF but is a silent no-op on IndexPreTransform(OPQ, IVF) (the
    factory string OPQ...,IVF...,PQ...). We extract the inner IVF and
    set nprobe on it directly.
    """
    ivf = faiss.extract_index_ivf(index)
    ivf.nprobe = int(nprobe)
    # FAISS top-k. Request more than k to absorb any ids that fall outside
    # the pool, then truncate.
    pool_set = set(int(x) for x in gt_pool_global_ids.tolist())
    over_k = min(k * 4, max(k + 1, 32))
    _scores, faiss_ids = index.search(query_vecs_fp32, over_k)

    # Exhaustive top-k: dot products vs the pool, take top-k indices, map
    # to global_ids
    sims = query_vecs_fp32 @ gt_pool_vecs_fp32.T  # (Q, P)
    top_local = np.argpartition(-sims, k, axis=1)[:, :k]
    gt_global_ids = gt_pool_global_ids[top_local]  # (Q, k)

    recalls = []
    for q in range(query_vecs_fp32.shape[0]):
        gt_set = set(int(x) for x in gt_global_ids[q].tolist())
        # Restrict FAISS results to ids that are in the pool, then take top-k
        f_in_pool = [int(x) for x in faiss_ids[q].tolist() if int(x) in pool_set]
        f_set = set(f_in_pool[:k])
        recalls.append(len(f_set & gt_set) / k)

    return {
        "k": k,
        "nprobe": nprobe,
        "n_queries": int(query_vecs_fp32.shape[0]),
        "recall_at_k": float(np.mean(recalls)),
        "recall_min": float(np.min(recalls)),
        "recall_p10": float(np.percentile(recalls, 10)),
    }


def _cli() -> None:
    from themisim.config import ARTIFACTS_ROOT

    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--memmap", type=Path, default=ARTIFACTS_ROOT / "vectors.f16.dat")
    p.add_argument("--manifest", type=Path, default=ARTIFACTS_ROOT / "manifest.parquet")
    p.add_argument("--meta", type=Path, default=ARTIFACTS_ROOT / "vectors.meta.json")
    p.add_argument("--out", type=Path, default=ARTIFACTS_ROOT / "index.faiss")
    p.add_argument("--nlist", type=int, default=DEFAULT_NLIST_PILOT)
    p.add_argument("--train-target", type=int, default=DEFAULT_TRAIN_TARGET)
    args = p.parse_args()

    import time

    meta = json.loads(args.meta.read_text())
    total_n = meta["shape"][0]
    print(f"Opening memmap: {args.memmap} shape ({total_n}, {N_FEATURES}) fp16")
    mm = open_memmap(args.memmap, total_n)

    def progress(stage, info):
        print(f"  [{stage}] {info}  (+{time.time() - t0:.0f}s)", flush=True)

    t0 = time.time()
    # Size the training set to the quantizer: an IVF with `nlist` centroids needs
    # tens of points per centroid to train well, so floor the target at
    # TRAIN_POINTS_PER_CENTROID * nlist. At nlist=65536 the 500K pilot default
    # gives ~7.6 points/centroid (FAISS warns, recall suffers); the floor lifts it
    # to ~4.2M (~64/centroid). Pilot nlist=4096 stays at the 500K default.
    train_target = max(args.train_target, TRAIN_POINTS_PER_CENTROID * args.nlist)
    print(
        f"Train target: {train_target:,} "
        f"(~{train_target / max(args.nlist, 1):.0f} points/centroid for nlist={args.nlist})",
        flush=True,
    )
    # Stream the (site, time_ns) keys out of the manifest and reservoir-sample the
    # training ids in a few GB of RAM. Building the full ~1B-row DataFrame and
    # group-by-ing it (the old path) peaked at ~120 GB and OOM-killed the session.
    print("Sampling training ids from manifest (streaming) ...", flush=True)
    train_ids = stratified_sample_from_parquet(
        args.manifest, target_n=train_target, seed=0, total_n_check=total_n
    )
    progress("train_ids", f"{len(train_ids):,}")

    print(f"Building {_build_factory_string(args.nlist)} ...", flush=True)
    train_and_build_index(
        mm, out_path=args.out,
        nlist=args.nlist,
        train_ids=train_ids,
        on_progress=progress,
    )

    size_bytes = args.out.stat().st_size
    print(
        f"Wrote {args.out} ({size_bytes / 1e6:.1f} MB, "
        f"{size_bytes / total_n:.1f} bytes/vector)"
    )


if __name__ == "__main__":
    _cli()
