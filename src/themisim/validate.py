"""Measure whether a built index actually retrieves the right frames, and report it.

``build_index`` tells you it finished; it never tells you whether the thing it
wrote is correct. This module answers that for any artifacts directory, by
comparing the index against ground truth it computes itself.

**The operating point matters.** :func:`themisim.search.SearchEngine.search`
does not return FAISS's top-k: it pulls the top ``prefilter`` candidates from the
IVF-PQ index and then *rescores them exactly* in fp32 against the memmap. Raw
FAISS recall is therefore not the number a user experiences — it is much lower —
so the gated metric here is the recall of the full prefilter+rerank pipeline.
The raw FAISS number is reported alongside it as a diagnostic, not as a verdict.

Checks:

* **integrity** — index, manifest and memmap describe the same vectors; the
  index is trained; site blocks are contiguous and ``time_ns`` is non-decreasing
  within each. Those last two are unchecked invariants that
  ``SearchEngine._filter_ranges`` binary-searches on.
* **vector health** — no NaN/inf and no zero-norm rows. ``normalize_masked``
  guards the *image* against the ``p1 == p99`` case; nothing guards the
  resulting *vector*, and a zero row matches everything at score 0.
* **self-retrieval** — split deliberately in two. On the exact path (a
  site-filtered query, which ``search`` scores exhaustively) a frame *must*
  retrieve itself at cosine 1.0; that is deterministic and gated hard. On the
  approximate IVF path it is stochastic, so it is reported and gated loosely.
* **retrieval quality** — recall@k against an exhaustive cosine scan over a
  ``(nprobe, prefilter)`` grid. Reported against *fraction of the database
  scanned*, because a pilot's ``nprobe/nlist`` is orders of magnitude larger
  than the full archive's and the two are not the same experiment.
* **encoder reference** — vectors this build produced vs. ones pinned at
  release, separating "your encoder differs from ours" from "the index is bad".
* **export round-trip** — the user-facing DataFrame agrees with the manifest.

Exhaustive ground truth means this is tractable only at pilot scale; the
retrieval checks refuse to run above :data:`MAX_EXHAUSTIVE_N`.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np

from themisim.concat import N_FEATURES
from themisim.index import evaluate_recall
from themisim.search import SearchEngine

#: Above this the exhaustive ground-truth pool stops being affordable: it holds
#: the whole database as fp32 plus an (n_queries x n) similarity matrix.
MAX_EXHAUSTIVE_N = 2_000_000

DEFAULT_K = 10
DEFAULT_N_QUERIES = 200
#: Swept together, not independently: at a fixed candidate budget, probing more
#: cells can *lower* end-to-end recall by crowding true neighbours out of the
#: prefilter. Reporting one without the other invites a wrong conclusion.
DEFAULT_NPROBE_SWEEP = (1, 8, 32, 64)
DEFAULT_PREFILTER_SWEEP = (100, 500, 2000)

#: Fallback gates when a spec doesn't pin its own. A spec's calibrated
#: thresholds, measured on a real build, are the real contract.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "exact_self_retrieval": 1.0,
    "ivf_self_retrieval": 0.90,
    "recall_at_10": 0.90,
    # fp16 storage alone caps agreement at ~0.9999999, and CPU-vs-CUDA differs
    # by ~7e-6. A gate of 0.999 would still admit a transposed channel or the
    # wrong resize filter, so it is set an order of magnitude tighter.
    "encoder_cosine_mean": 0.9999,
    "encoder_cosine_min": 0.9999,
}

#: Production code paths a pilot-scale index structurally cannot exercise,
#: reported explicitly rather than left for a reader to infer from silence.
COVERAGE_GAPS = [
    "IDSelectorRange-restricted IVF search: every pilot-scale filtered query is "
    "below search.DEFAULT_BRUTE_FORCE_MAX, so filters always take the exact path",
    "multi-GPU / multi-drive embed coordinator (themisim.embed._spawn_*)",
    "scale guards that no-op below their thresholds: concat.MANIFEST_BATCH_ROWS "
    "(20M rows/group), index.ADD_CHUNK_SIZE (100k), index.TRAIN_GATHER_CHUNK "
    "(2M), and the reservoir branch of stratified_sample_from_parquet. These are "
    "covered by the offline test suite with shrunken constants instead.",
    "GPU coarse quantizer in index.train_and_build_index",
    "IO_FLAG_MMAP paging behaviour of a multi-GB index",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _provenance(build_device: Optional[str] = None) -> dict:
    """Everything needed to adjudicate "I got a different number".

    ``build_device`` is the device the *index* was built on, which is not the
    same as what this host can do: a report generated on a CUDA box for a
    CPU-built index must say ``cpu``, or the encoder-reference cosine (which
    differs by ~1e-6 between CPU and CUDA) cannot be interpreted.
    """
    import numpy as _np
    import pandas as _pd

    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "numpy": _np.__version__,
        "pandas": _pd.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "faiss_threads": int(faiss.omp_get_max_threads()),
        "build_device": build_device or "unknown",
        "cuda_available": False,
    }
    try:
        env["faiss_compile_options"] = faiss.get_compile_options()
    except Exception:
        pass
    for mod in ("pyarrow", "PIL", "cdflib", "torchvision"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:  # torch is absent from query-only installs
        import torch

        env["torch"] = torch.__version__
        env["torch_threads"] = int(torch.get_num_threads())
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        env["torch"] = None
    try:
        from themisim import __version__ as _v

        env["themisim"] = _v
    except Exception:
        pass
    return env


def _recall_regime_caveat(nlist: int, nprobe: int) -> str:
    """State how much of the database this index actually scans.

    A pilot and the full archive are not the same experiment: an IVF with 107
    cells probed 64 deep touches most of the data, while the archive's 65536
    cells probed 64 deep touch a thousandth of it. Quoting a pilot recall
    without this is the most misleading thing the report could do.
    """
    frac = min(nprobe, nlist) / max(nlist, 1)
    return (
        f"recall regime: this index has nlist={nlist}, so nprobe={nprobe} scans "
        f"{frac:.1%} of the database; the full archive (nlist=65536) scans "
        f"{min(nprobe, 65536) / 65536:.2%} at the same nprobe. Pilot recall is an "
        "upper bound on archive recall, not an estimate of it."
    )


def _normalized(engine: SearchEngine, gids: np.ndarray) -> np.ndarray:
    """fp32 L2-normalized copies of the given memmap rows."""
    vecs = np.ascontiguousarray(np.asarray(engine.memmap[gids], dtype=np.float32))
    faiss.normalize_L2(vecs)
    return vecs


def _dtstr_of(engine: SearchEngine, gid: int) -> str:
    """The ``YYYYMMDDHH`` of the CDF a global_id came from."""
    return Path(engine.shard_of(int(gid))).name.replace(".f16.npy", "")


def _nlist_of(engine: SearchEngine) -> int:
    return int(engine._ivf.nlist)


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #
def check_integrity(engine: SearchEngine) -> dict:
    """Artifacts agree, the index is trained, and the layout invariants hold."""
    problems: List[str] = []
    if engine.index.ntotal != engine.total_n:
        problems.append(f"index.ntotal {engine.index.ntotal} != total_n {engine.total_n}")
    if engine.memmap.shape != (engine.total_n, N_FEATURES):
        problems.append(f"memmap shape {engine.memmap.shape} unexpected")
    if not engine.index.is_trained:
        problems.append("index reports is_trained=False")
    contiguous = bool(getattr(engine, "_sorted_layout", False))
    if not contiguous:
        problems.append(
            "site blocks are not contiguous; filtered queries silently fall back "
            "to an O(N) scan (the concat ordering invariant is violated)"
        )

    # time_ns must be non-decreasing inside each site block: _filter_ranges
    # binary-searches on exactly that assumption and nothing else verifies it.
    non_monotone: List[str] = []
    starts = engine.shard_starts()
    for site in engine.sites():
        code = engine._site_to_code[site]
        idx = np.flatnonzero(engine._site_codes == code)
        if idx.size == 0:
            continue
        s, e = int(idx[0]), int(idx[-1]) + 1
        block = np.asarray(engine._time_ns[s:e])
        if np.any(np.diff(block) < 0):
            non_monotone.append(site)
    if non_monotone:
        problems.append(f"time_ns not non-decreasing within site block(s): {non_monotone}")

    return {
        "total_n": int(engine.total_n),
        "index_ntotal": int(engine.index.ntotal),
        "is_trained": bool(engine.index.is_trained),
        "nlist": _nlist_of(engine),
        "n_shards": int(len(starts) - 1),
        "n_sites": len(engine.sites()),
        "contiguous_site_blocks": contiguous,
        "time_monotone_within_sites": not non_monotone,
        "problems": problems,
        "passed": not problems,
    }


def check_vector_health(engine: SearchEngine, *, chunk: int = 100_000) -> dict:
    """No NaN/inf rows, and no zero-norm rows.

    A zero row survives ``faiss.normalize_L2`` unchanged and then scores 0.0
    against every query — retrievable by nothing, and invisible unless checked.
    """
    non_finite = 0
    zero_norm = 0
    min_norm = np.inf
    for start in range(0, engine.total_n, chunk):
        block = np.asarray(engine.memmap[start : start + chunk], dtype=np.float32)
        finite_rows = np.isfinite(block).all(axis=1)
        non_finite += int((~finite_rows).sum())
        norms = np.linalg.norm(np.where(finite_rows[:, None], block, 0.0), axis=1)
        zero_norm += int((norms == 0).sum())
        if norms.size:
            min_norm = min(min_norm, float(norms.min()))
    return {
        "non_finite_rows": non_finite,
        "zero_norm_rows": zero_norm,
        "min_row_norm": None if min_norm is np.inf else float(min_norm),
        "passed": non_finite == 0 and zero_norm == 0,
    }


def check_self_retrieval(
    engine: SearchEngine,
    *,
    n_queries: int = DEFAULT_N_QUERIES,
    nprobe: int = 64,
    prefilter: int = 500,
    seed: int = 0,
) -> dict:
    """A frame queried by its own vector must come back first.

    Two paths, reported separately because they have different guarantees:

    ``exact`` — a site-filtered query. At pilot scale the matching subset is
    below ``brute_force_max``, so ``search`` scores it exhaustively and the
    frame's own cosine is exactly 1.0. Deterministic; any failure is a real
    ``global_id`` -> ``(site, datetime, frame)`` mapping bug.

    ``ivf`` — the unfiltered approximate path. The frame can genuinely miss the
    candidate pool, so this is a quality statistic, not a correctness one.

    ``diversify_seconds=0`` throughout: the default temporal diversification
    would otherwise be part of what is under test.
    """
    rng = np.random.default_rng(seed)
    n = min(n_queries, engine.total_n)
    gids = rng.choice(engine.total_n, size=n, replace=False)

    exact_hits = 0
    ivf_hits = 0
    exact_scores: List[float] = []
    for gid in gids:
        gid = int(gid)
        vec = np.asarray(engine.memmap[gid], dtype=np.float32)
        site = engine.site_of(gid)

        hits = engine.search(vec, k=1, sites=[site], diversify_seconds=0)
        if hits and hits[0].global_id == gid:
            exact_hits += 1
            exact_scores.append(float(hits[0].score))

        hits = engine.search(
            vec, k=1, nprobe=nprobe, prefilter=prefilter, diversify_seconds=0
        )
        if hits and hits[0].global_id == gid:
            ivf_hits += 1

    return {
        "n_queries": int(n),
        "nprobe": int(nprobe),
        "prefilter": int(prefilter),
        "exact_path_fraction": exact_hits / n,
        "ivf_path_fraction": ivf_hits / n,
        "exact_min_score": float(min(exact_scores)) if exact_scores else 0.0,
        "exact_mean_score": float(np.mean(exact_scores)) if exact_scores else 0.0,
        "passed": None,  # filled by the threshold pass
    }


def check_retrieval(
    engine: SearchEngine,
    *,
    k: int = DEFAULT_K,
    n_queries: int = DEFAULT_N_QUERIES,
    nprobe_sweep: Sequence[int] = DEFAULT_NPROBE_SWEEP,
    prefilter_sweep: Sequence[int] = DEFAULT_PREFILTER_SWEEP,
    seed: int = 0,
) -> dict:
    """Recall@k vs an exhaustive cosine scan, over a (nprobe, prefilter) grid.

    Ground truth is the exact top-k by cosine over the whole database — i.e.
    what a brute-force ``IndexFlatIP`` would return, computed directly.

    Two recalls are reported per grid point:

    ``faiss_recall`` — FAISS's own top-k from the compressed PQ codes. This is
    the raw index quality and is *not* what a user sees.

    ``pipeline_recall`` — the top-k after the exact fp32 rerank of the top
    ``prefilter`` candidates, which is exactly what ``SearchEngine.search``
    returns. This is the gated number.

    The reranked scores are read out of the same similarity matrix used for
    ground truth, so the whole grid costs one matrix product plus one FAISS
    search per grid point.
    """
    if engine.total_n > MAX_EXHAUSTIVE_N:
        return {
            "skipped": f"total_n {engine.total_n:,} > {MAX_EXHAUSTIVE_N:,}",
            "grid": [],
            "passed": None,
        }

    rng = np.random.default_rng(seed)
    n_q = min(n_queries, engine.total_n)
    q_gids = np.sort(rng.choice(engine.total_n, size=n_q, replace=False))

    pool_ids = np.arange(engine.total_n, dtype=np.int64)
    pool = _normalized(engine, pool_ids)
    queries = _normalized(engine, q_gids)

    # Exhaustive ground truth: exact cosine against every vector.
    sims = queries @ pool.T                      # (Q, N)
    k_eff = min(k, engine.total_n - 1)
    gt = np.argpartition(-sims, k_eff, axis=1)[:, :k_eff]
    gt_sets = [set(row.tolist()) for row in gt]
    gt_scores = np.take_along_axis(sims, gt, axis=1)
    gt_scores.sort(axis=1)
    gt_scores = gt_scores[:, ::-1]

    nlist = _nlist_of(engine)
    grid = []
    for nprobe in nprobe_sweep:
        eff_nprobe = min(int(nprobe), nlist)
        engine._ivf.nprobe = eff_nprobe
        for prefilter in prefilter_sweep:
            pf = min(int(prefilter), engine.total_n)
            _s, cand = engine.index.search(queries, pf)
            faiss_rec, pipe_rec, score_err = [], [], []
            for i in range(n_q):
                ids = cand[i][cand[i] >= 0]
                faiss_rec.append(len(set(ids[:k_eff].tolist()) & gt_sets[i]) / k_eff)
                if ids.size:
                    # Exact rerank == look the candidates up in `sims`; this is
                    # the same arithmetic SearchEngine._exact_scores performs.
                    cs = sims[i, ids]
                    top = ids[np.argsort(-cs)[:k_eff]]
                    pipe_rec.append(len(set(top.tolist()) & gt_sets[i]) / k_eff)
                    score_err.append(
                        float(np.abs(np.sort(cs)[::-1][:k_eff] - gt_scores[i][: len(top)]).mean())
                    )
                else:
                    pipe_rec.append(0.0)
            grid.append(
                {
                    "nprobe": eff_nprobe,
                    "prefilter": pf,
                    "fraction_scanned": eff_nprobe / nlist,
                    "faiss_recall_at_k": float(np.mean(faiss_rec)),
                    "pipeline_recall_at_k": float(np.mean(pipe_rec)),
                    "pipeline_recall_p10": float(np.percentile(pipe_rec, 10)),
                    "pipeline_recall_min": float(np.min(pipe_rec)),
                    "mean_abs_score_error": float(np.mean(score_err)) if score_err else None,
                }
            )
    return {
        "k": int(k_eff),
        "n_queries": int(n_q),
        "pool_size": int(engine.total_n),
        "nlist": nlist,
        "grid": grid,
        "passed": None,
    }


def check_raw_faiss_recall(
    engine: SearchEngine, *, k: int = DEFAULT_K, n_queries: int = 50,
    nprobe: int = 64, seed: int = 0,
) -> dict:
    """Headline raw-index recall via :func:`themisim.index.evaluate_recall`.

    Exercises the library's own recall helper against the same exhaustive pool,
    so the number a caller of that public function would get is reported here
    too. Diagnostic only — see :func:`check_retrieval` for the gated metric.
    """
    if engine.total_n > MAX_EXHAUSTIVE_N:
        return {"skipped": "database too large for exhaustive ground truth", "passed": None}
    rng = np.random.default_rng(seed)
    n_q = min(n_queries, engine.total_n)
    q_gids = np.sort(rng.choice(engine.total_n, size=n_q, replace=False))
    pool_ids = np.arange(engine.total_n, dtype=np.int64)
    out = evaluate_recall(
        engine.index,
        _normalized(engine, q_gids),
        _normalized(engine, pool_ids),
        pool_ids,
        k=min(k, engine.total_n - 1),
        nprobe=min(int(nprobe), _nlist_of(engine)),
    )
    out["passed"] = None
    return out


def check_encoder_reference(
    engine: SearchEngine, reference: Union[str, Path, None]
) -> dict:
    """Compare this build's vectors against ones pinned at release time.

    The reference stores ``(site, datetime, frame_idx)`` addresses plus the
    vectors the authors computed. Those frames are inside the pilot slice, so
    the comparison reads straight out of the freshly built memmap — no
    re-embedding, and no torch dependency here. A low cosine means the
    reviewer's CDF read, preprocessing or encoder differs from ours, which is a
    different failure from a badly built index.

    Perfect agreement is not expected across devices: fp16 storage alone caps
    cosine at ~0.9999999 and CPU-vs-CUDA inference differs by ~1e-6, which is
    why the report carries the device alongside the number.
    """
    if reference is None or not Path(reference).exists():
        return {"skipped": "no reference vectors available", "passed": None}

    from themisim.query import resolve_global_id

    with np.load(reference, allow_pickle=False) as npz:
        sites = [str(s) for s in npz["site"]]
        dts = [str(d) for d in npz["datetime"]]
        frames = npz["frame_idx"].astype(int)
        ref = np.ascontiguousarray(npz["vectors"].astype(np.float32))

    faiss.normalize_L2(ref)
    cosines: List[float] = []
    missing = 0
    for site, dt, frame, r in zip(sites, dts, frames, ref):
        try:
            gid = resolve_global_id(engine, site, dt, int(frame))
        except (KeyError, IndexError):
            missing += 1
            continue
        v = np.ascontiguousarray(
            np.asarray(engine.memmap[gid], dtype=np.float32).reshape(1, -1)
        )
        faiss.normalize_L2(v)
        cosines.append(float(v[0] @ r))

    if not cosines:
        return {"skipped": "no reference frames present in this index", "passed": None}
    return {
        "n_reference_frames": len(cosines),
        "n_missing_from_index": int(missing),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_min": float(np.min(cosines)),
        "passed": None,
    }


def check_export_roundtrip(engine: SearchEngine, *, seed: int = 0) -> dict:
    """The exported DataFrame must agree with the manifest it came from."""
    from themisim.export import themis_cdf_url, time_ns_utc_iso
    from themisim.query import query

    rng = np.random.default_rng(seed)
    gid = int(rng.integers(0, engine.total_n))
    site = engine.site_of(gid)
    dtstr = _dtstr_of(engine, gid)
    frame = int(engine.frame_idx_of(gid))

    df = query(site, dtstr, frame, engine=engine, results=5, diversify_seconds=0)
    problems: List[str] = []
    if df.empty:
        problems.append("query returned no rows")
    else:
        top = df.iloc[0]
        if top["site"] != site:
            problems.append(f"top site {top['site']!r} != {site!r}")
        if top["source_cdf"] != themis_cdf_url(site, dtstr):
            problems.append(f"source_cdf {top['source_cdf']!r} does not match the shard")
        if top["datetime"] != time_ns_utc_iso(engine.time_ns_of(gid)):
            problems.append("datetime column disagrees with the manifest time_ns")
        if float(top["score"]) < 0.999:
            problems.append(f"self score {float(top['score']):.4f} < 0.999")
    return {
        "probe": {"site": site, "datetime": dtstr, "frame": frame, "global_id": gid},
        "n_rows": int(len(df)),
        "problems": problems,
        "passed": not problems,
    }


def check_sampler(manifest_path: Union[str, Path], *, target_n: int = 1000) -> dict:
    """``stratified_sample_from_parquet`` is deterministic and well-formed.

    At pilot scale the sampler's reservoir branch never fires (the training
    target exceeds every bucket's population, so the "sample" is the whole
    database). This check at least pins determinism, sortedness and uniqueness
    on real data; the reservoir and chunking-invariance paths are covered by the
    offline test suite, which shrinks the constants to force them.
    """
    from themisim.index import stratified_sample_from_parquet

    a = stratified_sample_from_parquet(manifest_path, target_n=target_n, seed=0)
    b = stratified_sample_from_parquet(manifest_path, target_n=target_n, seed=0)
    problems: List[str] = []
    if not np.array_equal(a, b):
        problems.append("same seed produced different samples")
    if a.size != np.unique(a).size:
        problems.append("sample contains duplicate global_ids")
    if not np.all(np.diff(a) > 0):
        problems.append("sample is not sorted ascending")
    return {
        "target_n": int(target_n),
        "n_sampled": int(a.size),
        "deterministic": bool(np.array_equal(a, b)),
        "problems": problems,
        "passed": not problems,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def validate_index(
    artifacts: Union[str, Path],
    *,
    nprobe: int = 64,
    prefilter: int = 500,
    k: int = DEFAULT_K,
    n_queries: int = DEFAULT_N_QUERIES,
    nprobe_sweep: Optional[Sequence[int]] = None,
    prefilter_sweep: Optional[Sequence[int]] = None,
    seed: int = 0,
    thresholds: Optional[Dict[str, float]] = None,
    reference: Union[str, Path, None] = None,
    spec_name: Optional[str] = None,
    timings: Optional[Dict[str, float]] = None,
    build_device: Optional[str] = None,
    engine: Optional[SearchEngine] = None,
) -> dict:
    """Run every check against ``artifacts`` and return the report dict.

    ``thresholds`` overrides :data:`DEFAULT_THRESHOLDS` key by key. The report's
    ``passed`` field is the AND of every check that reached a verdict; checks
    that were skipped (no reference vectors, database too large for exhaustive
    ground truth) report ``None`` and neither pass nor fail the run.

    The engine is constructed directly rather than through
    ``themisim.query.get_engine`` so a validate-after-rebuild in the same
    process cannot pick up a cached engine over the previous memmap.
    """
    artifacts = Path(artifacts)
    gates = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if engine is None:
        engine = SearchEngine(artifacts)
    if reference is None:
        default_ref = Path(__file__).with_name("data") / "pilot_reference_vectors.npz"
        reference = default_ref if default_ref.exists() else None
    if nprobe_sweep is None:
        nprobe_sweep = tuple(sorted({*DEFAULT_NPROBE_SWEEP, int(nprobe)}))
    if prefilter_sweep is None:
        prefilter_sweep = tuple(sorted({*DEFAULT_PREFILTER_SWEEP, int(prefilter)}))

    checks = {
        "integrity": check_integrity(engine),
        "vector_health": check_vector_health(engine),
        "self_retrieval": check_self_retrieval(
            engine, n_queries=n_queries, nprobe=nprobe, prefilter=prefilter, seed=seed
        ),
        "retrieval": check_retrieval(
            engine,
            k=k,
            n_queries=n_queries,
            nprobe_sweep=nprobe_sweep,
            prefilter_sweep=prefilter_sweep,
            seed=seed,
        ),
        "raw_faiss_recall": check_raw_faiss_recall(engine, k=k, nprobe=nprobe, seed=seed),
        "encoder_reference": check_encoder_reference(engine, reference),
        "export_roundtrip": check_export_roundtrip(engine, seed=seed),
        "sampler": check_sampler(artifacts / "manifest.parquet"),
    }

    # --- apply the gates -------------------------------------------------- #
    sr = checks["self_retrieval"]
    sr["threshold_exact"] = gates["exact_self_retrieval"]
    sr["threshold_ivf"] = gates["ivf_self_retrieval"]
    sr["passed"] = (
        sr["exact_path_fraction"] >= gates["exact_self_retrieval"]
        and sr["exact_min_score"] >= 0.999
        and sr["ivf_path_fraction"] >= gates["ivf_self_retrieval"]
    )

    ret = checks["retrieval"]
    gate = gates.get(f"recall_at_{ret.get('k', k)}", gates["recall_at_10"])
    if ret.get("grid"):
        at_op = next(
            (
                g
                for g in ret["grid"]
                if g["nprobe"] == min(int(nprobe), ret["nlist"])
                and g["prefilter"] == min(int(prefilter), ret["pool_size"])
            ),
            ret["grid"][-1],
        )
        ret["at_operating_point"] = at_op
        ret["threshold"] = gate
        ret["passed"] = at_op["pipeline_recall_at_k"] >= gate

    enc = checks["encoder_reference"]
    if "cosine_mean" in enc:
        enc["threshold_mean"] = gates["encoder_cosine_mean"]
        enc["threshold_min"] = gates["encoder_cosine_min"]
        enc["passed"] = (
            enc["cosine_mean"] >= gates["encoder_cosine_mean"]
            and enc["cosine_min"] >= gates["encoder_cosine_min"]
        )

    verdicts = [c["passed"] for c in checks.values() if c.get("passed") is not None]

    lo, hi = engine.time_ns_bounds()
    return {
        "spec": spec_name,
        "artifacts": str(artifacts.resolve()),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": _provenance(build_device),
        "dataset": {
            "total_n": int(engine.total_n),
            "sites": engine.sites(),
            "time_span_utc": [
                _dt.datetime.fromtimestamp(t / 1e9, _dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                for t in (int(lo), int(hi))
            ],
        },
        "operating_point": {"nprobe": int(nprobe), "prefilter": int(prefilter), "k": int(k)},
        "thresholds": gates,
        "checks": checks,
        "coverage_gaps": [
            _recall_regime_caveat(_nlist_of(engine), int(nprobe)),
            *COVERAGE_GAPS,
        ],
        "timings_s": dict(timings or {}),
        "passed": bool(verdicts) and all(verdicts),
    }


def write_report(report: dict, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def _verdict(c: dict) -> str:
    p = c.get("passed")
    return "SKIP" if p is None else ("PASS" if p else "FAIL")


def format_report(report: dict) -> str:
    """Human-readable summary of :func:`validate_index`'s output."""
    checks = report["checks"]
    ds = report["dataset"]
    op = report["operating_point"]
    lines: List[str] = [
        f"index: {ds['total_n']:,} vectors, {len(ds['sites'])} sites "
        f"({', '.join(ds['sites'])}), {ds['time_span_utc'][0]} .. {ds['time_span_utc'][1]}",
        f"operating point: nprobe={op['nprobe']} prefilter={op['prefilter']} k={op['k']}"
        f"   built on {report['provenance'].get('build_device')}"
        f"  faiss={report['provenance'].get('faiss')}"
        f"  torch={report['provenance'].get('torch')}",
        "",
    ]

    ig = checks["integrity"]
    lines.append(
        f"  [{_verdict(ig)}] {'integrity':<22} ntotal={ig['index_ntotal']:,} "
        f"nlist={ig['nlist']} shards={ig['n_shards']} "
        f"contiguous={ig['contiguous_site_blocks']} monotone={ig['time_monotone_within_sites']}"
    )
    vh = checks["vector_health"]
    lines.append(
        f"  [{_verdict(vh)}] {'vector health':<22} {vh['non_finite_rows']} non-finite, "
        f"{vh['zero_norm_rows']} zero-norm (min norm {vh['min_row_norm']:.4f})"
    )
    sr = checks["self_retrieval"]
    lines.append(
        f"  [{_verdict(sr)}] {'self-retrieval':<22} exact path "
        f"{sr['exact_path_fraction']:.3f} (min score {sr['exact_min_score']:.5f}), "
        f"IVF path {sr['ivf_path_fraction']:.3f} over {sr['n_queries']} frames"
    )
    enc = checks["encoder_reference"]
    lines.append(
        f"  [{_verdict(enc)}] {'encoder reference':<22} "
        + (
            enc["skipped"]
            if enc.get("skipped")
            else f"cosine mean {enc['cosine_mean']:.6f} min {enc['cosine_min']:.6f} "
            f"over {enc['n_reference_frames']} pinned frames"
        )
    )
    ex = checks["export_roundtrip"]
    lines.append(
        f"  [{_verdict(ex)}] {'export round-trip':<22} "
        + ("; ".join(ex["problems"]) or f"{ex['n_rows']} rows consistent with manifest")
    )
    sm = checks["sampler"]
    lines.append(
        f"  [{_verdict(sm)}] {'training sampler':<22} "
        + ("; ".join(sm["problems"]) or f"{sm['n_sampled']:,} ids, deterministic, sorted, unique")
    )

    ret = checks["retrieval"]
    lines.append("")
    if ret.get("skipped"):
        lines.append(f"  [SKIP] retrieval quality     {ret['skipped']}")
    else:
        lines.append(
            f"  [{_verdict(ret)}] recall@{ret['k']} vs exhaustive scan of "
            f"{ret['pool_size']:,} vectors, {ret['n_queries']} queries "
            f"(gate {ret.get('threshold', 0):.3f} at the operating point)"
        )
        lines.append(
            "      nprobe  scanned  prefilter   pipeline  (raw faiss)   |score err|"
        )
        for g in ret["grid"]:
            mark = " <-" if g is ret.get("at_operating_point") else ""
            err = "n/a" if g["mean_abs_score_error"] is None else f"{g['mean_abs_score_error']:.4f}"
            lines.append(
                f"      {g['nprobe']:<7} {g['fraction_scanned'] * 100:5.1f}%  "
                f"{g['prefilter']:<10} {g['pipeline_recall_at_k']:.4f}     "
                f"{g['faiss_recall_at_k']:.4f}       {err}{mark}"
            )
        lines.append(
            "      pipeline = FAISS top-prefilter then exact fp32 rerank, which is"
        )
        lines.append(
            "      what SearchEngine.search returns; raw faiss is the index alone."
        )

    if report.get("timings_s"):
        lines += ["", "  timings: " + ", ".join(
            f"{k.replace('_s', '')} {v:.0f}s" for k, v in report["timings_s"].items()
        )]

    lines += ["", f"  RESULT: {'PASS' if report['passed'] else 'FAIL'}", ""]
    lines.append("  not exercised at this scale:")
    for gap in report["coverage_gaps"]:
        lines.append(f"    - {gap}")
    return "\n".join(lines)
