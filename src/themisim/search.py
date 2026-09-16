"""Query-time similarity search with exact rerank + temporal diversification.

Stage 8. The full path:

    query (any (512,) fp32, possibly un-normalized)
        -> L2-normalize
        -> FAISS top-`prefilter` (default 500)         coarse, OPQ-PQ
        -> load those rows from the fp16 memmap
        -> cast fp32, L2-normalize, dot with query     exact cosine rerank
        -> sort descending by score
        -> drop any hit within ±30 s of an already-kept hit at the SAME site
        -> keep top-`k` (default 24)

Why this shape:
  - FAISS-only top-k on the OPQ-PQ index has poor ranking, because adjacent
    3-second frames are near-duplicates and the PQ codes cannot separate them.
    The prefilter+rerank pattern absorbs that. With `nprobe=64` and
    diversification disabled, a prefilter of 500 places 93.7 % of the
    *exhaustive* top-10 into the candidate pool; raising it to 1000 reaches
    96.1 % and to 2000 reaches 97.3 %, while dropping it to 100 falls to
    79.7 %.

    That capture rate IS the recall@10 of the returned results, not merely an
    upper bound on it. The rerank scores every candidate exactly, so a vector
    in the global exact top-10 that reaches the pool has at most 9 vectors in
    the whole database scoring above it, hence at most 9 candidates, and it
    necessarily survives into the reranked top-10. The prefilter decides what
    can be found; the rerank only orders it and cannot lose a captured hit.
    (Measured over 500 queries against an exhaustive scan of the full
    1,009,488,343-vector index; capture rate and recall@10 agreed to four
    decimal places. See BENCHMARK.md.)

  - Recall depends on the (nprobe, prefilter) PAIR, not on prefilter alone —
    though prefilter dominates and nprobe saturates early. At nprobe=1,
    prefilter=500 yields 80.6 % (0.131 below the default); from nprobe=8 to
    nprobe=256 at prefilter=2000 recall moves by 0.0008, for 32x the cells
    scanned. nprobe=8 is the knee, not the default 64.
  - Temporal diversification stops the result grid from being filled with
    24 nearly-identical frames from one substorm minute. ±30 s on a
    3-second cadence drops 19 of every 20 in-arc frames; the remainder
    fall through to the next-best distinct match.

    NOTE this is deliberately destructive of recall as measured against an
    exhaustive top-10: the frames it discards ARE in that top-10. At the
    default `diversify_seconds=30`, recall@10 versus exhaustive search is
    0.238, against 0.937 with it disabled (n=500, same queries). The 0.937
    figure describes retrieval; diversification is a presentation choice
    applied afterwards. Benchmarks must set `diversify_seconds=0` or they are
    measuring the de-duplicator.

Public surface:
    Hit                           — dataclass with everything render() needs
    SearchEngine(art_dir).search(q) -> list[Hit]
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import faiss
import numpy as np

from themisim.concat import N_FEATURES, open_memmap

logger = logging.getLogger(__name__)

DEFAULT_K = 24
# Safety cap for threshold mode (min_score): a low cutoff over a large filtered
# subset could otherwise match an unbounded number of frames. Callers paginate,
# but we still bound the returned list so a single query can't build (and the
# CSV export can't serialize) a runaway result set.
THRESHOLD_HIT_CAP = 2000
DEFAULT_PREFILTER = 500
DEFAULT_NPROBE = 64
DEFAULT_DIVERSIFY_SECONDS = 30
NS_PER_SECOND = 1_000_000_000

# When a site/date filter is active we score the matching subset *exactly*
# rather than going through the IVF index. This is both necessary and cheap.
# Necessary: the IVF coarse quantizer clusters by feature similarity — which
# correlates strongly with time — so an IDSelector-restricted IVF search often
# finds zero matching candidates in the nprobe nearest cells when the date
# filter excludes the query's own period (verified: a month filter returned
# 0/500 at nprobe=64). Cheap: the manifest/memmap are ordered by global_id =
# concat order = site then time, so any (site, date-range) filter is a handful
# of CONTIGUOUS id ranges (see _filter_ranges) that read SEQUENTIALLY off the
# SSD — far faster than the scattered reads an unfiltered IVF rerank does.
#
# The cap below is the largest matching subset we'll score exactly before
# falling back to the IDSelector-restricted IVF path. The old value (200k) was
# sized for the HDD era; vectors.f16.dat now lives on SSD and matches are read
# as sequential slices, so a few million rows score in a few seconds. Raising
# it keeps the high-recall exact path serving typical filters (a site-month, a
# site-year) instead of dropping them to the recall-limited IVF path; only very
# broad filters (e.g. a multi-site multi-year window) exceed it.
DEFAULT_BRUTE_FORCE_MAX = 4_000_000

# When a filtered subset is too large to score exactly we probe the IVF with an
# IDSelector -- and that probe can come back completely empty, because the cells
# nearest the query need not intersect a time-disjoint filter at all. Measured
# on the full archive with a one-month, all-station filter and prefilter=500:
# 59% of queries returned nothing at nprobe=1, 7% at the default nprobe=64, and
# 1% even at nprobe=4096. Before falling back to an exact scan we retry once
# with nprobe multiplied by this factor (capped at nlist), which is cheap
# relative to reading gigabytes of vectors.
RESTRICTED_NPROBE_ESCALATION = 16

# Memory bound for the exact rerank: load/normalize/dot at most this many
# candidate rows at a time so a large filtered subset doesn't materialize a
# multi-GB fp32 copy of the memmap.
_RERANK_CHUNK = 262_144


def date_range_to_ns(
    start_date: Union[_dt.date, str], end_date: Union[_dt.date, str]
) -> Tuple[int, int]:
    """Convert an inclusive UTC calendar-day range to ``(lo_ns, hi_ns)``.

    Accepts ``datetime.date`` or ``'YYYY-MM-DD'`` strings. The returned
    bounds are inclusive on both ends: ``lo_ns`` is midnight UTC at the
    start of ``start_date`` and ``hi_ns`` is the last nanosecond of
    ``end_date`` (23:59:59.999999999 UTC), so every frame on the end day is
    kept. Suitable to drop straight into ``SearchEngine.search(
    time_ns_ranges=...)``.
    """
    if isinstance(start_date, str):
        start_date = _dt.date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = _dt.date.fromisoformat(end_date)
    lo = _dt.datetime(
        start_date.year, start_date.month, start_date.day, tzinfo=_dt.timezone.utc
    )
    hi_day = end_date + _dt.timedelta(days=1)
    hi = _dt.datetime(
        hi_day.year, hi_day.month, hi_day.day, tzinfo=_dt.timezone.utc
    )
    lo_ns = int(lo.timestamp()) * NS_PER_SECOND
    hi_ns = int(hi.timestamp()) * NS_PER_SECOND - 1
    return lo_ns, hi_ns


def _encode_chunk(col, vocab: dict) -> np.ndarray:
    """Dictionary-encode one Arrow string column to int codes against a running
    global ``{value: code}`` dict, so codes are consistent across row groups.

    ``col.combine_chunks()`` first contiguates the ChunkedArray (its
    ``to_numpy`` takes no kwargs, unlike Array's). Returns an int64 code array.
    """
    enc = col.combine_chunks().dictionary_encode()
    remap = np.array(
        [vocab.setdefault(v, len(vocab)) for v in enc.dictionary.to_pylist()],
        dtype=np.int64,
    )
    return remap[enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)]


def _load_manifest_arrays(manifest_path, total_n: int):
    """Load the manifest columns the engine needs, frugally, from a huge parquet.

    Reading the full manifest through pandas materializes ~1B Python ``str``
    objects for the ``site`` and ``shard_path`` columns (measured >50 GB), which
    would OOM alongside the ~72 GB index at full-archive scale. Instead stream it
    row group by row group into preallocated arrays, dictionary-encoding the two
    string columns to compact int codes plus a small code->value vocab. Peak adds
    only the ~18 GB of final arrays (no concat doubling, no per-row strings).

    Returns ``(time_ns, frame_idx, site_codes, site_vocab, site_to_code,
    shard_codes, shard_vocab)``.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(manifest_path))
    if pf.metadata.num_rows != total_n:
        raise ValueError(
            f"manifest rows {pf.metadata.num_rows} != memmap size {total_n}"
        )
    time_ns = np.empty(total_n, dtype=np.int64)
    frame_idx = np.empty(total_n, dtype=np.int32)
    site_codes = np.empty(total_n, dtype=np.int16)
    shard_codes = np.empty(total_n, dtype=np.int32)
    site_to_code: dict = {}
    shard_to_code: dict = {}
    pos = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(
            rg, columns=["time_ns", "frame_idx", "site", "shard_path"]
        )
        m = tbl.num_rows
        time_ns[pos : pos + m] = (
            tbl.column("time_ns").combine_chunks().to_numpy(zero_copy_only=False)
        )
        frame_idx[pos : pos + m] = (
            tbl.column("frame_idx").combine_chunks().to_numpy(zero_copy_only=False)
        )
        site_codes[pos : pos + m] = _encode_chunk(tbl.column("site"), site_to_code)
        if len(site_to_code) > np.iinfo(np.int16).max:
            raise ValueError("more distinct sites than int16 can encode")
        shard_codes[pos : pos + m] = _encode_chunk(
            tbl.column("shard_path"), shard_to_code
        )
        pos += m
        del tbl
    if pos != total_n:
        raise ValueError(f"filled {pos} rows != manifest size {total_n}")
    site_vocab = np.empty(len(site_to_code), dtype=object)
    for v, c in site_to_code.items():
        site_vocab[c] = v
    shard_vocab = np.empty(len(shard_to_code), dtype=object)
    for v, c in shard_to_code.items():
        shard_vocab[c] = v
    return (
        time_ns,
        frame_idx,
        site_codes,
        site_vocab,
        site_to_code,
        shard_codes,
        shard_vocab,
    )


# Cold-start accelerator. Re-deriving the manifest arrays from the multi-GB
# parquet on every process start (decompress 1B+ rows, convert columns, dict-
# encode the strings) takes minutes. The result is a pure function of the
# manifest, so we persist it once and memory-map it on subsequent starts —
# turning a multi-minute parse into a near-instant mmap. Bump _CACHE_VERSION if
# the array layout/dtypes ever change so old caches are ignored, not misread.
_CACHE_VERSION = 1


def _manifest_cache_stamp(manifest_path: Path, total_n: int) -> dict:
    """Identity of the manifest the cache was built from. Any change to the
    manifest's size, mtime, row count, or feature dim invalidates the cache."""
    st = manifest_path.stat()
    return {
        "version": _CACHE_VERSION,
        "manifest_size": int(st.st_size),
        "manifest_mtime_ns": int(st.st_mtime_ns),
        "total_n": int(total_n),
        "n_features": int(N_FEATURES),
    }


def _load_manifest_arrays_cached(
    manifest_path: Path, total_n: int, cache_dir: Path
):
    """:func:`_load_manifest_arrays` with a persistent disk cache.

    Fast path: a cache whose stamp matches the current manifest — memory-map
    the three large int arrays (read-only, paged in lazily and reclaimable, so
    this also lowers resident memory vs. holding them as anonymous arrays) and
    load the small vocabularies. Slow path (first run, or manifest changed):
    build from parquet as before, then write the cache for next time. A failed
    or corrupt cache never blocks startup — it just falls back to the build.
    """
    big = ("time_ns", "frame_idx", "site_codes", "shard_codes")
    small = ("site_vocab", "shard_vocab")
    paths = {name: cache_dir / f"{name}.npy" for name in (*big, *small)}
    meta_path = cache_dir / "meta.json"
    want = _manifest_cache_stamp(manifest_path, total_n)

    if meta_path.exists() and all(p.exists() for p in paths.values()):
        try:
            if json.loads(meta_path.read_text()) == want:
                arr = {n: np.load(paths[n], mmap_mode="r") for n in big}
                site_vocab = np.load(paths["site_vocab"], allow_pickle=True)
                shard_vocab = np.load(paths["shard_vocab"], allow_pickle=True)
                site_to_code = {v: i for i, v in enumerate(site_vocab.tolist())}
                logger.info("manifest arrays loaded from cache %s", cache_dir)
                return (
                    arr["time_ns"], arr["frame_idx"], arr["site_codes"],
                    site_vocab, site_to_code, arr["shard_codes"], shard_vocab,
                )
        except Exception:
            logger.exception("manifest cache unreadable; rebuilding")

    result = _load_manifest_arrays(manifest_path, total_n)
    (time_ns, frame_idx, site_codes, site_vocab,
     _site_to_code, shard_codes, shard_vocab) = result
    try:
        tmp = cache_dir.with_name(cache_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        np.save(tmp / "time_ns.npy", time_ns)
        np.save(tmp / "frame_idx.npy", frame_idx)
        np.save(tmp / "site_codes.npy", site_codes)
        np.save(tmp / "shard_codes.npy", shard_codes)
        np.save(tmp / "site_vocab.npy", site_vocab, allow_pickle=True)
        np.save(tmp / "shard_vocab.npy", shard_vocab, allow_pickle=True)
        # meta.json written last and gating the fast path, so a crash mid-write
        # leaves an incomplete cache that simply fails the validity check.
        (tmp / "meta.json").write_text(json.dumps(want))
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp.rename(cache_dir)
        logger.info("manifest arrays cached to %s", cache_dir)
    except Exception:
        logger.exception("failed to write manifest array cache at %s", cache_dir)
    return result


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and coalesce overlapping/adjacent ``[start, end)`` ranges, so a
    frame matched by more than one window (overlapping date ranges) is scored
    exactly once and the IVF selector sees a minimal range set."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


@dataclass(frozen=True)
class Hit:
    global_id: int
    score: float       # exact fp32 cosine after rerank
    site: str
    time_ns: int
    shard_path: str    # path to the .f16.npy shard the frame lives in
    frame_idx: int     # row index inside that shard


class SearchEngine:
    """Loads the index + manifest + memmap once; serves many queries.

    Construction is moderately expensive (faiss.read_index can take a few
    seconds on a large index, manifest deserialization a similar amount).
    Hold one engine instance per process. In Streamlit, wrap with
    st.cache_resource.
    """

    def __init__(
        self,
        artifacts_dir: Union[str, Path],
        *,
        index_filename: str = "index.faiss",
        manifest_filename: str = "manifest.parquet",
        memmap_filename: str = "vectors.f16.dat",
        meta_filename: str = "vectors.meta.json",
        brute_force_max: int = DEFAULT_BRUTE_FORCE_MAX,
    ) -> None:
        art = Path(artifacts_dir)
        self.brute_force_max = int(brute_force_max)
        meta = json.loads((art / meta_filename).read_text())
        total_n = int(meta["shape"][0])
        if int(meta["shape"][1]) != N_FEATURES:
            raise ValueError(
                f"meta dim {meta['shape'][1]} != expected {N_FEATURES}"
            )

        # Memory-map the index instead of reading it into anonymous RAM. At
        # full-archive scale index.faiss is ~72.7 GB; a plain read_index makes
        # that 72.7 GB resident and non-evictable, which (alongside the
        # manifest arrays) OOM-killed a 125 GB box. With IO_FLAG_MMAP the
        # OPQ matrix and HNSW coarse quantizer load to RAM (small) while the
        # IVFPQ inverted lists stay on disk and page in per query (nprobe of
        # nlist touched -> tiny working set); the OS evicts those pages under
        # pressure. Trade-off: the first query to touch a cold list faults
        # from disk (minor cold-start latency that warms quickly) — vastly
        # preferable to OOM, and the only way 72.7 GB fits on this box.
        self.index = faiss.read_index(
            str(art / index_filename), faiss.IO_FLAG_MMAP
        )
        # Checked before ntotal: an untrained index is also empty, and
        # "untrained" is the diagnosis while "ntotal 0 != N" is only a symptom.
        # In practice train_and_build_index always trains before writing, so
        # this catches a hand-assembled or truncated index file rather than a
        # pipeline bug -- which is exactly when a clear message earns its keep.
        if not self.index.is_trained:
            raise ValueError(
                f"{art / index_filename} holds an untrained index; "
                "rebuild it with themisim.index.train_and_build_index"
            )
        if self.index.ntotal != total_n:
            raise ValueError(
                f"index ntotal {self.index.ntotal} != memmap size {total_n}"
            )
        # The factory string is OPQ64_64,IVF{nlist}_HNSW32,PQ64x8 which
        # FAISS implements as IndexPreTransform(OPQMatrix, IndexIVFPQ).
        # The outer IndexPreTransform does NOT proxy `nprobe` — assigning
        # `self.index.nprobe = N` would just attach a stray attribute to
        # the wrapper while the underlying IVF keeps its default of 1.
        # Hold the inner IVF separately so we can set nprobe directly.
        self._ivf = faiss.extract_index_ivf(self.index)

        self.memmap = open_memmap(art / memmap_filename, total_n)

        self.total_n = total_n
        # Cache the manifest columns as compact numpy arrays for hot-path lookup.
        # site/shard_path are dictionary-encoded to int codes (+ small vocabs)
        # rather than held as ~1B-element object arrays of Python strings, which
        # would OOM with the index loaded — see _load_manifest_arrays.
        (
            self._time_ns,
            self._frame_idx,
            self._site_codes,
            self._site_vocab,
            self._site_to_code,
            self._shard_codes,
            self._shard_vocab,
        ) = _load_manifest_arrays_cached(
            art / manifest_filename,
            total_n,
            art / f"{manifest_filename}.arrays.cache",
        )
        self._init_site_blocks()

    def _init_site_blocks(self) -> None:
        """Locate each site's contiguous block of global_ids.

        The manifest is ordered by (site, then time): each site occupies one
        contiguous id block and site codes are assigned in block order, so
        ``site_codes`` is non-decreasing and the block boundaries are found by
        binary search — ``_site_bounds[c] .. _site_bounds[c+1]`` is site ``c``.
        ``_sorted_layout`` records whether that invariant actually holds (an
        O(n_sites) endpoint check, no billion-row scan); the fast filter path
        in _filter_ranges uses it, and falls back to a linear scan if it does
        not. Within each block ``time_ns`` is monotonic, which lets a date
        filter resolve to a sub-range by binary search too."""
        n_sites = len(self._site_vocab)
        bounds = np.searchsorted(
            self._site_codes, np.arange(n_sites + 1, dtype=self._site_codes.dtype)
        ).astype(np.int64)
        ok = (
            n_sites > 0
            and bounds[0] == 0
            and bounds[-1] == self.total_n
            and bool(np.all(np.diff(bounds) >= 0))
        )
        if ok:
            # Endpoints of each non-empty block must carry that block's code.
            for c in range(n_sites):
                s, e = int(bounds[c]), int(bounds[c + 1])
                if e > s and not (
                    int(self._site_codes[s]) == c
                    and int(self._site_codes[e - 1]) == c
                ):
                    ok = False
                    break
        self._site_bounds = bounds
        self._sorted_layout = ok

    # ----------------------- manifest accessors ----------------------- #
    # The manifest columns are stored compactly (site/shard_path are
    # dictionary-encoded to int codes + small vocabs — see
    # _load_manifest_arrays). These methods are the *only* supported way to
    # read a manifest field for a global_id: callers must never touch
    # self._site_codes / self._shard_vocab / self._time_ns / self._frame_idx
    # directly. Owning the representation here is what stops a storage change
    # from rippling out into app.py / render.py as a fresh round of Attribute
    # errors.

    def site_of(self, gid: int) -> str:
        """Site code (e.g. ``'fsmi'``) for a global_id."""
        return str(self._site_vocab[self._site_codes[gid]])

    def shard_of(self, gid: int) -> str:
        """Absolute path of the .f16.npy shard holding this global_id's frame."""
        return str(self._shard_vocab[self._shard_codes[gid]])

    def frame_idx_of(self, gid: int) -> int:
        """Row index of this global_id's frame inside its shard."""
        return int(self._frame_idx[gid])

    def time_ns_of(self, gid: int) -> int:
        """UTC timestamp (ns since the epoch) of this global_id's frame."""
        return int(self._time_ns[gid])

    def sites(self) -> List[str]:
        """All distinct site codes present, sorted. O(n_sites): reads the
        small vocab, never the billion-row code column."""
        return sorted(self._site_vocab.tolist())

    def time_ns_bounds(self) -> Tuple[int, int]:
        """``(min, max)`` frame timestamp in ns — the indexed UTC time span."""
        return int(self._time_ns.min()), int(self._time_ns.max())

    def shard_starts(self) -> np.ndarray:
        """Global_ids where a new shard/CDF begins, with a terminal
        ``total_n`` sentinel appended. Within one CDF the manifest rows are
        contiguous and frame_idx restarts at 0, so ``frame_idx == 0`` marks
        the boundaries; the Browse tab walks these to enumerate every
        (site, date, hour) range without an O(N) per-frame structure."""
        return np.append(
            np.where(self._frame_idx == 0)[0], self.total_n
        ).astype(np.int64)

    # ----------------------------- search ----------------------------- #

    def _filter_ranges(
        self,
        sites: Optional[Sequence[str]],
        time_ns_ranges: Optional[Sequence[Tuple[int, int]]],
    ) -> Optional[List[Tuple[int, int]]]:
        """Resolve site / date-range filters to a sorted, non-overlapping list
        of contiguous ``[start, end)`` global_id ranges — or ``None`` when no
        filter is active (caller takes the unfiltered IVF path), or ``[]`` when
        nothing matches.

        Semantics: a frame is kept when its site is in `sites` (if given) AND
        its time falls in one of `time_ns_ranges` (if given); within each
        category the test is OR, between the two it is AND.

        Because the manifest is ordered by (site, then time) — see
        `_init_site_blocks` — each (site, window) pair is one contiguous id
        range, found by two binary searches. This is O(n_sites * n_windows *
        log N) and never touches the billion-row arrays beyond a few paged-in
        cache lines, replacing the old full-array boolean-mask scan. If the
        sorted layout can't be verified, fall back to that scan."""
        if not sites and not time_ns_ranges:
            return None
        if not self._sorted_layout:
            return self._filter_ranges_scan(sites, time_ns_ranges)

        if sites:
            codes = sorted(
                {self._site_to_code[s] for s in sites if s in self._site_to_code}
            )
        else:
            codes = list(range(len(self._site_vocab)))

        ranges: List[Tuple[int, int]] = []
        for c in codes:
            s, e = int(self._site_bounds[c]), int(self._site_bounds[c + 1])
            if e <= s:
                continue
            if not time_ns_ranges:
                ranges.append((s, e))
                continue
            tslice = self._time_ns[s:e]  # mmap view, ascending within a site
            for lo, hi in time_ns_ranges:
                a = s + int(np.searchsorted(tslice, int(lo), side="left"))
                b = s + int(np.searchsorted(tslice, int(hi), side="right"))
                if b > a:
                    ranges.append((a, b))
        return _merge_ranges(ranges)

    def _filter_ranges_scan(
        self,
        sites: Optional[Sequence[str]],
        time_ns_ranges: Optional[Sequence[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        """Fallback range resolution via a full-array boolean mask — correct
        but O(N). Used only if the (site, time)-sorted layout can't be verified
        (it always could on the as-built archive)."""
        mask = np.ones(self.total_n, dtype=bool)
        if sites:
            codes = np.asarray(
                [self._site_to_code[s] for s in sites if s in self._site_to_code],
                dtype=self._site_codes.dtype,
            )
            mask &= np.isin(self._site_codes, codes)
        if time_ns_ranges:
            tmask = np.zeros(self.total_n, dtype=bool)
            for lo, hi in time_ns_ranges:
                tmask |= (self._time_ns >= int(lo)) & (self._time_ns <= int(hi))
            mask &= tmask
        ids = np.where(mask)[0]
        if ids.size == 0:
            return []
        cuts = np.flatnonzero(np.diff(ids) != 1)
        starts = np.concatenate(([0], cuts + 1))
        ends = np.concatenate((cuts + 1, [ids.size]))
        return [(int(ids[s]), int(ids[e - 1]) + 1) for s, e in zip(starts, ends)]

    def _ranges_selector(self, ranges: List[Tuple[int, int]]):
        """A faiss ``IDSelector`` accepting exactly the ids in `ranges`.

        One ``IDSelectorRange`` per contiguous range (O(1) membership, nothing
        materialized — vs. an ``IDSelectorBatch`` over millions of explicit
        ids), OR-folded for multiple ranges. The operands are pinned on the
        instance for the duration of the search because faiss does not take
        ownership of them."""
        sels = [faiss.IDSelectorRange(int(s), int(e)) for s, e in ranges]
        sel = sels[0]
        for nxt in sels[1:]:
            combined = faiss.IDSelectorOr(sel, nxt)
            sels.append(combined)
            sel = combined
        self._sel_refs = sels  # keep alive across index.search
        return sel

    def _restricted_ivf_candidates(
        self,
        q: np.ndarray,
        ranges: List[Tuple[int, int]],
        *,
        nprobe: int,
        prefilter: int,
    ) -> "np.ndarray | None":
        """IVF candidates within `ranges`, or None if the probe found nothing.

        Retries once at a much larger nprobe before giving up. Returning None
        is the signal that the caller must fall back to exact scoring: an empty
        candidate set here does not mean "no frames match the filter", it means
        the probed cells missed a subset that does match.
        """
        sel = self._ranges_selector(ranges)
        nlist = int(self._ivf.nlist)
        attempts = [int(nprobe)]
        escalated = min(nlist, int(nprobe) * RESTRICTED_NPROBE_ESCALATION)
        if escalated > int(nprobe):
            attempts.append(escalated)

        for attempt in attempts:
            self._ivf.nprobe = attempt
            params = faiss.SearchParametersIVF(nprobe=attempt, sel=sel)
            _scores, faiss_ids = self.index.search(q, prefilter, params=params)
            candidate_ids = faiss_ids[0]
            candidate_ids = candidate_ids[candidate_ids >= 0].astype(np.int64)
            if candidate_ids.size:
                return candidate_ids
        return None

    def _exact_scores(self, candidate_ids: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Exact-cosine score every candidate against the (already
        L2-normalized) query `q` (shape ``(1, d)``). Loads the candidate
        fp16 rows in chunks, casts fp32, L2-normalizes, and dots — chunking
        bounds peak memory so a large filtered subset can't blow up into a
        multi-GB fp32 copy of the memmap."""
        scores = np.empty(candidate_ids.size, dtype=np.float32)
        for start in range(0, candidate_ids.size, _RERANK_CHUNK):
            block = candidate_ids[start : start + _RERANK_CHUNK]
            vecs = np.ascontiguousarray(self.memmap[block].astype(np.float32))
            faiss.normalize_L2(vecs)
            scores[start : start + block.size] = vecs @ q[0]
        return scores

    def _exact_scores_ranges(
        self, ranges: List[Tuple[int, int]], q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Exact-cosine score every frame in the contiguous `ranges`, reading
        each range as a SEQUENTIAL memmap slice (`self.memmap[s:e]`) rather than
        the scattered gather a fancy-index of explicit ids would do — the win
        that makes exact scoring of a multi-million-row filtered subset fast on
        SSD. Returns ``(global_ids, scores)`` aligned; chunked to bound peak
        memory the same way `_exact_scores` is."""
        total = sum(e - s for s, e in ranges)
        ids = np.empty(total, dtype=np.int64)
        scores = np.empty(total, dtype=np.float32)
        pos = 0
        for s, e in ranges:
            for cs in range(s, e, _RERANK_CHUNK):
                ce = min(cs + _RERANK_CHUNK, e)
                vecs = np.ascontiguousarray(self.memmap[cs:ce].astype(np.float32))
                faiss.normalize_L2(vecs)
                n = ce - cs
                scores[pos : pos + n] = vecs @ q[0]
                ids[pos : pos + n] = np.arange(cs, ce, dtype=np.int64)
                pos += n
        return ids, scores

    def search(
        self,
        query_vec: np.ndarray,
        *,
        k: int = DEFAULT_K,
        min_score: Optional[float] = None,
        prefilter: int = DEFAULT_PREFILTER,
        nprobe: int = DEFAULT_NPROBE,
        diversify_seconds: int = DEFAULT_DIVERSIFY_SECONDS,
        sites: Optional[Sequence[str]] = None,
        time_ns_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> List[Hit]:
        """Return Hits ranked best first, temporally diversified.

        Two result-limit modes:
          `min_score is None`  count mode — return up to `k` top Hits.
          `min_score` set      threshold mode — return every (diversified) Hit
                               whose cosine score is >= `min_score`, ignoring
                               `k`, up to `THRESHOLD_HIT_CAP` for safety. In the
                               unfiltered IVF path the candidate pool is the
                               top `prefilter`, so raise `prefilter` to surface
                               more low-cutoff matches.

        `query_vec` is (512,) fp32. Normalization is handled internally —
        callers may pass either normalized or un-normalized vectors.

        Optional filters restrict the search to a subset of the archive:
          `sites`           keep only frames from these site codes (OR
                            within the list). None/empty -> all sites.
          `time_ns_ranges`  keep only frames whose `time_ns` falls in one
                            of these inclusive ``(lo_ns, hi_ns)`` windows
                            (OR within the list; see `date_range_to_ns`).
                            None/empty -> all dates.
        Site and date constraints combine with AND. When a filter is active
        the matching subset is scored exactly (see the module notes on
        `DEFAULT_BRUTE_FORCE_MAX`); the unfiltered call keeps the original
        IVF prefilter + rerank path unchanged.
        """
        if query_vec.shape != (N_FEATURES,):
            raise ValueError(
                f"query shape {query_vec.shape} != ({N_FEATURES},)"
            )
        q = np.ascontiguousarray(query_vec.astype(np.float32).reshape(1, -1))
        faiss.normalize_L2(q)

        ranges = self._filter_ranges(sites, time_ns_ranges)
        if ranges is None:
            # No filter: original IVF prefilter path.
            self._ivf.nprobe = int(nprobe)
            _scores_faiss, faiss_ids = self.index.search(q, prefilter)
            candidate_ids = faiss_ids[0]
            candidate_ids = candidate_ids[candidate_ids >= 0].astype(np.int64)
            if candidate_ids.size == 0:
                return []
            exact_scores = self._exact_scores(candidate_ids, q)
        elif not ranges:
            return []
        else:
            total = sum(e - s for s, e in ranges)
            if total <= self.brute_force_max:
                # High-recall exact path: score every matching frame, reading
                # the contiguous ranges as sequential SSD slices.
                candidate_ids, exact_scores = self._exact_scores_ranges(ranges, q)
            else:
                # Subset too large to score exactly: IVF restricted to the
                # contiguous ranges via IDSelectorRange. Recall is best-effort
                # here, because the IVF cells track time.
                restricted = self._restricted_ivf_candidates(
                    q, ranges, nprobe=nprobe, prefilter=prefilter
                )
                if restricted is None:
                    # The probe found nothing, but `ranges` is non-empty, so
                    # frames matching the filter certainly exist. Returning []
                    # here would be fast and wrong -- a user filtering to a
                    # month would silently get no results at all while millions
                    # of frames matched. Score the ranges exactly instead:
                    # expensive (a sequential read of the whole subset) but
                    # correct, and rare.
                    logger.warning(
                        "restricted IVF probe returned no candidates for %d ranges "
                        "(%d frames) at nprobe up to %d; falling back to an exact "
                        "scan of the filtered subset",
                        len(ranges), total,
                        min(int(self._ivf.nlist), int(nprobe) * RESTRICTED_NPROBE_ESCALATION),
                    )
                    candidate_ids, exact_scores = self._exact_scores_ranges(ranges, q)
                else:
                    candidate_ids = restricted
                    exact_scores = self._exact_scores(candidate_ids, q)

        order = np.argsort(-exact_scores, kind="stable")

        # Count mode stops at k; threshold mode keeps every hit at/above the
        # cutoff (bounded by THRESHOLD_HIT_CAP). `order` is descending, so once
        # a score drops below the cutoff nothing later can qualify.
        limit = THRESHOLD_HIT_CAP if min_score is not None else int(k)

        diversify_ns = int(diversify_seconds) * NS_PER_SECOND
        # Per-site sorted list of kept time_ns; bisect to test ±diversify
        kept_times: dict[str, list[int]] = {}
        hits: List[Hit] = []

        for rank in order:
            score = float(exact_scores[rank])
            if min_score is not None and score < min_score:
                break
            gid = int(candidate_ids[rank])
            t = int(self._time_ns[gid])
            site = str(self._site_vocab[self._site_codes[gid]])

            seen = kept_times.get(site)
            if seen is not None:
                # Linear scan is fine: kept_times[site] is at most k entries.
                if any(abs(t - tk) <= diversify_ns for tk in seen):
                    continue
            kept_times.setdefault(site, []).append(t)

            hits.append(
                Hit(
                    global_id=gid,
                    score=score,
                    site=site,
                    time_ns=t,
                    shard_path=str(self._shard_vocab[self._shard_codes[gid]]),
                    frame_idx=int(self._frame_idx[gid]),
                )
            )
            if len(hits) >= limit:
                break

        return hits


def load_default_engine(artifacts_dir: Optional[Union[str, Path]] = None) -> SearchEngine:
    """Convenience: load from the conventional artifacts dir (paths.ARTIFACTS_ROOT)."""
    from themisim.config import ARTIFACTS_ROOT

    return SearchEngine(Path(artifacts_dir) if artifacts_dir else ARTIFACTS_ROOT)
