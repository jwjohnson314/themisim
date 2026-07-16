"""High-level query API: similarity search by ``(site, datetime, frame)``.

This is the library's main entry point for end users — usable from an
interactive session, a Jupyter notebook, or another script:

    >>> from themisim import query
    >>> df = query("fsmi", "2015-03-18T06", 412, artifacts="data/artifacts")
    >>> df.head()

The query frame is addressed the same way the dashboard's "Browse" tab does:
``datetime`` names the hourly CDF and ``frame`` is the 0-based frame index
within that hour. Because every indexed frame's embedding is already stored in
the vectors memmap, the query vector is fetched directly (no model weights or
re-embedding needed at query time), then run through the same
:class:`~themisim.search.SearchEngine` the app uses. The returned
DataFrame is byte-for-byte the format the app exports (see
:mod:`themisim.export`).
"""
from __future__ import annotations

import datetime as _dt
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from themisim.export import hits_to_dataframe
from themisim.search import (
    DEFAULT_DIVERSIFY_SECONDS,
    DEFAULT_K,
    DEFAULT_NPROBE,
    DEFAULT_PREFILTER,
    SearchEngine,
)

# Browse index: {site: {YYYY-MM-DD: {HH: (gid_start, gid_end)}}}.
BrowseIndex = Dict[str, Dict[str, Dict[str, Tuple[int, int]]]]

# Process-global caches so repeated query() calls in a notebook pay the
# (multi-second) engine load and browse-index build only once per artifacts dir.
_ENGINE_CACHE: Dict[str, SearchEngine] = {}
_BROWSE_CACHE: "Dict[int, BrowseIndex]" = {}
_LOCK = threading.Lock()


def get_engine(artifacts: Union[str, Path]) -> SearchEngine:
    """Return a process-cached :class:`SearchEngine` for ``artifacts``.

    Loading the index + manifest takes a few seconds; this caches one engine
    per artifacts directory so successive queries reuse it.
    """
    key = str(Path(artifacts).resolve())
    with _LOCK:
        engine = _ENGINE_CACHE.get(key)
        if engine is None:
            engine = SearchEngine(key)
            _ENGINE_CACHE[key] = engine
        return engine


def build_browse_index(engine: SearchEngine) -> BrowseIndex:
    """Group manifest rows into ``{site: {date: {hour: (gid_start, gid_end)}}}``.

    Within one CDF/hour the manifest rows are contiguous and ``frame_idx``
    restarts at 0, so shard boundaries are exactly ``shard_starts()`` and no
    O(N) per-frame scan is needed. Mirrors ``app.py::build_browse_index``.
    """
    starts = engine.shard_starts()
    out: BrowseIndex = {}
    for i in range(len(starts) - 1):
        s, e = int(starts[i]), int(starts[i + 1])
        site = engine.site_of(s)
        name = Path(engine.shard_of(s)).name  # YYYYMMDDHH.f16.npy
        dt = name.replace(".f16.npy", "")
        if len(dt) != 10 or not dt.isdigit():
            continue
        date_str = f"{dt[:4]}-{dt[4:6]}-{dt[6:8]}"
        hour = dt[8:10]
        out.setdefault(site, {}).setdefault(date_str, {})[hour] = (s, e)
    return out


def _browse_index_for(engine: SearchEngine) -> BrowseIndex:
    key = id(engine)
    with _LOCK:
        bi = _BROWSE_CACHE.get(key)
        if bi is None:
            bi = build_browse_index(engine)
            _BROWSE_CACHE[key] = bi
        return bi


def _parse_hour(value: Union[str, _dt.datetime, _dt.date]) -> Tuple[str, str]:
    """Resolve a datetime spec to ``(YYYY-MM-DD, HH)`` identifying an hourly CDF.

    Accepts a ``datetime``/``date``, an ISO-ish string (``2015-03-18T06``,
    ``2015-03-18 06``, ``2015-03-18T06:20:36``, or just ``2015-03-18``), or a
    compact ``YYYYMMDDHH`` / ``YYYYMMDD`` digit string. Sub-hour fields are
    ignored — the CDF is hourly.
    """
    if isinstance(value, _dt.datetime):
        dt = value
    elif isinstance(value, _dt.date):
        dt = _dt.datetime(value.year, value.month, value.day)
    else:
        s = str(value).strip()
        dt = None
        if s.isdigit():
            if len(s) == 10:
                dt = _dt.datetime.strptime(s, "%Y%m%d%H")
            elif len(s) == 8:
                dt = _dt.datetime.strptime(s, "%Y%m%d")
        else:
            s2 = s.replace(" ", "T")
            for fmt in (
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M",
                "%Y-%m-%dT%H",
                "%Y-%m-%d",
            ):
                try:
                    dt = _dt.datetime.strptime(s2, fmt)
                    break
                except ValueError:
                    continue
        if dt is None:
            raise ValueError(
                f"could not parse datetime {value!r}; expected e.g. "
                "'2015-03-18T06', '2015-03-18 06', or '2015031806'"
            )
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


def resolve_global_id(
    engine: SearchEngine,
    site: str,
    datetime: Union[str, _dt.datetime, _dt.date],
    frame: int,
) -> int:
    """Map ``(site, datetime, frame)`` to the indexed vector's global id.

    Raises ``KeyError`` if the site/date/hour is not in the index and
    ``IndexError`` if ``frame`` is out of range for that hour, each with a
    message that lists what *is* available so the user can correct the query.
    """
    browse = _browse_index_for(engine)
    date_str, hour = _parse_hour(datetime)

    by_date = browse.get(site)
    if by_date is None:
        raise KeyError(
            f"site {site!r} not in index. Available sites: "
            f"{', '.join(sorted(browse))}"
        )
    by_hour = by_date.get(date_str)
    if by_hour is None:
        dates = sorted(by_date)
        raise KeyError(
            f"no frames for site {site!r} on {date_str}. "
            f"{len(dates)} indexed dates span {dates[0]}..{dates[-1]}."
        )
    rng = by_hour.get(hour)
    if rng is None:
        raise KeyError(
            f"no CDF for site {site!r} at {date_str}T{hour}. "
            f"Indexed hours that day: {', '.join(sorted(by_hour))}"
        )
    s, e = rng
    n_frames = e - s
    if not (0 <= int(frame) < n_frames):
        raise IndexError(
            f"frame {frame} out of range; {site} {date_str}T{hour} has "
            f"{n_frames} frames (valid 0..{n_frames - 1})"
        )
    return s + int(frame)


def query(
    site: str,
    datetime: Union[str, _dt.datetime, _dt.date],
    frame: int,
    *,
    artifacts: Optional[Union[str, Path]] = None,
    results: int = DEFAULT_K,
    min_score: Optional[float] = None,
    prefilter: int = DEFAULT_PREFILTER,
    nprobe: int = DEFAULT_NPROBE,
    diversify_seconds: int = DEFAULT_DIVERSIFY_SECONDS,
    engine: Optional[SearchEngine] = None,
) -> pd.DataFrame:
    """Find the frames most similar to ``(site, datetime, frame)``.

    Parameters
    ----------
    site : str
        Four-letter THEMIS site code (e.g. ``"fsmi"``).
    datetime : str | datetime | date
        Identifies the hourly CDF; see :func:`_parse_hour` for accepted forms.
    frame : int
        0-based frame index within that hour's CDF.
    artifacts : str | Path, optional
        Directory holding ``index.faiss`` / ``manifest.parquet`` /
        ``vectors.f16.dat`` / ``vectors.meta.json``. Defaults to
        ``$THEMIS_ASI_ARTIFACTS`` or ``./data/artifacts``. Ignored when an
        explicit ``engine`` is supplied.
    results, prefilter, nprobe, diversify_seconds : int
        Search parameters, identical in meaning to the dashboard controls.
    min_score : float, optional
        Two ways to bound the result set. ``None`` (the default) is count
        mode: return the top ``results`` matches. A value in ``[0, 1]`` is
        threshold mode: ignore ``results`` and return *every* (diversified)
        match whose cosine similarity is at or above the cutoff, best first,
        capped at ``THRESHOLD_HIT_CAP`` for safety. In threshold mode the
        candidate pool is still the top ``prefilter`` FAISS matches, so raise
        ``prefilter`` to surface more low-cutoff hits.
    engine : SearchEngine, optional
        Reuse a pre-built engine instead of the cached one (advanced use).

    Returns
    -------
    pandas.DataFrame
        Columns ``site, datetime, score, source_cdf`` — the same table (and
        formatting) the dashboard's CSV export produces, best match first.
        The top row is normally the query frame itself (score ≈ 1.0).
    """
    if engine is None:
        if artifacts is None:
            from themisim.config import default_artifacts_root

            artifacts = default_artifacts_root()
        engine = get_engine(artifacts)

    global_id = resolve_global_id(engine, site, datetime, frame)
    query_vec = engine.memmap[global_id].astype(np.float32)
    hits = engine.search(
        query_vec,
        k=int(results),
        min_score=None if min_score is None else float(min_score),
        prefilter=int(prefilter),
        nprobe=int(nprobe),
        diversify_seconds=int(diversify_seconds),
    )
    return hits_to_dataframe(hits)
