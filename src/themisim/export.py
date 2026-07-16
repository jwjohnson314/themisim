"""Turn search ``Hit``s into the export table the THEMIS dashboard produces.

The column set, ordering, and string formats here are intentionally identical
to the app's "Download results CSV" feature (``app.py::_hits_to_dataframe`` and
its helpers) so a CSV written by this library is interchangeable with one
exported from the web UI:

    site, datetime, score, source_cdf

``datetime`` is a UTC wall-clock string, ``score`` is the exact fp32 cosine
similarity after rerank (1.0 == identical), and ``source_cdf`` is the public
URL of the CDF the matched frame came from.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from themisim.search import Hit

#: Base of the public THEMIS all-sky imager archive (no trailing slash).
THEMIS_BASE = "http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi"

EXPORT_COLUMNS = ["site", "datetime", "score", "source_cdf"]


def themis_cdf_url(site: str, dtstr: str) -> str:
    """Public download URL for a CDF.

    ``dtstr`` is the ``YYYYMMDDHH`` string baked into the shard filename.
    """
    yyyy, mm = dtstr[:4], dtstr[4:6]
    return f"{THEMIS_BASE}/{site}/{yyyy}/{mm}/thg_l1_asf_{site}_{dtstr}_v01.cdf"


def hit_cdf_url(hit: Hit) -> str:
    dtstr = Path(hit.shard_path).name.replace(".f16.npy", "")
    return themis_cdf_url(hit.site, dtstr)


def time_ns_utc_iso(time_ns: int) -> str:
    return _dt.datetime.fromtimestamp(
        time_ns / 1_000_000_000, tz=_dt.timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def hit_utc_iso(hit: Hit) -> str:
    return time_ns_utc_iso(hit.time_ns)


def hits_to_dataframe(hits: Iterable[Hit]) -> pd.DataFrame:
    """The same four columns the dashboard results grid shows, same order."""
    return pd.DataFrame(
        [
            {
                "site": h.site,
                "datetime": hit_utc_iso(h),
                "score": float(h.score),
                "source_cdf": hit_cdf_url(h),
            }
            for h in hits
        ],
        columns=EXPORT_COLUMNS,
    )


def results_csv_filename(site: Optional[str], time_ns: Optional[int]) -> str:
    """``themis-similarity-search-<site>-<YYYYMMDDTHHMMSSZ>.csv`` for an
    in-index query frame; a plain fallback otherwise. Mirrors the dashboard's
    download filename so exports from either path are named consistently."""
    if site and time_ns is not None:
        ts = _dt.datetime.fromtimestamp(
            time_ns / 1_000_000_000, tz=_dt.timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")
        return f"themis-similarity-search-{site}-{ts}.csv"
    return "themis-similarity-search.csv"
