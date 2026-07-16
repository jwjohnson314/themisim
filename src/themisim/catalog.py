"""Discover CDFs on the public THEMIS archive and build a download work-list.

Distilled from the original project's ``completeness.py`` (which also did
multi-USB local diffing). The public library only needs to (1) crawl the
Berkeley Apache autoindex to find which CDFs exist for a set of sites / dates,
and (2) turn that listing into the ``(url, target_path, size_bytes)`` work-list
that :mod:`themisim.download` consumes.

Crawling is seeded per site (and per year when a date range is given) so a
filtered run does not walk the entire ~1M-file tree.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set

import pandas as pd

from themisim.config import ARCHIVE_BASE_URL

# Strict on `thg_l1_asf_<site>_<YYYYMMDDHH>_v01.cdf`. The tree also carries
# `thg_l1_ast_*` thumbnails; ignore those (the index is built from asf only).
#
# v01 only, deliberately: the rest of the pipeline is v01-throughout — the
# inventory walk globs `*_v01.cdf` (inventory.py), the embed filename parser
# requires `_v01` (embed.py), and the CDF download URL is reconstructed with a
# hardcoded `_v01.cdf` suffix (export.py). Matching any `_v<NN>` here would let
# the crawler queue a reprocessed `_v02.cdf` for download that every downstream
# stage then silently drops. Keeping the crawl v01-only makes the whole pipeline
# consistent; supporting reprocessed versions would require making the shard
# naming and URL reconstruction version-aware end to end.
CDF_NAME_RE = re.compile(r"^thg_l1_asf_([a-z]{4})_(\d{10})_v01\.cdf$")
HREF_RE = re.compile(rb'href="([^"?#][^"]*)"', re.I)

WORKLIST_COLUMNS = ["site", "datetime", "filename", "url", "target_path", "size_bytes"]


def _opener(user_agent: str) -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", user_agent)]
    return opener


def _list_directory(
    url: str, opener: urllib.request.OpenerDirector, timeout: int = 30
) -> List[str]:
    """All href values on an Apache autoindex page (no filtering)."""
    with opener.open(url, timeout=timeout) as r:
        body = r.read()
    return [m.decode("utf-8", "ignore") for m in HREF_RE.findall(body)]


def _seed_urls(
    base_url: str,
    sites: Optional[Sequence[str]],
    start: Optional[str],
    end: Optional[str],
) -> List[str]:
    """Starting directories for the crawl.

    With no site filter we crawl from ``base_url`` (the whole archive). With
    sites we crawl each ``<base>/<site>/``; if a year range is derivable from
    ``start``/``end`` we descend to ``<base>/<site>/<YYYY>/`` so a narrow date
    request touches only the relevant years.
    """
    if not base_url.endswith("/"):
        base_url += "/"
    if not sites:
        return [base_url]

    years: Optional[List[int]] = None
    if start or end:
        y0 = int(start[:4]) if start else 1990
        y1 = int(end[:4]) if end else 2100
        years = list(range(y0, y1 + 1))

    seeds: List[str] = []
    for site in sites:
        site_base = urllib.parse.urljoin(base_url, f"{site}/")
        if years is None:
            seeds.append(site_base)
        else:
            for y in years:
                seeds.append(urllib.parse.urljoin(site_base, f"{y}/"))
    return seeds


def _in_range(dtstr: str, start: Optional[str], end: Optional[str]) -> bool:
    """Is the ``YYYYMMDDHH`` stamp within ``[start, end]``?

    ``start``/``end`` are compared as left-anchored prefixes of ``dtstr``
    (``YYYY``, ``YYYY-MM`` or ``YYYYMM`` etc.), so ``--start 2015-03`` keeps
    everything from March 2015 onward.
    """
    def norm(s: str) -> str:
        return s.replace("-", "")

    if start and dtstr < norm(start).ljust(len(dtstr), "0"):
        return False
    if end:
        e = norm(end)
        # Inclusive upper bound: pad with '9' so a YYYYMM end keeps that month.
        if dtstr > e.ljust(len(dtstr), "9"):
            return False
    return True


def crawl(
    sites: Optional[Sequence[str]] = None,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    base_url: str = ARCHIVE_BASE_URL,
    rate_limit_s: float = 0.3,
    user_agent: str = "themis-asi-search/0.1",
    cache_path: Optional[Path] = None,
    on_dir: Optional[Callable[[int, int, int, str], None]] = None,
) -> pd.DataFrame:
    """BFS the archive autoindex and return discovered CDFs.

    Returns a DataFrame with columns ``site, datetime, filename, remote_url``.
    If ``cache_path`` is given the listing is checkpointed there and resumed on
    a re-run (an interrupted crawl loses at most a few directory listings).
    """
    opener = _opener(user_agent)
    rows: List[dict] = []
    seen: Set[str] = set()

    if cache_path is not None and Path(cache_path).exists():
        prev = pd.read_parquet(cache_path)
        rows.extend(prev.to_dict("records"))
        seen.update(prev["remote_url"].tolist())

    queue: List[str] = _seed_urls(base_url, sites, start, end)
    dirs_done = 0
    last_ckpt = 0

    while queue:
        url = queue.pop(0)
        time.sleep(rate_limit_s)
        try:
            hrefs = _list_directory(url, opener)
        except (urllib.error.URLError, TimeoutError) as exc:
            if on_dir is not None:
                on_dir(dirs_done, len(queue), len(rows), f"WARN {url}: {exc}")
            continue

        for h in hrefs:
            if h in ("../", "..") or h.startswith("?") or h.startswith("/"):
                continue
            child = urllib.parse.urljoin(url, h)
            if h.endswith("/"):
                queue.append(child)
                continue
            m = CDF_NAME_RE.match(h)
            if not m:
                continue
            if not _in_range(m.group(2), start, end):
                continue
            if child in seen:
                continue
            seen.add(child)
            rows.append(
                {
                    "site": m.group(1),
                    "datetime": m.group(2),
                    "filename": h,
                    "remote_url": child,
                }
            )

        dirs_done += 1
        if on_dir is not None:
            on_dir(dirs_done, len(queue), len(rows), url)
        if cache_path is not None and dirs_done - last_ckpt >= 50:
            _write(rows, Path(cache_path))
            last_ckpt = dirs_done

    if cache_path is not None:
        _write(rows, Path(cache_path))
    return pd.DataFrame(rows, columns=["site", "datetime", "filename", "remote_url"])


def _write(rows: List[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("remote_url").reset_index(drop=True).to_parquet(
        out_path, engine="pyarrow", index=False
    )


def build_worklist(crawl_df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Turn a crawl listing into a download work-list.

    Each CDF's local target mirrors the archive layout
    ``<data_root>/<site>/<YYYY>/<MM>/<filename>``. ``size_bytes`` is left at 0;
    the downloader HEADs each URL for the authoritative size.
    """
    data_root = Path(data_root)
    rows = []
    for r in crawl_df.itertuples(index=False):
        yyyy, mm = r.datetime[:4], r.datetime[4:6]
        target = data_root / r.site / yyyy / mm / r.filename
        rows.append(
            {
                "site": r.site,
                "datetime": r.datetime,
                "filename": r.filename,
                "url": r.remote_url,
                "target_path": str(target),
                "size_bytes": 0,
            }
        )
    return pd.DataFrame(rows, columns=WORKLIST_COLUMNS)
