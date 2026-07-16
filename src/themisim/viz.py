"""Visualize query results in a grid, analogous to the dashboard display.

:func:`visualize_results` takes the DataFrame returned by
:func:`themisim.query.query` and lays the matched frames out in a grid
with per-tile captions (site / time / score), the same way the Streamlit app
shows its results.

The query DataFrame carries only ``site, datetime, score, source_cdf`` — not the
pixels — so the visualizer fetches each result's source CDF (from a local
``data_root`` if present, otherwise downloading it to a cache), reads the frame
whose timestamp matches the row, and renders it with the canonical THEMIS
normalization (percentile stretch + circular mask), shown as grayscale exactly
like the app's display.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from themisim.download import download_one
from themisim.embed import _read_cdf, parse_cdf_filename
from themisim.preprocess import normalize_masked


def _parse_export_datetime(value: str) -> int:
    """Parse an export ``datetime`` string back to ns since the epoch."""
    dt = _dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").replace(
        tzinfo=_dt.timezone.utc
    )
    return int(dt.timestamp() * 1_000_000_000)


def _local_cdf_path(base: Path, url: str) -> Path:
    """Where the CDF behind ``url`` lives/should live under ``base``.

    Mirrors the archive layout ``<base>/<site>/<YYYY>/<MM>/<filename>`` so a
    cache and a ``data_root`` built by the pipeline share the same paths.
    """
    filename = url.rsplit("/", 1)[-1]
    site, yyyy, dtstr = parse_cdf_filename(filename)
    return base / site / yyyy / dtstr[4:6] / filename


def _ensure_cdf(url: str, base: Path, download: bool) -> Optional[Path]:
    """Return a local path to the CDF, downloading it if needed (or None)."""
    path = _local_cdf_path(base, url)
    if path.exists():
        return path
    if not download:
        return None
    res = download_one(url, str(path))
    return path if res.status in ("downloaded", "skipped") else None


def _frame_for_row(
    cdf_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]],
    url: str,
    base: Path,
    download: bool,
    time_ns: int,
) -> Optional[np.ndarray]:
    """Normalized (256, 256) display frame for one result row, or None.

    CDFs are read at most once (cached on ``url``); the frame nearest the row's
    timestamp is selected.
    """
    if url not in cdf_cache:
        path = _ensure_cdf(url, base, download)
        if path is None:
            cdf_cache[url] = None
        else:
            try:
                cdf_cache[url] = _read_cdf(path)
            except Exception:
                cdf_cache[url] = None
    cached = cdf_cache[url]
    if cached is None:
        return None
    imgs, times = cached
    idx = int(np.argmin(np.abs(times.astype(np.int64) - time_ns)))
    return normalize_masked(imgs[idx].astype(np.float32))


def visualize_results(
    df: pd.DataFrame,
    *,
    data_root: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    cols: int = 6,
    tile_size: float = 2.2,
    cmap: str = "gray",
    download: bool = True,
    title: Optional[str] = None,
):
    """Render query results in a grid, like the dashboard's results view.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`themisim.query.query` (columns
        ``site, datetime, score, source_cdf``), best match first.
    data_root : str | Path, optional
        Root of locally-available CDFs (the same tree ``themis-download``
        writes). Frames are read from here when present.
    cache_dir : str | Path, optional
        Where to download CDFs not found under ``data_root``. Defaults to
        ``data_root`` if given, else ``./themis-cdf-cache``.
    cols : int
        Grid columns (the app uses 6).
    tile_size : float
        Per-tile size in inches.
    cmap : str
        Matplotlib colormap (grayscale by default, matching the app display).
    download : bool
        Fetch missing CDFs from the archive. Set False for offline use.
    title : str, optional
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if df is None or len(df) == 0:
        raise ValueError("no results to visualize (empty DataFrame)")

    base = Path(cache_dir or data_root or "themis-cdf-cache")

    n = len(df)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * tile_size, rows * tile_size), squeeze=False
    )
    flat = axes.ravel()

    cdf_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
    for ax, (_, row) in zip(flat, df.iterrows()):
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            frame = _frame_for_row(
                cdf_cache,
                row["source_cdf"],
                base,
                download,
                _parse_export_datetime(row["datetime"]),
            )
        except Exception:
            frame = None

        if frame is None:
            ax.text(0.5, 0.5, "frame\nunavailable", ha="center", va="center",
                    fontsize=8, transform=ax.transAxes)
        else:
            ax.imshow(frame, cmap=cmap, vmin=0.0, vmax=1.0)

        hhmmss = row["datetime"].split(" ")[1]
        ax.set_title(
            f"{row['site']}  {hhmmss}\nscore {row['score']:.3f}", fontsize=8
        )

    for ax in flat[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
