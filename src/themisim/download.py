"""Resumable downloader for the archive backfill work-list.

Consumes a work-list parquet (see scripts that build ``worklist-*.parquet``)
and downloads each ``thg_l1_asf_*_v01.cdf`` from themis.ssl.berkeley.edu to its
assigned ``target_path``. Designed to be run once per drive in parallel and to
be interrupted/resumed freely.

Per file the worker:
  1. HEADs the URL for the authoritative Content-Length (the work-list's
     ``size_bytes`` is only MB-rounded from the directory listing, so it is
     used for planning, never for verification).
  2. If ``target_path`` already exists at the exact expected size -> skip.
  3. Otherwise streams to ``target_path + ".part"``, resuming via a Range
     request when a partial ``.part`` is already present, then verifies the
     final size and atomically renames into place.

Robust to interruption (``.part`` files resume), transient HTTP errors
(bounded retries), and a full disk (ENOSPC sets a shared stop flag so the
run halts cleanly instead of spewing errors). Failed rows are written to a
sibling ``<worklist-stem>.failures.parquet`` for a later re-run.

Required work-list columns: ``url``, ``target_path``, ``size_bytes``.
Optional: ``drive_root`` (enables ``--drive-root`` filtering).

CLI:
    python -m themisim.download --parquet worklist.parquet \
        --drive-root /path/to/drive --workers 6
"""
from __future__ import annotations

import argparse
import errno
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

USER_AGENT = "themis-backfill-downloader/1.0"
CHUNK = 1 << 20  # 1 MiB streaming chunks
REQUIRED_COLUMNS = ("url", "target_path", "size_bytes")


ARCHIVE_HOST = "themis.ssl.berkeley.edu"


def _https(url: str) -> str:
    """Upgrade archive URLs to https. The archive 302-redirects http->https;
    following that redirect on a Range request corrupts resumes (urllib
    reissues the GET and appends a full body onto the partial file).
    Requesting https up front avoids the redirect. Scoped to the archive host
    so non-archive URLs (e.g. local test servers) are left untouched."""
    if url.startswith(f"http://{ARCHIVE_HOST}"):
        return "https://" + url[len("http://"):]
    return url


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class FileResult:
    url: str
    target_path: str
    status: str  # "downloaded" | "skipped" | "failed" | "stopped"
    bytes_written: int = 0
    expected_bytes: Optional[int] = None
    error: str = ""


# --------------------------------------------------------------------------- #
# http helpers
# --------------------------------------------------------------------------- #
def head_content_length(url: str, *, tries: int = 3, timeout: int = 45) -> Optional[int]:
    """Authoritative file size via HEAD, or None if it can't be determined."""
    url = _https(url)
    for i in range(tries):
        try:
            req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                cl = resp.headers.get("Content-Length")
                return int(cl) if cl is not None else None
        except (HTTPError, URLError, OSError):
            if i == tries - 1:
                return None
            time.sleep(1.5 * (i + 1))
    return None


def _stream_to_part(
    url: str,
    part: Path,
    expected: int,
    *,
    timeout: int,
) -> int:
    """Download ``url`` fully into ``part`` (truncating any existing partial).

    We deliberately do NOT use HTTP Range to resume partials. The archive
    server mishandles Range requests intermittently under concurrent load
    (it occasionally answers a ``bytes=N-`` request with the full body, or
    302-redirects in a way that makes urllib re-issue the GET), which appended
    a second copy onto the partial and produced oversized, never-matching
    files. Always fetching the whole object to a fresh ``.part`` is simple and
    correct; an interrupted file just restarts from zero on the next attempt.

    Returns the final size of ``part``. Raises on network/IO error so the
    caller can apply retry/stop policy.
    """
    url = _https(url)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        with open(part, "wb") as fh:  # "wb" truncates any stale/partial .part
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
    return part.stat().st_size


def download_one(
    url: str,
    target_path: str,
    *,
    tries: int = 4,
    timeout: int = 120,
    dry_run: bool = False,
    stop: Optional[threading.Event] = None,
) -> FileResult:
    """Download a single file with HEAD-verify, resume and atomic rename."""
    target = Path(target_path)
    part = target.with_suffix(target.suffix + ".part")

    if stop is not None and stop.is_set():
        return FileResult(url, target_path, "stopped")

    expected = head_content_length(url)
    if expected is None:
        return FileResult(url, target_path, "failed", error="HEAD failed / no Content-Length")

    if target.exists() and target.stat().st_size == expected:
        return FileResult(url, target_path, "skipped", expected_bytes=expected)

    if dry_run:
        return FileResult(url, target_path, "downloaded", expected_bytes=expected)

    target.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for i in range(tries):
        if stop is not None and stop.is_set():
            return FileResult(url, target_path, "stopped", expected_bytes=expected)
        try:
            final = _stream_to_part(url, part, expected, timeout=timeout)
            if final != expected:
                last_err = f"size mismatch: got {final}, expected {expected}"
                continue  # resume on next attempt
            os.replace(part, target)  # atomic within the same filesystem
            return FileResult(
                url, target_path, "downloaded", bytes_written=final, expected_bytes=expected
            )
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.ENOSPC:
                if stop is not None:
                    stop.set()
                return FileResult(
                    url, target_path, "stopped", expected_bytes=expected, error="disk full (ENOSPC)"
                )
            last_err = f"{type(exc).__name__}: {exc}"
        except (HTTPError, URLError) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        if i < tries - 1:
            time.sleep(2.0 * (i + 1))
    return FileResult(url, target_path, "failed", expected_bytes=expected, error=last_err)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_worklist(parquet: Path, drive_root: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"work-list {parquet} missing columns: {missing}")
    if drive_root is not None:
        if "drive_root" not in df.columns:
            raise ValueError("--drive-root given but work-list has no 'drive_root' column")
        df = df[df["drive_root"] == drive_root].reset_index(drop=True)
    return df


def run(
    parquet: Path,
    *,
    drive_root: Optional[str] = None,
    workers: int = 6,
    limit: Optional[int] = None,
    dry_run: bool = False,
    on_event: Optional[Callable[[FileResult, int, int], None]] = None,
) -> pd.DataFrame:
    """Download every row of the work-list; return a per-file results frame.

    ``on_event(result, n_done, n_total)`` fires once per completed file.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    df = load_worklist(parquet, drive_root)
    if limit is not None:
        df = df.head(limit)
    total = len(df)
    stop = threading.Event()
    results: list[FileResult] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                download_one,
                row.url,
                row.target_path,
                dry_run=dry_run,
                stop=stop,
            ): row.url
            for row in df.itertuples(index=False)
        }
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            if on_event is not None:
                on_event(res, done, total)

    return pd.DataFrame([vars(r) for r in results])


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--parquet", type=Path, required=True, help="work-list parquet")
    p.add_argument("--drive-root", default=None, help="only rows with this drive_root")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="cap files (smoke tests)")
    p.add_argument("--dry-run", action="store_true", help="HEAD + skip-check only, no writes")
    args = p.parse_args()

    state = {"bytes": 0, "downloaded": 0, "skipped": 0, "failed": 0, "stopped": 0, "t0": time.time()}

    def on_event(res: FileResult, done: int, total: int) -> None:
        state[res.status] = state.get(res.status, 0) + 1
        state["bytes"] += res.bytes_written
        if res.status in ("failed", "stopped"):
            print(f"  [{res.status}] {res.url}  {res.error}")
        if done % 50 == 0 or done == total:
            dt = max(time.time() - state["t0"], 1e-6)
            gb = state["bytes"] / 1e9
            print(
                f"  {done:,}/{total:,}  dl={state['downloaded']:,} skip={state['skipped']:,} "
                f"fail={state['failed']:,}  {gb:.1f} GB  {gb/dt*3600:.0f} GB/h"
            )

    print(f"Loading {args.parquet}" + (f" (drive_root={args.drive_root})" if args.drive_root else ""))
    res = run(
        args.parquet,
        drive_root=args.drive_root,
        workers=args.workers,
        limit=args.limit,
        dry_run=args.dry_run,
        on_event=on_event,
    )

    print("\n=== summary ===")
    print(res.groupby("status").size().to_string() if len(res) else "(no rows)")
    print(f"downloaded bytes: {res['bytes_written'].sum()/1e12:.3f} TB")

    failed = res[res.status.isin(["failed", "stopped"])]
    if len(failed):
        out = args.parquet.with_suffix(".failures.parquet")
        failed.to_parquet(out, index=False)
        print(f"\n{len(failed):,} failed/stopped rows -> {out}")


if __name__ == "__main__":
    _cli()
