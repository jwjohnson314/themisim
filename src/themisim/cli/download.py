"""``themis-download`` — fetch THEMIS ASI CDFs from the Berkeley archive.

Examples
--------
Full archive (every site, every date)::

    themis-download --data-root ./data/cdf

A single site and month::

    themis-download --data-root ./data/cdf --sites fsmi --start 2015-03 --end 2015-03
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from themisim.config import default_data_root


def _split_sites(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [s.strip().lower() for s in value.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="where to store CDFs (default: $THEMIS_ASI_DATA_ROOT or ./data/cdf)",
    )
    p.add_argument(
        "--sites",
        default=None,
        help="comma-separated 4-letter site codes (default: all sites)",
    )
    p.add_argument("--start", default=None, help="earliest date, e.g. 2015 or 2015-03")
    p.add_argument("--end", default=None, help="latest date, e.g. 2016 or 2016-12")
    p.add_argument("--workers", type=int, default=6, help="parallel downloads")
    p.add_argument("--limit", type=int, default=None, help="cap files (smoke tests)")
    args = p.parse_args(argv)

    # Imported here so `themis-download --help` doesn't pay the import cost.
    from themisim.pipeline import download_archive

    state = {"t0": time.time()}

    def on_dir(done: int, queued: int, found: int, msg: str) -> None:
        if done % 25 == 0 or msg.startswith("WARN"):
            print(f"[crawl] dirs={done} queued={queued} found={found:,}  {msg}", flush=True)

    def on_event(res, done: int, total: int) -> None:
        if res.status in ("failed", "stopped"):
            print(f"  [{res.status}] {res.url}  {res.error}", flush=True)
        if done % 50 == 0 or done == total:
            dt = max(time.time() - state["t0"], 1e-6)
            print(f"  {done:,}/{total:,} files  ({done / dt:.1f}/s)", flush=True)

    print(f"Crawling archive (sites={args.sites or 'ALL'}, "
          f"start={args.start}, end={args.end}) ...", flush=True)
    res = download_archive(
        args.data_root,
        sites=_split_sites(args.sites),
        start=args.start,
        end=args.end,
        workers=args.workers,
        limit=args.limit,
        on_dir=on_dir,
        on_event=on_event,
    )

    print("\n=== summary ===")
    if len(res):
        print(res.groupby("status").size().to_string())
        print(f"bytes downloaded: {res['bytes_written'].sum() / 1e9:.2f} GB")
    else:
        print("(no files matched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
