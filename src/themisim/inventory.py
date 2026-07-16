"""Walk THEMIS data drives and build a global inventory.

The inventory is the input to Stage 5's bulk embed loop. One row per
unique (site, datetime) CDF, recording where to read it from. Drives
are searched in declared order; if the same logical CDF exists on
multiple drives, the first encounter wins.

Robust to flaky drives: if rglob or stat hits an OSError mid-walk
(e.g., USB drive disconnecting), the partial results from that drive
are kept and the walker moves on. The caller is told via on_error.

Schema (parquet, pyarrow-backed):
    site             string  4-letter site code, e.g. "fsmi"
    datetime         string  YYYYMMDDHH (one CDF per UTC hour)
    drive_path       string  absolute resolved path on this machine
    file_size_bytes  int64
    status           string  "unprocessed" | "done" | "failed"

CLI:
    python -m themisim.inventory --out <parquet_path>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import pandas as pd

from themisim.embed import parse_cdf_filename

# The public tool downloads into a single ``--data-root`` tree, so there is no
# built-in drive list; callers pass the root(s) explicitly to
# ``build_inventory``. The embed module's streaming coordinator groups work by
# filesystem device id (or an explicit ``--drive-root`` list), so it no longer
# needs a hardcoded roots constant here.

# Re-export for backwards compatibility with existing CLI code; new code
# should import directly from themisim.config to avoid pulling torch
# into modules that don't need it (faiss-cpu and torch fight over CUDA
# loader symbols if torch isn't imported first).
from themisim.config import ARTIFACTS_ROOT, DEFAULT_INVENTORY_PATH  # noqa: F401

INVENTORY_COLUMNS = ["site", "datetime", "drive_path", "file_size_bytes", "status"]


def _walk_drive(drive: Path) -> Iterator[Path]:
    """Yield CDF paths under a drive. Caller handles OSError from rglob/stat."""
    if not drive.exists():
        return
    yield from drive.rglob("thg_l1_asf_*_v01.cdf")


def build_inventory(
    drives: Optional[List[Union[str, Path]]] = None,
    *,
    on_progress: Optional[Callable[[Path, int], None]] = None,
    on_error: Optional[Callable[[Path, OSError], None]] = None,
) -> pd.DataFrame:
    """Walk each drive and return an inventory DataFrame.

    on_progress(drive_path, n_found_added) — called once per drive.
    on_error(drive_path, exc) — called when an OSError (e.g., I/O error
        from a flaky USB drive) interrupts a walk; partial results from
        that drive are kept and the next drive is attempted.
    Missing drives (path doesn't exist) are silently skipped.
    """
    drives = drives or []
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for drive in drives:
        drive_path = Path(drive)
        before = len(rows)
        try:
            for cdf_path in _walk_drive(drive_path):
                try:
                    site, _yyyy, dtstr = parse_cdf_filename(cdf_path)
                    size = cdf_path.stat().st_size
                    resolved = str(cdf_path.resolve())
                except (ValueError, OSError):
                    # ValueError: filename doesn't match the THEMIS pattern.
                    # OSError: stat on a single dead inode — skip the file,
                    # keep walking; if rglob itself dies later the outer
                    # except handles it.
                    continue
                key = (site, dtstr)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "site": site,
                        "datetime": dtstr,
                        "drive_path": resolved,
                        "file_size_bytes": size,
                        "status": "unprocessed",
                    }
                )
        except OSError as exc:
            if on_error is not None:
                on_error(drive_path, exc)
        if on_progress is not None:
            on_progress(drive_path, len(rows) - before)

    df = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    if df.empty:
        # Preserve column dtypes even when no rows were found.
        df = df.astype(
            {
                "site": "string",
                "datetime": "string",
                "drive_path": "string",
                "file_size_bytes": "int64",
                "status": "string",
            }
        )
        return df
    return df.astype(
        {
            "site": "string",
            "datetime": "string",
            "drive_path": "string",
            "file_size_bytes": "int64",
            "status": "string",
        }
    )


def write_inventory(df: pd.DataFrame, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def read_inventory(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_parquet(Path(path), engine="pyarrow")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--out", type=Path, default=DEFAULT_INVENTORY_PATH)
    p.add_argument(
        "--drive",
        action="append",
        default=None,
        help="Root(s) to walk for CDFs (repeatable). Typically your "
        "--data-root; required (no default).",
    )
    args = p.parse_args()

    def progress(drive_path, n_added):
        print(f"  {drive_path}: +{n_added} unique CDFs")

    def on_error(drive_path, exc):
        print(f"  {drive_path}: I/O error mid-walk ({exc}); kept partial results")

    print(f"Walking {len(args.drive or [])} drives...")
    df = build_inventory(drives=args.drive, on_progress=progress, on_error=on_error)
    write_inventory(df, args.out)
    print(f"\nWrote {len(df):,} rows to {args.out}")
    print(f"Distinct sites: {df['site'].nunique()}")
    print(df.groupby("site").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    _cli()
