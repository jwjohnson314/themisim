"""Concatenate per-CDF shards into a single memmap + global manifest.

Stage 6 of the pipeline. Inputs are the fp16 shards + JSON sidecars
written by Stage 3/5. Output is:

    vectors.f16.dat    np.memmap, dtype=fp16, shape=(total_N, 512)
    manifest.parquet   one row per vector:
                         global_id    int64   == row index in memmap
                         site         string  4-letter site code
                         time_ns      int64   ns since 1970
                         shard_path   string  abs path to source shard
                         frame_idx    int32   index inside that shard
    vectors.meta.json  {shape, dtype, n_shards, sha256_of_paths_list}

Shards are streamed in lexicographic path order, which because shard
paths are <site>/<YYYY>/<YYYYMMDDHH>.f16.npy is also chronological
within each site. The order is deterministic — re-running concat on
the same shard set produces an identical memmap and manifest, byte for
byte. The manifest is sorted ascending by global_id by construction.

Memory footprint: O(n_shards) sidecars + per-shard load (one shard at
a time). Manifest columns are pre-allocated as numpy arrays sized to
total_n, so peak python overhead is a few hundred MB even at full
archive scale (~756M rows).

CLI:
    python -m themisim.concat \
        --shards-dir ./data/artifacts/shards \
        --out-dir   ./data/artifacts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

N_FEATURES = 512  # SimCLR encoder output dimension
DTYPE = np.float16
BYTES_PER_VECTOR = N_FEATURES * np.dtype(DTYPE).itemsize  # 1024

MANIFEST_COLUMNS = ["global_id", "site", "time_ns", "shard_path", "frame_idx"]

# Rows buffered before each manifest row group is flushed to parquet. At full
# archive scale the manifest is ~1B rows; materializing every column at once
# (as an early version did) peaks well over 100 GB on the shard_path string
# column and OOM-kills the process. Streaming ~20M-row batches caps the arrow
# conversion spike at a couple of GB while parquet's own dictionary encoding
# still dedups the (few hundred-K unique) shard_path values on disk.
MANIFEST_BATCH_ROWS = 20_000_000


class _ManifestWriter:
    """Append per-shard manifest rows and stream them to parquet in batches.

    Holds at most one ``MANIFEST_BATCH_ROWS`` batch in memory. ``shard_path``
    and ``site`` are kept as ``string`` columns (parquet dictionary-encodes the
    repeats on disk), so the read-back schema is byte-for-byte the same as the
    old single-shot ``to_parquet`` and downstream consumers are unchanged.
    """

    def __init__(self, manifest_path: Path, batch_rows: int = MANIFEST_BATCH_ROWS):
        self.manifest_path = Path(manifest_path)
        self.batch_rows = batch_rows
        self._writer: "pq.ParquetWriter | None" = None
        self._schema: "pa.Schema | None" = None
        self._gid: List[np.ndarray] = []
        self._site: List[np.ndarray] = []
        self._time: List[np.ndarray] = []
        self._path: List[np.ndarray] = []
        self._frame: List[np.ndarray] = []
        self._buffered = 0

    def add_shard(
        self, gid_start: int, site: str, time_ns: np.ndarray, shard_path: str, n: int
    ) -> None:
        self._gid.append(np.arange(gid_start, gid_start + n, dtype=np.int64))
        self._site.append(np.full(n, site, dtype=object))
        self._time.append(np.asarray(time_ns, dtype=np.int64))
        self._path.append(np.full(n, shard_path, dtype=object))
        self._frame.append(np.arange(n, dtype=np.int32))
        self._buffered += n
        if self._buffered >= self.batch_rows:
            self._flush()

    def _flush(self) -> None:
        if self._buffered == 0:
            return
        df = pd.DataFrame(
            {
                "global_id": np.concatenate(self._gid),
                "site": np.concatenate(self._site),
                "time_ns": np.concatenate(self._time),
                "shard_path": np.concatenate(self._path),
                "frame_idx": np.concatenate(self._frame),
            }
        ).astype({"site": "string", "shard_path": "string"})
        table = pa.Table.from_pandas(df, schema=self._schema, preserve_index=False)
        if self._writer is None:
            # First batch fixes the schema (incl. the pandas metadata that makes
            # site / shard_path read back as StringDtype) for all later batches.
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.manifest_path, self._schema)
        self._writer.write_table(table)
        self._gid.clear()
        self._site.clear()
        self._time.clear()
        self._path.clear()
        self._frame.clear()
        self._buffered = 0

    def close(self) -> None:
        self._flush()
        if self._writer is None:
            # No rows were ever added; still emit a valid empty manifest.
            empty = pd.DataFrame(
                {
                    "global_id": np.empty(0, np.int64),
                    "site": pd.array([], dtype="string"),
                    "time_ns": np.empty(0, np.int64),
                    "shard_path": pd.array([], dtype="string"),
                    "frame_idx": np.empty(0, np.int32),
                }
            )
            empty.to_parquet(self.manifest_path, engine="pyarrow", index=False)
            return
        self._writer.close()
        self._writer = None


def discover_shards(shards_dir: Union[str, Path]) -> List[Path]:
    """Return shard paths in deterministic (lexicographic) order."""
    return sorted(Path(shards_dir).rglob("*.f16.npy"))


def sidecar_for(shard_path: Path) -> Path:
    return shard_path.with_name(shard_path.name.replace(".f16.npy", ".json"))


def _read_sidecar(shard_path: Path) -> dict:
    return json.loads(sidecar_for(shard_path).read_text())


def build_memmap_and_manifest(
    shards_dir: Union[str, Path],
    out_dir: Union[str, Path],
) -> Tuple[Path, Path, Path, int]:
    """Stream shards into vectors.f16.dat and write manifest.parquet + meta.

    Returns (memmap_path, manifest_path, meta_path, total_n).
    """
    shards_dir = Path(shards_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = discover_shards(shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f"no *.f16.npy shards found under {shards_dir}")

    # First pass: read sidecars to get totals up front.
    sidecars = [_read_sidecar(sp) for sp in shard_paths]
    n_per_shard = np.array([sc["n_frames"] for sc in sidecars], dtype=np.int64)
    total_n = int(n_per_shard.sum())

    memmap_path = out_dir / "vectors.f16.dat"
    manifest_path = out_dir / "manifest.parquet"
    meta_path = out_dir / "vectors.meta.json"

    # Allocate the memmap.
    mm = np.memmap(memmap_path, dtype=DTYPE, mode="w+", shape=(total_n, N_FEATURES))

    # Stream the manifest alongside the memmap fill so neither the vectors nor
    # any manifest column is ever fully resident — peak overhead stays at one
    # shard + one manifest batch.
    writer = _ManifestWriter(manifest_path)
    offset = 0
    for sp, sc in zip(shard_paths, sidecars):
        n = int(sc["n_frames"])
        arr = np.load(sp)
        if arr.shape != (n, N_FEATURES):
            raise ValueError(
                f"{sp}: expected shape ({n}, {N_FEATURES}), got {arr.shape}"
            )
        if arr.dtype != DTYPE:
            raise ValueError(f"{sp}: expected dtype {DTYPE}, got {arr.dtype}")
        if len(sc["time_ns"]) != n:
            raise ValueError(
                f"{sp}: sidecar time_ns length {len(sc['time_ns'])} != n_frames {n}"
            )

        mm[offset : offset + n] = arr
        writer.add_shard(offset, sc["site"], sc["time_ns"], str(sp.resolve()), n)
        offset += n

    mm.flush()
    del mm  # release the OS handle (before to_parquet so an OOM there can't
    #         corrupt the now-complete memmap — that exact failure killed the
    #         first full-archive run; see the streaming writer above)
    writer.close()

    _write_meta(meta_path, total_n, shard_paths, shards_dir)
    return memmap_path, manifest_path, meta_path, total_n


def _write_meta(
    meta_path: Path,
    total_n: int,
    shard_paths: List[Path],
    shards_dir: Union[str, Path],
) -> None:
    paths_digest = hashlib.sha256(
        b"\n".join(str(p.resolve()).encode() for p in shard_paths)
    ).hexdigest()
    meta = {
        "shape": [total_n, N_FEATURES],
        "dtype": str(np.dtype(DTYPE)),
        "n_shards": len(shard_paths),
        "shards_dir": str(Path(shards_dir).resolve()),
        "shard_paths_sha256": paths_digest,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def rebuild_manifest(
    shards_dir: Union[str, Path],
    out_dir: Union[str, Path],
    max_consecutive_skips: int = 1024,
) -> Tuple[Path, Path, int]:
    """Rebuild manifest.parquet + vectors.meta.json against an EXISTING memmap.

    Use this when ``vectors.f16.dat`` is already written and valid but the
    manifest/meta are missing or stale (e.g. the manifest write OOM-crashed
    after the memmap was flushed). The ~1 TB memmap is read-only ground truth
    and is never rewritten.

    The memmap was filled from ``sorted(rglob("*.f16.npy"))`` at concat time.
    Shards can have been *added* since (lexicographically anywhere, not just at
    the end), so a fresh sort would misalign every ``global_id``. We instead
    walk the current sorted shards and, for each, check whether its first/last
    vector equals the memmap rows at the running offset; a shard that does not
    match is an addition made after concat and is skipped without advancing the
    offset. ``total_n`` comes from the memmap file size, the authoritative count.

    Returns (manifest_path, meta_path, total_n).
    """
    shards_dir = Path(shards_dir)
    out_dir = Path(out_dir)
    memmap_path = out_dir / "vectors.f16.dat"
    manifest_path = out_dir / "manifest.parquet"
    meta_path = out_dir / "vectors.meta.json"

    if not memmap_path.exists():
        raise FileNotFoundError(f"no memmap to rebuild against at {memmap_path}")
    size = memmap_path.stat().st_size
    if size % BYTES_PER_VECTOR != 0:
        raise ValueError(
            f"{memmap_path} size {size} is not a multiple of {BYTES_PER_VECTOR}"
        )
    total_n = size // BYTES_PER_VECTOR
    mm = np.memmap(memmap_path, dtype=DTYPE, mode="r", shape=(total_n, N_FEATURES))

    shard_paths = discover_shards(shards_dir)
    if not shard_paths:
        raise FileNotFoundError(f"no *.f16.npy shards found under {shards_dir}")

    # Write to a temp file and atomically rename on success so a crash mid-pass
    # can never leave a half-written manifest in place of the old one.
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    writer = _ManifestWriter(tmp_manifest)
    used: List[Path] = []
    skipped: List[Path] = []
    offset = 0
    consecutive_skips = 0

    for sp in shard_paths:
        if offset >= total_n:
            # Every memmap row is accounted for; the rest are post-concat adds.
            skipped.append(sp)
            continue
        sc = _read_sidecar(sp)
        n = int(sc["n_frames"])
        if len(sc["time_ns"]) != n:
            raise ValueError(
                f"{sp}: sidecar time_ns length {len(sc['time_ns'])} != n_frames {n}"
            )
        # mmap the shard header only; read just the boundary rows we compare.
        arr = np.load(sp, mmap_mode="r")
        matches = (
            arr.shape == (n, N_FEATURES)
            and offset + n <= total_n
            and np.array_equal(np.asarray(arr[0]), mm[offset])
            and np.array_equal(np.asarray(arr[-1]), mm[offset + n - 1])
        )
        del arr
        if not matches:
            skipped.append(sp)
            consecutive_skips += 1
            if consecutive_skips > max_consecutive_skips:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive shards failed to align with the "
                    f"memmap at offset {offset:,}/{total_n:,}. The on-disk shard set has "
                    f"likely diverged from the one the memmap was built from (a shard may "
                    f"have been deleted). Aborting rather than writing a corrupt manifest."
                )
            continue
        consecutive_skips = 0
        writer.add_shard(offset, sc["site"], sc["time_ns"], str(sp.resolve()), n)
        offset += n
        used.append(sp)

    writer.close()

    if offset != total_n:
        tmp_manifest.unlink(missing_ok=True)
        raise RuntimeError(
            f"aligned only {offset:,}/{total_n:,} memmap rows from {len(used):,} shards "
            f"({len(skipped):,} skipped). A shard the memmap needs is missing under "
            f"{shards_dir}; manifest not written (existing manifest left intact)."
        )

    tmp_manifest.replace(manifest_path)
    # Meta describes exactly the shards that compose the memmap, in order.
    _write_meta(meta_path, total_n, used, shards_dir)
    print(
        f"Rebuilt manifest: {total_n:,} rows from {len(used):,} shards "
        f"({len(skipped):,} post-concat shards skipped).\n"
        f"Manifest: {manifest_path}\nMeta:     {meta_path}"
    )
    return manifest_path, meta_path, total_n


def open_memmap(memmap_path: Union[str, Path], total_n: int) -> np.memmap:
    """Read-only memmap, shape (total_n, N_FEATURES)."""
    return np.memmap(memmap_path, dtype=DTYPE, mode="r", shape=(total_n, N_FEATURES))


def _cli() -> None:
    from themisim.config import ARTIFACTS_ROOT

    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--shards-dir", type=Path, default=ARTIFACTS_ROOT / "shards")
    p.add_argument("--out-dir", type=Path, default=ARTIFACTS_ROOT)
    p.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rebuild manifest.parquet + vectors.meta.json against the existing "
        "vectors.f16.dat without rewriting the memmap. Use after a manifest-write "
        "crash on an already-complete memmap.",
    )
    args = p.parse_args()

    if args.manifest_only:
        rebuild_manifest(args.shards_dir, args.out_dir)
        return

    memmap_path, manifest_path, meta_path, total_n = build_memmap_and_manifest(
        args.shards_dir, args.out_dir
    )
    bytes_written = memmap_path.stat().st_size
    print(
        f"Wrote {total_n:,} vectors ({bytes_written / 1e9:.2f} GB) to {memmap_path}\n"
        f"Manifest:  {manifest_path}\n"
        f"Meta:      {meta_path}"
    )


if __name__ == "__main__":
    _cli()
