"""Single-CDF and bulk-inventory embedding pipelines.

`embed_cdf` reads one THEMIS L1 ASF CDF, preprocesses every frame, runs the
SimCLR encoder in batched fp32, and writes a fp16 shard plus a JSON sidecar.
`embed_inventory` iterates an inventory DataFrame, calling embed_cdf per row,
with resume support (skips CDFs whose shard + sidecar already exist) and
per-CDF try/except so one bad file doesn't kill the run.

Atomic writes (.tmp + rename) so an interrupted run never leaves a
half-written shard in place.

Output layout:
    <out_dir>/<site>/<YYYY>/<YYYYMMDDHH>.f16.npy   shape (N, 512), dtype float16
    <out_dir>/<site>/<YYYY>/<YYYYMMDDHH>.json      one-shard manifest

Sidecar schema (JSON):
    {
        "site": str,                # 4-letter site code, e.g. "fsmi"
        "datetime": str,            # YYYYMMDDHH
        "n_frames": int,            # rows in the shard
        "time_ns": [int, ...],      # int64 ns since 1970, length n_frames
        "source_sha256": str,       # sha256 of the source CDF
        "source_path": str,         # absolute path the CDF was read from
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import cdflib
import numpy as np
import pandas as pd
import torch
from torch import nn

from themisim.preprocess import preprocess_frame

DEFAULT_BATCH_SIZE = 128
N_FEATURES = 512

_FNAME_RE = re.compile(r"thg_l1_asf_(?P<site>[a-z]{4})_(?P<dt>\d{10})_v01\.cdf$")


def parse_cdf_filename(cdf_path: Union[str, Path]) -> Tuple[str, str, str]:
    """Returns (site, yyyy, yyyymmddhh) parsed from the CDF filename."""
    m = _FNAME_RE.search(str(cdf_path))
    if not m:
        raise ValueError(f"unrecognized CDF filename: {cdf_path}")
    site = m.group("site")
    dt = m.group("dt")
    return site, dt[:4], dt


def shard_paths(out_dir: Union[str, Path], cdf_path: Union[str, Path]) -> Tuple[Path, Path]:
    """Return (shard_path, sidecar_path) where they would be written for cdf_path."""
    site, yyyy, dtstr = parse_cdf_filename(cdf_path)
    base = Path(out_dir) / site / yyyy
    return base / f"{dtstr}.f16.npy", base / f"{dtstr}.json"


def _read_cdf(cdf_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (imgs uint16 [N,256,256], time_ns int64 [N])."""
    site, _, _ = parse_cdf_filename(cdf_path)
    h = cdflib.cdfread.CDF(str(cdf_path))
    imgs = h[f"thg_asf_{site}"][:]
    if imgs.ndim == 4:
        # 2024+ THEMIS layout: image cube is (1, row, col, N_frames) with the
        # frame axis last, and the per-frame timestamps live in `_time`
        # (Unix seconds) rather than `_epoch` (which holds only a single,
        # unusable start value in these files). Normalize to (N, row, col).
        imgs = np.ascontiguousarray(np.moveaxis(imgs[0], -1, 0))
        secs = np.asarray(h[f"thg_asf_{site}_time"][:], dtype="float64")
        time_ns = (secs * 1e9).astype(np.int64)
    else:
        # Legacy layout: (N, row, col) (or a single (row, col) frame), with
        # per-frame CDF_EPOCH timestamps in `_epoch`. Unchanged so existing
        # shards stay byte-consistent.
        if imgs.ndim == 2:
            imgs = imgs[None, ...]
        epoch = h[f"thg_asf_{site}_epoch"][:]
        times = cdflib.cdfepoch.to_datetime(epoch)
        time_ns = np.asarray(times, dtype="datetime64[ns]").astype(np.int64)
    # Guard the frame geometry at the single read funnel, after normalizing
    # both layouts to (N, 256, 256). Both the streaming (MultiCDFDataset.
    # __iter__) and non-streaming embed paths obtain frames via _read_cdf, and
    # both wrap this call in a skip-and-log try/except, so raising here turns a
    # malformed/mis-axised CDF into a logged skip of that one CDF rather than an
    # uncaught broadcast error in the per-frame loop that aborts the whole drive
    # worker.
    if imgs.ndim != 3 or imgs.shape[1:] != (256, 256):
        raise ValueError(
            f"{cdf_path}: expected image var shape (N, 256, 256), got "
            f"{imgs.shape}"
        )
    if len(time_ns) != imgs.shape[0]:
        raise ValueError(
            f"{cdf_path}: time/img length mismatch {len(time_ns)} vs {imgs.shape[0]}"
        )
    return imgs, time_ns


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


class _FrameDataset(torch.utils.data.Dataset):
    """Wraps an in-memory uint16 array of frames for DataLoader-driven preprocessing.

    Lifetime is bound to a single embed_cdf call. Workers fork from the
    parent process, so the array is shared via copy-on-write — no
    serialization of 150 MB happens at worker spawn.
    """

    def __init__(self, imgs: np.ndarray) -> None:
        self.imgs = imgs

    def __len__(self) -> int:
        return self.imgs.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return preprocess_frame(self.imgs[idx])


def embed_cdf(
    cdf_path: Union[str, Path],
    model: nn.Module,
    out_dir: Union[str, Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
) -> Path:
    """Embed every frame in `cdf_path` and persist as fp16 shard + JSON sidecar.

    `model` is the SimCLR encoder (output shape (B, 512)) on the desired device.
    `num_workers > 0` spawns DataLoader workers to parallelize CPU-side
    preprocessing (PIL Resize + ToTensor) so it overlaps with GPU forward.
    Returns the shard path on success.
    """
    cdf_path = Path(cdf_path)
    out_dir = Path(out_dir)
    shard_path, sidecar_path = shard_paths(out_dir, cdf_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    imgs, time_ns = _read_cdf(cdf_path)
    n_frames = imgs.shape[0]
    device = next(model.parameters()).device

    out_fp16 = np.empty((n_frames, N_FEATURES), dtype=np.float16)
    if num_workers > 0:
        loader = torch.utils.data.DataLoader(
            _FrameDataset(imgs),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
        idx = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                h = model(batch)
                n = h.shape[0]
                out_fp16[idx : idx + n] = h.detach().to(torch.float16).cpu().numpy()
                idx += n
    else:
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            batch = torch.stack(
                [preprocess_frame(imgs[i]) for i in range(start, end)]
            ).to(device, non_blocking=True)
            with torch.no_grad():
                h = model(batch)
            out_fp16[start:end] = h.detach().to(torch.float16).cpu().numpy()

    return _write_shard_atomic(
        out_dir=out_dir,
        cdf_path=cdf_path,
        out_fp16=out_fp16,
        time_ns=time_ns,
        source_sha256=_sha256_file(cdf_path),
    )


def _write_shard_atomic(
    out_dir: Union[str, Path],
    cdf_path: Union[str, Path],
    out_fp16: np.ndarray,
    time_ns: np.ndarray,
    source_sha256: str,
) -> Path:
    """Stage shard + sidecar to .tmp and rename. Shared by embed_cdf and the
    streaming consumer so atomicity is identical regardless of how the
    fp16 buffer was filled.

    `time_ns` may be a list[int] or np.ndarray; both serialize to the
    same JSON form."""
    cdf_path = Path(cdf_path)
    shard_path, sidecar_path = shard_paths(out_dir, cdf_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "site": parse_cdf_filename(cdf_path)[0],
        "datetime": parse_cdf_filename(cdf_path)[2],
        "n_frames": int(out_fp16.shape[0]),
        "time_ns": [int(t) for t in time_ns],
        "source_sha256": source_sha256,
        "source_path": str(cdf_path.resolve()),
    }

    # Atomic write: stage to .tmp, then rename. We pass an open file handle
    # to np.save so it doesn't second-guess the suffix and append `.npy`.
    shard_tmp = shard_path.with_suffix(shard_path.suffix + ".tmp")
    sidecar_tmp = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    with open(shard_tmp, "wb") as f:
        np.save(f, out_fp16, allow_pickle=False)
    sidecar_tmp.write_text(json.dumps(sidecar))
    shard_tmp.rename(shard_path)
    sidecar_tmp.rename(sidecar_path)
    return shard_path


# ---------------------------- bulk embedding ---------------------------- #


FAILURES_COLUMNS = [
    "site",
    "datetime",
    "drive_path",
    "exception_type",
    "exception_msg",
    "traceback",
]


def shard_is_ready(out_dir: Union[str, Path], cdf_path: Union[str, Path]) -> bool:
    """Both shard and sidecar present (and non-empty) — Stage 5 resume key."""
    shard_path, sidecar_path = shard_paths(out_dir, cdf_path)
    return (
        shard_path.exists()
        and sidecar_path.exists()
        and shard_path.stat().st_size > 0
        and sidecar_path.stat().st_size > 0
    )


def embed_inventory(
    inventory_df: pd.DataFrame,
    model: nn.Module,
    out_dir: Union[str, Path],
    *,
    resume: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    failures_path: Optional[Union[str, Path]] = None,
    on_progress: Optional[Callable[[int, int, str, pd.Series], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Embed every CDF listed in `inventory_df` into `out_dir`.

    Resume: when `resume=True`, a CDF whose shard + sidecar already exist
    on disk is skipped (its status becomes 'done' without re-embedding).
    Shard existence is the source of truth; the inventory's `status`
    column is informational. This makes the pipeline crash-safe — kill
    the process at any point, restart, and only the missing shards run.

    Failures: a per-CDF try/except records each failure as a row in the
    returned `failures_df`. If `failures_path` is given, the parquet is
    rewritten after each failure so a crash mid-run still leaves a
    record.

    Returns
    -------
    updated_inventory_df : pd.DataFrame
        Same as input but with `status` set per CDF outcome.
    failures_df : pd.DataFrame
        One row per CDF that raised. Empty if everything succeeded.
    """
    out_dir = Path(out_dir)
    failures_path_p = Path(failures_path) if failures_path is not None else None

    statuses = list(inventory_df["status"].astype(str))
    failures: list[dict] = []

    for i, row in enumerate(inventory_df.itertuples(index=False)):
        cdf_path = Path(row.drive_path)

        if resume and shard_is_ready(out_dir, cdf_path):
            statuses[i] = "done"
            if on_progress is not None:
                on_progress(i, len(inventory_df), "skipped", row)
            continue

        try:
            embed_cdf(
                cdf_path,
                model,
                out_dir,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            statuses[i] = "done"
            evt = "embedded"
        except Exception as exc:
            statuses[i] = "failed"
            failures.append(
                {
                    "site": row.site,
                    "datetime": row.datetime,
                    "drive_path": str(cdf_path),
                    "exception_type": type(exc).__name__,
                    "exception_msg": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            evt = "failed"
            if failures_path_p is not None:
                # Recording a failure must not itself crash the worker. The
                # underlying drive is mounted errors=remount-ro, so a single
                # ext4 hiccup briefly makes the whole filesystem read-only.
                # If we propagate the recording error, we lose ALL subsequent
                # CDFs the worker was going to handle. The in-memory list is
                # the source of truth; the parquet snapshot is best-effort.
                try:
                    failures_path_p.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(failures, columns=FAILURES_COLUMNS).to_parquet(
                        failures_path_p, engine="pyarrow", index=False
                    )
                except OSError as record_exc:
                    print(
                        f"[warn] could not persist failures parquet "
                        f"({type(record_exc).__name__}: {record_exc}); "
                        f"keeping in-memory record"
                    )

        if on_progress is not None:
            on_progress(i, len(inventory_df), evt, row)

    updated = inventory_df.copy()
    updated["status"] = pd.Series(statuses, dtype="string", index=inventory_df.index)
    failures_df = pd.DataFrame(failures, columns=FAILURES_COLUMNS)
    return updated, failures_df


# ---------------- Stage 10.5: streaming embed pipeline ---------------- #
#
# `embed_inventory` (above) is the eager Stage-5 loop: per-CDF
# DataLoader spawn + tear-down, GPU sees ~25% utilization, 4-5 s of the
# 5-6 s wall-clock per CDF is preprocessing + cold-cache I/O. The
# streaming pipeline below replaces it for full-archive runs:
#
#   - One MultiCDFDataset walks the worker's full ordered slice of the
#     inventory and yields per-frame items. DataLoader workers are
#     spawned ONCE per process and persist for the entire run, so
#     preprocessing stays warm and the GPU is never blocked on worker
#     spawn.
#   - The consumer routes per-frame outputs back to per-CDF buffers by
#     `cdf_idx`. When all frames for a CDF have arrived, the shard is
#     written atomically — exactly the same on-disk format as embed_cdf.
#   - shard_is_ready() remains the source of truth; resume semantics
#     are unchanged. Killing mid-run loses no completed CDFs.
#   - Failures (OSError on a CDF read, etc.) are captured per-DataLoader
#     -worker into JSONL files in a temp dir and aggregated at the end.
#     This keeps the dataset's __iter__ generator clean (no mixed item
#     types in the stream) at the cost of a tiny disk hop per failure.
#
# torch.compile and per-drive subprocess fan-out are layered on top of
# this streaming core (see `_cli`'s --streaming and --no-compile flags).


@dataclass(frozen=True)
class _CdfRecord:
    cdf_idx: int      # dense 0..len-1, used as routing key
    cdf_path: Path
    site: str
    datetime: str     # YYYYMMDDHH


class MultiCDFDataset(torch.utils.data.IterableDataset):
    """Stream every frame of an ordered list of CDFs.

    Each item is a dict (consumed by `_collate_streaming`):
        cdf_idx, frame_idx, n_frames, frame, time_ns,
        site, datetime, source_path, source_sha256

    `source_sha256` is set on the first frame of each CDF and "" on the
    rest — the consumer keeps the non-empty value.

    Worker sharding: with `num_workers > 0`, each DataLoader worker
    handles every Nth CDF (slot = worker_id, stride = num_workers). The
    parent process's slice is sorted by (site, datetime) before
    construction, so each worker reads its CDFs in mostly-sequential
    on-disk order — that's the fix for cross-drive seek thrashing.

    Per-CDF failure handling: if reading a CDF raises (OSError on a
    flaky drive, malformed file, …), a JSONL record is appended to
    `failure_log_dir/worker_<wid>.jsonl` and the worker moves to the
    next CDF. The dataset never yields an item for a failed CDF, which
    keeps the per-frame schema clean.
    """

    def __init__(
        self,
        cdf_records: List[_CdfRecord],
        failure_log_dir: Union[str, Path],
    ) -> None:
        self.cdf_records = cdf_records
        self.failure_log_dir = Path(failure_log_dir)
        self.failure_log_dir.mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[Dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            wid, n_workers = 0, 1
        else:
            wid, n_workers = worker_info.id, worker_info.num_workers
        my_records = self.cdf_records[wid::n_workers]

        log_path = self.failure_log_dir / f"worker_{wid}.jsonl"
        # 'a' so a worker that gets re-iterated (we don't, but defensive)
        # doesn't drop earlier failures.
        log = open(log_path, "a", buffering=1)
        try:
            for rec in my_records:
                try:
                    imgs, time_ns = _read_cdf(rec.cdf_path)
                    sha = _sha256_file(rec.cdf_path)
                except Exception as exc:
                    log.write(
                        json.dumps(
                            {
                                "site": rec.site,
                                "datetime": rec.datetime,
                                "drive_path": str(rec.cdf_path),
                                "exception_type": type(exc).__name__,
                                "exception_msg": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        + "\n"
                    )
                    continue

                n_frames = imgs.shape[0]
                for fidx in range(n_frames):
                    yield {
                        "cdf_idx": rec.cdf_idx,
                        "frame_idx": fidx,
                        "n_frames": n_frames,
                        "frame": preprocess_frame(imgs[fidx]),
                        "time_ns": int(time_ns[fidx]),
                        "site": rec.site,
                        "datetime": rec.datetime,
                        "source_path": str(rec.cdf_path),
                        "source_sha256": sha if fidx == 0 else "",
                    }
        finally:
            log.close()


def _collate_streaming(batch: List[Dict]) -> Dict:
    """Stack frames; pass through per-frame metadata as parallel lists."""
    return {
        "frames": torch.stack([b["frame"] for b in batch]),
        "cdf_idx": [b["cdf_idx"] for b in batch],
        "frame_idx": [b["frame_idx"] for b in batch],
        "n_frames": [b["n_frames"] for b in batch],
        "time_ns": [b["time_ns"] for b in batch],
        "site": [b["site"] for b in batch],
        "datetime": [b["datetime"] for b in batch],
        "source_path": [b["source_path"] for b in batch],
        "source_sha256": [b["source_sha256"] for b in batch],
    }


def _read_failure_logs(failure_log_dir: Path) -> List[Dict]:
    out: List[Dict] = []
    for log_path in sorted(failure_log_dir.glob("worker_*.jsonl")):
        for line in log_path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def embed_inventory_streaming(
    inventory_df: pd.DataFrame,
    model: nn.Module,
    out_dir: Union[str, Path],
    *,
    resume: bool = True,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    pad_last_batch: bool = False,
    failures_path: Optional[Union[str, Path]] = None,
    on_progress: Optional[Callable[[int, int, str, pd.Series], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stream-embed an inventory slice using a single persistent DataLoader.

    Drop-in for `embed_inventory` with two differences:

    1. DataLoader workers are spawned once and persist across CDFs,
       eliminating the per-CDF spawn cost that dominates the eager
       loop.
    2. Frames from multiple CDFs may be batched together; outputs are
       routed to per-CDF buffers by `cdf_idx` and shards are written
       atomically as each CDF completes.

    `pad_last_batch=True` zero-pads the final partial batch up to
    `batch_size` so a `torch.compile(mode='reduce-overhead')` encoder
    sees only static shapes (CUDA graphs require this). The padded
    rows' outputs are sliced off before routing.

    `failures_path` is written once at the end of the run (not
    incrementally per failure as `embed_inventory` does). shard_is_ready
    is still the resume source of truth, so a kill mid-run loses no
    completed shards — only the failures parquet snapshot.
    """
    out_dir = Path(out_dir)
    failures_path_p = Path(failures_path) if failures_path is not None else None

    # Build (and resume-filter) the dense work list. cdf_idx is dense so it
    # can index small per-CDF buffers without a hash table.
    cdf_records: List[_CdfRecord] = []
    skipped_keys: set = set()
    inventory_keys = []
    for row in inventory_df.itertuples(index=False):
        cdf_path = Path(row.drive_path)
        inventory_keys.append((row.site, row.datetime))
        if resume and shard_is_ready(out_dir, cdf_path):
            skipped_keys.add((row.site, row.datetime))
            continue
        cdf_records.append(
            _CdfRecord(
                cdf_idx=len(cdf_records),
                cdf_path=cdf_path,
                site=row.site,
                datetime=row.datetime,
            )
        )

    statuses: Dict[Tuple[str, str], str] = {
        k: ("done" if k in skipped_keys else "unprocessed") for k in inventory_keys
    }

    if on_progress is not None:
        for i, k in enumerate(inventory_keys):
            if k in skipped_keys:
                row = inventory_df.iloc[i]
                on_progress(i, len(inventory_df), "skipped", row)

    if not cdf_records:
        updated = inventory_df.copy()
        updated["status"] = pd.Series(
            [statuses[k] for k in inventory_keys],
            dtype="string",
            index=inventory_df.index,
        )
        return updated, pd.DataFrame(columns=FAILURES_COLUMNS)

    device = next(model.parameters()).device

    fail_dir = Path(tempfile.mkdtemp(prefix="themis_streaming_fail_"))
    try:
        dataset = MultiCDFDataset(cdf_records, failure_log_dir=fail_dir)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=(device.type == "cuda"),
            collate_fn=_collate_streaming,
        )

        # Per-CDF accumulator. Keyed by cdf_idx; entries deleted as soon as
        # the CDF's shard is written, so memory stays bounded by the number
        # of *concurrently in-flight* CDFs (= num_workers, since each
        # DataLoader worker reads one CDF at a time).
        per_cdf: Dict[int, Dict] = {}
        completed_cdf_idx: set = set()

        with torch.no_grad():
            for batch in loader:
                frames = batch["frames"]
                actual_b = frames.shape[0]
                if pad_last_batch and actual_b < batch_size:
                    pad = torch.zeros(
                        batch_size - actual_b,
                        *frames.shape[1:],
                        dtype=frames.dtype,
                    )
                    frames_in = torch.cat([frames, pad], dim=0)
                else:
                    frames_in = frames
                frames_in = frames_in.to(device, non_blocking=True)
                h_full = model(frames_in)
                h = h_full[:actual_b]
                h_fp16 = h.detach().to(torch.float16).cpu().numpy()

                for i in range(actual_b):
                    cidx = batch["cdf_idx"][i]
                    fidx = batch["frame_idx"][i]
                    n = batch["n_frames"][i]
                    if cidx not in per_cdf:
                        per_cdf[cidx] = {
                            "buf": np.empty((n, N_FEATURES), dtype=np.float16),
                            "time_ns": [0] * n,
                            "received": 0,
                            "site": batch["site"][i],
                            "datetime": batch["datetime"][i],
                            "source_path": batch["source_path"][i],
                            "source_sha256": "",
                        }
                    s = per_cdf[cidx]
                    s["buf"][fidx] = h_fp16[i]
                    s["time_ns"][fidx] = batch["time_ns"][i]
                    s["received"] += 1
                    if batch["source_sha256"][i]:
                        s["source_sha256"] = batch["source_sha256"][i]

                    if s["received"] == s["buf"].shape[0]:
                        _write_shard_atomic(
                            out_dir=out_dir,
                            cdf_path=Path(s["source_path"]),
                            out_fp16=s["buf"],
                            time_ns=s["time_ns"],
                            source_sha256=s["source_sha256"],
                        )
                        completed_cdf_idx.add(cidx)
                        statuses[(s["site"], s["datetime"])] = "done"
                        if on_progress is not None:
                            # Best-effort progress hook: we don't have the
                            # original DataFrame row here, so synthesize the
                            # fields the callers actually use.
                            row = pd.Series(
                                {
                                    "site": s["site"],
                                    "datetime": s["datetime"],
                                    "drive_path": s["source_path"],
                                }
                            )
                            on_progress(
                                len(completed_cdf_idx),
                                len(cdf_records),
                                "embedded",
                                row,
                            )
                        del per_cdf[cidx]

        # CDFs that started but never finished (worker died, pipeline killed
        # mid-stream): mark as failed-incomplete. Their shards are NOT on
        # disk yet — shard_is_ready will return False on the next run, so
        # they'll be retried.
        incomplete: List[Dict] = []
        for cidx, s in per_cdf.items():
            incomplete.append(
                {
                    "site": s["site"],
                    "datetime": s["datetime"],
                    "drive_path": s["source_path"],
                    "exception_type": "IncompleteStream",
                    "exception_msg": (
                        f"received {s['received']}/{s['buf'].shape[0]} frames "
                        "before stream ended"
                    ),
                    "traceback": "",
                }
            )
            statuses[(s["site"], s["datetime"])] = "failed"

        worker_failures = _read_failure_logs(fail_dir)
        for wf in worker_failures:
            statuses[(wf["site"], wf["datetime"])] = "failed"

        all_failures = worker_failures + incomplete
    finally:
        # Clean up failure log temp dir.
        for p in fail_dir.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            fail_dir.rmdir()
        except OSError:
            pass

    if failures_path_p is not None and all_failures:
        try:
            failures_path_p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_failures, columns=FAILURES_COLUMNS).to_parquet(
                failures_path_p, engine="pyarrow", index=False
            )
        except OSError as exc:
            print(
                f"[warn] could not persist failures parquet "
                f"({type(exc).__name__}: {exc}); keeping in-memory record"
            )

    updated = inventory_df.copy()
    updated["status"] = pd.Series(
        [statuses[k] for k in inventory_keys],
        dtype="string",
        index=inventory_df.index,
    )
    return updated, pd.DataFrame(all_failures, columns=FAILURES_COLUMNS)


# ----------- Stage 10.5: per-drive grouping + torch.compile ----------- #


def _root_key_for_path(p: str, roots: List[str]) -> str:
    """Longest explicit ``drive_root`` prefix containing ``p``, or "" if none.

    String-prefix match with a path-separator guard, so it doesn't depend on
    the drive being mounted at lookup time. The "" sentinel lets callers warn
    about rows outside every declared root rather than silently dropping them."""
    best = ""
    for d in roots:
        dd = d.rstrip("/")
        if p == dd or p.startswith(dd + "/"):
            if best == "" or len(dd) > len(best):
                best = dd
    return best


def _device_key_for_path(p: str, cache: dict) -> str:
    """Filesystem-device group key for a CDF path (``dev<st_dev>``), or "".

    Every file on one physical drive/mount shares an ``st_dev``, so grouping by
    it puts each drive's CDFs in one seek-sequential worker with no hardcoded
    drive list. We stat the *parent directory* (memoized) rather than each file,
    so a full-archive inventory costs one stat per site/year/month dir (a few
    thousand) instead of one per CDF (~1M). A path whose parent can't be statted
    (drive not mounted) maps to the "" sentinel."""
    parent = str(Path(p).parent)
    if parent not in cache:
        try:
            cache[parent] = os.stat(parent).st_dev
        except OSError:
            cache[parent] = None
    dev = cache[parent]
    return "" if dev is None else f"dev{dev}"


def group_inventory_by_drive(
    df: pd.DataFrame, drive_roots: Optional[List[str]] = None
) -> List[Tuple[str, pd.DataFrame]]:
    """Return ``[(group_key, slice_df), ...]`` in deterministic order.

    Grouping strategy:

    - ``drive_roots`` given (an explicit ``--drive-root`` list): each CDF is
      grouped under the longest matching root prefix; group order follows the
      declared ``drive_roots`` order. Rows outside every root fall to "".
    - ``drive_roots`` omitted (the default): group by filesystem device id
      (``st_dev``) so each physical drive/mount becomes one group with no
      configuration; group order is sorted by device key for determinism.

    Determinism matters: the streaming coordinator computes this list and spawns
    one worker per group, and each worker recomputes it to pick its own slice by
    index — both must agree. Every slice is sorted by (site, datetime) ascending
    for mostly-sequential disk access. Rows in the "" sentinel group are surfaced
    (not dropped) so callers can warn about them.

    A single data-root on one device collapses to exactly one group, which is
    what un-breaks the streaming path for the common single-mount install."""
    if drive_roots:
        norm_roots = [d.rstrip("/") for d in drive_roots]
        keys = df["drive_path"].astype(str).map(
            lambda p: _root_key_for_path(p, norm_roots)
        )
        order = norm_roots + [""]
    else:
        cache: dict = {}
        keys = df["drive_path"].astype(str).map(
            lambda p: _device_key_for_path(p, cache)
        )
        uniq = sorted(k for k in keys.unique() if k != "")
        order = uniq + ([""] if (keys == "").any() else [])

    out: List[Tuple[str, pd.DataFrame]] = []
    seen: set = set()
    for k in order:
        if k in seen:
            continue
        seen.add(k)
        sel = df[keys == k]
        if sel.empty:
            continue
        sel = sel.sort_values(["site", "datetime"]).reset_index(drop=True)
        out.append((k, sel))
    return out


def maybe_compile_encoder(
    encoder: nn.Module,
    *,
    use_compile: bool,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """Wrap encoder with torch.compile for a ~20-30% forward speedup, or
    return it untouched. `mode='reduce-overhead'` uses CUDA graphs and
    requires static input shapes (the streaming consumer pads its last
    partial batch to satisfy this)."""
    if not use_compile:
        return encoder
    if not torch.cuda.is_available():
        # CPU path: compile is mostly a no-op or slower; skip.
        return encoder
    return torch.compile(encoder, mode=mode)


# ----------------------------- bulk CLI ------------------------------- #


def _filter_inventory(
    df: pd.DataFrame,
    sites: Optional[List[str]] = None,
    years: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    out = df
    if sites:
        out = out[out["site"].isin(sites)]
    if years:
        out = out[out["datetime"].str[:4].isin(years)]
    if limit is not None:
        out = out.head(limit)
    return out.reset_index(drop=True)


def _cli() -> None:
    from themisim.config import DEFAULT_CHECKPOINT_NAME, default_weights_dir
    from themisim.inventory import (
        ARTIFACTS_ROOT,
        DEFAULT_INVENTORY_PATH,
        read_inventory,
        write_inventory,
    )
    from themisim.model import load_simclr

    p = argparse.ArgumentParser(description="Bulk-embed CDFs from inventory.")
    p.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY_PATH)
    p.add_argument("--out-dir", type=Path, default=ARTIFACTS_ROOT / "shards")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=default_weights_dir() / DEFAULT_CHECKPOINT_NAME,
    )
    p.add_argument(
        "--site",
        action="append",
        default=None,
        help="Filter to one or more sites (repeatable).",
    )
    p.add_argument(
        "--year",
        action="append",
        default=None,
        help="Filter to one or more YYYY years (repeatable).",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for CPU-side preprocessing per GPU process.",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override CUDA device, e.g. cuda:1. Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="If >1, fan out the work across N GPU processes (cuda:0..N-1). "
        "Mutually exclusive with --device.",
    )
    p.add_argument(
        "--worker-shard",
        type=int,
        default=None,
        help="Internal: this process is shard k of --gpus total. Set by the "
        "auto-spawn loop. Do not pass manually.",
    )
    p.add_argument(
        "--streaming",
        action="store_true",
        help="Stage 10.5: use the streaming pipeline (one process per "
        "drive, persistent DataLoader workers, optional torch.compile). "
        "Mutually exclusive with --gpus / --worker-shard.",
    )
    p.add_argument(
        "--worker-drive",
        type=int,
        default=None,
        help="Internal (streaming): index into the per-drive group list "
        "spawned by the streaming coordinator. Do not pass manually.",
    )
    p.add_argument(
        "--worker-device",
        type=str,
        default=None,
        help="Internal (streaming): cuda device for this drive worker, "
        "e.g. cuda:0. Set by the coordinator's GPU round-robin.",
    )
    p.add_argument(
        "--drive-root",
        action="append",
        default=None,
        help="Streaming: explicit physical-drive root(s) to group work by "
        "(repeatable). Default: group by filesystem device id (st_dev), so "
        "each mount becomes one seek-sequential worker automatically.",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Streaming-only: skip torch.compile on the encoder. Useful "
        "for A/B'ing the compile speedup or sidestepping its 30-60 s "
        "warmup on small runs.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap rows for testing.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--failures-out",
        type=Path,
        default=ARTIFACTS_ROOT / "failures.parquet",
    )
    p.add_argument(
        "--no-write-back",
        action="store_true",
        help="Do not merge updated status back into the inventory parquet.",
    )
    args = p.parse_args()

    if args.streaming and (args.gpus > 1 or args.worker_shard is not None):
        p.error("--streaming is mutually exclusive with --gpus / --worker-shard")
    if args.gpus > 1 and args.device is not None:
        p.error("--gpus and --device are mutually exclusive")
    if args.streaming and args.worker_drive is None:
        # Top-level coordinator: spawn one worker per non-empty drive.
        return _spawn_drive_workers(args)
    if args.streaming and args.worker_drive is not None:
        return _run_drive_worker(args)
    if args.gpus > 1 and args.worker_shard is None:
        # Top-level coordinator: spawn one worker per GPU.
        return _spawn_gpu_workers(args)

    inv_full = read_inventory(args.inventory)
    work = _filter_inventory(inv_full, sites=args.site, years=args.year, limit=args.limit)

    # Multi-GPU shard split: each worker takes every Nth row, modulo the
    # total worker count. Clean even split, no coordination required (resume
    # is handled by shard_is_ready).
    if args.gpus > 1 and args.worker_shard is not None:
        work = work.iloc[args.worker_shard :: args.gpus].reset_index(drop=True)

    print(
        f"Inventory: {len(inv_full):,} total rows; selecting {len(work):,} for "
        f"this run (sites={args.site}, years={args.year}, limit={args.limit}, "
        f"shard={args.worker_shard}/{args.gpus})"
    )
    if work.empty:
        print("Nothing to do.")
        return

    if args.device is not None:
        device = torch.device(args.device)
    elif args.worker_shard is not None:
        device = torch.device(f"cuda:{args.worker_shard}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = load_simclr(args.checkpoint, device=device)

    n_emb = n_skip = n_fail = 0
    t0 = time.time()
    last_print = t0

    def progress(i, total, evt, row):
        nonlocal n_emb, n_skip, n_fail, last_print
        if evt == "embedded":
            n_emb += 1
        elif evt == "skipped":
            n_skip += 1
        elif evt == "failed":
            n_fail += 1
        now = time.time()
        if now - last_print >= 5.0 or i + 1 == total:
            elapsed = now - t0
            rate = (n_emb + n_fail) / max(elapsed, 1e-6)
            remaining = total - (i + 1)
            eta = remaining / max(rate, 1e-6) if rate > 0 else float("inf")
            print(
                f"[{i + 1}/{total}] embedded={n_emb} skipped={n_skip} failed={n_fail} "
                f"  {rate:.2f} CDF/s  eta {eta / 60:.1f} min  "
                f"({row.site} {row.datetime})"
            )
            last_print = now

    # Per-shard failures path so multi-GPU workers don't stomp each other.
    failures_out = args.failures_out
    if args.worker_shard is not None and failures_out is not None:
        failures_out = failures_out.with_name(
            failures_out.stem + f".shard{args.worker_shard}of{args.gpus}.parquet"
        )

    updated, failures = embed_inventory(
        work,
        model,
        args.out_dir,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        failures_path=failures_out if failures_out else None,
        on_progress=progress,
    )

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed / 60:.1f} min. "
        f"embedded={n_emb} skipped={n_skip} failed={n_fail}"
    )

    if not args.no_write_back:
        # Multi-GPU workers must not write the full inventory concurrently;
        # only the coordinator does it after join().
        if args.worker_shard is None:
            keymap = dict(
                zip(zip(updated["site"], updated["datetime"]), updated["status"])
            )
            new_status = [
                keymap.get((s, d), st)
                for s, d, st in zip(
                    inv_full["site"], inv_full["datetime"], inv_full["status"]
                )
            ]
            inv_full = inv_full.copy()
            inv_full["status"] = pd.Series(new_status, dtype="string")
            write_inventory(inv_full, args.inventory)
            print(f"Updated status persisted to {args.inventory}")

    if not failures.empty and failures_out:
        print(f"{len(failures)} failures recorded at {failures_out}")


def _spawn_gpu_workers(args) -> None:
    """Run one worker process per GPU, each taking a stride of the inventory."""
    import subprocess
    import sys

    print(f"Spawning {args.gpus} worker processes (one per GPU).")

    base_cmd = [
        sys.executable,
        "-m",
        "themisim.embed",
        "--inventory",
        str(args.inventory),
        "--out-dir",
        str(args.out_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--gpus",
        str(args.gpus),
    ]
    for site in args.site or []:
        base_cmd += ["--site", site]
    for year in args.year or []:
        base_cmd += ["--year", year]
    if args.limit is not None:
        base_cmd += ["--limit", str(args.limit)]
    if args.no_resume:
        base_cmd.append("--no-resume")
    if args.failures_out:
        base_cmd += ["--failures-out", str(args.failures_out)]
    # Workers MUST not write back the inventory; only the coordinator does
    # that after they finish, to avoid concurrent parquet writes.
    base_cmd.append("--no-write-back")

    procs = []
    for k in range(args.gpus):
        cmd = base_cmd + ["--worker-shard", str(k)]
        print(f"  launching shard {k}: {' '.join(cmd)}")
        procs.append(subprocess.Popen(cmd))

    rc_all = 0
    for k, p in enumerate(procs):
        rc = p.wait()
        print(f"  shard {k} exited rc={rc}")
        rc_all = rc_all or rc

    if rc_all != 0:
        sys.exit(rc_all)

    # Coordinator phase: re-derive status from shard files (source of truth)
    # and write the inventory back once.
    if not args.no_write_back:
        from themisim.inventory import read_inventory, write_inventory

        inv_full = read_inventory(args.inventory)
        statuses = []
        for s, d, p_, st in zip(
            inv_full["site"],
            inv_full["datetime"],
            inv_full["drive_path"],
            inv_full["status"],
        ):
            if shard_is_ready(args.out_dir, p_):
                statuses.append("done")
            else:
                statuses.append(st)
        inv_full = inv_full.copy()
        inv_full["status"] = pd.Series(statuses, dtype="string")
        write_inventory(inv_full, args.inventory)
        print(f"Coordinator wrote merged status to {args.inventory}")


# ----------- Stage 10.5: streaming coordinator + per-drive worker ---------- #


def _spawn_drive_workers(args) -> None:
    """Stage 10.5 coordinator: one subprocess per non-empty drive, GPU
    assigned round-robin. Each worker calls _run_drive_worker on its
    own drive slice.

    Drive-failure isolation comes for free here: a fatal error in one
    drive's worker only kills its own process. The other drives keep
    making progress, and the coordinator collects exit codes at the
    end."""
    import subprocess
    import sys

    from themisim.inventory import read_inventory

    inv_full = read_inventory(args.inventory)
    work = _filter_inventory(
        inv_full, sites=args.site, years=args.year, limit=args.limit
    )
    groups = group_inventory_by_drive(work, drive_roots=args.drive_root)
    dropped = sum(len(g) for d, g in groups if not d)
    if dropped:
        print(
            f"[warn] {dropped:,} CDF(s) could not be assigned to a drive group "
            f"(parent unstattable / outside every --drive-root) and are skipped."
        )
    groups = [(d, g) for d, g in groups if d]  # skip "" sentinel
    if not groups:
        print("No CDFs to embed (after filtering).")
        return

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        f"Streaming coordinator: {len(groups)} drive(s) × "
        f"{n_gpus or 'cpu'} GPU(s). Compile: "
        f"{'off' if args.no_compile else 'on'}."
    )

    base_cmd = [
        sys.executable,
        "-m",
        "themisim.embed",
        "--inventory",
        str(args.inventory),
        "--out-dir",
        str(args.out_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--streaming",
    ]
    for site in args.site or []:
        base_cmd += ["--site", site]
    for year in args.year or []:
        base_cmd += ["--year", year]
    if args.limit is not None:
        base_cmd += ["--limit", str(args.limit)]
    if args.no_resume:
        base_cmd.append("--no-resume")
    if args.failures_out:
        base_cmd += ["--failures-out", str(args.failures_out)]
    if args.no_compile:
        base_cmd.append("--no-compile")
    for root in args.drive_root or []:
        base_cmd += ["--drive-root", root]
    base_cmd.append("--no-write-back")

    procs = []
    for k, (drive_root, group_df) in enumerate(groups):
        gpu_id = (k % n_gpus) if n_gpus > 0 else None
        device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
        cmd = base_cmd + [
            "--worker-drive",
            str(k),
            "--worker-device",
            device,
        ]
        print(
            f"  drive {k} {drive_root}: {len(group_df):,} CDFs on {device}"
        )
        procs.append((drive_root, subprocess.Popen(cmd)))

    rc_all = 0
    for drive_root, proc in procs:
        rc = proc.wait()
        print(f"  drive {drive_root}: exited rc={rc}")
        rc_all = rc_all or rc

    if rc_all != 0:
        # Don't exit early — surface the failure but still let the
        # coordinator merge the partial successes back into the
        # inventory.
        print(f"[warn] one or more drive workers exited non-zero ({rc_all})")

    if not args.no_write_back:
        from themisim.inventory import read_inventory, write_inventory

        inv_full = read_inventory(args.inventory)
        statuses = [
            "done" if shard_is_ready(args.out_dir, p_) else st
            for p_, st in zip(inv_full["drive_path"], inv_full["status"])
        ]
        inv_full = inv_full.copy()
        inv_full["status"] = pd.Series(statuses, dtype="string")
        write_inventory(inv_full, args.inventory)
        print(f"Coordinator wrote merged status to {args.inventory}")

    if rc_all != 0:
        sys.exit(rc_all)


def _run_drive_worker(args) -> None:
    """One process, one drive, one GPU. Loads the encoder (optionally
    torch.compile-d), runs `embed_inventory_streaming` on its drive's
    slice."""
    from themisim.inventory import read_inventory
    from themisim.model import load_simclr

    inv_full = read_inventory(args.inventory)
    work = _filter_inventory(
        inv_full, sites=args.site, years=args.year, limit=args.limit
    )
    groups = group_inventory_by_drive(work, drive_roots=args.drive_root)
    groups = [(d, g) for d, g in groups if d]
    if args.worker_drive >= len(groups):
        print(
            f"[drive worker {args.worker_drive}] no group at this index "
            f"(have {len(groups)} drives)"
        )
        return
    drive_root, drive_df = groups[args.worker_drive]
    if drive_df.empty:
        print(f"[drive worker {args.worker_drive}] empty slice for {drive_root}")
        return

    if args.worker_device:
        device = torch.device(args.worker_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[drive worker {args.worker_drive}] {drive_root}: "
        f"{len(drive_df):,} CDFs on {device}, compile="
        f"{'off' if args.no_compile else 'on'}"
    )

    encoder = load_simclr(args.checkpoint, device=device)
    use_compile = (not args.no_compile) and (device.type == "cuda")
    encoder = maybe_compile_encoder(encoder, use_compile=use_compile)

    # Per-drive failures path so concurrent workers don't stomp each other.
    failures_out = args.failures_out
    if failures_out is not None:
        failures_out = failures_out.with_name(
            failures_out.stem + f".drive{args.worker_drive}.parquet"
        )

    n_emb = n_skip = n_fail = 0
    t0 = time.time()
    last_print = t0

    def progress(i, total, evt, row):
        nonlocal n_emb, n_skip, n_fail, last_print
        if evt == "embedded":
            n_emb += 1
        elif evt == "skipped":
            n_skip += 1
        elif evt == "failed":
            n_fail += 1
        now = time.time()
        if now - last_print >= 5.0 or i + 1 == total:
            elapsed = now - t0
            rate = (n_emb + n_fail) / max(elapsed, 1e-6)
            remaining = total - (i + 1)
            eta = remaining / max(rate, 1e-6) if rate > 0 else float("inf")
            print(
                f"[drive {args.worker_drive}][{i + 1}/{total}] "
                f"emb={n_emb} skip={n_skip} fail={n_fail} "
                f"{rate:.2f} CDF/s eta {eta / 60:.1f} min "
                f"({getattr(row, 'site', '?')} {getattr(row, 'datetime', '?')})"
            )
            last_print = now

    embed_inventory_streaming(
        drive_df,
        encoder,
        args.out_dir,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pad_last_batch=use_compile,
        failures_path=failures_out if failures_out else None,
        on_progress=progress,
    )

    elapsed = time.time() - t0
    print(
        f"[drive {args.worker_drive}] done in {elapsed / 60:.1f} min — "
        f"emb={n_emb} skip={n_skip} fail={n_fail}"
    )


if __name__ == "__main__":
    _cli()
