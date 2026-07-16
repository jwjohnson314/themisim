"""End-to-end build orchestration: download -> embed -> concat -> index.

These are the library-level functions behind the ``themis-download`` and
``themis-build-index`` CLIs. They are importable so the whole pipeline can be
driven from a script or notebook, not just the command line.

Import-order note: this module imports the torch-backed embed/model code before
the faiss-backed index/search code. torch and faiss both ship CUDA shared libs
and corrupt each other's loader if imported in the wrong order; importing torch
first is the safe order.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pandas as pd

from themisim import catalog, download
# torch-backed imports first (see module docstring).
from themisim.embed import (
    DEFAULT_BATCH_SIZE,
    embed_inventory,
    embed_inventory_streaming,
    maybe_compile_encoder,
)
from themisim.inventory import build_inventory
from themisim.model import load_simclr
# faiss-backed imports after torch.
from themisim.concat import build_memmap_and_manifest, open_memmap
from themisim.index import (
    DEFAULT_NLIST_FULL,
    DEFAULT_NLIST_PILOT,
    DEFAULT_TRAIN_TARGET,
    TRAIN_POINTS_PER_CENTROID,
    stratified_sample_from_parquet,
    train_and_build_index,
)


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if a GPU is visible, else ``"cpu"``.

    Any explicit value is returned unchanged.
    """
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def auto_nlist(total_n: int) -> int:
    """Pick an IVF cell count appropriate for ``total_n`` vectors.

    Full archive (~1B) -> 65536, the pilot range (>=1M) -> 4096, and smaller
    sets (tests, single-site builds) -> ~sqrt(n) so training stays feasible.
    """
    if total_n >= 100_000_000:
        return DEFAULT_NLIST_FULL
    if total_n >= 1_000_000:
        return DEFAULT_NLIST_PILOT
    return max(1, min(DEFAULT_NLIST_PILOT, int(math.sqrt(max(total_n, 1)))))


def download_archive(
    data_root: Union[str, Path],
    *,
    sites: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    workers: int = 6,
    limit: Optional[int] = None,
    cache_path: Optional[Union[str, Path]] = None,
    on_dir: Optional[Callable] = None,
    on_event: Optional[Callable] = None,
) -> pd.DataFrame:
    """Crawl the archive for the requested sites/dates and download the CDFs.

    Files land under ``<data_root>/<site>/<YYYY>/<MM>/``. Resumable: existing,
    correctly-sized files are skipped. Returns the per-file results frame.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_path) if cache_path else data_root / "_crawl_cache.parquet"

    crawl_df = catalog.crawl(
        sites, start=start, end=end, cache_path=cache, on_dir=on_dir
    )
    worklist = catalog.build_worklist(crawl_df, data_root)
    wl_path = data_root / "_worklist.parquet"
    worklist.to_parquet(wl_path, index=False)

    return download.run(wl_path, workers=workers, limit=limit, on_event=on_event)


def build_index(
    data_root: Union[str, Path],
    artifacts: Union[str, Path],
    checkpoint: Union[str, Path],
    *,
    nlist: Union[int, str] = "auto",
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 4,
    train_target: int = DEFAULT_TRAIN_TARGET,
    use_gpu_quantizer: bool = True,
    streaming: bool = True,
    use_compile: bool = True,
    on_progress: Optional[Callable[[str, object], None]] = None,
) -> Path:
    """Embed every downloaded CDF, concatenate, and build the FAISS index.

    Returns the path to ``index.faiss``. All three stages resume cleanly:
    already-embedded shards are skipped, and concat/index always rebuild from
    the current shard set.

    ``streaming`` (default) uses the persistent-DataLoader embed path
    (:func:`~themisim.embed.embed_inventory_streaming`) — workers are
    spawned once and stay warm, and (when ``use_compile`` and a GPU are present)
    the encoder is ``torch.compile``-d — which is dramatically faster than the
    eager per-CDF loop on a large archive. Set ``streaming=False`` for the eager
    loop. This runs the embed in a single process on one device; for a
    multi-GPU / multi-physical-drive full-archive build, pre-embed with the
    coordinator (``python -m themisim.embed --streaming``, which fans
    out one worker per GPU/drive) and then call this — the embed step will find
    every shard already on disk and skip straight to concat + index.
    """
    data_root = Path(data_root)
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    shards_dir = artifacts / "shards"

    def report(stage: str, info: object) -> None:
        if on_progress is not None:
            on_progress(stage, info)

    # 1. Inventory the downloaded CDFs (single data-root tree).
    inv = build_inventory(drives=[data_root])
    if inv.empty:
        raise FileNotFoundError(f"no THEMIS CDFs found under {data_root}")
    report("inventory", f"{len(inv):,} CDFs")

    # 2. Load the encoder (GPU if available, else CPU).
    dev = resolve_device(device)
    report("device", dev)
    model = load_simclr(checkpoint, device=dev)

    # 3. Embed -> per-hour fp16 shards (+ JSON sidecars). The streaming path
    # keeps DataLoader workers warm across CDFs (and torch.compiles the encoder
    # on GPU), a large speedup over the eager per-CDF loop; both write identical
    # shards and share the shard_is_ready resume key.
    failures_path = artifacts / "embed_failures.parquet"
    if streaming:
        compile_on = use_compile and dev == "cuda"
        enc = maybe_compile_encoder(model, use_compile=compile_on)
        report("embed_mode", f"streaming (compile={'on' if compile_on else 'off'})")
        embed_inventory_streaming(
            inv,
            enc,
            shards_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pad_last_batch=compile_on,
            failures_path=failures_path,
        )
    else:
        report("embed_mode", "eager")
        embed_inventory(
            inv,
            model,
            shards_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            failures_path=failures_path,
        )
    report("embedded", str(shards_dir))

    # 4. Concatenate shards -> memmap + manifest + meta.
    memmap_path, manifest_path, _meta_path, total_n = build_memmap_and_manifest(
        shards_dir, artifacts
    )
    report("concat", f"{total_n:,} vectors")

    # 5. Train + build the index.
    nl = auto_nlist(total_n) if nlist == "auto" else int(nlist)
    target = max(train_target, TRAIN_POINTS_PER_CENTROID * nl)
    train_ids = stratified_sample_from_parquet(
        manifest_path, target_n=target, seed=0, total_n_check=total_n
    )
    mm = open_memmap(memmap_path, total_n)
    out_path = artifacts / "index.faiss"
    train_and_build_index(
        mm,
        out_path=out_path,
        nlist=nl,
        train_ids=train_ids,
        use_gpu_quantizer=use_gpu_quantizer,
        on_progress=on_progress,
    )
    report("index", str(out_path))
    return out_path
