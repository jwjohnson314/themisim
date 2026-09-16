"""Pinned, reproducible slices of the THEMIS archive ("pilot indexes").

The production index covers ~1M CDFs (>100 TB) and takes GPU-days to build,
which makes it impossible for a reviewer to reproduce or even smoke-test. A
*pilot* is a small, explicitly enumerated slice of the **real** archive that
builds on a CPU laptop in minutes and exercises every stage of the production
pipeline — download, inventory, embed, concat, train/add, query.

What makes a pilot reproducible is that a :class:`PilotSpec` pins an explicit
list of CDFs together with each file's size and SHA-256. Nothing is discovered
at build time: the archive crawler is not used at all (URLs are reconstructed
with :func:`themisim.export.themis_cdf_url`), so two reviewers a year apart
fetch byte-identical inputs even if the archive's directory listings change.

Specs ship in ``themisim/data/pilot_specs.json``. See :mod:`themisim.validate`
for the quality report a pilot build produces, and ``themis-pilot`` for the CLI.

Note on import order: this module is imported through the ``themisim`` package,
whose ``__init__`` imports torch before anything pulls in faiss. Do not import
it by a path that bypasses that.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from themisim.catalog import WORKLIST_COLUMNS
from themisim.export import themis_cdf_url

#: Packaged registry of frozen slice definitions.
SPECS_PATH = Path(__file__).with_name("data") / "pilot_specs.json"

#: Conservative reviewer-machine rates used for the up-front cost estimate.
#: Measured on a 24-core desktop: 17.5 MB/s single-stream from the archive and
#: 89 encoder frames/s on 8 CPU threads. These defaults are deliberately about
#: half that, so the printed estimate is an upper bound for most machines
#: rather than a best case.
EST_DOWNLOAD_MB_PER_S = 10.0
EST_EMBED_FRAMES_PER_S = 50.0

#: Frames in a complete THEMIS hour (3-second cadence). Only used for the
#: pre-download estimate; the realized count comes from the shards.
NOMINAL_FRAMES_PER_CDF = 1200


@dataclass(frozen=True)
class PilotCdf:
    """One pinned CDF: where it lives and what bytes it must contain."""

    site: str
    datetime: str  # YYYYMMDDHH
    size_bytes: int
    sha256: str

    @property
    def filename(self) -> str:
        return f"thg_l1_asf_{self.site}_{self.datetime}_v01.cdf"

    @property
    def url(self) -> str:
        return themis_cdf_url(self.site, self.datetime)

    def target_path(self, data_root: Union[str, Path]) -> Path:
        """Archive-mirroring location under ``data_root``.

        Matches :func:`themisim.catalog.build_worklist` exactly
        (``<data_root>/<site>/<YYYY>/<MM>/<filename>``) so a pilot tree and a
        crawled tree are interchangeable.
        """
        return (
            Path(data_root)
            / self.site
            / self.datetime[:4]
            / self.datetime[4:6]
            / self.filename
        )


@dataclass(frozen=True)
class PilotSpec:
    """A frozen slice definition."""

    name: str
    description: str
    cdfs: Tuple[PilotCdf, ...]
    #: IVF cell count, pinned so the build is reproducible. 0 means "not yet
    #: frozen" and defers to ``pipeline.auto_nlist``; ``--freeze`` records the
    #: value the first real build chose.
    nlist: int
    nprobe: int
    #: The other half of the operating point. Recall depends on the
    #: (nprobe, prefilter) pair, not on nprobe alone, so both are pinned.
    prefilter: int = 500
    thresholds: Dict[str, float] = field(default_factory=dict)
    #: Vector count measured when the spec was frozen; ``None`` until a first
    #: real build fills it in.
    expected_vectors: Optional[int] = None

    @property
    def n_files(self) -> int:
        return len(self.cdfs)

    @property
    def download_bytes(self) -> int:
        return sum(c.size_bytes for c in self.cdfs)

    @property
    def sites(self) -> List[str]:
        return sorted({c.site for c in self.cdfs})

    @property
    def year_months(self) -> List[str]:
        return sorted({c.datetime[:6] for c in self.cdfs})

    def estimate(
        self,
        *,
        mb_per_s: float = EST_DOWNLOAD_MB_PER_S,
        frames_per_s: float = EST_EMBED_FRAMES_PER_S,
    ) -> Dict[str, float]:
        """Rough wall-clock budget, in seconds, for a CPU-only reviewer run."""
        n_vec = self.expected_vectors or self.n_files * NOMINAL_FRAMES_PER_CDF
        download_s = self.download_bytes / 1e6 / mb_per_s
        embed_s = n_vec / frames_per_s
        # Concat is a linear copy of a few hundred MB; index train+add measured
        # at ~44 s for 12K vectors and scales roughly linearly at this size.
        index_s = 5.0 + n_vec * 44.0 / 12_000.0
        return {
            "download_s": download_s,
            "embed_s": embed_s,
            "index_s": index_s,
            "total_s": download_s + embed_s + index_s,
        }


@dataclass
class PilotResult:
    """What :func:`build_pilot` produced."""

    name: str
    data_root: Path
    artifacts: Path
    index_path: Path
    total_n: int
    n_shards: int
    timings: Dict[str, float]
    report: Optional[dict] = None

    @property
    def ok(self) -> bool:
        """True when no validation ran, or validation passed every threshold."""
        return self.report is None or bool(self.report.get("passed", False))


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_SPEC_CACHE: Optional[Dict[str, PilotSpec]] = None


def _spec_from_json(name: str, raw: dict) -> PilotSpec:
    cdfs = tuple(
        PilotCdf(
            site=c["site"],
            datetime=c["datetime"],
            size_bytes=int(c["size_bytes"]),
            sha256=c["sha256"],
        )
        for c in raw["cdfs"]
    )
    return PilotSpec(
        name=name,
        description=raw["description"],
        cdfs=cdfs,
        nlist=int(raw["nlist"]),
        nprobe=int(raw["nprobe"]),
        prefilter=int(raw.get("prefilter", 500)),
        thresholds=dict(raw.get("thresholds", {})),
        expected_vectors=raw.get("expected_vectors"),
    )


def _load_specs(path: Union[str, Path, None] = None) -> Dict[str, PilotSpec]:
    global _SPEC_CACHE
    if path is None and _SPEC_CACHE is not None:
        return _SPEC_CACHE
    p = Path(path) if path is not None else SPECS_PATH
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    specs = {name: _spec_from_json(name, body) for name, body in raw.items()}
    if path is None:
        _SPEC_CACHE = specs
    return specs


def list_pilots() -> List[PilotSpec]:
    """Every registered pilot spec, smallest first."""
    return sorted(_load_specs().values(), key=lambda s: s.n_files)


def get_pilot(name: str) -> PilotSpec:
    """Look up one spec by name, with an informative error if it's unknown."""
    specs = _load_specs()
    try:
        return specs[name]
    except KeyError:
        raise KeyError(
            f"unknown pilot spec {name!r}; available: {sorted(specs)}"
        ) from None


def _as_spec(spec: Union[str, PilotSpec]) -> PilotSpec:
    return get_pilot(spec) if isinstance(spec, str) else spec


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def default_pilot_root(name: str) -> Path:
    """Isolated root for one pilot: ``./data/pilot/<name>``.

    Isolation is load-bearing, not tidiness. ``concat.discover_shards`` is an
    unfiltered ``rglob("*.f16.npy")`` with no allow-list, so a pilot sharing an
    artifacts directory with any other build would silently absorb that build's
    shards and produce an index that is not the pinned slice.
    """
    return Path.cwd() / "data" / "pilot" / name


def pilot_paths(
    name: str,
    data_root: Union[str, Path, None] = None,
    artifacts: Union[str, Path, None] = None,
) -> Tuple[Path, Path]:
    """Resolve ``(data_root, artifacts)`` for a pilot, defaulting to isolation."""
    root = default_pilot_root(name)
    return (
        Path(data_root) if data_root is not None else root / "cdf",
        Path(artifacts) if artifacts is not None else root / "artifacts",
    )


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def worklist_for(spec: Union[str, PilotSpec], data_root: Union[str, Path]) -> pd.DataFrame:
    """The download work-list for a spec, in ``catalog.WORKLIST_COLUMNS`` shape.

    Built directly from the pinned CDF list — the archive crawler is not used,
    which is what makes the input set reproducible.
    """
    spec = _as_spec(spec)
    rows = [
        {
            "site": c.site,
            "datetime": c.datetime,
            "filename": c.filename,
            "url": c.url,
            "target_path": str(c.target_path(data_root)),
            "size_bytes": c.size_bytes,
        }
        for c in spec.cdfs
    ]
    return pd.DataFrame(rows, columns=WORKLIST_COLUMNS)


def download_pilot(
    spec: Union[str, PilotSpec],
    data_root: Union[str, Path, None] = None,
    *,
    workers: int = 4,
    on_event: Optional[Callable] = None,
) -> pd.DataFrame:
    """Fetch exactly the CDFs a spec pins. Resumable; returns the results frame."""
    from themisim import download

    spec = _as_spec(spec)
    if data_root is None:
        data_root, _ = pilot_paths(spec.name)
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    wl_path = data_root / "_pilot_worklist.parquet"
    worklist_for(spec, data_root).to_parquet(wl_path, index=False)
    return download.run(wl_path, workers=workers, on_event=on_event)


def verify_slice(
    spec: Union[str, PilotSpec],
    data_root: Union[str, Path, None] = None,
    *,
    checksums: bool = True,
) -> dict:
    """Check the downloaded tree is exactly the pinned slice.

    Sizes are always checked (cheap); ``checksums`` additionally SHA-256s every
    file, which is what actually proves the reviewer holds the same bytes the
    spec was frozen against. Hashing runs at roughly 1 GB/s, so even the larger
    slice costs a few seconds.

    Returns a report dict; ``report["ok"]`` is True when nothing is missing,
    mis-sized or mis-hashed.
    """
    from themisim.weights import sha256_file

    spec = _as_spec(spec)
    if data_root is None:
        data_root, _ = pilot_paths(spec.name)

    missing: List[str] = []
    wrong_size: List[dict] = []
    wrong_hash: List[str] = []
    for c in spec.cdfs:
        p = c.target_path(data_root)
        if not p.exists():
            missing.append(c.filename)
            continue
        actual = p.stat().st_size
        if actual != c.size_bytes:
            wrong_size.append(
                {"file": c.filename, "expected": c.size_bytes, "actual": actual}
            )
            continue
        if checksums and sha256_file(p) != c.sha256:
            wrong_hash.append(c.filename)

    return {
        "n_files": spec.n_files,
        "checksums_verified": bool(checksums),
        "missing": missing,
        "wrong_size": wrong_size,
        "wrong_hash": wrong_hash,
        "ok": not (missing or wrong_size or wrong_hash),
    }


# --------------------------------------------------------------------------- #
# pre-flight / post-flight guards
# --------------------------------------------------------------------------- #
def preflight(
    spec: Union[str, PilotSpec],
    data_root: Union[str, Path],
    artifacts: Union[str, Path],
    *,
    need_download: bool = True,
) -> None:
    """Refuse to start a build that would not produce the pinned slice.

    Isolated directories are the first line of defence, but they are advisory:
    a reviewer can point ``--data-root`` at a tree they already have, and
    ``build_index`` will inventory it wholesale (``build_inventory`` rglobs
    ``thg_l1_asf_*_v01.cdf`` with no filter) while ``concat.discover_shards``
    rglobs every shard under the artifacts directory. Either would quietly
    produce a valid index over the wrong data. Catching it here, before the
    download, is far kinder than a count mismatch an hour later.

    Raises ``RuntimeError`` on contamination or insufficient disk space.
    """
    import shutil

    from themisim.inventory import build_inventory

    spec = _as_spec(spec)
    data_root, artifacts = Path(data_root), Path(artifacts)
    pinned = {(c.site, c.datetime) for c in spec.cdfs}

    if need_download:
        target = data_root if data_root.exists() else data_root.parent
        while not target.exists() and target != target.parent:
            target = target.parent
        free = shutil.disk_usage(target).free
        needed = int(spec.download_bytes * 1.15)
        if free < needed:
            raise RuntimeError(
                f"need ~{needed / 1e9:.1f} GB free under {data_root} for pilot "
                f"{spec.name!r} but only {free / 1e9:.1f} GB is available"
            )

    if data_root.exists():
        inv = build_inventory(drives=[data_root])
        present = {(r.site, r.datetime) for r in inv.itertuples()}
        extra = present - pinned
        if extra:
            sample = ", ".join(f"{s}/{d}" for s, d in sorted(extra)[:5])
            raise RuntimeError(
                f"{data_root} holds {len(extra)} CDF(s) the {spec.name!r} spec "
                f"does not pin (e.g. {sample}). build_index inventories the whole "
                "tree, so the index would not be this slice. Use the isolated "
                f"default ({default_pilot_root(spec.name) / 'cdf'}) or an empty "
                "--data-root."
            )

    shards_dir = artifacts / "shards"
    if shards_dir.exists():
        found = {
            (p.parent.parent.name, p.name.replace(".f16.npy", ""))
            for p in shards_dir.rglob("*.f16.npy")
        }
        extra = found - pinned
        if extra:
            sample = ", ".join(f"{s}/{d}" for s, d in sorted(extra)[:5])
            raise RuntimeError(
                f"{shards_dir} holds {len(extra)} shard(s) outside the "
                f"{spec.name!r} spec (e.g. {sample}). concat globs every shard "
                "under this directory, so the index would not be this slice. "
                f"Use the isolated default ({default_pilot_root(spec.name) / 'artifacts'})."
            )


def read_embed_failures(artifacts: Union[str, Path]) -> "pd.DataFrame | None":
    """The per-CDF failures ``build_index`` logs but does not raise on.

    ``embed_inventory_streaming`` catches per-CDF exceptions, records them and
    keeps going, and ``build_index`` discards its return value. A slice where a
    third of the CDFs failed to read therefore produces a perfectly valid index
    over the wrong data, silently. Surfacing this file is how the pilot turns
    that into something a reader can see.
    """
    path = Path(artifacts) / "embed_failures.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df if len(df) else None


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_pilot(
    spec: Union[str, PilotSpec],
    *,
    data_root: Union[str, Path, None] = None,
    artifacts: Union[str, Path, None] = None,
    checkpoint: Union[str, Path, None] = None,
    device: str = "auto",
    batch_size: int = 128,
    num_workers: int = 4,
    workers: int = 4,
    download: bool = True,
    verify_checksums: bool = True,
    strict_checksums: bool = False,
    validate: bool = True,
    on_progress: Optional[Callable[[str, object], None]] = None,
) -> PilotResult:
    """Download, build and (by default) validate one pilot slice.

    Delegates the actual build to :func:`themisim.pipeline.build_index`
    unchanged — the point of a pilot is to exercise the production path, not a
    parallel one. What this adds is the pinned input set, isolated output
    directories, a post-build check that the realized artifacts match the spec,
    and the validation report.

    ``checkpoint=None`` fetches and SHA-256-verifies the encoder weights.
    """
    spec = _as_spec(spec)
    data_root, artifacts = pilot_paths(spec.name, data_root, artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}

    def report(stage: str, info: object) -> None:
        if on_progress is not None:
            on_progress(stage, info)

    preflight(spec, data_root, artifacts, need_download=download)
    report("preflight", f"{data_root} and {artifacts} hold only this slice")

    if download:
        t0 = time.time()
        res = download_pilot(spec, data_root, workers=workers)
        timings["download_s"] = time.time() - t0
        failed = res[res["status"].isin(["failed", "stopped"])] if len(res) else res
        if len(failed):
            raise RuntimeError(
                f"{len(failed)} of {spec.n_files} pilot CDFs did not download; "
                f"first error: {failed.iloc[0].get('error', '')!r}"
            )
        report("download", f"{spec.n_files} CDFs in {timings['download_s']:.0f}s")

    t0 = time.time()
    verification = verify_slice(spec, data_root, checksums=verify_checksums)
    timings["verify_s"] = time.time() - t0
    if verification["missing"]:
        raise RuntimeError(
            f"{len(verification['missing'])} of {spec.n_files} pinned CDFs are "
            f"missing under {data_root}: "
            f"{', '.join(verification['missing'][:5])}"
        )
    drift = verification["wrong_size"] + verification["wrong_hash"]
    if drift:
        # The archive occasionally reprocesses an hour in place. Failing hard on
        # that would make a published spec permanently unbuildable the day it
        # happened, which is the opposite of what pinning is for: the pin exists
        # to make divergence *visible*, not to make the tool brittle. Warn
        # loudly, record it in the report so a reviewer can interpret an
        # encoder-cosine deviation, and let --strict-checksums escalate when
        # exact byte reproduction is the actual goal.
        detail = (
            f"{len(verification['wrong_size'])} wrong size, "
            f"{len(verification['wrong_hash'])} wrong checksum"
        )
        if strict_checksums:
            raise RuntimeError(
                f"downloaded tree does not match the pinned slice ({detail}); "
                "the archive may have reprocessed these hours"
            )
        report(
            "warning",
            f"{detail} vs the frozen spec - the archive appears to have "
            "reprocessed these hours, so results may differ from the pinned "
            "reference vectors",
        )
    report(
        "verify",
        f"{spec.n_files} CDFs checked"
        + (" (sha256)" if verify_checksums else " (size only)"),
    )

    if checkpoint is None:
        from themisim.weights import fetch_weights

        checkpoint = fetch_weights()
        report("weights", str(checkpoint))

    from themisim.pipeline import build_index, resolve_device

    dev = resolve_device(device)
    t0 = time.time()
    index_path = build_index(
        data_root,
        artifacts,
        checkpoint,
        nlist=spec.nlist if spec.nlist > 0 else "auto",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        # Tie the FAISS coarse quantizer to the requested device. Left at its
        # default, a `--device cpu` run still ran k-means on the GPU, which
        # makes the reported CPU wall clock an understatement of what a
        # CPU-only reviewer actually pays.
        use_gpu_quantizer=(dev != "cpu"),
        on_progress=on_progress,
    )
    timings["build_s"] = time.time() - t0

    meta = json.loads((artifacts / "vectors.meta.json").read_text())
    total_n, n_shards = int(meta["shape"][0]), int(meta["n_shards"])

    failures = read_embed_failures(artifacts)
    if failures is not None:
        for row in failures.itertuples():
            report(
                "embed_failure",
                f"{row.site}/{row.datetime}: {row.exception_type}: {row.exception_msg}",
            )

    # A shard-count mismatch means the build swept in files the spec does not
    # pin (or missed some) — the index is not this slice, so refuse it. A
    # vector-count mismatch is only a warning: the archive occasionally
    # reprocesses an hour with a different frame count.
    if n_shards != spec.n_files:
        detail = (
            f" {len(failures)} CDF(s) failed to embed (see "
            f"{artifacts / 'embed_failures.parquet'})."
            if failures is not None
            else ""
        )
        raise RuntimeError(
            f"built index covers {n_shards} shards but spec {spec.name!r} pins "
            f"{spec.n_files} CDFs.{detail}"
        )
    if spec.expected_vectors is not None and total_n != spec.expected_vectors:
        report(
            "warning",
            f"{total_n:,} vectors != {spec.expected_vectors:,} recorded when the "
            "spec was frozen (an archive hour may have been reprocessed)",
        )

    result = PilotResult(
        name=spec.name,
        data_root=data_root,
        artifacts=artifacts,
        index_path=index_path,
        total_n=total_n,
        n_shards=n_shards,
        timings=timings,
    )

    if validate:
        from themisim.validate import validate_index

        t0 = time.time()
        result.report = validate_index(
            artifacts,
            nprobe=spec.nprobe,
            prefilter=spec.prefilter,
            thresholds=spec.thresholds,
            spec_name=spec.name,
            timings=timings,
            build_device=dev,
        )
        # Record any divergence from the frozen bytes in the report itself, so
        # the JSON a reviewer files alongside the paper says whether their
        # inputs were the ones the spec was calibrated against.
        result.report["slice_verification"] = verification
        timings["validate_s"] = time.time() - t0

    return result
