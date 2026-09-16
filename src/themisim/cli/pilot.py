"""``themis-pilot`` — build and validate a small, pinned slice of the archive.

A pilot index is a reproducible subset of the real THEMIS archive, small enough
to download and build on a CPU laptop, that exercises the same pipeline the
full-archive build uses and finishes by measuring its own retrieval quality. It
exists so the software can be assessed without the ~100 TB archive or the
GPU-days a full build costs.

Examples::

    themis-pilot --list
    themis-pilot --spec tiny
    themis-pilot --spec small --report pilot_report.json
    themis-pilot --spec small --validate-only

Exits non-zero if any validation threshold is not met.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _fmt_bytes(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _fmt_mins(seconds: float) -> str:
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def _print_catalogue() -> None:
    from themisim.pilot import list_pilots

    print("Available pilot specs:\n")
    for spec in list_pilots():
        est = spec.estimate()
        print(f"  {spec.name}")
        print(f"    {spec.description}")
        print(
            f"    {spec.n_files} CDFs, {_fmt_bytes(spec.download_bytes)} download, "
            f"sites {', '.join(spec.sites)}, months {', '.join(spec.year_months)}"
        )
        vec = f"{spec.expected_vectors:,}" if spec.expected_vectors else "~unknown"
        print(f"    {vec} vectors, nlist={spec.nlist}")
        print(
            f"    estimated {_fmt_mins(est['total_s'])} on a CPU laptop "
            f"(download {_fmt_mins(est['download_s'])}, "
            f"embed {_fmt_mins(est['embed_s'])}, "
            f"index {_fmt_mins(est['index_s'])})"
        )
        print()


def _parse_cdf_list(value: str) -> list[tuple[str, str]]:
    """``fsmi:2015031806,gill:2015031807`` -> [(site, datetime), ...]."""
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        site, _, dt = item.partition(":")
        if not dt:
            raise ValueError(f"expected <site>:<YYYYMMDDHH>, got {item!r}")
        out.append((site.strip().lower(), dt.strip()))
    return out


def _discover(args) -> int:
    """Maintainer path: turn a candidate CDF list into a frozen spec entry.

    Downloads each candidate (a SHA-256 needs the bytes anyway), records its
    true size and digest, and writes the spec entry. ``expected_vectors`` and
    the calibrated thresholds are filled in later by ``--freeze``, which needs a
    real build to measure them.
    """
    import pandas as pd

    from themisim import download
    from themisim.catalog import WORKLIST_COLUMNS
    from themisim.export import themis_cdf_url
    from themisim.pilot import pilot_paths
    from themisim.weights import sha256_file

    cdfs = _parse_cdf_list(args.cdfs)
    data_root, _ = pilot_paths(args.spec, args.data_root, args.artifacts)
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"Discovering {len(cdfs)} CDFs into {data_root} ...", flush=True)

    rows = [
        {
            "site": site,
            "datetime": dt,
            "filename": f"thg_l1_asf_{site}_{dt}_v01.cdf",
            "url": themis_cdf_url(site, dt),
            "target_path": str(
                data_root / site / dt[:4] / dt[4:6] / f"thg_l1_asf_{site}_{dt}_v01.cdf"
            ),
            "size_bytes": 0,
        }
        for site, dt in cdfs
    ]
    wl = data_root / "_discover_worklist.parquet"
    pd.DataFrame(rows, columns=WORKLIST_COLUMNS).to_parquet(wl, index=False)
    res = download.run(wl, workers=args.workers)
    bad = res[res["status"].isin(["failed", "stopped"])]
    if len(bad):
        print(f"ERROR: {len(bad)} candidates could not be downloaded:")
        for _, r in bad.iterrows():
            print(f"  {r['url']}  {r['error']}")
        return 1

    entries = []
    for row in rows:
        p = Path(row["target_path"])
        entries.append(
            {
                "site": row["site"],
                "datetime": row["datetime"],
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
        print(f"  {p.name}  {entries[-1]['size_bytes'] / 1e6:.1f} MB  {entries[-1]['sha256'][:16]}...")

    specs_path = Path(args.specs_file)
    specs = json.loads(specs_path.read_text()) if specs_path.exists() else {}
    existing = specs.get(args.spec, {})
    specs[args.spec] = {
        "description": args.description or existing.get("description", ""),
        "nlist": existing.get("nlist", 0),
        "nprobe": existing.get("nprobe", 64),
        "prefilter": existing.get("prefilter", 500),
        "expected_vectors": existing.get("expected_vectors"),
        "thresholds": existing.get("thresholds", {}),
        "cdfs": entries,
    }
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    specs_path.write_text(json.dumps(specs, indent=2) + "\n")
    print(f"\nWrote spec {args.spec!r} ({len(entries)} CDFs) to {specs_path}")
    print("Now run with --freeze to measure expected_vectors and calibrate thresholds.")
    return 0


def _freeze(args, result) -> None:
    """Maintainer path: record measured values and calibrated gates in the spec.

    Thresholds are set just below what a real build actually achieved, so they
    catch regressions without being a guess. Encoder-reproducibility gates stay
    fixed — they are about numerical agreement, not about this slice.
    """
    import numpy as np

    from themisim.pipeline import auto_nlist

    report = result.report
    specs_path = Path(args.specs_file)
    specs = json.loads(specs_path.read_text())
    entry = specs[args.spec]

    sr = report["checks"]["self_retrieval"]
    ret = report["checks"]["retrieval"].get("at_operating_point")
    thresholds = {
        # The exact path is deterministic, so it is gated at exactly 1.0 rather
        # than at "what we happened to measure".
        "exact_self_retrieval": 1.0,
        "ivf_self_retrieval": round(max(0.80, sr["ivf_path_fraction"] - 0.05), 3),
        "encoder_cosine_mean": 0.9999,
        "encoder_cosine_min": 0.9999,
    }
    if ret is not None:
        gate = round(max(0.50, ret["pipeline_recall_at_k"] - 0.05), 3)
        thresholds[f"recall_at_{report['checks']['retrieval']['k']}"] = gate
        thresholds["recall_at_10"] = gate

    entry["expected_vectors"] = int(result.total_n)
    entry["nlist"] = int(report["checks"]["integrity"]["nlist"])
    entry["prefilter"] = int(report["operating_point"]["prefilter"])
    entry["thresholds"] = thresholds
    specs_path.write_text(json.dumps(specs, indent=2) + "\n")
    print(f"\nFroze spec {args.spec!r}: expected_vectors={result.total_n:,}, "
          f"thresholds={thresholds}")

    # Pin a handful of vectors from this build so a reviewer can prove their
    # encoder path reproduces ours without re-embedding anything.
    if args.reference_out:
        from themisim.search import SearchEngine

        eng = SearchEngine(result.artifacts)
        rng = np.random.default_rng(0)
        gids = np.sort(rng.choice(eng.total_n, size=min(50, eng.total_n), replace=False))
        np.savez_compressed(
            args.reference_out,
            site=np.array([eng.site_of(int(g)) for g in gids]),
            datetime=np.array(
                [Path(eng.shard_of(int(g))).name.replace(".f16.npy", "") for g in gids]
            ),
            frame_idx=np.array([eng.frame_idx_of(int(g)) for g in gids], dtype=np.int32),
            vectors=np.asarray(eng.memmap[gids], dtype=np.float16),
        )
        print(f"Wrote {len(gids)} reference vectors to {args.reference_out}")


def _render_figure(result, out_path: Path) -> None:
    """Contact sheet for one query, rendered from the pilot's own CDFs.

    ``visualize_results`` reads frames back out of the local CDF tree
    (``download=False``), so this is fully offline. A grid of visually similar
    aurora is the most direct evidence that the embedding captures morphology
    rather than pixel identity -- which no recall number can show.
    """
    try:
        from themisim.viz import visualize_results
    except ImportError as exc:
        print(f"\n[figure] skipped: {exc}. Install with: pip install 'themisim[notebook]'")
        return

    from themisim.query import query
    from themisim.search import SearchEngine

    eng = SearchEngine(result.artifacts)
    gid = result.total_n // 2
    site = eng.site_of(gid)
    dtstr = Path(eng.shard_of(gid)).name.replace(".f16.npy", "")
    frame = eng.frame_idx_of(gid)
    # A 10-minute diversification window instead of the 30-second default. At
    # the 3-second cadence, adjacent frames are near-duplicates of each other,
    # so a default query fills the sheet with the query's own minute and
    # demonstrates temporal autocorrelation rather than visual similarity.
    # Widening the window forces matches from other hours and other stations,
    # which is the claim the figure is supposed to support.
    df = query(
        site, dtstr, frame, engine=eng, results=12,
        diversify_seconds=600, prefilter=2000,
    )
    try:
        fig = visualize_results(
            df,
            data_root=result.data_root,
            download=False,
            title=f"pilot '{result.name}': frames most similar to {site} {dtstr} frame {frame}",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        print(f"\n[figure] wrote {out_path}")
    except Exception as exc:  # rendering is a nicety, never a build failure
        print(f"\n[figure] skipped: {type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--list", action="store_true", help="show the available specs and exit")
    p.add_argument("--spec", default=None, help="which pilot to build, e.g. 'tiny'")
    p.add_argument("--data-root", type=Path, default=None,
                   help="CDF tree (default: ./data/pilot/<spec>/cdf)")
    p.add_argument("--artifacts", type=Path, default=None,
                   help="index output (default: ./data/pilot/<spec>/artifacts)")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="SimCLR checkpoint .tar (default: fetch/verify into the weights dir)")
    p.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'")
    p.add_argument("--workers", type=int, default=4, help="parallel downloads")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--skip-download", action="store_true",
                   help="assume the CDFs are already present")
    p.add_argument("--no-verify-checksums", action="store_true",
                   help="check file sizes only, not SHA-256")
    p.add_argument("--strict-checksums", action="store_true",
                   help="fail if any CDF's bytes differ from the frozen spec "
                        "(default: warn, since the archive may reprocess an hour)")
    p.add_argument("--no-validate", action="store_true", help="build without the quality report")
    p.add_argument("--validate-only", action="store_true",
                   help="re-run validation against an existing pilot build")
    p.add_argument("--report", type=Path, default=None,
                   help="where to write pilot_report.json (default: under --artifacts)")
    p.add_argument("--figure", type=Path, default=None,
                   help="also render a contact sheet of one query's results "
                        "(needs matplotlib: pip install 'themisim[notebook]')")
    # Maintainer-only: freeze a new slice into the packaged spec registry.
    p.add_argument("--discover", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--freeze", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--cdfs", default=None, help=argparse.SUPPRESS)
    p.add_argument("--description", default=None, help=argparse.SUPPRESS)
    p.add_argument("--reference-out", type=Path, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--specs-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pilot_specs.json",
        help=argparse.SUPPRESS,
    )
    args = p.parse_args(argv)

    # Imported here so `themis-pilot --help` doesn't pay the torch/faiss cost.
    if args.list or (args.spec is None and not args.discover):
        _print_catalogue()
        return 0 if args.list else 2

    if args.discover:
        if not args.cdfs:
            p.error("--discover requires --cdfs")
        return _discover(args)

    from themisim.pilot import get_pilot, pilot_paths
    from themisim.validate import format_report, validate_index, write_report

    spec = get_pilot(args.spec)
    data_root, artifacts = pilot_paths(spec.name, args.data_root, args.artifacts)
    report_path = args.report or (artifacts / "pilot_report.json")

    if args.validate_only:
        report = validate_index(
            artifacts, nprobe=spec.nprobe, thresholds=spec.thresholds, spec_name=spec.name
        )
        print(format_report(report))
        write_report(report, report_path)
        print(f"\nWrote {report_path}")
        return 0 if report["passed"] else 1

    est = spec.estimate()
    print(f"=== pilot '{spec.name}' ===")
    print(f"  {spec.description}")
    print(f"  {spec.n_files} CDFs ({_fmt_bytes(spec.download_bytes)}) -> {data_root}")
    print(f"  artifacts -> {artifacts}")
    print(
        f"  estimated {_fmt_mins(est['total_s'])} on a CPU laptop "
        f"(download {_fmt_mins(est['download_s'])}, embed {_fmt_mins(est['embed_s'])}, "
        f"index {_fmt_mins(est['index_s'])})"
    )
    print()

    t0 = time.time()

    def progress(stage: str, info: object) -> None:
        print(f"  [{stage}] {info}  (+{time.time() - t0:.0f}s)", flush=True)

    from themisim.pilot import build_pilot

    result = build_pilot(
        spec,
        data_root=data_root,
        artifacts=artifacts,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        workers=args.workers,
        download=not args.skip_download,
        verify_checksums=not args.no_verify_checksums,
        strict_checksums=args.strict_checksums,
        validate=not args.no_validate,
        on_progress=progress,
    )

    print(f"\nBuilt {result.index_path} "
          f"({result.index_path.stat().st_size / 1e6:.1f} MB, "
          f"{result.total_n:,} vectors from {result.n_shards} CDFs) "
          f"in {_fmt_mins(time.time() - t0)}")

    if result.report is not None:
        print()
        print(format_report(result.report))
        write_report(result.report, report_path)
        print(f"\nWrote {report_path}")

    if args.figure is not None:
        _render_figure(result, args.figure)

    if args.freeze and result.report is not None:
        _freeze(args, result)

    print(f"\nQuery it with:\n  themis-query --site {result.report['dataset']['sites'][0] if result.report else '<site>'} "
          f"--datetime <YYYY-MM-DDTHH> --frame 0 --artifacts {artifacts}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
