"""``themis-query`` — similarity search by ``(site, datetime, frame)``.

Writes (or prints) a CSV identical to the dashboard's export:
``site, datetime, score, source_cdf``.

Example::

    themis-query --site fsmi --datetime 2015-03-18T06 --frame 412 \\
        --artifacts ./data/artifacts --output results.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from themisim.config import default_artifacts_root
from themisim.search import (
    DEFAULT_DIVERSIFY_SECONDS,
    DEFAULT_K,
    DEFAULT_NPROBE,
    DEFAULT_PREFILTER,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--site", required=True, help="4-letter site code, e.g. fsmi")
    p.add_argument(
        "--datetime",
        required=True,
        help="hourly CDF, e.g. 2015-03-18T06 (sub-hour fields ignored)",
    )
    p.add_argument("--frame", type=int, required=True, help="0-based frame in the hour")
    p.add_argument("--artifacts", type=Path, default=default_artifacts_root())
    p.add_argument(
        "--results",
        type=int,
        default=DEFAULT_K,
        help="count mode: number of top matches to return (ignored with --min-score)",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="threshold mode: return every match with cosine similarity >= this "
        "cutoff (0..1), best first, instead of a fixed --results count",
    )
    p.add_argument("--prefilter", type=int, default=DEFAULT_PREFILTER)
    p.add_argument("--nprobe", type=int, default=DEFAULT_NPROBE)
    p.add_argument("--diversify-seconds", type=int, default=DEFAULT_DIVERSIFY_SECONDS)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path. Default: auto-named in the cwd; '-' prints to stdout",
    )
    args = p.parse_args(argv)

    from themisim.export import results_csv_filename
    from themisim.query import get_engine, query, resolve_global_id

    engine = get_engine(args.artifacts)
    # Resolve once so we can name the output file the way the dashboard does.
    gid = resolve_global_id(engine, args.site, args.datetime, args.frame)
    df = query(
        args.site,
        args.datetime,
        args.frame,
        results=args.results,
        min_score=args.min_score,
        prefilter=args.prefilter,
        nprobe=args.nprobe,
        diversify_seconds=args.diversify_seconds,
        engine=engine,
    )

    if str(args.output) == "-":
        print(df.to_csv(index=False))
        return 0

    if args.output is None:
        out = Path.cwd() / results_csv_filename(
            args.site, int(engine.time_ns_of(gid))
        )
    else:
        out = args.output
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
