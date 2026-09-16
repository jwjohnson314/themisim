"""``themis-benchmark`` — measure query latency, throughput and recall.

Reports the deployed pipeline (index probe + exact fp32 rerank) against an exact
brute-force scan of the same database, with the hardware, storage placement and
every hyperparameter recorded alongside.

Examples::

    themis-benchmark --artifacts data/pilot/small/artifacts
    themis-benchmark --artifacts ~/Desktop/artifacts --no-brute-force
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from themisim.config import default_artifacts_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--artifacts", type=Path, default=default_artifacts_root())
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--prefilter", type=int, default=500)
    p.add_argument("--n-latency", type=int, default=50, help="warm latency samples")
    p.add_argument("--n-cold", type=int, default=10,
                   help="cold-cache latency samples (each evicts the artifacts)")
    p.add_argument("--n-recall", type=int, default=50,
                   help="queries scored against the brute-force baseline")
    p.add_argument("--no-brute-force", action="store_true",
                   help="skip the exact baseline (it costs one full scan of the memmap)")
    p.add_argument("--chunk-rows", type=int, default=1_000_000,
                   help="rows per chunk in the brute-force sweep")
    p.add_argument("--no-sweep", action="store_true",
                   help="skip the (nprobe, prefilter) recall grid")
    p.add_argument("--ground-truth", type=Path, default=None,
                   help="save the exact top-k here (.npz) so later runs can "
                        "score new operating points without rescanning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None, help="write the JSON report here")
    args = p.parse_args(argv)

    # Deferred so `--help` doesn't pay the faiss/torch import cost.
    from themisim.benchmark import format_benchmark, run_benchmark

    t0 = time.time()

    def progress(stage: str, info: object) -> None:
        print(f"  [{stage}] {info}  (+{time.time() - t0:.0f}s)", flush=True)

    report = run_benchmark(
        args.artifacts,
        k=args.k,
        nprobe=args.nprobe,
        prefilter=args.prefilter,
        n_latency=args.n_latency,
        n_cold=args.n_cold,
        n_recall=args.n_recall,
        brute_force=not args.no_brute_force,
        chunk_rows=args.chunk_rows,
        sweep=not args.no_sweep,
        ground_truth_path=args.ground_truth,
        seed=args.seed,
        on_progress=progress,
    )
    print()
    print(format_benchmark(report))

    out = args.output or (args.artifacts / "benchmark_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
