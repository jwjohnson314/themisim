"""``themis-build-index`` — embed downloaded CDFs and build the FAISS index.

Runs the full build: inventory -> embed (SimCLR encoder) -> concatenate ->
train + add to an OPQ-IVF-PQ index. Uses the GPU when available and falls back
to CPU otherwise. Resumable — already-embedded hours are skipped.

Example::

    themis-build-index --data-root ./data/cdf --artifacts ./data/artifacts \\
        --checkpoint ./weights/aurora-fm-no-finetune.tar
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from themisim.config import default_artifacts_root, default_data_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument("--artifacts", type=Path, default=default_artifacts_root())
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="SimCLR checkpoint .tar (default: fetch/verify into the weights dir)",
    )
    p.add_argument(
        "--nlist",
        default="auto",
        help="IVF cells; 'auto' scales with the vector count (default: auto)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' (GPU if present, else CPU), 'cuda', or 'cpu'",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    p.add_argument(
        "--no-gpu-quantizer",
        action="store_true",
        help="train the IVF k-means on CPU even if GPUs are present",
    )
    p.add_argument(
        "--no-streaming",
        action="store_true",
        help="use the eager per-CDF embed loop instead of the faster "
        "persistent-worker streaming path",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile on the encoder during streaming embed "
        "(avoids its warmup; only affects GPU runs)",
    )
    args = p.parse_args(argv)

    from themisim.pipeline import build_index

    checkpoint = args.checkpoint
    if checkpoint is None:
        from themisim.weights import fetch_weights

        print("No --checkpoint given; fetching/verifying weights ...", flush=True)
        checkpoint = fetch_weights()

    t0 = time.time()

    def progress(stage: str, info: object) -> None:
        print(f"  [{stage}] {info}  (+{time.time() - t0:.0f}s)", flush=True)

    nlist = args.nlist if args.nlist == "auto" else int(args.nlist)
    out = build_index(
        args.data_root,
        args.artifacts,
        checkpoint,
        nlist=nlist,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_gpu_quantizer=not args.no_gpu_quantizer,
        streaming=not args.no_streaming,
        use_compile=not args.no_compile,
        on_progress=progress,
    )
    size = out.stat().st_size
    print(f"\nWrote {out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
