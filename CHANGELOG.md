# Changelog

Notable changes to THEMISim. This project follows [Semantic
Versioning](https://semver.org/).

## 0.2.0 — 2026-09-16

This release provides adds functionality for verifying THEMISim. 
0.2.0 adds a pinned slice of the real archive, a validation report
the build produces about itself, a benchmark suite measured
against an exact brute-force baseline, and test coverage of the whole
scientific path from pixels to ranked results.

### Added

**Pilot indexes** — `themis-pilot`, `themisim.build_pilot`,
`themisim.list_pilots`.

- A `PilotSpec` pins an explicit list of CDFs, each with its size and SHA-256.
  URLs are reconstructed rather than crawled, so two people a year apart fetch
  byte-identical inputs regardless of how the archive's directory listings
  change.
- Two slices ship: `tiny` (10 CDFs, 11,457 vectors, ~1.2 GB, ~3 min on 8 CPU
  threads) and `small` (48 CDFs, 56,955 vectors, ~6.0 GB, ~12 min). Both span
  two geomagnetic storms, four stations, two year-months, and **both on-disk
  CDF layouts**, and both include ragged (partial) hours.
- Builds default to isolated directories, with a pre-flight that refuses to
  start if the tree holds anything the spec does not pin — `build_index`
  inventories a whole data root and `concat` globs every shard beneath an
  artifacts directory, so contamination would otherwise yield a valid index
  over the wrong data, silently.
- Per-CDF embed failures, which `build_index` logs and continues past, are now
  surfaced rather than left in a parquet file nobody reads.

**Validation** — `themisim.validate_index`, usable on any artifacts directory.

- Checks artifact agreement, index trained-ness, contiguous site blocks and
  monotone `time_ns`, NaN/zero-norm vectors, self-retrieval on both the exact
  and approximate paths, recall@k against exhaustive ground truth, encoder
  reproducibility against pinned reference vectors, export round-trip, and
  training-sampler determinism.
- Writes `pilot_report.json` with full provenance and per-stage timings, and
  exits non-zero on a missed threshold so it works as a CI gate.
- Reports explicitly what a pilot-scale index **cannot** demonstrate, rather
  than leaving a reader to infer coverage from silence.

**Benchmarking** — `themis-benchmark`, `themisim.run_benchmark`.

- Latency (median/p95/p99, cold and warm), throughput, an index-probe vs
  exact-rerank breakdown with candidate-locality statistics, a
  `(nprobe, prefilter)` recall grid, and an exact brute-force baseline.
- Ground truth is persisted, so new operating points can be scored without
  repeating the full scan.
- Results and analysis in `BENCHMARK.md`; raw reports, ground truth and
  generated LaTeX tables in `benchmarks/`.

**Documentation** — an architecture diagram in the README and docs; new
`docs/pilot.rst` and `docs/benchmark.rst` pages; a "Reviewing this software"
section in the README.

**Testing and CI**

- **The first CI workflow that runs pytest** (`tests.yml`, Python 3.11/3.12),
  plus a manually dispatched real-data pilot build (`pilot.yml`).
- An **end-to-end test from synthetic frames**: THEMIS-format CDFs written in
  both on-disk layouts, served over localhost, then run through the genuine
  chain — download, inventory, CDF read, preprocessing, SimCLR encoder, fp16
  shards, concat, OPQ-IVF-PQ train/add, query — asserting a frame retrieves
  itself at cosine 1.0 and that the exported table agrees with the manifest.
  The pipeline can now be verified with no THEMIS data at all.
- New coverage for preprocessing (including the flat-frame NaN guard), encoder
  loading and checkpoint verification, the downloader (against a localhost HTTP
  server), build orchestration, concat, the training sampler and the search
  path. Scale-guard branches are exercised with shrunken constants so code that
  only fires on the full archive actually runs.
- `slow` and `pilot` pytest markers. The default run is offline; `pytest -m
  pilot` adds a real archive build.
- Suite went from 5 test modules to 18, and from no coverage of the build path
  to covering it end to end.

### Changed

- **Dependency constraints corrected**: `numpy>=1.24,<2` → `numpy>=1.25`, and
  `faiss-cpu>=1.8` → `faiss-cpu>=1.9`. Nothing in this package required NumPy
  1.x; the cap was standing in for faiss-cpu 1.8.x, which is built against the
  NumPy 1.x C ABI and fails to import under NumPy 2. Constraining faiss is the
  narrow fix, and an upper bound on NumPy in a published package is contagious.
  Indexes written by faiss 1.8.0 are read correctly by 1.9+.
- **A pinned CDF whose bytes changed upstream is now a warning**, recorded in
  the report under `slice_verification`, rather than a hard failure — a
  reprocessed archive hour should not make a published spec permanently
  unbuildable. Missing files remain fatal; `--strict-checksums` restores strict
  behaviour.
- **`visualize_results` tile captions carry the full date** alongside station,
  time and score. Previously a caption showed only `HH:MM:SS`, so results from
  different years were indistinguishable — routine, since result sets span the
  archive. The grid uses `constrained_layout` so the taller captions are never
  clipped, and caption size scales with tile size.
- Package data (`themisim/data/*.json`, `*.npz`) is now declared, without which
  the pilot specs and reference vectors are absent from the wheel.
- `matplotlib` added to the `dev` extra so the visualizer test runs in CI
  instead of skipping.
- `.gitignore` now covers the CDF cache `visualize_results` creates when called
  without a `data_root`, which reaches gigabytes without anyone deciding to.

### Fixed

- **Filtered queries could return an empty result set while millions of frames
  matched.** When a filtered subset exceeds `brute_force_max`, the search
  restricts an IVF probe with an `IDSelector`, and that probe can find nothing:
  IVF cells cluster by feature similarity, which correlates with time, so the
  cells nearest a query need not intersect a time-disjoint filter. Measured on
  the production index with a one-month filter, this affected **7% of queries at
  the default `nprobe`** (59% at `nprobe=1`, 1% even at `nprobe=4096`). The
  search now escalates `nprobe` 16x and, if still empty, scores the filtered
  ranges exactly. Re-measured: 0 of 50 queries return empty.
- **`query._BROWSE_CACHE` was keyed on `id(engine)`.** After an engine was
  garbage collected, CPython could reuse its address for an engine over a
  *different* artifacts directory, and `resolve_global_id` would silently answer
  from the wrong archive. Now a `WeakKeyDictionary`.
- **`SearchEngine` now rejects an untrained index** at load with a clear
  message, instead of failing obscurely inside a later `faiss.search`.

### Measured

Full index, 1,009,488,343 vectors, 500 queries against an exhaustive scan
(`nprobe=64`, `prefilter=500`, `diversify_seconds=0`) — see `BENCHMARK.md`:

| | |
|---|---|
| recall@10 vs exhaustive | 0.9374 |
| exact top-1 agreement | 0.9680 |
| top-1 cosine gap | mean 0.000024 |
| latency, steady state | 409 ms (p95 815 ms) |
| latency, repeated query | 130 ms |
| throughput | 9.0 queries/s |
| exact brute-force pass | 150.5 min for 1.034 TB |
| speedup over brute force | ~21,900x single query, ~163x batched |

Two findings worth carrying forward:

- **`prefilter` governs recall; `nprobe` saturates at 8.** Going from
  `nprobe=8` to `nprobe=256` at `prefilter=2000` gains 0.0008 for 32x the cells
  scanned, while `prefilter` 500 → 2000 gains 0.036.
- **Recall must be measured with `diversify_seconds=0`.** At the library default
  of 30, recall@10 against an exhaustive top-10 is 0.238 rather than 0.937,
  because the near-duplicate frames diversification removes *are* most of the
  exhaustive top-10. That is a presentation choice applied after retrieval, not
  a retrieval failure.

### Known limitations

- The brute-force baseline is a straightforward implementation, not a tuned
  one; it does not saturate the storage device, so the quoted speedups are
  against an honest but unoptimised reference.
- The default `nprobe=64` is unchanged in this release despite the sweep
  suggesting `nprobe=8` is the knee; changing it affects every user and
  warrants confirmation on an independent query sample.

## 0.1.0

Initial release: download the THEMIS ASI archive, build a FAISS index from the
pretrained SimCLR encoder, and query it by `(site, datetime, frame)`.
