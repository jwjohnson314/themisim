# THEMISim index benchmark

Measured performance of the deployed search pipeline against an exact
brute-force scan of the same vectors, on the production index
(1,009,488,343 frames) and on the reviewer-scale `small` pilot (56,955 frames).

Raw reports with full provenance are in [`benchmarks/`](benchmarks/):

| file | contents |
|---|---|
| `benchmarks/full-index-1.009B.json` | complete report, production index |
| `benchmarks/pilot-small-57k.json` | complete report, `small` pilot |
| `benchmarks/full-index-ground-truth.npz` | exact top-10 for all 500 production queries |
| `benchmarks/benchmark-tables.tex` | LaTeX tables + figure, generated from the JSON |
| `benchmarks/full-index-1.009B.log` | run log |

**Reproduce:**

```bash
themis-benchmark --artifacts <artifacts-dir> \
    --k 10 --nprobe 64 --prefilter 500 \
    --n-latency 50 --n-cold 10 --n-recall 500 \
    --ground-truth gt.npz --output report.json
```

The ground-truth file is kept because it costs a full scan: any new operating
point can be scored from it in minutes rather than repeating the 2.5-hour pass.

---

## What is being measured

`SearchEngine.search` is not a bare FAISS lookup. It pulls the top `prefilter`
candidates from the compressed index and then rescores them **exactly** in fp32
against the stored vectors. Every figure below is for that whole pipeline.

Three consequences shape how to read the numbers.

**Recall and candidate-capture rate are the same quantity.** Because the rerank
is exact, the fraction of the true top-10 reaching the candidate pool *is* the
recall@10 of the results: a vector in the global exact top-10 that reaches the
pool has at most nine vectors in the database scoring above it, hence at most
nine candidates, so it cannot be displaced. `prefilter` decides what can be
found; the rerank only orders it. Measured directly — capture rate and recall@10
agree to four decimal places.

**Diversification must be disabled when measuring recall.** The near-duplicate
frames `diversify_seconds` removes *are* most of the exhaustive top-10: at the
library default of 30, recall@10 against exhaustive search is **0.238** versus
**0.937** with it disabled (same 500 queries). That is a presentation choice
applied after retrieval, not a retrieval failure, but a benchmark that leaves it
on is scoring the de-duplicator.

**There is no "warm" state for diverse queries at this scale.** The vector store
is 962.7 GB against 125 GiB of RAM. Repeating one query is fast; answering a
*different* query is not, because its rerank rows and manifest entries have
never been read. This dominates the latency numbers and is stated explicitly
below rather than averaged away.

---

## 1. Hardware

| | |
|---|---|
| CPU | Intel Core i9-9920X, 12 cores / 24 threads, 3.5 GHz |
| RAM | 125 GiB |
| GPUs | 3 x Quadro RTX 5000 — **present but unused**; the query path is CPU-only |
| FAISS | 1.8.0, 12 OMP threads |

### Storage placement

| artifact | size | device | cold sequential read |
|---|---|---|---|
| `index.faiss` | 72.72 GB | **SSD** | 487 MB/s |
| `vectors.f16.dat` | 1033.72 GB | **HDD** | 213 MB/s |
| `manifest.parquet` | 12.54 GB | HDD | 227 MB/s |

The index probe reads from SSD; the exact rerank reads from a spinning disk
holding a file ~8x larger than RAM. An isolated cold random row read from that
file costs **6.1 ms**.

---

## 2. Hyperparameters

### Index (production)

| parameter | value |
|---|---|
| factory | `OPQ64_64,IVF65536_HNSW32,PQ64x8` |
| metric | inner product on L2-normalized vectors (cosine) |
| `nlist` | 65,536 |
| vectors | 1,009,488,343 x 512-D, float16 |
| PQ codes | 64 bytes/vector |
| loaded with | `faiss.IO_FLAG_MMAP` |

The pilot index is identical except `nlist = 238` and 56,955 vectors.

### Build

Training sample stratified by (site, year_month), `seed = 0`;
`train_target = 4,194,304` (= 64 points/centroid x `nlist`); `add_chunk_size = 100,000`.

### Query

`k = 10`; `nprobe = 64`; `prefilter = 500`; `diversify_seconds = 0` for all
measurements unless stated; rerank chunk 262,144 rows; `brute_force_max = 4,000,000`.

### Baseline

Exact cosine over all rows, one sequential pass, 1,000,000-row chunks,
fp16 -> fp32 and L2-normalized per chunk — identical arithmetic to the rerank.

### Sampling

500 queries (200 for the pilot), uniform without replacement over `global_id`,
`seed = 0`. Each query is an indexed frame's own stored vector.

---

## 3. Latency

Each configuration was measured in its **own process**, with the page cache
dropped first, so no configuration warms another and nothing is mapped at start.

| operating point | first query, fresh process | steady state, diverse queries | repeated query |
|---|---|---|---|
| **(64, 500)** default | **573 ms** (p95 762) | **409 ms** (p95 815, p99 1656) | **130 ms** (p95 148) |
| **(256, 1000)** | **1181 ms** (p95 1751) | **887 ms** (p95 1562, p99 2886) | — |

*n = 10 for the fresh-process column, 49 for steady state, 10 for repeated.*

Read the three columns as three different questions. The first is what a user
pays on the first query after a server starts. The second is the realistic
steady state — each query is new, so its rerank rows and manifest entries have
never been read, and no amount of running longer makes that cheaper. The third
is the same query issued twice, which is the only genuinely warm case and gives
a **4.4x** cache effect.

Engine construction in a fresh process takes **681 ms** (p95 1.8 s). Read that
as "time to first query" rather than "everything is now resident": the 18 GB of
manifest columns are opened with `np.load(mmap_mode="r")`, so most of that cost
is deferred into per-query page faults.

Enabling `diversify_seconds = 30` changes latency by under 2 % at both operating
points; its cost is a linear scan over at most `k` kept timestamps.

### Where the time goes

| stage | median | p95 |
|---|---|---|
| index probe | **116.7 ms** | 155.3 ms |
| exact rerank | **1.0 ms** | 1.3 ms |

The probe dominates; the rerank is ~1 % of the query. That inverts the obvious
prediction that scattered reads from a spinning disk would dominate, and the
reason is locality: rerank candidates land in a median of **173 contiguous id
runs spanning 2.2 %** of the database, not 500 scattered seeks. `global_id` is
assigned in time order and IVF cells correlate with time, so the candidate set
is nearly contiguous and readahead absorbs it.

> The breakdown runs on queries the latency pass has already touched, so 1.0 ms
> is a warm-cache rerank. On a first-touch query it costs materially more; that
> cost is inside the 573 ms and 409 ms figures, not excluded from them.

---

## 4. Throughput and the brute-force baseline

**Index:** 9.00 queries/s (single client, FAISS using all cores per call).

**Brute force:** one sequential pass over 1.034 TB took **150.5 min** at 114 MB/s
effective. A lone query needs the whole pass: **9,032 s**. Batched over 500
queries sharing one pass: **18.1 s/query** = 0.055 q/s.

| comparison | speedup |
|---|---|
| single query (steady state) | **21,854x** |
| batched throughput | **163x** |

> **The baseline is unoptimised and the speedups should be quoted with that
> stated.** It sustained 114 MB/s against a device measured at 213 MB/s, because
> each chunk serialises read, fp16->fp32 conversion, normalisation and matrix
> multiply with no overlap: the disk idles during arithmetic and the CPU idles
> during I/O. A pipelined baseline would be roughly 2x faster, cutting the
> speedups to ~11,000x and ~80x. Those are the defensible numbers.

---

## 5. Accuracy vs exact ground truth

`nprobe = 64`, `prefilter = 500`, n = 500:

| metric | value |
|---|---|
| **recall@10** | **0.9374** |
| recall p10 | 0.8000 |
| worst query | **0.0000** |
| exact top-1 match | 0.9680 |
| top-1 cosine gap | mean 0.000024, worst 0.003716 |

**The similarity cost of a miss is negligible.** When the pipeline returns a
different top result, it is 0.00002 worse in cosine — indistinguishable in a
512-D embedding of auroral morphology.

**The tail is real.** At least one query in 500 returned none of the true top-10,
and the 10th percentile is 0.80. A mean of 0.937 conceals that, which is why p10
and the minimum are reported beside it.

---

## 6. Operating-point sweep

Recall against the same exact ground truth, ANN settings varied. Because ground
truth was already paid for, each point costs only its own approximate queries.

| `nprobe` | cells scanned | pf=100 | pf=500 | pf=1000 | pf=2000 |
|---|---|---|---|---|---|
| 1 | 0.0015% | 0.704 | 0.806 | 0.826 | 0.836 |
| 8 | 0.012% | 0.797 | 0.938 | 0.961 | 0.973 |
| 32 | 0.049% | 0.797 | 0.937 | 0.961 | 0.973 |
| 64 | 0.098% | 0.797 | **0.937** | 0.961 | 0.973 |
| 256 | 0.391% | 0.797 | 0.937 | 0.961 | 0.973 |

**`prefilter` controls recall; `nprobe` saturates at 8.**

- `nprobe` 8 -> 256 at `prefilter = 2000`: **+0.0008 for 32x the cells scanned**.
- `prefilter` 500 -> 2000 at `nprobe = 64`: **+0.036**.
- Dropping to `nprobe = 1` costs **0.131**.

The library default of `nprobe = 64` does 8x the work of `nprobe = 8` for nothing
measurable. On this evidence `nprobe = 8, prefilter = 2000` reaches 0.973 — above
the current default's 0.937 — while probing an eighth as many cells.

---

## 7. Filtered queries

Filtered queries (by station or date range) bypass the approximate index:
`global_id` is assigned station-major then chronologically, so any
(station, date-range) predicate reduces to a few contiguous identifier ranges
read as sequential slices. A one-month, all-station filter resolves to **17
contiguous ranges** covering 0.53 % of the index.

Subsets below `brute_force_max = 4,000,000` frames are scored exactly. Above it,
an `IDSelectorRange`-restricted IVF probe is used — and that probe can return
**nothing**, because the cells nearest the query need not intersect a
time-disjoint filter. Measured with a one-month filter (5,338,463 frames, above
the cap) and queries drawn from outside that window, `prefilter = 500`, n = 100:

| `nprobe` | queries returning zero candidates |
|---|---|
| 1 | 59 % |
| 8 | 25 % |
| **64 (default)** | **7 %** |
| 256 | 1 % |
| 4096 | 1 % |

Before this was measured, such queries returned an empty result set to the user
while millions of frames matched the filter. The search now escalates `nprobe`
16x and, if the probe is still empty, scores the filtered ranges exactly.
Re-measured on the production index with the same filter: **0 of 50 queries
return empty** (previously ~7 %), median latency 194 ms, with the exact fallback
engaging once and costing 27 s.

---

## 8. Scaling: production vs pilot

| | production (1.009B) | pilot (57k) |
|---|---|---|
| `nlist` | 65,536 | 238 |
| cells scanned at `nprobe=64` | 0.098% | 26.9% |
| steady-state latency | 409 ms | 28 ms |
| probe / rerank | 116.7 / 1.0 ms (1%) | 17.2 / 1.2 ms (6%) |
| candidate locality | 173 runs, 2.20% | 50 runs, 3.27% |
| throughput | 9.00 q/s | 45.2 q/s |
| brute-force pass | 150.5 min | 0.2 s |
| speedup, single query | 21,854x | 6x |
| **speedup, batched throughput** | **163x** | **0.04x** |
| recall@10 | 0.9374 (n=500) | 0.9850 (n=200) |

Two conclusions.

**At 57k vectors the index is 25x *slower* than a linear scan for batch work.**
Its fixed overhead is not amortised until the database is large; the approach
earns its keep at production scale and not before. That is a claim an ANN index
should have to demonstrate rather than assert.

**Pilot recall is an upper bound on production recall, not an estimate of it.**
At equal `nprobe` the pilot scans 26.9 % of its database against the production
index's 0.098 %. These are not the same experiment, and pilot recall (0.985) is
correspondingly higher than production recall (0.937).

---

## 9. Threats to validity

- **The brute-force baseline is unoptimised** (§4); a pipelined implementation
  would roughly halve the quoted speedups.
- **n = 500 queries** on the production index. The mean is well determined; the
  tail (one query at 0/10) rests on few observations.
- **The rerank timing in the stage breakdown is warm-biased** (§3).
- **Cold measurement is only possible in a fresh process.**
  `posix_fadvise(DONTNEED)` cannot evict pages mapped into a live address space,
  and the index is memory-mapped, so an in-process "cold" measurement after warm
  queries is only partially cold. All cold figures here come from a fresh
  process per sample.
- **Single machine, single configuration.** Storage placement dominates; an
  all-SSD or in-RAM deployment would look substantially different.
- **Per-operating-point latency is cache-sensitive.** Each configuration was
  measured in an isolated process for this reason; figures from a single
  sequential run are not comparable across configurations.
