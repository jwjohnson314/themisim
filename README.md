# THEMISim

Similarity search over [THEMIS](https://themis.ssl.berkeley.edu) all-sky imager
(ASI) auroral imagery. A SimCLR-trained ResNet-18 encoder turns each 256×256
frame into a 512-D feature vector where proximity as measured by cosine similarity corresponds to visual and morphological similarity rather than pixel-level identity; ~1 billion of these are indexed with FAISS (OPQ-IVF-PQ). Given any indexed frame, the library quickly and efficiently returns the most visually similar frames across the whole archive.

This software does three things:

1. **Download** THEMIS ASI CDFs from the Berkeley archive and encoder weights from HuggingFace.
2. **Build** a FAISS index from the pretrained encoder weights.
3. **Query** the index by `(site, datetime, frame)`, returning a tidy
   `DataFrame`/CSV.

## Install

```bash
pip install themisim
```

`faiss-cpu` is pulled in as a dependency (the query path is CPU-only). If you
already have a GPU `faiss` from conda, install with `--no-deps` to keep it.
A CUDA-enabled PyTorch makes index building much faster but is optional — every
stage falls back to CPU.

## Quick start — querying an existing index

```python
from themisim import query

df = query("fsmi", "2015-03-18T06", 412, artifacts="data/artifacts")
df.head()
#    site             datetime     score                                         source_cdf
# 0  fsmi  2015-03-18 06:20:36  1.000000  http://themis.ssl.berkeley.edu/.../thg_l1_asf_fsmi_2015031806_v01.cdf
# ...
```

- `datetime` names the **hourly** CDF (`YYYY-MM-DDTHH`); sub-hour fields are
  ignored. `2015-03-18 06`, `2015031806`, and a `datetime` object also work.
- `frame` is the 0-based frame index within that hour (THEMIS runs at a
  3-second cadence, ~1200 frames/hour).
- The top result is normally the query frame itself (`score ≈ 1.0`).

Search parameters:

| arg | default | meaning |
|-----|---------|---------|
| `results` | 24 | count mode: number of results to return |
| `min_score` | `None` | threshold mode: return *every* match scoring ≥ this cutoff (overrides `results`) |
| `prefilter` | 500 | FAISS candidates pulled before exact cosine rerank |
| `nprobe` | 64 | IVF cells inspected per query |
| `diversify_seconds` | 30 | drop near-duplicate frames within ±N s at the same site (0 disables) |

### Bounding the result set: count vs. threshold

There are two ways to decide how many results come back:

- **Count** (default) — return the top `results` matches.
- **Threshold** — pass `min_score` (a cosine similarity in `[0, 1]`) and the
  search ignores `results`, returning *every* (temporally diversified) match
  scoring at or above the cutoff, best first. The result set is capped at 2000
  for safety. Because the candidate pool is the top `prefilter` FAISS matches,
  raise `prefilter` to surface more low-cutoff hits.

```python
# every frame at least 0.9 cosine-similar to the query, not just the top 24
df = query("fsmi", "2015-03-18T06", 412, artifacts="data/artifacts",
           min_score=0.9)
```

### From the command line

```bash
themis-query --site fsmi --datetime 2015-03-18T06 --frame 412 \
    --artifacts data/artifacts --output results.csv

# threshold mode: every match scoring >= 0.9, instead of a fixed --results count
themis-query --site fsmi --datetime 2015-03-18T06 --frame 412 \
    --artifacts data/artifacts --min-score 0.9 --output results.csv
```

Output columns are `site, datetime, score, source_cdf` — identical to the
dashboard's CSV export.

## Build an index from scratch

Note that the full THEMIS ASI archive is ~1M CDFs (over 100 TB) and building the index takes on the order of GPU-days; the
prebuilt index (about 72 GB) is **not** currently downloadable. If you are interested in obtaining it please email me to discuss.


```bash
# 1. fetch the encoder weights (44 MB)
python -c "from themisim import fetch_weights; print(fetch_weights())"

# 2. one-shot pipeline (download + build)
SITES=fsmi START=2015-03 END=2015-03 ./scripts/run_pipeline.sh
```

or step by step:

```bash
themis-download   --data-root data/cdf --sites fsmi --start 2015-03 --end 2015-03
themis-build-index --data-root data/cdf --artifacts data/artifacts \
                   --checkpoint weights/aurora-fm-no-finetune.tar
```

The equivalent Python API:

```python
from themisim import download_archive, build_index

download_archive("data/cdf", sites=["fsmi"], start="2015-03", end="2015-03")
build_index("data/cdf", "data/artifacts",
            "weights/aurora-fm-no-finetune.tar")
```

`build_index` runs inventory → embed → concat → train/build, resuming cleanly
if interrupted (already-embedded hours are skipped). `--nlist auto` scales the
IVF cell count to the dataset size.

## Model weights

`fetch_weights()` downloads and SHA-256-verifies the ~44 MB SimCLR checkpoint
into `./weights` (override with `$THEMIS_ASI_WEIGHTS`). It is pulled from the
public Hugging Face repo
[`Jwjohnson314/Aurora-FM`](https://huggingface.co/Jwjohnson314/Aurora-FM); override
the URL via `$THEMIS_ASI_WEIGHTS_URL` or pass `url=`. If you already have the
`.tar`, drop it in the weights directory and it will be verified and reused
without a network call.

## Paths / configuration

Resolved from explicit arguments, then environment variables, then defaults:

| what | env var | default |
|------|---------|---------|
| downloaded CDFs | `THEMIS_ASI_DATA_ROOT` | `./data/cdf` |
| index artifacts | `THEMIS_ASI_ARTIFACTS` | `./data/artifacts` |
| model weights | `THEMIS_ASI_WEIGHTS` | `./weights` |

An artifacts directory contains `index.faiss`, `manifest.parquet`,
`vectors.f16.dat`, and `vectors.meta.json`.

## License, data & citation

The **code** in this repository is MIT-licensed — see [LICENSE](LICENSE).

The assets this tool downloads carry their own terms, which you must honor when
publishing results:

- **Model weights** (`Jwjohnson314/Aurora-FM`) are released under
  **CC-BY-4.0** — attribution required.
- **THEMIS ASI data** is provided by the THEMIS mission (UC Berkeley / NASA) and
  is subject to the
  [THEMIS data use & citation policy](https://themis.ssl.berkeley.edu/roadrules.shtml).
  Acknowledge the mission and instrument teams in any publication.
