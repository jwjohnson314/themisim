Performance
===========

``themis-benchmark`` measures query latency, throughput and recall for the
deployed search pipeline, against an exact brute-force scan of the same vectors.
It exists for the same reason the pilot index does: the performance claims this
library makes should be checkable by someone who did not build it.

The full measured results, including hardware, storage placement and every
hyperparameter, live in `BENCHMARK.md
<https://github.com/jwjohnson314/themisim/blob/main/BENCHMARK.md>`_, with the raw
JSON reports alongside it under ``benchmarks/``. This page describes what is
being measured and how to reproduce it.

What is measured
----------------

:meth:`~themisim.search.SearchEngine.search` is not a bare FAISS lookup. It pulls
the top ``prefilter`` candidates from the compressed index and then rescores them
*exactly* in fp32 against the stored vectors. Every latency and recall figure is
for that whole pipeline, because that is what a user experiences; timing the
FAISS call alone would flatter the system and describe something nobody runs.

Three consequences shape how the numbers should be read.

**Recall and capture rate are the same quantity.** Because the rerank is exact,
the fraction of the true top-10 that reaches the candidate pool is also the
recall@10 of the returned results. A vector in the global exact top-10 that
reaches the pool has at most nine vectors in the entire database scoring above
it, hence at most nine candidates, so it cannot be displaced. ``prefilter``
decides what can be found; the rerank only orders it.

**Diversification must be disabled when measuring recall.** The near-duplicate
frames that ``diversify_seconds`` removes *are* most of the exhaustive top-10, so
at the library default of 30 the measured recall@10 against exhaustive search is
about 0.26 rather than about 0.91. That is a presentation choice applied after
retrieval, not a retrieval failure — but a benchmark that leaves it on is scoring
the de-duplicator.

**Storage placement dominates latency.** The rerank reads rows from arbitrary
offsets in a vector store far larger than RAM. Whether that store is on SSD or a
spinning disk changes query latency by more than any tuning parameter, so the
report records the backing device and measured throughput of each artifact.

Reproducing
-----------

.. code-block:: bash

   themis-benchmark --artifacts <artifacts-dir> \
       --k 10 --nprobe 64 --prefilter 500 \
       --n-latency 50 --n-cold 10 --n-recall 500 \
       --ground-truth gt.npz --output report.json

Ground truth costs one full sequential pass over the vector store, so it is saved
rather than discarded: a later run can score new operating points from the
``.npz`` for the price of the approximate queries alone, without rescanning.

On a pilot index the whole thing takes seconds:

.. code-block:: bash

   themis-benchmark --artifacts data/pilot/small/artifacts

What the report contains
------------------------

* **Latency** — median, p95, p99 for the full pipeline, cold and warm. Cold
  measurements evict every artifact with ``posix_fadvise(DONTNEED)`` first; the
  warm comparison re-runs the *identical* query set, because individual queries
  differ several-fold in cost depending on where their rerank candidates sit on
  disk, and comparing different query sets would confound cache state with query
  difficulty.
* **Stage breakdown** — how much of a query is the index probe versus the exact
  rerank, plus how contiguous the rerank candidates are. Since ``global_id`` is
  assigned in time order and IVF cells correlate with time, candidates tend to
  fall in a few contiguous runs rather than scattering, which matters enormously
  on rotational storage.
* **Throughput** — sustained warm queries per second through the public API.
* **Operating-point sweep** — recall@k over a ``(nprobe, prefilter)`` grid
  against the same exact ground truth. The two are swept together because they
  interact: at a fixed candidate budget, probing more cells can *lower*
  end-to-end recall by crowding true neighbours out of the prefilter.
* **Brute-force baseline** — wall clock for one exact pass, and the resulting
  speedups.
* **Provenance** — CPU, RAM, backing devices, library versions, thread counts,
  and every hyperparameter, so a differing result can be diagnosed rather than
  argued about.

Limits
------

The brute-force baseline is a straightforward implementation, not a tuned one: it
serialises read, conversion, normalisation and matrix multiply with no overlap,
so it does not saturate the storage device. The speedups it yields are therefore
against an honest but unoptimised reference, and are quoted with that stated. See
the threats-to-validity section of ``BENCHMARK.md`` for the full list.

Python API
----------

.. code-block:: python

   from themisim import run_benchmark

   report = run_benchmark("data/artifacts", k=10, nprobe=64, prefilter=500)
   print(report["accuracy"]["recall_at_k"], report["latency_warm"]["median_s"])
