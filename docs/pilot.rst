Pilot indexes
=============

The published index covers the whole THEMIS all-sky imager archive: roughly one
million CDFs, over 100 TB of imagery, about a billion 512-dimensional vectors
and a 72 GB ``index.faiss``. Building it takes GPU-days. To allow testing the 
software without committing to the full build, we have provided a pilot index: 
a small, *pinned* slice of the real archive that builds on
an ordinary CPU laptop in minutes, runs through exactly the same pipeline the
full build uses, and finishes by measuring its own retrieval quality against
exhaustive ground truth.

.. code-block:: bash

   themis-pilot --list          # what is available and what it costs
   themis-pilot --spec small    # download, build, validate

The command is idempotent: downloads resume, already-embedded hours are skipped,
and it is safe to re-run after an interrupted connection.

Available slices
----------------

.. list-table::
   :header-rows: 1

   * - Spec
     - CDFs
     - Vectors
     - Download
     - Purpose
   * - ``tiny``
     - 10
     - 11,457
     - 1.2 GB
     - The smallest slice that still trains the production quantizer without
       FAISS warnings (the floor is ~9,984 vectors: ``PQ64x8`` needs 39x256
       training points). Built in 3 min on 8 CPU threads. Used by CI and by
       ``pytest -m pilot``.
   * - ``small``
     - 48
     - 56,955
     - 6.0 GB
     - The reviewer's slice: four sites, two storms, both CDF layouts. Built in
       12 min on 8 CPU threads.

``tiny`` is a strict subset of ``small``, which is what lets a single 46 KB
file of pinned reference vectors validate the encoder path for both.

Both draw on two well-observed geomagnetic storms — the St Patrick's Day storm
of 18 March 2015 and the G4 storm of 25 March 2024 — so the retrieved frames
show real auroral structure rather than a thousand near-identical dark frames.

What the slices are chosen to exercise
--------------------------------------

The CDF lists are not arbitrary. Each was selected to reach code that a naive
subset would miss, and :mod:`tests.test_pilot_spec` asserts the properties still
hold:

* **Two or more sites.** The training sampler buckets by ``(site, year_month)``,
  and the contiguous-site-block layout that
  :meth:`~themisim.search.SearchEngine.search` binary-searches on is meaningless
  with only one block.
* **Two or more year-months**, for the other half of that bucket key.
* **At least one CDF from 2024 or later.** ``themisim.embed._read_cdf`` has two
  mutually exclusive layout branches — the legacy ``(N, 256, 256)`` cube with
  ``_epoch`` timestamps, and the 2024+ ``(1, row, col, N)`` cube with ``_time``
  in Unix seconds. Only real modern data reaches the second one.
* **Overlapping UT hours across sites**, so a query at one station can retrieve
  the same auroral structure seen from another. Without it every match is a
  within-site near-duplicate, which demonstrates temporal autocorrelation rather
  than visual similarity.
* **Consecutive hours at one site**, to cross a shard boundary.
* **Partial (night-edge) hours**, whose frame counts differ from the nominal
  1200, so nothing downstream may assume a constant.

If the archive changes
----------------------

Each pinned CDF carries its size and SHA-256, and ``themis-pilot`` checks both
before building. A file that is *missing* is a hard error. A file whose bytes
have *changed* is a warning by default, recorded in the report under
``slice_verification``, because the THEMIS archive occasionally reprocesses an
hour in place and a published spec should not become unbuildable the day that
happens. Pass ``--strict-checksums`` to make any divergence fatal when exact
byte-for-byte reproduction is the point.

The validation report
---------------------

Every build writes ``pilot_report.json`` alongside the index and prints a
summary. ``themis-pilot`` exits non-zero if any calibrated threshold is missed,
so it works as a one-command verdict.

**Retrieval quality is reported at the operating point users actually get.**
:meth:`~themisim.search.SearchEngine.search` does not return FAISS's top-k: it
pulls the top ``prefilter`` candidates from the compressed index and then
rescores them *exactly* in fp32 against the stored vectors. The difference is
not small. On the ``tiny`` slice, raw FAISS recall@10 is about 0.44 while the
recall of the full pipeline is 0.94 to 1.00. The report shows both, and gates on
the second.

It also sweeps ``nprobe`` and ``prefilter`` *together*, because they interact:
at a fixed candidate budget, probing more cells can *lower* end-to-end recall by
crowding true neighbours out of the prefilter. Reporting one without the other
invites the wrong conclusion.

Checks performed:

* **integrity** — index, manifest and memmap describe the same vectors; the
  index is trained; each site occupies one contiguous block of global ids and
  ``time_ns`` is non-decreasing within it.
* **vector health** — no NaN/infinite rows and no zero-norm rows. A zero vector
  survives normalization and then matches nothing, silently.
* **self-retrieval** — a frame queried by its own vector must come back first.
  Reported separately for the exact path (deterministic; gated at 1.0) and the
  approximate IVF path (stochastic; gated loosely).
* **retrieval quality** — recall@k against an exhaustive cosine scan.
* **encoder reference** — vectors this build produced, compared against ones
  pinned at release. Separates "your encoder differs from ours" from "the index
  is bad".
* **export round-trip** — the returned DataFrame agrees with the manifest.
* **training sampler** — deterministic, sorted and unique on real data.

The report carries full provenance (Python, numpy, faiss, torch, Pillow, cdflib
versions, platform, thread counts, and the device the index was built on) and
per-stage timings, so a run that produces different numbers can be diagnosed
rather than argued about.

What a pilot cannot show
------------------------

The report states this itself, in a ``coverage_gaps`` section, rather than
leaving a reader to infer it:

* **The recall regime is different.** A pilot with ``nlist=107`` probed 64 deep
  scans most of its database; the full archive with ``nlist=65536`` probed 64
  deep scans about a thousandth of it. Pilot recall is an *upper bound* on
  archive recall, not an estimate of it.
* Several scale guards no-op below their thresholds — the 20-million-row
  manifest batching, the 100k-row index add chunking, the 2-million-row training
  gather, and the reservoir branch of the training sampler. Those are covered
  instead by the offline test suite, which shrinks the constants to force the
  branches to execute.
* Filtered queries at pilot scale always take the exact path, so the
  ``IDSelectorRange``-restricted IVF fallback is never reached.
* The multi-GPU embed coordinator and the memory-mapped paging behaviour of a
  multi-gigabyte index are out of reach by construction.

Measuring performance
---------------------

``themis-benchmark`` runs against a pilot index the same way it runs against the
production one, and on a pilot the exact brute-force baseline takes under a
second. See :doc:`benchmark`.

Querying a pilot index
----------------------

A pilot's artifacts directory is an ordinary artifacts directory:

.. code-block:: bash

   themis-query --site fsmi --datetime 2024-03-25T06 --frame 400 \
       --artifacts data/pilot/small/artifacts

.. code-block:: python

   from themisim import query
   df = query("fsmi", "2024-03-25T06", 400, artifacts="data/pilot/small/artifacts")

Pass ``--figure results.png`` to ``themis-pilot`` to render a contact sheet of
one query's matches from the CDFs already on disk (requires
``pip install 'themisim[notebook]'``).

Python API
----------

.. code-block:: python

   from themisim import list_pilots, build_pilot, validate_index

   for spec in list_pilots():
       print(spec.name, spec.n_files, spec.estimate()["total_s"])

   result = build_pilot("tiny", device="cpu")
   print(result.ok, result.total_n)

   # validate_index works on any artifacts directory, pilot or not
   report = validate_index("data/artifacts", nprobe=64, prefilter=500)
