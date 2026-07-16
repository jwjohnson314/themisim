Quickstart
==========

Query an existing index
-----------------------

.. code-block:: python

   from themisim import query

   df = query("fsmi", "2015-03-18T06", 412, artifacts="data/artifacts")
   df.head()

- ``datetime`` names the **hourly** CDF (``YYYY-MM-DDTHH``); sub-hour fields are
  ignored. ``2015-03-18 06``, ``2015031806``, and a ``datetime`` object also work.
- ``frame`` is the 0-based frame index within that hour (THEMIS runs at a
  3-second cadence, ~1200 frames/hour).
- The top result is normally the query frame itself (``score ≈ 1.0``).

Pass ``min_score`` to switch from count mode to threshold mode — every match at
or above a cosine cutoff, best first:

.. code-block:: python

   df = query("fsmi", "2015-03-18T06", 412, artifacts="data/artifacts",
              min_score=0.9)

From the command line
---------------------

.. code-block:: bash

   themis-query --site fsmi --datetime 2015-03-18T06 --frame 412 \
       --artifacts data/artifacts --output results.csv

Build an index from scratch
---------------------------

.. code-block:: python

   from themisim import download_archive, build_index

   download_archive("data/cdf", sites=["fsmi"], start="2015-03", end="2015-03")
   build_index("data/cdf", "data/artifacts",
               "weights/aurora-fm-no-finetune.tar")

The ~44 MB SimCLR checkpoint is fetched (and SHA-256 verified) on demand:

.. code-block:: python

   from themisim import fetch_weights
   fetch_weights()

See the `README <https://github.com/jwjohnson314/themisim>`_ for the full
pipeline, configuration environment variables, and data-use terms.
