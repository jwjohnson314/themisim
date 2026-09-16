API reference
=============

.. currentmodule:: themisim

Everything below is importable directly from the top-level ``themisim`` package.

Querying
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   query
   resolve_global_id
   SearchEngine
   Hit

Exporting results
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   hits_to_dataframe
   results_csv_filename

Building an index
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   download_archive
   build_index
   fetch_weights

Pilot indexes
-------------

A pilot is a small, pinned slice of the real archive that builds in minutes and
reports its own retrieval quality, so the software can be assessed without the
full ~100 TB archive. See :doc:`pilot`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   list_pilots
   build_pilot
   validate_index
   run_benchmark

Visualization
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   visualize_results
