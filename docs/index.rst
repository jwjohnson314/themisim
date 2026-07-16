themisim
========

Similarity search over `THEMIS <https://themis.ssl.berkeley.edu>`_ all-sky
imager (ASI) auroral imagery. A SimCLR-trained ResNet-18 encoder turns each
256×256 frame into a 512-D feature vector whose cosine proximity tracks visual
and morphological similarity rather than pixel-level identity; those vectors are
indexed with FAISS (OPQ-IVF-PQ). Given any indexed frame, ``themisim`` returns
the most visually similar frames across the archive.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api

Installation
------------

.. code-block:: bash

   pip install themisim

``faiss-cpu`` is pulled in automatically (the query path is CPU-only). A
CUDA-enabled PyTorch speeds up index building but is optional — every stage
falls back to CPU.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
