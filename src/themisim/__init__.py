"""THEMIS all-sky imager similarity search.

A standalone, pip-installable extraction of the THEMIS similarity-search
engine: download the archive, build a FAISS index from the pretrained SimCLR
encoder, and query it by ``(site, datetime, frame)``.

Quick start (query an existing index)::

    from themisim import query
    df = query("fsmi", "2015-03-18T06", 412, artifacts="data/artifacts")

Build an index from scratch::

    from themisim import download_archive, build_index
    download_archive("data/cdf", sites=["fsmi"], start="2015-03", end="2015-03")
    build_index("data/cdf", "data/artifacts", "weights/<checkpoint>.tar")
"""
from __future__ import annotations

# Establish the torch-before-faiss import order before anything pulls in faiss
# (the query stack imports faiss). torch and faiss ship conflicting CUDA libs
# and break each other's loader if faiss is imported first; importing torch
# here (when present) is a no-op safety net. Absent torch (query-only installs)
# this is simply skipped and faiss loads alone, which is fine.
try:  # pragma: no cover - environment dependent
    import torch as _torch  # noqa: F401
except Exception:  # pragma: no cover
    pass

from themisim.export import (  # noqa: E402
    hits_to_dataframe,
    results_csv_filename,
)
from themisim.query import query, resolve_global_id  # noqa: E402
from themisim.search import Hit, SearchEngine  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "query",
    "resolve_global_id",
    "SearchEngine",
    "Hit",
    "hits_to_dataframe",
    "results_csv_filename",
    "download_archive",
    "build_index",
    "fetch_weights",
    "visualize_results",
    "__version__",
]


def __getattr__(name: str):
    """Lazily expose the build pipeline + weights helper.

    Kept out of the eager import path so ``import themisim`` stays
    cheap (and torch-free) for query-only use; ``build_index`` /
    ``download_archive`` pull in the embed/index stack only when first used.
    """
    if name in ("build_index", "download_archive"):
        from themisim import pipeline

        return getattr(pipeline, name)
    if name == "fetch_weights":
        from themisim.weights import fetch_weights

        return fetch_weights
    if name == "visualize_results":
        from themisim.viz import visualize_results

        return visualize_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
