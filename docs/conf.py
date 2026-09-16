"""Sphinx configuration for the themisim documentation.

Built by GitHub Actions and published to GitHub Pages. Autodoc pulls the API
reference straight from the package's NumPy-style docstrings; the heavy binary
dependencies (torch, faiss, …) are mocked so the docs build without installing
the full runtime stack.
"""
from __future__ import annotations

import os
import sys

# Make the package importable for autodoc without requiring an install.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------

project = "themisim"
author = "Jeremiah Johnson"
copyright = "2026, Jeremiah Johnson"

try:  # prefer the installed distribution's version when available
    from importlib.metadata import version as _dist_version

    release = _dist_version("themisim")
except Exception:  # not installed (e.g. building from a bare checkout)
    release = "0.2.0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy / Google style docstrings
    "sphinx.ext.intersphinx",   # cross-links to numpy/pandas/torch docs
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"

# Mock the heavy ML/binary deps so the docs build skips the torch/faiss stack.
# NB: don't mock pyarrow/PIL — pandas reads ``pyarrow.__version__`` at its own
# import time and a mock breaks it, so those stay real (cheap wheels anyway).
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "faiss",
    "cdflib",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "themisim"
