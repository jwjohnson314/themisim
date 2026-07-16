"""Filesystem locations for the build/query artifacts.

Generalized from the original project's ``paths.py`` (which hard-coded a
single workstation's ``/home/...`` layout) so the library is portable. Nothing
here is machine specific: the artifacts directory is resolved from, in order,
an explicit argument passed by the caller, the ``THEMIS_ASI_ARTIFACTS``
environment variable, or ``./data/artifacts`` under the current directory.

Kept torch-free so the indexing / concat CLIs (numpy + faiss + pandas only) can
import these constants without dragging the encoder stack — and its CUDA shared
libraries — onto the import path. ``torch`` and ``faiss`` both ship CUDA libs
that break each other's loader when imported in the wrong order; this module
stays out of that hazard.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_ARTIFACTS = "THEMIS_ASI_ARTIFACTS"
ENV_DATA_ROOT = "THEMIS_ASI_DATA_ROOT"
ENV_WEIGHTS = "THEMIS_ASI_WEIGHTS"

#: Public THEMIS all-sky imager archive (Apache autoindex).
ARCHIVE_BASE_URL = "http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/"

#: Default model checkpoint filename.
DEFAULT_CHECKPOINT_NAME = "aurora-fm-no-finetune.tar"


def default_artifacts_root() -> Path:
    """Where the index / vectors / manifest live.

    ``$THEMIS_ASI_ARTIFACTS`` if set, else ``./data/artifacts``.
    """
    env = os.environ.get(ENV_ARTIFACTS)
    return Path(env).expanduser() if env else Path.cwd() / "data" / "artifacts"


def default_data_root() -> Path:
    """Where downloaded CDFs live.

    ``$THEMIS_ASI_DATA_ROOT`` if set, else ``./data/cdf``.
    """
    env = os.environ.get(ENV_DATA_ROOT)
    return Path(env).expanduser() if env else Path.cwd() / "data" / "cdf"


def default_weights_dir() -> Path:
    """Where the model checkpoint is cached.

    ``$THEMIS_ASI_WEIGHTS`` if set, else ``./weights``.
    """
    env = os.environ.get(ENV_WEIGHTS)
    return Path(env).expanduser() if env else Path.cwd() / "weights"


# Constants the copied engine CLIs (concat/index/inventory/embed ``_cli``)
# reference by name. They are convenience defaults only; every library entry
# point also accepts explicit paths.
ARTIFACTS_ROOT = default_artifacts_root()
DEFAULT_INVENTORY_PATH = ARTIFACTS_ROOT / "inventory.parquet"
