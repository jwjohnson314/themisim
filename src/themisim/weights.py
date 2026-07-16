"""Fetch (and verify) the pretrained SimCLR encoder checkpoint.

The ~44 MB checkpoint is **not** committed to the repository — it is published
on Hugging Face (``Jwjohnson314/Aurora-FM``, CC-BY-4.0) and pulled on demand.
``fetch_weights`` caches it under ``./weights`` (or ``$THEMIS_ASI_WEIGHTS``) and
checks its SHA-256 so a truncated or wrong download is caught before it reaches
the encoder.

The download URL is ``DEFAULT_WEIGHTS_URL`` (the public HF ``resolve`` link,
overridable via ``$THEMIS_ASI_WEIGHTS_URL`` or the ``url=`` argument). Users who
already have the ``.tar`` can drop it in the weights directory and
``fetch_weights`` will verify and reuse it without a network call.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional, Union
from urllib.request import Request, urlopen

from themisim.config import DEFAULT_CHECKPOINT_NAME, default_weights_dir

# Pinned SHA-256 of aurora-fm-no-finetune.tar
# (45,965,389 bytes). Used to verify any local or downloaded copy.
EXPECTED_SHA256 = "f9c97e906bee7a34f0727377f8b3194086080fa9573390db44daab4ce2c64e33"

# Public download URL for the checkpoint (Hugging Face ``resolve`` link, which
# 302-redirects to the CDN; urllib follows it). Override via
# ``$THEMIS_ASI_WEIGHTS_URL`` or the ``url=`` argument.
DEFAULT_WEIGHTS_URL = os.environ.get(
    "THEMIS_ASI_WEIGHTS_URL",
    "https://huggingface.co/Jwjohnson314/Aurora-FM/resolve/main/aurora-fm-no-finetune.tar",
)

_USER_AGENT = "themis-asi-search/0.1"
_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Union[str, Path], chunk: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _verify(path: Path, expected_sha256: Optional[str]) -> bool:
    if expected_sha256 is None:
        return True
    return sha256_file(path) == expected_sha256.lower()


def fetch_weights(
    dest_dir: Optional[Union[str, Path]] = None,
    *,
    url: Optional[str] = None,
    expected_sha256: Optional[str] = EXPECTED_SHA256,
    force: bool = False,
) -> Path:
    """Return a local path to the verified checkpoint, downloading if needed.

    Parameters
    ----------
    dest_dir : str | Path, optional
        Directory to cache the checkpoint in. Defaults to
        ``$THEMIS_ASI_WEIGHTS`` or ``./weights``.
    url : str, optional
        Override the download URL. Defaults to ``DEFAULT_WEIGHTS_URL`` /
        ``$THEMIS_ASI_WEIGHTS_URL``.
    expected_sha256 : str, optional
        Verify the file against this digest (default: the pinned hash). Pass
        ``None`` to skip verification.
    force : bool
        Re-download even if a valid local copy exists.
    """
    dest_dir = Path(dest_dir) if dest_dir is not None else default_weights_dir()
    dest = dest_dir / DEFAULT_CHECKPOINT_NAME

    if dest.exists() and not force:
        if _verify(dest, expected_sha256):
            return dest
        raise RuntimeError(
            f"existing checkpoint {dest} failed SHA-256 verification; "
            f"delete it and retry, or pass force=True to re-download"
        )

    download_url = url or DEFAULT_WEIGHTS_URL
    if not download_url:
        raise RuntimeError(
            f"no checkpoint at {dest} and no download URL configured. Either "
            f"place {DEFAULT_CHECKPOINT_NAME} in {dest_dir}, pass url=..., or "
            f"set $THEMIS_ASI_WEIGHTS_URL (see README)."
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = Request(download_url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            fh.write(chunk)

    if not _verify(part, expected_sha256):
        got = sha256_file(part)
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"downloaded checkpoint failed SHA-256 verification "
            f"(expected {expected_sha256}, got {got})"
        )
    part.replace(dest)
    return dest
