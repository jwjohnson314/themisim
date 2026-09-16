"""Encoder loading and checkpoint verification.

Two distinct failure modes are covered. ``load_simclr`` loads with
``strict=True``, so a checkpoint whose keys drift from the ``SimCLR`` class
fails loudly — that is the intended behaviour and it should stay that way,
because silently loading a partially-initialised encoder would produce
plausible-looking embeddings that mean nothing.

``fetch_weights`` must fail *closed*: a truncated or substituted checkpoint has
to be rejected rather than embedded into a billion vectors. The tests below use
a locally generated checkpoint, so nothing here touches the network.
"""
from __future__ import annotations

import hashlib

import pytest
import torch

from themisim.model import N_FEATURES, PROJECTION_DIM, SimCLR, load_simclr
from themisim.weights import DEFAULT_CHECKPOINT_NAME, fetch_weights, sha256_file


# --------------------------------------------------------------------------- #
# load_simclr
# --------------------------------------------------------------------------- #
def test_loads_and_returns_the_encoder_not_the_wrapper(synthetic_checkpoint):
    model = load_simclr(synthetic_checkpoint, device="cpu")
    # The projector is training-only; query time must get the encoder.
    assert not hasattr(model, "projector")
    assert not model.training, "must be in eval() mode; dropout/BN would drift"


def test_produces_the_contracted_512_d_output(synthetic_checkpoint):
    model = load_simclr(synthetic_checkpoint, device="cpu")
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, N_FEATURES)
    assert out.dtype == torch.float32


def test_final_layer_is_identity_so_features_are_not_classified(synthetic_checkpoint):
    """`fc` is replaced by Identity; a 1000-way ImageNet head would be wrong."""
    model = load_simclr(synthetic_checkpoint, device="cpu")
    assert model.fc.__class__.__name__ == "Identity"


def test_inference_is_deterministic(synthetic_checkpoint):
    model = load_simclr(synthetic_checkpoint, device="cpu")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_batch_size_does_not_change_the_embedding(synthetic_checkpoint):
    """Batching is an implementation detail; it must not move the vectors.

    If it did, shards embedded with different batch sizes would be mutually
    inconsistent and the index would be built on incomparable vectors.
    """
    model = load_simclr(synthetic_checkpoint, device="cpu")
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        whole = model(x)
        split = torch.cat([model(x[:1]), model(x[1:])])
    torch.testing.assert_close(whole, split, rtol=1e-5, atol=1e-5)


def test_a_mismatched_checkpoint_is_rejected(tmp_path, synthetic_checkpoint):
    """strict=True is deliberate: silent partial loading is the danger."""
    state = torch.load(synthetic_checkpoint, map_location="cpu", weights_only=True)
    state.pop(next(iter(state)))
    bad = tmp_path / "missing-key.tar"
    torch.save(state, bad)
    with pytest.raises(RuntimeError, match="Missing key|size mismatch|Unexpected key"):
        load_simclr(bad, device="cpu")


def test_simclr_wrapper_shape_matches_the_published_checkpoint_layout():
    import torchvision

    model = SimCLR(torchvision.models.resnet18(weights=None), PROJECTION_DIM, N_FEATURES)
    keys = model.state_dict().keys()
    assert any(k.startswith("encoder.") for k in keys)
    assert any(k.startswith("projector.") for k in keys)


# --------------------------------------------------------------------------- #
# fetch_weights — must fail closed
# --------------------------------------------------------------------------- #
def _place(dest_dir, payload: bytes):
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / DEFAULT_CHECKPOINT_NAME
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_existing_file_with_the_right_digest_is_reused_without_network(tmp_path):
    path, digest = _place(tmp_path / "w", b"pretend checkpoint")
    got = fetch_weights(tmp_path / "w", expected_sha256=digest, url=None)
    assert got == path


def test_existing_file_with_a_wrong_digest_is_refused(tmp_path):
    """A corrupted local checkpoint must not be silently embedded."""
    _place(tmp_path / "w", b"corrupted")
    with pytest.raises(RuntimeError, match="failed SHA-256"):
        fetch_weights(tmp_path / "w", expected_sha256="0" * 64, url=None)


def test_verification_can_be_skipped_explicitly(tmp_path):
    path, _ = _place(tmp_path / "w", b"unverified")
    assert fetch_weights(tmp_path / "w", expected_sha256=None, url=None) == path


def test_missing_file_and_no_url_explains_the_options(tmp_path, monkeypatch):
    """``url=None`` means "use the default", not "do not download".

    The default is the pinned Hugging Face URL, so this path is only reachable
    when that default is itself empty -- which is why the module-level constant
    has to be patched rather than the argument. Worth pinning: passing
    ``url=None`` expecting an offline call would quietly fetch 44 MB instead.
    """
    monkeypatch.setattr("themisim.weights.DEFAULT_WEIGHTS_URL", "")
    with pytest.raises(RuntimeError, match="no checkpoint at"):
        fetch_weights(tmp_path / "empty", url=None, expected_sha256=None)


def test_a_download_with_the_wrong_digest_is_rejected_and_discarded(tmp_path):
    """Fail closed on a substituted checkpoint, and leave nothing behind.

    Served over ``file://`` so the verification path is exercised without the
    network. A partial file surviving a failed verification would be loaded on
    the next run as though it were valid.
    """
    source = tmp_path / "served.tar"
    source.write_bytes(b"not the real checkpoint")
    dest_dir = tmp_path / "w"

    with pytest.raises(RuntimeError, match="failed SHA-256 verification"):
        fetch_weights(dest_dir, url=source.resolve().as_uri(), expected_sha256="0" * 64)

    assert not (dest_dir / DEFAULT_CHECKPOINT_NAME).exists()
    assert not list(dest_dir.glob("*.part")), "failed download must not linger"


def test_a_download_with_the_right_digest_is_kept(tmp_path):
    payload = b"a plausible checkpoint"
    source = tmp_path / "served.tar"
    source.write_bytes(payload)
    dest_dir = tmp_path / "w"

    got = fetch_weights(
        dest_dir,
        url=source.resolve().as_uri(),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert got.read_bytes() == payload


def test_sha256_file_matches_hashlib(tmp_path):
    payload = b"themis" * 1000
    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert sha256_file(f) == hashlib.sha256(payload).hexdigest()
