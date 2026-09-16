"""Pilot build guards: the index must be the pinned slice, or the build fails.

Isolated directories are only advisory. ``build_index`` inventories a whole
``--data-root`` tree and ``concat.discover_shards`` globs every shard under the
artifacts directory, so a contaminated directory yields a perfectly valid index
over the wrong data. These tests cover the guards that turn that into a loud
failure. Nothing here touches the network.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import SYNTHETIC_SHARDS, make_vectors, write_shard
from themisim import pilot


@pytest.fixture
def spec():
    return pilot.get_pilot("tiny")


def _empty_tree(tmp_path):
    (tmp_path / "cdf").mkdir()
    (tmp_path / "artifacts").mkdir()
    return tmp_path / "cdf", tmp_path / "artifacts"


def test_paths_default_to_an_isolated_root():
    data_root, artifacts = pilot.pilot_paths("tiny")
    assert data_root.parts[-3:] == ("pilot", "tiny", "cdf")
    assert artifacts.parts[-3:] == ("pilot", "tiny", "artifacts")
    assert data_root.parent == artifacts.parent


def test_explicit_paths_win():
    data_root, artifacts = pilot.pilot_paths("tiny", "/a", "/b")
    assert (str(data_root), str(artifacts)) == ("/a", "/b")


def test_preflight_accepts_an_empty_tree(spec, tmp_path):
    data_root, artifacts = _empty_tree(tmp_path)
    pilot.preflight(spec, data_root, artifacts, need_download=False)


def test_preflight_rejects_foreign_cdfs(spec, tmp_path):
    """A reviewer pointing --data-root at an existing archive tree.

    ``build_inventory`` rglobs the whole tree unfiltered, so this would build a
    valid index over data the spec does not pin.
    """
    data_root, artifacts = _empty_tree(tmp_path)
    intruder = data_root / "zzzz" / "2011" / "07"
    intruder.mkdir(parents=True)
    (intruder / "thg_l1_asf_zzzz_2011071500_v01.cdf").write_bytes(b"x")

    with pytest.raises(RuntimeError, match="does not pin"):
        pilot.preflight(spec, data_root, artifacts, need_download=False)


def test_preflight_rejects_foreign_shards(spec, tmp_path):
    """``concat.discover_shards`` globs every shard under the artifacts dir."""
    data_root, artifacts = _empty_tree(tmp_path)
    write_shard(
        artifacts / "shards", "zzzz", "2011071500",
        make_vectors(4, np.random.default_rng(0)),
    )
    with pytest.raises(RuntimeError, match="outside the"):
        pilot.preflight(spec, data_root, artifacts, need_download=False)


def test_preflight_refuses_without_disk_space(spec, tmp_path, monkeypatch):
    import shutil

    data_root, artifacts = _empty_tree(tmp_path)
    monkeypatch.setattr(
        shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(1, 1, 1024)
    )
    with pytest.raises(RuntimeError, match="GB free"):
        pilot.preflight(spec, data_root, artifacts, need_download=True)


def test_verify_slice_reports_missing_files(spec, tmp_path):
    report = pilot.verify_slice(spec, tmp_path, checksums=False)
    assert report["ok"] is False
    assert len(report["missing"]) == spec.n_files


def test_verify_slice_detects_a_wrong_size(spec, tmp_path):
    cdf = spec.cdfs[0]
    target = cdf.target_path(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the real file")
    report = pilot.verify_slice(spec, tmp_path, checksums=False)
    assert [w["file"] for w in report["wrong_size"]] == [cdf.filename]
    assert report["ok"] is False


def test_verify_slice_detects_a_wrong_digest(tmp_path):
    """Right size, wrong bytes: only the checksum can tell them apart.

    Uses a one-file synthetic spec rather than a real one so the test does not
    have to write 116 MB of padding to match a real CDF's length.
    """
    from themisim.pilot import PilotCdf, PilotSpec

    payload = b"the bytes we froze"
    decoy = b"x" * len(payload)
    one = PilotSpec(
        name="one",
        description="single-file spec",
        cdfs=(PilotCdf("fsmi", "2015031806", len(payload), _sha(payload)),),
        nlist=8,
        nprobe=8,
    )
    target = one.cdfs[0].target_path(tmp_path)
    target.parent.mkdir(parents=True)

    target.write_bytes(payload)
    assert pilot.verify_slice(one, tmp_path, checksums=True)["ok"] is True

    target.write_bytes(decoy)
    assert pilot.verify_slice(one, tmp_path, checksums=False)["ok"] is True, (
        "a size-only check cannot see this, which is why checksums are the default"
    )
    checked = pilot.verify_slice(one, tmp_path, checksums=True)
    assert checked["wrong_hash"] == [one.cdfs[0].filename]
    assert checked["ok"] is False


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_read_embed_failures_is_none_when_clean(tmp_path):
    assert pilot.read_embed_failures(tmp_path) is None


def test_read_embed_failures_surfaces_silently_skipped_cdfs(tmp_path):
    """``build_index`` logs per-CDF embed failures and carries on regardless.

    A slice where a third of the hours failed to read still produces a valid
    index, so the failures file has to be surfaced or the wrongness is invisible.
    """
    pd.DataFrame(
        [{"site": "fsmi", "datetime": "2015031806", "drive_path": "/x",
          "exception_type": "ValueError", "exception_msg": "bad geometry",
          "traceback": ""}]
    ).to_parquet(tmp_path / "embed_failures.parquet", index=False)
    failures = pilot.read_embed_failures(tmp_path)
    assert failures is not None and len(failures) == 1


def test_build_pilot_rejects_a_short_slice(tmp_path, monkeypatch, shards_dir):
    """If fewer CDFs were embedded than the spec pins, refuse the index.

    Simulated by handing build_pilot a spec that pins more files than the
    pre-seeded shards provide, with download and verification stubbed out.
    """
    from themisim.pilot import PilotCdf, PilotSpec

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "shards").symlink_to(shards_dir, target_is_directory=True)

    fake = PilotSpec(
        name="fake",
        description="pins one more hour than the shards provide",
        cdfs=tuple(
            PilotCdf(site, dt, 1, "0" * 64) for site, dt, _ in SYNTHETIC_SHARDS
        ) + (PilotCdf("cccc", "2015031809", 1, "0" * 64),),
        nlist=8,
        nprobe=8,
    )
    monkeypatch.setattr(pilot, "preflight", lambda *a, **k: None)
    # Mirror verify_slice's real return shape: build_pilot inspects the
    # individual failure lists, not just "ok".
    monkeypatch.setattr(
        pilot, "verify_slice",
        lambda *a, **k: {"missing": [], "wrong_size": [], "wrong_hash": [], "ok": True},
    )
    monkeypatch.setattr(
        "themisim.pipeline.build_index",
        lambda *a, **k: _fake_build(shards_dir, artifacts),
    )

    with pytest.raises(RuntimeError, match="pins 7 CDFs"):
        pilot.build_pilot(
            fake, data_root=tmp_path / "cdf", artifacts=artifacts,
            checkpoint="unused", download=False, verify_checksums=False,
            validate=False,
        )


def test_archive_drift_warns_by_default_but_fails_under_strict(tmp_path, monkeypatch, shards_dir):
    """A reprocessed archive hour must not make a published spec unbuildable.

    Pinning digests exists to make divergence visible, not to brick the tool the
    day Berkeley regenerates a file. Default: warn and record. --strict: fail.
    """
    from themisim.pilot import PilotCdf, PilotSpec

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "shards").symlink_to(shards_dir, target_is_directory=True)
    spec = PilotSpec(
        name="fake",
        description="matches the synthetic shard set exactly",
        cdfs=tuple(PilotCdf(site, dt, 1, "0" * 64) for site, dt, _ in SYNTHETIC_SHARDS),
        nlist=8,
        nprobe=8,
    )
    drifted = {"missing": [], "wrong_size": [], "wrong_hash": ["a.cdf"], "ok": False}
    monkeypatch.setattr(pilot, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(pilot, "verify_slice", lambda *a, **k: drifted)
    monkeypatch.setattr(
        "themisim.pipeline.build_index",
        lambda *a, **k: _fake_build(shards_dir, artifacts),
    )
    kw = dict(
        data_root=tmp_path / "cdf", artifacts=artifacts, checkpoint="unused",
        download=False, validate=False,
    )

    events = []
    result = pilot.build_pilot(spec, on_progress=lambda s, i: events.append((s, i)), **kw)
    assert result.n_shards == len(SYNTHETIC_SHARDS)
    assert any(stage == "warning" and "wrong checksum" in str(info) for stage, info in events)

    with pytest.raises(RuntimeError, match="reprocessed"):
        pilot.build_pilot(spec, strict_checksums=True, **kw)


def test_missing_files_are_always_fatal(tmp_path, monkeypatch):
    """Absent input is unrecoverable however lenient the checksum policy is."""
    from themisim.pilot import PilotCdf, PilotSpec

    spec = PilotSpec(
        name="fake", description="x",
        cdfs=(PilotCdf("aaaa", "2015031806", 1, "0" * 64),), nlist=8, nprobe=8,
    )
    monkeypatch.setattr(pilot, "preflight", lambda *a, **k: None)
    monkeypatch.setattr(
        pilot, "verify_slice",
        lambda *a, **k: {"missing": ["a.cdf"], "wrong_size": [], "wrong_hash": [], "ok": False},
    )
    with pytest.raises(RuntimeError, match="missing"):
        pilot.build_pilot(
            spec, data_root=tmp_path, artifacts=tmp_path / "art",
            checkpoint="unused", download=False, validate=False,
        )


def _fake_build(shards_dir, artifacts):
    """Run the real concat, skipping only the torch-dependent embed stage."""
    from themisim.concat import build_memmap_and_manifest

    build_memmap_and_manifest(shards_dir, artifacts)
    return artifacts / "index.faiss"


# --------------------------------------------------------------------------- #
# the real thing
# --------------------------------------------------------------------------- #
@pytest.mark.pilot
def test_tiny_pilot_builds_and_validates(tmp_path):
    """End-to-end against the live archive. Needs network and several minutes."""
    result = pilot.build_pilot("tiny", device="cpu", validate=True)
    assert result.ok, json.dumps(result.report["checks"], indent=2, default=str)
    assert result.total_n == pilot.get_pilot("tiny").expected_vectors
