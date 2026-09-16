"""The frozen pilot specs must stay internally consistent and reachable.

A spec is a promise that a specific set of bytes on the Berkeley archive can be
turned into a specific index. These tests are offline: they check the promise is
*well-formed* (parseable, non-duplicated, round-trips through the same filename
grammar every other stage uses) and that the slices still satisfy the coverage
criteria they were chosen for. Whether the archive still serves those bytes is
checked by the ``pilot``-marked tests, which need the network.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from themisim import pilot
from themisim.catalog import CDF_NAME_RE
from themisim.embed import parse_cdf_filename

SPECS = pilot.list_pilots()


def test_registry_is_not_empty():
    assert SPECS, "no pilot specs are registered; the packaged JSON is missing"
    assert {s.name for s in SPECS} >= {"tiny", "small"}


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_spec_is_well_formed(spec):
    assert spec.description.strip()
    assert spec.n_files == len(spec.cdfs) > 0
    assert spec.nprobe > 0 and spec.prefilter > 0
    assert spec.expected_vectors and spec.expected_vectors > 0, "run --freeze"
    assert spec.nlist > 0, "nlist must be pinned by --freeze, not left to auto"
    assert spec.thresholds, "a spec without calibrated gates cannot fail"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_cdf_addresses_round_trip_the_archive_filename_grammar(spec):
    """Every pinned CDF must survive crawl -> download -> embed name parsing.

    ``catalog.CDF_NAME_RE`` is deliberately ``_v01``-only and
    ``embed.parse_cdf_filename`` must agree with it, so a spec entry that any
    stage would silently drop is caught here rather than after a 6 GB download.
    """
    for cdf in spec.cdfs:
        assert CDF_NAME_RE.match(cdf.filename), cdf.filename
        site, year, dtstr = parse_cdf_filename(cdf.filename)
        assert (site, dtstr) == (cdf.site, cdf.datetime)
        assert cdf.url.endswith(f"/{cdf.site}/{year}/{cdf.datetime[4:6]}/{cdf.filename}")
        assert re.fullmatch(r"[a-z]{4}", cdf.site)
        assert re.fullmatch(r"\d{10}", cdf.datetime)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_no_duplicate_hours(spec):
    keys = [(c.site, c.datetime) for c in spec.cdfs]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_every_cdf_is_pinned_by_size_and_digest(spec):
    for cdf in spec.cdfs:
        assert cdf.size_bytes > 1_000_000, cdf.filename
        assert re.fullmatch(r"[0-9a-f]{64}", cdf.sha256), cdf.filename
    assert len({c.sha256 for c in spec.cdfs}) == spec.n_files, "duplicate content"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_slice_meets_its_coverage_criteria(spec):
    """The slices were chosen to exercise specific code, not at random.

    Losing any of these silently would make the pilot cover less than the paper
    claims, so they are asserted rather than left in a comment.
    """
    # >=2 sites: stratified_sample_from_parquet buckets by site, and
    # SearchEngine._init_site_blocks needs more than one block to mean anything.
    assert len(spec.sites) >= 2

    # >=2 year-months: the training sampler's bucket key is (site, year_month).
    assert len(spec.year_months) >= 2

    # >=1 CDF from 2024+: embed._read_cdf has two mutually exclusive layout
    # branches and only real 2024+ data reaches the second one.
    assert any(c.datetime[:4] >= "2024" for c in spec.cdfs)

    # Overlapping UT hours across sites, so cross-station retrieval is possible
    # at all. Without it every match is a within-site near-duplicate and the
    # results demonstrate temporal autocorrelation rather than similarity.
    by_hour: dict[str, set[str]] = {}
    for c in spec.cdfs:
        by_hour.setdefault(c.datetime, set()).add(c.site)
    assert any(len(sites) >= 2 for sites in by_hour.values())

    # Consecutive hours at one site, so shard-boundary handling is exercised.
    hours = sorted((c.site, c.datetime) for c in spec.cdfs)
    assert any(
        s1 == s2 and int(d2) - int(d1) == 1 for (s1, d1), (s2, d2) in zip(hours, hours[1:])
    )


def test_expected_vectors_is_not_a_round_multiple_of_1200():
    """At least one slice must contain a ragged (partial) hour.

    A full THEMIS hour is ~1200 frames; night-edge hours are shorter. If every
    hour were full, nothing would catch code that assumes a constant frame count.
    """
    assert any(s.expected_vectors % 1200 != 0 for s in SPECS)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_estimate_is_positive_and_ordered(spec):
    est = spec.estimate()
    assert est["total_s"] > 0
    assert est["total_s"] >= est["download_s"] + est["embed_s"]


def test_unknown_spec_names_what_is_available():
    with pytest.raises(KeyError, match="tiny"):
        pilot.get_pilot("does-not-exist")


def test_worklist_matches_the_crawler_layout(tmp_path):
    """A pilot tree and a crawled tree must be interchangeable on disk."""
    from themisim.catalog import WORKLIST_COLUMNS

    spec = pilot.get_pilot("tiny")
    wl = pilot.worklist_for(spec, tmp_path)
    assert list(wl.columns) == WORKLIST_COLUMNS
    assert len(wl) == spec.n_files
    for row in wl.itertuples():
        # <data_root>/<site>/<YYYY>/<MM>/<filename>, as catalog.build_worklist
        rel = str(row.target_path).replace(str(tmp_path) + "/", "")
        assert rel == f"{row.site}/{row.datetime[:4]}/{row.datetime[4:6]}/{row.filename}"


def test_tiny_is_a_subset_of_small():
    """One pinned reference-vector file has to serve both slices.

    ``check_encoder_reference`` addresses frames by ``(site, datetime,
    frame_idx)``, so the reference frames must exist in every slice it is
    validated against. Keeping ``tiny`` a strict subset of ``small`` is what
    makes a single 46 KB file cover both; if the slices diverged, validating
    ``small`` would silently SKIP the encoder check instead of failing.
    """
    tiny = {(c.site, c.datetime) for c in pilot.get_pilot("tiny").cdfs}
    small = {(c.site, c.datetime) for c in pilot.get_pilot("small").cdfs}
    assert tiny < small


def test_reference_vectors_are_packaged_and_addressable():
    """The pinned vectors must exist and point into the tiny slice."""
    import numpy as np

    path = Path(pilot.__file__).with_name("data") / "pilot_reference_vectors.npz"
    assert path.exists(), "run --freeze --reference-out to regenerate"
    with np.load(path, allow_pickle=False) as npz:
        assert npz["vectors"].dtype == np.float16
        assert npz["vectors"].shape[1] == 512
        n = len(npz["vectors"])
        assert n >= 25
        assert len(npz["site"]) == len(npz["datetime"]) == len(npz["frame_idx"]) == n
        pinned = {(c.site, c.datetime) for c in pilot.get_pilot("tiny").cdfs}
        for site, dt in zip(npz["site"], npz["datetime"]):
            assert (str(site), str(dt)) in pinned
