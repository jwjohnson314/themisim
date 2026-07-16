"""Crawl + work-list construction, with the network stubbed out."""
from pathlib import Path

import pytest

from themisim import catalog
from themisim.config import ARCHIVE_BASE_URL

BASE = ARCHIVE_BASE_URL  # ends with '/'

# A tiny fake Apache autoindex tree for site fsmi.
_TREE = {
    BASE + "fsmi/": ["../", "2015/"],
    BASE + "fsmi/2015/": ["03/"],
    BASE + "fsmi/2015/03/": [
        "thg_l1_asf_fsmi_2015031806_v01.cdf",
        "thg_l1_asf_fsmi_2015031807_v02.cdf",   # reprocessed version, must be ignored
        "thg_l1_ast_fsmi_2015031806_v01.cdf",  # thumbnail, must be ignored
        "index.html?C=N;O=D",                   # query link, must be ignored
    ],
}


@pytest.fixture
def stub_listing(monkeypatch):
    def fake_list(url, opener, timeout=30):
        return _TREE.get(url, [])

    monkeypatch.setattr(catalog, "_list_directory", fake_list)


def test_crawl_finds_only_asf(stub_listing):
    df = catalog.crawl(["fsmi"], rate_limit_s=0.0)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["site"] == "fsmi"
    assert row["datetime"] == "2015031806"
    assert row["filename"] == "thg_l1_asf_fsmi_2015031806_v01.cdf"


def test_crawl_ignores_non_v01(stub_listing):
    # The whole pipeline is v01-throughout (inventory glob, embed parser, and
    # the reconstructed download URL all assume _v01), so the crawl must not
    # queue a reprocessed _v02 file the downstream stages would silently drop.
    df = catalog.crawl(["fsmi"], rate_limit_s=0.0)
    assert list(df["filename"]) == ["thg_l1_asf_fsmi_2015031806_v01.cdf"]
    assert not df["filename"].str.contains("_v02").any()


def test_crawl_date_filter_excludes(stub_listing):
    # File is March 2015; an April-2015 lower bound should drop it.
    df = catalog.crawl(["fsmi"], start="2015-04", rate_limit_s=0.0)
    assert len(df) == 0


def test_build_worklist_layout(stub_listing, tmp_path):
    df = catalog.crawl(["fsmi"], rate_limit_s=0.0)
    wl = catalog.build_worklist(df, tmp_path)
    assert list(wl.columns) == catalog.WORKLIST_COLUMNS
    row = wl.iloc[0]
    expected = tmp_path / "fsmi" / "2015" / "03" / "thg_l1_asf_fsmi_2015031806_v01.cdf"
    assert Path(row["target_path"]) == expected
    assert row["url"].endswith("thg_l1_asf_fsmi_2015031806_v01.cdf")
    assert row["size_bytes"] == 0


def test_in_range():
    assert catalog._in_range("2015031806", None, None)
    assert catalog._in_range("2015031806", "2015-03", "2015-03")
    assert not catalog._in_range("2015031806", "2015-04", None)
    assert not catalog._in_range("2015031806", None, "2015-02")
