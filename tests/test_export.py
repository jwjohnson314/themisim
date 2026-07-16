"""The export table must match the dashboard's CSV format exactly."""
import datetime as _dt

from themisim.export import (
    EXPORT_COLUMNS,
    hit_cdf_url,
    hits_to_dataframe,
    results_csv_filename,
    themis_cdf_url,
    time_ns_utc_iso,
)
from themisim.search import Hit


def _ns(year, month, day, hour, minute, second):
    dt = _dt.datetime(year, month, day, hour, minute, second, tzinfo=_dt.timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _hit(score=1.0):
    return Hit(
        global_id=412,
        score=score,
        site="fsmi",
        time_ns=_ns(2015, 3, 18, 6, 20, 36),
        shard_path="/data/artifacts/shards/fsmi/2015/2015031806.f16.npy",
        frame_idx=412,
    )


def test_url_layout():
    assert themis_cdf_url("fsmi", "2015031806") == (
        "http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/"
        "fsmi/2015/03/thg_l1_asf_fsmi_2015031806_v01.cdf"
    )
    assert hit_cdf_url(_hit()).endswith("thg_l1_asf_fsmi_2015031806_v01.cdf")


def test_time_format():
    assert time_ns_utc_iso(_ns(2015, 3, 18, 6, 20, 36)) == "2015-03-18 06:20:36 UTC"


def test_dataframe_columns_and_values():
    df = hits_to_dataframe([_hit(1.0), _hit(0.5)])
    assert list(df.columns) == EXPORT_COLUMNS == ["site", "datetime", "score", "source_cdf"]
    assert df.iloc[0]["site"] == "fsmi"
    assert df.iloc[0]["datetime"] == "2015-03-18 06:20:36 UTC"
    assert df.iloc[0]["score"] == 1.0
    assert df.iloc[0]["source_cdf"].startswith("http://themis.ssl.berkeley.edu/")


def test_empty_dataframe_keeps_schema():
    df = hits_to_dataframe([])
    assert list(df.columns) == EXPORT_COLUMNS
    assert len(df) == 0


def test_csv_filename():
    name = results_csv_filename("fsmi", _ns(2015, 3, 18, 6, 20, 36))
    assert name == "themis-similarity-search-fsmi-20150318T062036Z.csv"
    assert results_csv_filename(None, None) == "themis-similarity-search.csv"
