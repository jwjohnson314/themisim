"""group_inventory_by_drive: explicit --drive-root and st_dev grouping.

Replaces the retired hardcoded DEFAULT_DRIVES behaviour. Both the streaming
coordinator and each worker call this and must agree on the ordered group list,
so the ordering assertions here guard that determinism.
"""
import pandas as pd

from themisim.embed import group_inventory_by_drive


def _inv(paths):
    return pd.DataFrame(
        {
            "site": [p.rsplit("/", 2)[-2][:4] for p in paths],
            "datetime": ["2015031806"] * len(paths),
            "drive_path": paths,
        }
    )


def test_explicit_drive_roots_group_and_order():
    df = _inv(
        [
            "/mnt/b/fsmi/x.cdf",
            "/mnt/a/atha/x.cdf",
            "/mnt/a/fsmi/x.cdf",
        ]
    )
    groups = group_inventory_by_drive(df, drive_roots=["/mnt/a", "/mnt/b"])
    keys = [k for k, _ in groups]
    # Declared order preserved (a before b), regardless of row order.
    assert keys == ["/mnt/a", "/mnt/b"]
    a = dict(groups)["/mnt/a"]
    assert len(a) == 2
    # Each slice is sorted by (site, datetime): atha before fsmi.
    assert list(a["site"]) == ["atha", "fsmi"]


def test_rows_outside_every_root_go_to_sentinel():
    df = _inv(["/mnt/a/fsmi/x.cdf", "/elsewhere/fsmi/x.cdf"])
    groups = group_inventory_by_drive(df, drive_roots=["/mnt/a"])
    by_key = dict(groups)
    assert set(by_key) == {"/mnt/a", ""}
    assert len(by_key[""]) == 1  # surfaced, not silently dropped


def test_trailing_slash_in_root_is_normalized():
    df = _inv(["/mnt/a/fsmi/x.cdf"])
    groups = group_inventory_by_drive(df, drive_roots=["/mnt/a/"])
    assert [k for k, _ in groups] == ["/mnt/a"]


def test_stdev_grouping_single_device(tmp_path):
    # Real dirs under one tmp filesystem all share an st_dev -> exactly one
    # group, which is what un-breaks streaming for a single-mount install.
    d = tmp_path / "fsmi" / "2015"
    d.mkdir(parents=True)
    paths = [str(d / "a.cdf"), str(d / "b.cdf")]
    groups = group_inventory_by_drive(_inv(paths))  # no drive_roots -> st_dev
    assert len(groups) == 1
    key, slice_df = groups[0]
    assert key.startswith("dev")
    assert len(slice_df) == 2
