"""Resolution math for (site, datetime, frame) -> global_id, no real index."""
import datetime as _dt

import pytest

from themisim.query import _parse_hour, resolve_global_id


class StubEngine:
    """Minimal stand-in exposing only what build_browse_index touches.

    Two contiguous one-hour shards for site 'fsmi':
      gids    0..1199  -> 2015031806 (hour 06)
      gids 1200..2399  -> 2015031807 (hour 07)
    """

    _SHARDS = {
        0: "/art/shards/fsmi/2015/2015031806.f16.npy",
        1200: "/art/shards/fsmi/2015/2015031807.f16.npy",
    }

    def shard_starts(self):
        return [0, 1200, 2400]

    def site_of(self, gid):
        return "fsmi"

    def shard_of(self, gid):
        return self._SHARDS[gid]


@pytest.fixture
def engine():
    return StubEngine()


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2015-03-18T06", ("2015-03-18", "06")),
        ("2015-03-18 06", ("2015-03-18", "06")),
        ("2015-03-18T06:20:36", ("2015-03-18", "06")),
        ("2015031806", ("2015-03-18", "06")),
        (_dt.datetime(2015, 3, 18, 6, 20, 36), ("2015-03-18", "06")),
    ],
)
def test_parse_hour(value, expected):
    assert _parse_hour(value) == expected


def test_parse_hour_bad():
    with pytest.raises(ValueError):
        _parse_hour("not-a-date")


def test_resolve_basic(engine):
    assert resolve_global_id(engine, "fsmi", "2015-03-18T06", 412) == 412
    assert resolve_global_id(engine, "fsmi", "2015-03-18T07", 0) == 1200


def test_resolve_frame_out_of_range(engine):
    with pytest.raises(IndexError):
        resolve_global_id(engine, "fsmi", "2015-03-18T06", 1200)
    with pytest.raises(IndexError):
        resolve_global_id(engine, "fsmi", "2015-03-18T06", -1)


def test_resolve_unknown_site(engine):
    with pytest.raises(KeyError):
        resolve_global_id(engine, "nope", "2015-03-18T06", 0)


def test_resolve_unknown_hour(engine):
    with pytest.raises(KeyError):
        resolve_global_id(engine, "fsmi", "2015-03-18T09", 0)
