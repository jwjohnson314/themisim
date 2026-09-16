"""Resolution math for (site, datetime, frame) -> global_id, no real index."""
import datetime as _dt
from importlib import import_module

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


def test_browse_cache_entry_dies_with_its_engine():
    """The browse-index cache must not survive the engine it was built for.

    It used to be keyed on ``id(engine)``, which is only unique while that
    object is alive. Once an engine was collected CPython could hand the same
    address to a *different* engine over a *different* artifacts directory, and
    ``resolve_global_id`` would then answer from the wrong archive's browse
    index. Keying the cache weakly on the engine makes that impossible.
    """
    import gc
    import weakref

    # NB: neither `from themisim import query` nor `import themisim.query as q`
    # gives the module -- themisim/__init__.py binds the exported *function*
    # over the submodule attribute, and `import ... as` prefers the attribute.
    q = import_module("themisim.query")

    engine = StubEngine()
    assert q._browse_index_for(engine) is q._browse_index_for(engine)  # cached
    before = len(q._BROWSE_CACHE)
    assert before >= 1

    ref = weakref.ref(engine)
    del engine
    gc.collect()

    assert ref() is None, "engine should be collectable; the cache must not pin it"
    assert len(q._BROWSE_CACHE) == before - 1


def test_browse_indexes_are_per_engine():
    """Two engines over different data must not share a browse index."""
    # NB: neither `from themisim import query` nor `import themisim.query as q`
    # gives the module -- themisim/__init__.py binds the exported *function*
    # over the submodule attribute, and `import ... as` prefers the attribute.
    q = import_module("themisim.query")

    class OtherEngine(StubEngine):
        _SHARDS = {0: "/art/shards/gill/2016/2016011903.f16.npy"}

        def shard_starts(self):
            return [0, 900]

        def site_of(self, gid):
            return "gill"

    a, b = StubEngine(), OtherEngine()
    assert set(q._browse_index_for(a)) == {"fsmi"}
    assert set(q._browse_index_for(b)) == {"gill"}
