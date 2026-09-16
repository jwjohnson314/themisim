"""Tile captions must identify a frame unambiguously.

A similarity grid invites the reader to compare tiles, so a caption that cannot
distinguish two results is worse than no caption. The failure this guards
against is real: the captions used to show only ``HH:MM:SS``, which made two
frames nine years apart — an ordinary occurrence, since the pilot slices span
2015 and 2024 — look identical.

The caption builder is a pure function so these assertions need no matplotlib;
the render test below is skipped when it is absent.
"""
from __future__ import annotations

import pandas as pd
import pytest

from themisim.viz import _caption_fontsize, _tile_caption


def test_caption_carries_site_date_time_and_score():
    caption = _tile_caption("fsmi", "2024-03-25 04:59:09 UTC", 0.98765)
    assert caption == "fsmi  2024-03-25\n04:59:09 UTC\nscore 0.988"
    for field in ("fsmi", "2024-03-25", "04:59:09", "0.988"):
        assert field in caption


def test_caption_distinguishes_frames_from_different_years():
    """The regression this file exists for: same clock time, different date."""
    a = _tile_caption("fsmi", "2015-03-18 04:59:09 UTC", 1.0)
    b = _tile_caption("fsmi", "2024-03-25 04:59:09 UTC", 1.0)
    assert a != b
    assert "2015-03-18" in a and "2024-03-25" in b


def test_caption_is_three_short_lines_not_one_long_one():
    """Legibility: a tile is ~2 inches wide, so no line may be sprawling."""
    caption = _tile_caption("fsmi", "2024-03-25 04:59:09 UTC", 0.5)
    lines = caption.split("\n")
    assert len(lines) == 3
    assert max(len(line) for line in lines) <= 18


def test_caption_survives_an_unexpected_datetime_format():
    """A malformed caption must not take down an otherwise valid render."""
    assert _tile_caption("gill", "2024-03-25", 0.5) == "gill  2024-03-25\nscore 0.500"
    assert _tile_caption("gill", "", 0.25) == "gill\nscore 0.250"


@pytest.mark.parametrize("score,expected", [(1.0, "1.000"), (0.0, "0.000"), (0.9995, "1.000")])
def test_score_is_always_three_decimals(score, expected):
    assert _tile_caption("fsmi", "2024-03-25 04:59:09 UTC", score).endswith(expected)


def test_fontsize_scales_with_tile_but_stays_readable():
    assert _caption_fontsize(2.2) == pytest.approx(7.92)
    assert _caption_fontsize(0.5) == 7.0    # floor: never illegible
    assert _caption_fontsize(10.0) == 11.0  # ceiling: never absurd
    assert _caption_fontsize(4.0) > _caption_fontsize(2.0)


def test_rendered_titles_match_the_caption_builder(tmp_path):
    """End-to-end through matplotlib, with no CDFs available.

    ``download=False`` against an empty tree makes every frame unavailable, so
    the tiles render the placeholder — but the titles are set regardless, which
    is exactly what is under test.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from themisim.viz import visualize_results

    df = pd.DataFrame(
        [
            {"site": "fsmi", "datetime": "2015-03-18 06:20:00 UTC", "score": 1.0,
             "source_cdf": "http://x/thg_l1_asf_fsmi_2015031806_v01.cdf"},
            {"site": "gill", "datetime": "2024-03-25 07:01:21 UTC", "score": 0.9412,
             "source_cdf": "http://x/thg_l1_asf_gill_2024032507_v01.cdf"},
        ]
    )
    fig = visualize_results(df, data_root=tmp_path, download=False, cols=2)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == [
        "fsmi  2015-03-18\n06:20:00 UTC\nscore 1.000",
        "gill  2024-03-25\n07:01:21 UTC\nscore 0.941",
    ]
    # The extra row height must go to the caption, not come out of the images.
    width, height = fig.get_size_inches()
    assert height > 2.2, "row height should make room for the three-line caption"
