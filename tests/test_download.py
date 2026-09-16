"""The downloader, against a real HTTP server on localhost.

Nothing here touches the Berkeley archive. A ``http.server`` instance bound to
127.0.0.1 exercises the parts that matter and that a mock would paper over: the
HEAD size probe, the ``.part`` staging and atomic rename, the size-mismatch
retry, and the resume-by-skipping behaviour a reviewer on a flaky connection
depends on.

``download._https`` only rewrites archive URLs, so localhost passes through
untouched -- which is what makes this possible.
"""
from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from themisim import download


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output readable
        pass


@pytest.fixture
def server(tmp_path):
    """Serve ``tmp_path/www`` on an ephemeral localhost port."""
    root = tmp_path / "www"
    root.mkdir()
    handler = partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


PAYLOAD = b"THEMIS" * 5000  # 30 KB


def test_head_reports_the_true_size(server):
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    assert download.head_content_length(f"{base}/f.cdf") == len(PAYLOAD)


def test_head_on_a_missing_file_returns_none(server):
    _root, base = server
    assert download.head_content_length(f"{base}/nope.cdf", tries=1) is None


def test_downloads_and_renames_atomically(server, tmp_path):
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    target = tmp_path / "out" / "f.cdf"

    res = download.download_one(f"{base}/f.cdf", str(target))
    assert res.status == "downloaded"
    assert res.bytes_written == len(PAYLOAD)
    assert target.read_bytes() == PAYLOAD
    assert not list(target.parent.glob("*.part")), "staging file must not survive"


def test_existing_correct_file_is_skipped_not_refetched(server, tmp_path):
    """This is what makes an interrupted download resumable."""
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    target = tmp_path / "f.cdf"
    target.write_bytes(PAYLOAD)

    res = download.download_one(f"{base}/f.cdf", str(target))
    assert res.status == "skipped"
    assert res.bytes_written == 0


def test_truncated_local_file_is_refetched(server, tmp_path):
    """A half-written file from a killed run must not be mistaken for complete."""
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    target = tmp_path / "f.cdf"
    target.write_bytes(PAYLOAD[:100])

    res = download.download_one(f"{base}/f.cdf", str(target))
    assert res.status == "downloaded"
    assert target.read_bytes() == PAYLOAD


def test_missing_remote_file_fails_without_creating_a_target(server, tmp_path):
    _root, base = server
    target = tmp_path / "f.cdf"
    res = download.download_one(f"{base}/absent.cdf", str(target), tries=1)
    assert res.status == "failed"
    assert "HEAD failed" in res.error
    assert not target.exists()


def test_dry_run_touches_nothing(server, tmp_path):
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    target = tmp_path / "f.cdf"
    res = download.download_one(f"{base}/f.cdf", str(target), dry_run=True)
    assert res.status == "downloaded" and res.expected_bytes == len(PAYLOAD)
    assert not target.exists()


def test_stop_event_halts_before_fetching(server, tmp_path):
    """ENOSPC sets this event; in-flight work must stop rather than thrash."""
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    target = tmp_path / "f.cdf"
    stop = threading.Event()
    stop.set()
    res = download.download_one(f"{base}/f.cdf", str(target), stop=stop)
    assert res.status == "stopped"
    assert not target.exists()


# --------------------------------------------------------------------------- #
# run(): the work-list driver
# --------------------------------------------------------------------------- #
def _worklist(tmp_path, base, names):
    rows = [
        {"site": "aaaa", "datetime": "2015031806", "filename": n,
         "url": f"{base}/{n}", "target_path": str(tmp_path / "dl" / n),
         "size_bytes": 0}
        for n in names
    ]
    path = tmp_path / "wl.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_run_downloads_every_entry(server, tmp_path):
    root, base = server
    names = [f"f{i}.cdf" for i in range(4)]
    for n in names:
        (root / n).write_bytes(PAYLOAD)

    res = download.run(_worklist(tmp_path, base, names), workers=2)
    assert len(res) == 4
    assert set(res["status"]) == {"downloaded"}
    for n in names:
        assert (tmp_path / "dl" / n).read_bytes() == PAYLOAD


def test_run_is_idempotent(server, tmp_path):
    """Re-running after an interruption must skip, not re-download."""
    root, base = server
    (root / "f.cdf").write_bytes(PAYLOAD)
    wl = _worklist(tmp_path, base, ["f.cdf"])

    assert set(download.run(wl, workers=1)["status"]) == {"downloaded"}
    second = download.run(wl, workers=1)
    assert set(second["status"]) == {"skipped"}
    assert second["bytes_written"].sum() == 0


def test_run_reports_failures_without_aborting_the_batch(server, tmp_path):
    """One bad URL must not cost the other 999 files in a real run."""
    root, base = server
    (root / "good.cdf").write_bytes(PAYLOAD)
    res = download.run(_worklist(tmp_path, base, ["good.cdf", "missing.cdf"]),
                       workers=2)
    by_status = res.set_index("target_path")["status"].to_dict()
    assert sorted(by_status.values()) == ["downloaded", "failed"]


def test_run_limit_caps_the_batch(server, tmp_path):
    root, base = server
    names = [f"f{i}.cdf" for i in range(5)]
    for n in names:
        (root / n).write_bytes(PAYLOAD)
    res = download.run(_worklist(tmp_path, base, names), workers=2, limit=2)
    assert len(res) == 2


def test_worklist_missing_a_required_column_is_rejected(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame([{"url": "http://x/y"}]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        download.load_worklist(path)
