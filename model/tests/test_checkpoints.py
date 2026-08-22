"""Tests for smartfour.checkpoints — best{n}.pt versioning helpers."""

from smartfour.checkpoints import best_versions, latest_best


def test_best_versions_ascending_ignoring_other_files(tmp_path):
    (tmp_path / "best10.pt").write_bytes(b"x")
    (tmp_path / "best2.pt").write_bytes(b"x")
    (tmp_path / "best1.pt").write_bytes(b"x")
    (tmp_path / "best.pt").write_bytes(b"x")  # unversioned: not part of the scheme
    (tmp_path / "random.pt").write_bytes(b"x")
    (tmp_path / "bestX.pt").write_bytes(b"x")
    (tmp_path / "best2.pt.bak").write_bytes(b"x")
    assert best_versions(tmp_path) == [1, 2, 10]
    assert latest_best(tmp_path) == tmp_path / "best10.pt"


def test_best_versions_skip_directories(tmp_path):
    (tmp_path / "best3.pt").mkdir()
    assert best_versions(tmp_path) == []


def test_empty_dir_has_no_latest(tmp_path):
    assert best_versions(tmp_path) == []
    assert latest_best(tmp_path) is None


def test_missing_dir_has_no_latest(tmp_path):
    assert best_versions(tmp_path / "nope") == []
    assert latest_best(tmp_path / "nope") is None
