"""Tests for hgere.commands.download_dataset — verify_file behaviour."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hgere.commands.download_dataset import verify_file


def _write_file(path: Path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.md5(content).hexdigest()


class TestVerifyFile:
    def test_returns_true_on_match(self, tmp_path: Path) -> None:
        content = b"hello dataset"
        path = tmp_path / "train.jsonl"
        expected = _write_file(path, content)

        assert verify_file(path, expected) is True

    def test_returns_false_on_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "train.jsonl"
        _write_file(path, b"hello dataset")

        assert verify_file(path, "0" * 32) is False

    def test_prints_ok_on_match(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        content = b"hello dataset"
        path = tmp_path / "train.jsonl"
        expected = _write_file(path, content)

        verify_file(path, expected)

        assert "[ok]" in capsys.readouterr().out

    def test_warns_data_may_have_changed_on_mismatch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        path = tmp_path / "train.jsonl"
        actual = _write_file(path, b"hello dataset")

        verify_file(path, "0" * 32)

        err = capsys.readouterr().err
        assert "data may have changed" in err
        assert "0" * 32 in err  # expected hash shown
        assert actual in err  # actual hash shown
