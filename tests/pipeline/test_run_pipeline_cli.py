"""Tests for commands/run_pipeline.py CLI helpers and end-to-end routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hgere.commands.run_pipeline import (
    _collect_input_files,
    _resolve_output_path,
    _validate_input_output_types,
    cli,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, docs: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(d) for d in docs) + "\n")


def _make_doc(key: str) -> dict[str, Any]:
    return {
        "doc_key": key,
        "sentences": [["Hello", "world"]],
        "ner": [[]],
        "relations": [[]],
    }


def _make_enriched(doc: dict[str, Any]) -> dict[str, Any]:
    return dict(doc, predicted_ner=[[]], predicted_rel=[[]])


# ---------------------------------------------------------------------------
# _collect_input_files
# ---------------------------------------------------------------------------


class TestCollectInputFiles:
    def test_single_file_returns_itself(self, tmp_path: Path) -> None:
        f = tmp_path / "docs.jsonl"
        f.touch()
        assert _collect_input_files(f) == [f]

    def test_directory_returns_json_and_jsonl(self, tmp_path: Path) -> None:
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "b.json").touch()
        (tmp_path / "readme.txt").touch()
        result = _collect_input_files(tmp_path)
        names = {f.name for f in result}
        assert names == {"a.jsonl", "b.json"}

    def test_directory_ignores_other_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "data.csv").touch()
        (tmp_path / "notes.md").touch()
        assert _collect_input_files(tmp_path) == []

    def test_directory_returns_sorted_paths(self, tmp_path: Path) -> None:
        (tmp_path / "z.jsonl").touch()
        (tmp_path / "a.jsonl").touch()
        (tmp_path / "m.jsonl").touch()
        result = _collect_input_files(tmp_path)
        assert result == sorted(result)


# ---------------------------------------------------------------------------
# _resolve_output_path
# ---------------------------------------------------------------------------


class TestResolveOutputPath:
    def test_explicit_output_returned_as_is(self, tmp_path: Path) -> None:
        input_path = tmp_path / "docs.jsonl"
        explicit = tmp_path / "out.jsonl"
        assert _resolve_output_path(input_path, str(explicit)) == explicit

    def test_single_file_default_adds_predictions_suffix(self, tmp_path: Path) -> None:
        input_path = tmp_path / "docs.jsonl"
        result = _resolve_output_path(input_path, None)
        assert result == tmp_path / "docs_predictions.jsonl"

    def test_single_file_json_extension_preserved(self, tmp_path: Path) -> None:
        input_path = tmp_path / "data.json"
        result = _resolve_output_path(input_path, None)
        assert result == tmp_path / "data_predictions.json"

    def test_directory_default_adds_predictions_to_name(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "my_docs"
        input_dir.mkdir()
        result = _resolve_output_path(input_dir, None)
        assert result == tmp_path / "my_docs_predictions"

    def test_directory_explicit_output_returned_as_is(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "docs"
        input_dir.mkdir()
        output_dir = tmp_path / "results"
        assert _resolve_output_path(input_dir, str(output_dir)) == output_dir


# ---------------------------------------------------------------------------
# _validate_input_output_types
# ---------------------------------------------------------------------------


class TestValidateInputOutputTypes:
    def _make_logger(self) -> Any:
        import logging

        return logging.getLogger("test")

    def test_both_files_passes(self, tmp_path: Path) -> None:
        f_in = tmp_path / "in.jsonl"
        f_in.touch()
        f_out = tmp_path / "out.jsonl"
        f_out.touch()
        # Should not raise / exit
        _validate_input_output_types(f_in, f_out, self._make_logger())

    def test_both_dirs_passes(self, tmp_path: Path) -> None:
        d_in = tmp_path / "in"
        d_in.mkdir()
        d_out = tmp_path / "out"
        d_out.mkdir()
        _validate_input_output_types(d_in, d_out, self._make_logger())

    def test_file_input_dir_output_exits(self, tmp_path: Path) -> None:
        f_in = tmp_path / "in.jsonl"
        f_in.touch()
        d_out = tmp_path / "out"
        d_out.mkdir()
        with pytest.raises(SystemExit):
            _validate_input_output_types(f_in, d_out, self._make_logger())

    def test_dir_input_file_output_exits(self, tmp_path: Path) -> None:
        d_in = tmp_path / "in"
        d_in.mkdir()
        f_out = tmp_path / "out.jsonl"
        f_out.touch()
        with pytest.raises(SystemExit):
            _validate_input_output_types(d_in, f_out, self._make_logger())

    def test_nonexistent_output_inferred_from_input_dir(self, tmp_path: Path) -> None:
        """Non-existent output path is treated as same type as input."""
        d_in = tmp_path / "in"
        d_in.mkdir()
        d_out = tmp_path / "out_does_not_exist"
        # Both are dirs (inferred) → no exit
        _validate_input_output_types(d_in, d_out, self._make_logger())

    def test_nonexistent_output_file_with_file_input_passes(
        self, tmp_path: Path
    ) -> None:
        f_in = tmp_path / "in.jsonl"
        f_in.touch()
        f_out = tmp_path / "out_does_not_exist.jsonl"
        # input is a file → output inferred as file → no exit
        _validate_input_output_types(f_in, f_out, self._make_logger())


# ---------------------------------------------------------------------------
# CLI end-to-end (mocked pipeline)
# ---------------------------------------------------------------------------


def _make_mock_pipeline(docs: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.process_documents.side_effect = lambda batch: [_make_enriched(d) for d in batch]
    return mock


class TestCliSingleFile:
    def test_single_file_writes_output(self, tmp_path: Path) -> None:
        docs = [_make_doc("d0"), _make_doc("d1")]
        input_file = tmp_path / "docs.jsonl"
        _write_jsonl(input_file, docs)
        output_file = tmp_path / "out.jsonl"

        mock_pipeline = _make_mock_pipeline(docs)
        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_file),
                "--output", str(output_file),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        assert output_file.exists()
        lines = [json.loads(l) for l in output_file.read_text().splitlines() if l]
        assert len(lines) == 2
        assert all("predicted_ner" in d for d in lines)

    def test_single_file_default_output_path(self, tmp_path: Path) -> None:
        docs = [_make_doc("d0")]
        input_file = tmp_path / "docs.jsonl"
        _write_jsonl(input_file, docs)
        expected_output = tmp_path / "docs_predictions.jsonl"

        mock_pipeline = _make_mock_pipeline(docs)
        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_file),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        assert expected_output.exists()

    def test_missing_input_exits(self, tmp_path: Path) -> None:
        with (
            patch("hgere.commands.run_pipeline.Pipeline"),
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(tmp_path / "nonexistent.jsonl"),
                "--output", str(tmp_path / "out.jsonl"),
            ]),
            pytest.raises(SystemExit),
        ):
            cli()


class TestCliDirectoryMode:
    def test_directory_processes_all_jsonl_files(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        docs_a = [_make_doc("a0")]
        docs_b = [_make_doc("b0"), _make_doc("b1")]
        _write_jsonl(input_dir / "a.jsonl", docs_a)
        _write_jsonl(input_dir / "b.jsonl", docs_b)

        output_dir = tmp_path / "output"
        mock_pipeline = _make_mock_pipeline(docs_a + docs_b)

        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_dir),
                "--output", str(output_dir),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        assert (output_dir / "a.jsonl").exists()
        assert (output_dir / "b.jsonl").exists()

    def test_directory_preserves_filenames(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_jsonl(input_dir / "split_train.jsonl", [_make_doc("x")])
        output_dir = tmp_path / "output"

        mock_pipeline = _make_mock_pipeline([_make_doc("x")])
        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_dir),
                "--output", str(output_dir),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        assert (output_dir / "split_train.jsonl").exists()

    def test_directory_default_output_path(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "my_docs"
        input_dir.mkdir()
        _write_jsonl(input_dir / "data.jsonl", [_make_doc("x")])
        expected_output_dir = tmp_path / "my_docs_predictions"

        mock_pipeline = _make_mock_pipeline([_make_doc("x")])
        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_dir),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        assert (expected_output_dir / "data.jsonl").exists()

    def test_directory_skips_non_json_files(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_jsonl(input_dir / "data.jsonl", [_make_doc("x")])
        (input_dir / "readme.txt").write_text("ignore me")
        output_dir = tmp_path / "output"

        mock_pipeline = _make_mock_pipeline([_make_doc("x")])
        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_dir),
                "--output", str(output_dir),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        assert not (output_dir / "readme.txt").exists()

    def test_directory_pipeline_loaded_once(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_jsonl(input_dir / "a.jsonl", [_make_doc("a")])
        _write_jsonl(input_dir / "b.jsonl", [_make_doc("b")])
        output_dir = tmp_path / "output"

        mock_pipeline = _make_mock_pipeline([_make_doc("a"), _make_doc("b")])
        with (
            patch("hgere.commands.run_pipeline.Pipeline") as MockPipeline,
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_dir),
                "--output", str(output_dir),
            ]),
        ):
            MockPipeline.from_yaml.return_value = mock_pipeline
            cli()

        MockPipeline.from_yaml.assert_called_once()

    def test_type_mismatch_dir_input_file_output_exits(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _write_jsonl(input_dir / "data.jsonl", [_make_doc("x")])
        output_file = tmp_path / "output.jsonl"
        output_file.touch()

        with (
            patch("hgere.commands.run_pipeline.Pipeline"),
            patch.object(sys, "argv", [
                "run-pipeline",
                "--config", "fake.yaml",
                "--input", str(input_dir),
                "--output", str(output_file),
            ]),
            pytest.raises(SystemExit),
        ):
            cli()
