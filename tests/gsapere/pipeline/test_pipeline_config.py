"""Tests for PipelineConfig label_set / label_sets handling and output path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from gsapere.commands.run_pipeline import (
    _effective_output_file,
    _output_path_for_label_set,
)
from gsapere.pipeline.config import PipelineConfig


def _minimal_pipeline_dict(**overrides) -> dict:
    base = {
        "pruner": {
            "model_dir": "saves/pruner",
            "base_model_name_or_path": "allenai/scibert_scivocab_uncased",
        },
        "hgere": {
            "model_dir": "saves/hgere",
            "base_model_name_or_path": "allenai/scibert_scivocab_uncased",
        },
    }
    base.update(overrides)
    return base


class TestPipelineConfigLabelSets:
    def test_single_label_set_backward_compat(self) -> None:
        cfg = PipelineConfig.model_validate(_minimal_pipeline_dict(label_set="scier"))
        assert cfg.label_sets == ["scier"]
        assert cfg.label_set == "scier"

    def test_label_sets_list(self) -> None:
        cfg = PipelineConfig.model_validate(
            _minimal_pipeline_dict(label_sets=["scier", "scinlp"])
        )
        assert cfg.label_sets == ["scier", "scinlp"]
        assert cfg.label_set is None

    def test_neither_raises(self) -> None:
        with pytest.raises(Exception):
            PipelineConfig.model_validate(_minimal_pipeline_dict())

    def test_both_raises(self) -> None:
        with pytest.raises(Exception):
            PipelineConfig.model_validate(
                _minimal_pipeline_dict(
                    label_set="scier", label_sets=["scier", "scinlp"]
                )
            )

    def test_single_entry_label_sets(self) -> None:
        cfg = PipelineConfig.model_validate(_minimal_pipeline_dict(label_sets=["gsap"]))
        assert cfg.label_sets == ["gsap"]


class TestOutputPathForLabelSet:
    def test_inserts_label_set_as_subfolder(self) -> None:
        p = Path("output/predictions.jsonl")
        assert _output_path_for_label_set(p, "scier") == Path(
            "output/scier/predictions.jsonl"
        )

    def test_works_with_json_extension(self) -> None:
        p = Path("out/docs.json")
        assert _output_path_for_label_set(p, "scinlp") == Path("out/scinlp/docs.json")

    def test_preserves_filename(self) -> None:
        p = Path("/data/results/file.jsonl")
        result = _output_path_for_label_set(p, "gsap")
        assert result.parent == Path("/data/results/gsap")
        assert result.name == "file.jsonl"


class TestEffectiveOutputFile:
    def test_no_label_set_no_seed(self) -> None:
        p = Path("out/data.jsonl")
        assert _effective_output_file(p, None, None) == p

    def test_label_set_only(self) -> None:
        p = Path("out/data.jsonl")
        assert _effective_output_file(p, "scier", None) == Path("out/scier/data.jsonl")

    def test_seed_only(self) -> None:
        p = Path("out/data.jsonl")
        assert _effective_output_file(p, None, 42) == Path("out/42/data.jsonl")

    def test_label_set_and_seed(self) -> None:
        p = Path("out/data.jsonl")
        assert _effective_output_file(p, "gsap", 42) == Path("out/gsap/42/data.jsonl")
