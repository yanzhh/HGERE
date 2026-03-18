"""Tests for pipeline/config.py — FinalPruningConfig, PrunerConfig, HGEREConfig, PipelineConfig."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hgere.pipeline.config import (
    FinalPruningConfig,
    HGEREConfig,
    PipelineConfig,
    PrunerConfig,
)


class TestFinalPruningConfig:
    def test_default_method_is_threshold(self) -> None:
        cfg = FinalPruningConfig()
        assert cfg.method == "threshold"

    def test_default_threshold_value(self) -> None:
        cfg = FinalPruningConfig()
        assert cfg.threshold == 0.0005

    def test_topk_method_with_defaults(self) -> None:
        cfg = FinalPruningConfig(method="topk")
        assert cfg.method == "topk"
        assert cfg.topk_ratio == 0.5
        assert cfg.min_mentions_by_sentence == 3
        assert cfg.max_mentions_by_sentence == 18

    def test_threshold_method_none_threshold_raises(self) -> None:
        with pytest.raises(ValidationError):
            FinalPruningConfig(method="threshold", threshold=None)

    def test_override_threshold(self) -> None:
        cfg = FinalPruningConfig(method="threshold", threshold=0.01)
        assert cfg.threshold == 0.01

    def test_override_topk_params(self) -> None:
        cfg = FinalPruningConfig(
            method="topk",
            topk_ratio=0.3,
            min_mentions_by_sentence=2,
            max_mentions_by_sentence=12,
        )
        assert cfg.topk_ratio == 0.3
        assert cfg.min_mentions_by_sentence == 2
        assert cfg.max_mentions_by_sentence == 12


class TestPrunerConfig:
    def test_required_fields(self) -> None:
        cfg = PrunerConfig(
            model_dir="some/path",
            base_model_name_or_path="scibert",
        )
        assert cfg.model_dir == "some/path"
        assert cfg.base_model_name_or_path == "scibert"

    def test_do_lower_case_default_is_true(self) -> None:
        cfg = PrunerConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.do_lower_case is True

    def test_prune_config_default_is_none(self) -> None:
        cfg = PrunerConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.prune_config is None

    def test_final_pruning_default_is_threshold_method(self) -> None:
        cfg = PrunerConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.final_pruning.method == "threshold"

    def test_default_model_type(self) -> None:
        cfg = PrunerConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.model_type == "bertspanmarkerpruner"

    def test_default_batch_size(self) -> None:
        cfg = PrunerConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.per_gpu_eval_batch_size == 32

    def test_override_final_pruning(self) -> None:
        cfg = PrunerConfig(
            model_dir="x",
            base_model_name_or_path="y",
            final_pruning=FinalPruningConfig(method="topk", topk_ratio=0.3),
        )
        assert cfg.final_pruning.method == "topk"
        assert cfg.final_pruning.topk_ratio == 0.3

    def test_missing_model_dir_raises(self) -> None:
        with pytest.raises(ValidationError):
            PrunerConfig(base_model_name_or_path="y")  # type: ignore[call-arg]


class TestHGEREConfig:
    def test_required_fields(self) -> None:
        cfg = HGEREConfig(
            model_dir="some/path",
            base_model_name_or_path="scibert",
        )
        assert cfg.model_dir == "some/path"

    def test_default_model_type(self) -> None:
        cfg = HGEREConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.model_type == "hyper"

    def test_default_dims(self) -> None:
        cfg = HGEREConfig(model_dir="x", base_model_name_or_path="y")
        assert cfg.ent_dim == 400
        assert cfg.rel_dim == 400
        assert cfg.mem_dim == 400

    def test_missing_model_dir_raises(self) -> None:
        with pytest.raises(ValidationError):
            HGEREConfig(base_model_name_or_path="y")  # type: ignore[call-arg]


class TestPipelineConfig:
    def test_from_yaml_minimal(self, minimal_config_yaml: Path) -> None:
        cfg = PipelineConfig.from_yaml(minimal_config_yaml)
        assert cfg.label_set == "gsap"
        assert isinstance(cfg.pruner, PrunerConfig)
        assert isinstance(cfg.hgere, HGEREConfig)

    def test_from_yaml_full(self, full_config_yaml: Path) -> None:
        cfg = PipelineConfig.from_yaml(full_config_yaml)
        assert cfg.pruner.final_pruning.threshold == 0.0005
        assert cfg.pruner.max_seq_length == 256
        assert cfg.hgere.factor_type == "tersibcop"
        assert cfg.hgere.ent_dim == 400

    def test_from_yaml_topk_final_pruning(self, tmp_path: Path) -> None:
        config = {
            "label_set": "gsap",
            "pruner": {
                "model_dir": "x",
                "base_model_name_or_path": "y",
                "final_pruning": {
                    "method": "topk",
                    "topk_ratio": 0.3,
                    "min_mentions_by_sentence": 2,
                    "max_mentions_by_sentence": 12,
                },
            },
            "hgere": {"model_dir": "x", "base_model_name_or_path": "y"},
        }
        path = tmp_path / "topk.yaml"
        path.write_text(yaml.dump(config))
        cfg = PipelineConfig.from_yaml(path)
        assert cfg.pruner.final_pruning.method == "topk"
        assert cfg.pruner.final_pruning.topk_ratio == 0.3

    def test_from_yaml_string_path(self, minimal_config_yaml: Path) -> None:
        cfg = PipelineConfig.from_yaml(str(minimal_config_yaml))
        assert cfg.label_set == "gsap"

    def test_missing_label_set_raises(self, tmp_path: Path) -> None:
        config = {
            "pruner": {"model_dir": "x", "base_model_name_or_path": "y"},
            "hgere": {"model_dir": "x", "base_model_name_or_path": "y"},
        }
        path = tmp_path / "bad_config.yaml"
        path.write_text(yaml.dump(config))
        with pytest.raises(ValidationError):
            PipelineConfig.from_yaml(path)

    def test_missing_pruner_section_raises(self, tmp_path: Path) -> None:
        config = {
            "label_set": "gsap",
            "hgere": {"model_dir": "x", "base_model_name_or_path": "y"},
        }
        path = tmp_path / "bad_config2.yaml"
        path.write_text(yaml.dump(config))
        with pytest.raises(ValidationError):
            PipelineConfig.from_yaml(path)

    def test_pruner_defaults_applied_via_yaml(self, tmp_path: Path) -> None:
        """Fields omitted in YAML should receive default values."""
        config = {
            "label_set": "gsap",
            "pruner": {"model_dir": "x", "base_model_name_or_path": "y"},
            "hgere": {"model_dir": "x", "base_model_name_or_path": "y"},
        }
        path = tmp_path / "defaults.yaml"
        path.write_text(yaml.dump(config))
        cfg = PipelineConfig.from_yaml(path)
        assert cfg.pruner.final_pruning.threshold == 0.0005
        assert cfg.pruner.do_lower_case is True
        assert cfg.pruner.prune_config is None
        assert cfg.hgere.max_seq_length == 512
