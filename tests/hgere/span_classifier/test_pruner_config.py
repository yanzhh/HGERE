"""Tests for hgere.span_classifier.config — PrunerTrainConfig models."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hgere.span_classifier.config import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    PrunerTrainConfig,
    PrunerTrainParams,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data))
    return path


def _minimal_pruner_section() -> dict:
    return {
        "model_dir": "saves/pruner",
        "base_model_name_or_path": "pretrained_models/scibert",
        "train_params": {
            "data_dir": "datasets/gsap",
            "learning_rate": 1e-6,
            "num_train_epochs": 8,
            "per_gpu_train_batch_size": 8,
        },
    }


def _minimal_full_yaml() -> dict:
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "label_set": "gsap",
        "pruner": _minimal_pruner_section(),
        "hgere": {},  # ignored by PrunerTrainConfig.from_yaml
    }


# ---------------------------------------------------------------------------
# PrunerTrainParams
# ---------------------------------------------------------------------------


class TestPrunerTrainParams:
    def test_required_fields_must_be_provided(self) -> None:
        with pytest.raises(Exception):
            PrunerTrainParams()  # data_dir, learning_rate, etc. are required

    def test_defaults_applied(self) -> None:
        params = PrunerTrainParams(
            data_dir="d",
            learning_rate=1e-5,
            num_train_epochs=3,
            per_gpu_train_batch_size=8,
        )
        assert params.seed == 42
        assert params.pruner_loss == "bce"
        assert params.fp16 is False
        assert params.evaluate_during_training is True
        assert params.rulebased_pruner_file is None

    def test_literal_field_rejects_bad_value(self) -> None:
        with pytest.raises(Exception):
            PrunerTrainParams(
                data_dir="d",
                learning_rate=1e-5,
                num_train_epochs=3,
                per_gpu_train_batch_size=8,
                pruner_loss="invalid",
            )

    def test_literal_field_accepts_valid_values(self) -> None:
        for loss in ("bce", "focal"):
            p = PrunerTrainParams(
                data_dir="d",
                learning_rate=1e-5,
                num_train_epochs=3,
                per_gpu_train_batch_size=8,
                pruner_loss=loss,
            )
            assert p.pruner_loss == loss


# ---------------------------------------------------------------------------
# PrunerTrainConfig — direct construction
# ---------------------------------------------------------------------------


class TestPrunerTrainConfigDirect:
    def test_valid_config_constructed(self) -> None:
        cfg = PrunerTrainConfig(
            label_set="gsap",
            model_dir="saves/pruner",
            base_model_name_or_path="pretrained_models/scibert",
            train_params=PrunerTrainParams(
                data_dir="datasets/gsap",
                learning_rate=1e-6,
                num_train_epochs=8,
                per_gpu_train_batch_size=8,
            ),
        )
        assert cfg.label_set == "gsap"
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION
        assert cfg.do_lower_case is True  # default

    def test_unsupported_schema_version_raises(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            PrunerTrainConfig(
                schema_version="0.0",
                label_set="gsap",
                model_dir="saves/pruner",
                base_model_name_or_path="pretrained_models/scibert",
                train_params=PrunerTrainParams(
                    data_dir="d",
                    learning_rate=1e-5,
                    num_train_epochs=3,
                    per_gpu_train_batch_size=8,
                ),
            )

    def test_supported_versions_constant_non_empty(self) -> None:
        assert len(SUPPORTED_SCHEMA_VERSIONS) >= 1
        assert CURRENT_SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS


# ---------------------------------------------------------------------------
# PrunerTrainConfig.from_yaml — standalone pruner file
# ---------------------------------------------------------------------------


class TestFromYamlStandalone:
    def test_loads_flat_yaml(self, tmp_path: Path) -> None:
        data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "label_set": "gsap",
            **_minimal_pruner_section(),
        }
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        cfg = PrunerTrainConfig.from_yaml(path)
        assert cfg.label_set == "gsap"
        assert cfg.model_dir == "saves/pruner"

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        data = {"label_set": "gsap"}  # model_dir etc. missing
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        with pytest.raises(Exception):
            PrunerTrainConfig.from_yaml(path)

    def test_unsupported_version_raises(self, tmp_path: Path) -> None:
        data = {
            "schema_version": "99.0",
            "label_set": "gsap",
            **_minimal_pruner_section(),
        }
        path = _write_yaml(tmp_path / "cfg.yaml", data)
        with pytest.raises(ValueError, match="schema_version"):
            PrunerTrainConfig.from_yaml(path)


# ---------------------------------------------------------------------------
# PrunerTrainConfig.from_yaml — full pipeline training file
# ---------------------------------------------------------------------------


class TestFromYamlFullPipelineFile:
    def test_extracts_pruner_section(self, tmp_path: Path) -> None:
        data = _minimal_full_yaml()
        path = _write_yaml(tmp_path / "train.yaml", data)
        cfg = PrunerTrainConfig.from_yaml(path)
        assert cfg.label_set == "gsap"
        assert cfg.model_dir == "saves/pruner"

    def test_label_set_hoisted_from_top_level(self, tmp_path: Path) -> None:
        data = _minimal_full_yaml()
        data["label_set"] = "scier"
        path = _write_yaml(tmp_path / "train.yaml", data)
        cfg = PrunerTrainConfig.from_yaml(path)
        assert cfg.label_set == "scier"

    def test_schema_version_hoisted_from_top_level(self, tmp_path: Path) -> None:
        data = _minimal_full_yaml()
        data["schema_version"] = CURRENT_SCHEMA_VERSION
        path = _write_yaml(tmp_path / "train.yaml", data)
        cfg = PrunerTrainConfig.from_yaml(path)
        assert cfg.schema_version == CURRENT_SCHEMA_VERSION

    def test_hgere_section_ignored(self, tmp_path: Path) -> None:
        data = _minimal_full_yaml()
        data["hgere"] = {"garbage": "value"}
        path = _write_yaml(tmp_path / "train.yaml", data)
        # Should not raise — hgere section is irrelevant to PrunerTrainConfig
        cfg = PrunerTrainConfig.from_yaml(path)
        assert cfg.model_dir == "saves/pruner"
