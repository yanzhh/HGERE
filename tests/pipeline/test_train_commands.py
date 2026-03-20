"""Tests for train_span_classifier_by_config and train_hgere CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from hgere.commands.train_hgere import cli as train_hgere_by_config_cli
from hgere.commands.train_span_classifier import (
    cli as train_span_classifier_by_config_cli,
)
from hgere.hgere.config import HGERETrainConfig, HGERETrainParams
from hgere.span_classifier.config import PrunerTrainConfig, PrunerTrainParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.dump(data))


def _make_pruner_config(
    model_dir: str = "saves/pruner",
    base_model: str = "pretrained_models/scibert",
    label_set: str = "gsap",
    **train_overrides: Any,
) -> PrunerTrainConfig:
    """Build a minimal PrunerTrainConfig for testing."""
    train_defaults: dict[str, Any] = {
        "data_dir": "datasets/gsap",
        "learning_rate": 1e-6,
        "num_train_epochs": 8,
        "per_gpu_train_batch_size": 8,
    }
    train_defaults.update(train_overrides)
    return PrunerTrainConfig(
        label_set=label_set,
        model_dir=model_dir,
        base_model_name_or_path=base_model,
        train_params=PrunerTrainParams(**train_defaults),
    )


_HGERE_TRAIN_PARAM_FIELDS: frozenset[str] = frozenset(
    HGERETrainParams.model_fields.keys()
)


def _make_hgere_config(
    model_dir: str = "saves/hgere",
    base_model: str = "pretrained_models/scibert",
    label_set: str = "gsap",
    ner_prediction_dir: str = "saves/pruner/output",
    **overrides: Any,
) -> HGERETrainConfig:
    """Build a minimal HGERETrainConfig for testing.

    Keyword overrides are automatically routed to either the top-level config
    or train_params based on which model they belong to.
    """
    top_overrides: dict[str, Any] = {}
    train_overrides: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in _HGERE_TRAIN_PARAM_FIELDS:
            train_overrides[key] = value
        else:
            top_overrides[key] = value

    train_defaults: dict[str, Any] = {
        "learning_rate": 1e-5,
        "num_train_epochs": 8,
        "per_gpu_train_batch_size": 18,
    }
    train_defaults.update(train_overrides)
    return HGERETrainConfig(
        label_set=label_set,
        model_dir=model_dir,
        base_model_name_or_path=base_model,
        ner_prediction_dir=ner_prediction_dir,
        train_params=HGERETrainParams(**train_defaults),
        **top_overrides,
    )


def _minimal_flat_hgere_yaml() -> dict[str, Any]:
    """A flat standalone HGERE training YAML (not pipeline-style)."""
    return {
        "schema_version": "1.0",
        "label_set": "gsap",
        "model_dir": "saves/hgere",
        "base_model_name_or_path": "pretrained_models/scibert",
        "ner_prediction_dir": "saves/pruner/output",
        "train_params": {
            "learning_rate": 1e-5,
            "num_train_epochs": 8,
            "per_gpu_train_batch_size": 18,
        },
    }


def _minimal_flat_pruner_yaml() -> dict[str, Any]:
    """A flat standalone pruner training YAML (not pipeline-style)."""
    return {
        "schema_version": "1.0",
        "label_set": "gsap",
        "model_dir": "saves/pruner",
        "base_model_name_or_path": "pretrained_models/scibert",
        "train_params": {
            "data_dir": "datasets/gsap",
            "learning_rate": 1e-6,
            "num_train_epochs": 8,
            "per_gpu_train_batch_size": 8,
        },
    }


# ---------------------------------------------------------------------------
# train-span-classifier CLI tests
# ---------------------------------------------------------------------------


class TestTrainSpanClassifierCli:
    """cli() is an alias for main() — tests verify it delegates correctly."""

    def _write_config(
        self, tmp_path: Path, overrides: dict[str, Any] | None = None
    ) -> Path:
        data = _minimal_flat_pruner_yaml()
        if overrides:
            data.update(overrides)
        p = tmp_path / "train.yaml"
        _write_yaml(p, data)
        return p

    def test_calls_main(self, tmp_path: Path) -> None:
        """cli() delegates to main() with no arguments (main reads sys.argv)."""
        config_path = self._write_config(tmp_path)
        with (
            patch("hgere.commands.train_span_classifier.main") as mock_main,
            patch.object(
                sys, "argv", ["train-span-classifier", "--config", str(config_path)]
            ),
        ):
            train_span_classifier_by_config_cli()
        mock_main.assert_called_once_with()

    def test_missing_config_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "train-span-classifier",
                    "--config",
                    str(tmp_path / "missing.yaml"),
                ],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_span_classifier_by_config_cli()
        assert exc_info.value.code != 0

    def test_no_config_flag_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(sys, "argv", ["train-span-classifier"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_span_classifier_by_config_cli()
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# train-hgere CLI tests
# ---------------------------------------------------------------------------


class TestTrainHgereCli:
    """cli() is now an alias for main() — tests verify it delegates correctly."""

    def _write_config(
        self, tmp_path: Path, overrides: dict[str, Any] | None = None
    ) -> Path:
        data = _minimal_flat_hgere_yaml()
        if overrides:
            data.update(overrides)
        p = tmp_path / "train.yaml"
        _write_yaml(p, data)
        return p

    def test_calls_main(self, tmp_path: Path) -> None:
        """cli() delegates to main() with no arguments (main reads sys.argv)."""
        config_path = self._write_config(tmp_path)
        with (
            patch("hgere.commands.train_hgere.main") as mock_main,
            patch.object(
                sys, "argv", ["train-hgere-by-config", "--config", str(config_path)]
            ),
        ):
            train_hgere_by_config_cli()
        mock_main.assert_called_once_with()

    def test_missing_config_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["train-hgere-by-config", "--config", str(tmp_path / "missing.yaml")],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_hgere_by_config_cli()
        assert exc_info.value.code != 0

    def test_no_config_flag_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(sys, "argv", ["train-hgere-by-config"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_hgere_by_config_cli()
        assert exc_info.value.code != 0
