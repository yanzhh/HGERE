"""Integration tests for train-span-classifier and train-hgere entry points.

These tests exercise main() end-to-end through sys.argv to verify that
config loading and error handling work correctly at the entry-point level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from gsapere.commands.train_hgere import main as train_hgere_main
from gsapere.commands.train_pruner import main as train_pruner_main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.dump(data))


def _minimal_flat_pruner_yaml() -> dict[str, Any]:
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


def _minimal_flat_hgere_yaml() -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# train-span-classifier
# ---------------------------------------------------------------------------


class TestTrainSpanClassifier:
    def _write_config(self, tmp_path: Path) -> Path:
        p = tmp_path / "train.yaml"
        _write_yaml(p, _minimal_flat_pruner_yaml())
        return p

    def test_positional_config_loads(self, tmp_path: Path) -> None:
        """Positional YAML arg loads config and calls run_train_pruner."""
        config_path = self._write_config(tmp_path)
        with patch(
            "gsapere.commands.train_pruner.run_train_pruner"
        ) as mock_run:
            train_pruner_main([str(config_path)])
        mock_run.assert_called_once()

    def test_nonexistent_config_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            train_pruner_main([str(tmp_path / "missing.yaml")])
        assert exc_info.value.code != 0

    def test_no_args_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            train_pruner_main([])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# train-hgere
# ---------------------------------------------------------------------------


class TestTrainHgere:
    def _write_config(self, tmp_path: Path) -> Path:
        data = _minimal_flat_hgere_yaml()
        data["model_dir"] = str(tmp_path / "model")
        p = tmp_path / "train.yaml"
        _write_yaml(p, data)
        return p

    def test_positional_config_loads(self, tmp_path: Path) -> None:
        """Positional YAML arg loads config and calls setup_training."""
        from unittest.mock import MagicMock

        config_path = self._write_config(tmp_path)
        with (
            patch("gsapere.commands.train_hgere.setup_training") as mock_setup,
            patch("gsapere.commands.train_hgere.get_logger", return_value=MagicMock()),
            patch(
                "gsapere.commands.train_hgere.get_last_checkpoint", return_value=(None, 0)
            ),
        ):
            train_hgere_main([str(config_path)])
        mock_setup.assert_called_once()

    def test_nonexistent_config_exits(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc_info:
            train_hgere_main([str(tmp_path / "missing.yaml")])
        assert exc_info.value.code != 0

    def test_no_args_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            train_hgere_main([])
        assert exc_info.value.code != 0
