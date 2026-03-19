"""Tests for train_span_classifier_by_config and train_hgere CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hgere.commands.train_hgere import cli as train_hgere_by_config_cli
from hgere.commands.train_hgere import model_to_argv as hgere_model_to_argv
from hgere.commands.train_span_classifier_by_config import (
    cli as train_span_classifier_by_config_cli,
)
from hgere.commands.train_span_classifier_by_config import model_to_argv
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


# ---------------------------------------------------------------------------
# model_to_argv (pruner)
# ---------------------------------------------------------------------------


class TestPrunerModelToArgv:
    def test_model_dir_and_output_dir_both_present(self) -> None:
        config = _make_pruner_config()
        argv = model_to_argv(config)
        assert "--model_dir" in argv
        assert "--output_dir" in argv
        assert argv[argv.index("--model_dir") + 1] == "saves/pruner"
        assert argv[argv.index("--output_dir") + 1] == "saves/pruner"

    def test_label_set_added(self) -> None:
        config = _make_pruner_config(label_set="scier")
        argv = model_to_argv(config)
        assert "--label_set" in argv
        assert argv[argv.index("--label_set") + 1] == "scier"

    def test_do_train_added_by_default(self) -> None:
        config = _make_pruner_config()
        argv = model_to_argv(config)
        assert "--do_train" in argv

    def test_do_lower_case_boolean_flag(self) -> None:
        config = _make_pruner_config()
        argv = model_to_argv(config)
        # do_lower_case defaults to True in PrunerTrainConfig
        assert "--do_lower_case" in argv

    def test_fp16_boolean_flag(self) -> None:
        config = _make_pruner_config(fp16=True)
        argv = model_to_argv(config)
        assert "--fp16" in argv

    def test_false_boolean_flag_not_included(self) -> None:
        config = _make_pruner_config(fp16=False)
        argv = model_to_argv(config)
        assert "--fp16" not in argv

    def test_none_values_omitted(self) -> None:
        config = _make_pruner_config(rulebased_pruner_file=None)
        argv = model_to_argv(config)
        assert "--rulebased-pruner-file" not in argv
        assert "--rulebased_pruner_file" not in argv

    def test_rulebased_pruner_file_remapped(self) -> None:
        config = _make_pruner_config(rulebased_pruner_file="gsap_rulebased.json")
        argv = model_to_argv(config)
        assert "--rulebased-pruner-file" in argv
        assert argv[argv.index("--rulebased-pruner-file") + 1] == "gsap_rulebased.json"

    def test_learning_rate_included(self) -> None:
        config = _make_pruner_config(learning_rate=2e-6)
        argv = model_to_argv(config)
        assert "--learning_rate" in argv
        assert argv[argv.index("--learning_rate") + 1] == str(2e-6)

    def test_schema_version_excluded(self) -> None:
        config = _make_pruner_config()
        argv = model_to_argv(config)
        assert "--schema_version" not in argv


# ---------------------------------------------------------------------------
# model_to_argv (hgere)
# ---------------------------------------------------------------------------


class TestHGEREModelToArgv:
    def test_model_dir_remapped_to_output_dir(self) -> None:
        config = _make_hgere_config()
        argv = hgere_model_to_argv(config)
        assert "--output_dir" in argv
        assert argv[argv.index("--output_dir") + 1] == "saves/hgere"
        assert "--model_dir" not in argv

    def test_base_model_remapped_to_model_name_or_path(self) -> None:
        config = _make_hgere_config()
        argv = hgere_model_to_argv(config)
        assert "--model_name_or_path" in argv
        assert "--base_model_name_or_path" not in argv

    def test_label_set_added(self) -> None:
        config = _make_hgere_config(label_set="scier")
        argv = hgere_model_to_argv(config)
        assert "--label_set" in argv
        assert argv[argv.index("--label_set") + 1] == "scier"

    def test_n_iter_remapped_to_iter(self) -> None:
        config = _make_hgere_config()
        argv = hgere_model_to_argv(config)
        assert "--iter" in argv
        assert "--n_iter" not in argv

    def test_schema_version_excluded(self) -> None:
        config = _make_hgere_config()
        argv = hgere_model_to_argv(config)
        assert "--schema_version" not in argv

    def test_do_train_present_by_default(self) -> None:
        config = _make_hgere_config()
        argv = hgere_model_to_argv(config)
        assert "--do_train" in argv

    def test_eval_dev_eval_test_present_by_default(self) -> None:
        config = _make_hgere_config()
        argv = hgere_model_to_argv(config)
        assert "--eval_dev" in argv
        assert "--eval_test" in argv

    def test_layernorm_bool_flag(self) -> None:
        config = _make_hgere_config(layernorm=True, layernorm_1st=True)
        argv = hgere_model_to_argv(config)
        assert "--layernorm" in argv
        assert "--layernorm_1st" in argv

    def test_log_wandb_bool_flag(self) -> None:
        config = _make_hgere_config(log_wandb=True)
        argv = hgere_model_to_argv(config)
        assert "--log_wandb" in argv

    def test_false_bool_flag_not_included(self) -> None:
        config = _make_hgere_config(fp16=False)
        argv = hgere_model_to_argv(config)
        assert "--fp16" not in argv

    def test_fp16_bool_flag_included_when_true(self) -> None:
        config = _make_hgere_config(fp16=True)
        argv = hgere_model_to_argv(config)
        assert "--fp16" in argv

    def test_none_values_omitted(self) -> None:
        config = _make_hgere_config()
        # run_name is Optional[str] defaulting to None
        argv = hgere_model_to_argv(config)
        assert "--run_name" not in argv

    def test_learning_rate_included(self) -> None:
        config = _make_hgere_config(learning_rate=2e-5)
        argv = hgere_model_to_argv(config)
        assert "--learning_rate" in argv
        assert argv[argv.index("--learning_rate") + 1] == str(2e-5)

    def test_ner_prediction_dir_included(self) -> None:
        config = _make_hgere_config(ner_prediction_dir="path/to/predictions")
        argv = hgere_model_to_argv(config)
        assert "--ner_prediction_dir" in argv
        assert argv[argv.index("--ner_prediction_dir") + 1] == "path/to/predictions"

    def test_loss_re_weight_alpha_included(self) -> None:
        config = _make_hgere_config(loss_re_weight_alpha=0.9)
        argv = hgere_model_to_argv(config)
        assert "--loss_re_weight_alpha" in argv


# ---------------------------------------------------------------------------
# CLI end-to-end (subprocess mocked)
# ---------------------------------------------------------------------------


class TestTrainSpanClassifierByConfigCli:
    def _write_config(
        self, tmp_path: Path, overrides: dict[str, Any] | None = None
    ) -> Path:
        data = _minimal_flat_pruner_yaml()
        if overrides:
            data.update(overrides)
        p = tmp_path / "train.yaml"
        _write_yaml(p, data)
        return p

    def test_calls_train_span_classifier(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        mock_result = MagicMock(returncode=0)
        with (
            patch(
                "hgere.commands.train_span_classifier_by_config.subprocess.run",
                return_value=mock_result,
            ) as mock_run,
            patch.object(
                sys,
                "argv",
                ["train-span-classifier-by-config", "--config", str(config_path)],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_span_classifier_by_config_cli()
        assert exc_info.value.code == 0
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "uv"
        assert "train-span-classifier" in cmd

    def test_missing_config_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(
                sys,
                "argv",
                [
                    "train-span-classifier-by-config",
                    "--config",
                    str(tmp_path / "missing.yaml"),
                ],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_span_classifier_by_config_cli()
        assert exc_info.value.code != 0

    def test_propagates_subprocess_returncode(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        mock_result = MagicMock(returncode=1)
        with (
            patch(
                "hgere.commands.train_span_classifier_by_config.subprocess.run",
                return_value=mock_result,
            ),
            patch.object(
                sys,
                "argv",
                ["train-span-classifier-by-config", "--config", str(config_path)],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_span_classifier_by_config_cli()
        assert exc_info.value.code == 1

    def test_no_config_flag_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(sys, "argv", ["train-span-classifier-by-config"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_span_classifier_by_config_cli()
        assert exc_info.value.code != 0


class TestTrainHgereCli:
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
        config_path = self._write_config(tmp_path)
        with (
            patch("hgere.commands.train_hgere.main") as mock_main,
            patch.object(
                sys, "argv", ["train-hgere-by-config", "--config", str(config_path)]
            ),
        ):
            train_hgere_by_config_cli()
        mock_main.assert_called_once()
        argv = mock_main.call_args[0][0]
        assert isinstance(argv, list)

    def test_output_dir_in_argv(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        with (
            patch("hgere.commands.train_hgere.main") as mock_main,
            patch.object(
                sys, "argv", ["train-hgere-by-config", "--config", str(config_path)]
            ),
        ):
            train_hgere_by_config_cli()
        argv = mock_main.call_args[0][0]
        assert "--output_dir" in argv
        assert argv[argv.index("--output_dir") + 1] == "saves/hgere"

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

    def test_model_name_or_path_in_argv(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path)
        with (
            patch("hgere.commands.train_hgere.main") as mock_main,
            patch.object(
                sys, "argv", ["train-hgere-by-config", "--config", str(config_path)]
            ),
        ):
            train_hgere_by_config_cli()
        argv = mock_main.call_args[0][0]
        assert "--model_name_or_path" in argv
        assert "--base_model_name_or_path" not in argv

    def test_no_config_flag_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(sys, "argv", ["train-hgere-by-config"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_hgere_by_config_cli()
        assert exc_info.value.code != 0
