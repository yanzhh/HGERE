"""Tests for train_pruner and train_hgere CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from hgere.commands.train_hgere import _config_to_argv as hgere_config_to_argv
from hgere.commands.train_hgere import cli as train_hgere_cli
from hgere.commands.train_pruner import _config_to_argv as pruner_config_to_argv
from hgere.commands.train_pruner import cli as train_pruner_cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.dump(data))


@pytest.fixture
def minimal_pruner_cfg() -> dict[str, Any]:
    return {
        "model_dir": "saves/pruner",
        "base_model_name_or_path": "pretrained_models/scibert",
        "model_type": "bertspanmarkerpruner",
        "do_lower_case": True,
        "max_seq_length": 256,
        "max_pair_length": 64,
        "max_mention_ori_length": 12,
        "per_gpu_eval_batch_size": 32,
        "final_pruning": {"method": "threshold", "threshold": 0.0005},
    }


@pytest.fixture
def minimal_pruner_train_params() -> dict[str, Any]:
    return {
        "data_dir": "datasets/gsap",
        "train_file": "train.jsonl",
        "dev_file": "dev.jsonl",
        "test_file": "test.jsonl",
        "learning_rate": 1e-6,
        "num_train_epochs": 8,
        "per_gpu_train_batch_size": 8,
        "seed": 44,
        "fp16": True,
        "local_rank": -1,
        "pruner_loss": "focal",
    }


@pytest.fixture
def minimal_hgere_cfg() -> dict[str, Any]:
    return {
        "model_dir": "saves/hgere",
        "base_model_name_or_path": "pretrained_models/scibert",
        "model_type": "hyper",
        "do_lower_case": True,
        "max_seq_length": 512,
        "max_pair_length": 18,
        "factor_type": "tersibcop",
        "factor_encoder": "biaf",
        "ent_dim": 400,
        "rel_dim": 400,
        "mem_dim": 400,
        "iter": 3,
        "layernorm": True,
        "layernorm_1st": True,
    }


@pytest.fixture
def minimal_hgere_train_params() -> dict[str, Any]:
    return {
        "data_dir": "datasets/gsap",
        "train_file": "train.jsonl",
        "dev_file": "dev.jsonl",
        "test_file": "test.jsonl",
        "ner_prediction_dir": "saves/pruner/output",
        "learning_rate": 1e-5,
        "num_train_epochs": 8,
        "per_gpu_train_batch_size": 18,
        "seed": 45,
        "fp16": True,
        "local_rank": -1,
        "loss_re_weight_alpha": 0.9,
        "log_wandb": True,
    }


# ---------------------------------------------------------------------------
# pruner_config_to_argv
# ---------------------------------------------------------------------------


class TestPrunerConfigToArgv:
    def test_model_dir_passed_as_model_dir_and_output_dir(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        assert "--model-dir" in argv
        assert "--output-dir" in argv
        model_dir_val = argv[argv.index("--model-dir") + 1]
        output_dir_val = argv[argv.index("--output-dir") + 1]
        assert model_dir_val == "saves/pruner"
        assert output_dir_val == "saves/pruner"

    def test_label_set_added(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        assert "--label-set" in argv
        assert argv[argv.index("--label-set") + 1] == "gsap"

    def test_do_train_added_by_default(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        assert "--do-train" in argv

    def test_do_lower_case_boolean_flag(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        assert "--do-lower-case" in argv

    def test_fp16_boolean_flag(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        assert "--fp16" in argv

    def test_false_boolean_flag_not_included(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        params = {**minimal_pruner_train_params, "fp16": False}
        argv = pruner_config_to_argv(minimal_pruner_cfg, params, "gsap")
        assert "--fp16" not in argv

    def test_none_values_omitted(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        params = {**minimal_pruner_train_params, "rulebased_pruner_file": None}
        argv = pruner_config_to_argv(minimal_pruner_cfg, params, "gsap")
        assert "--rulebased-pruner-file" not in argv

    def test_final_pruning_skipped(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        # final_pruning is a dict — should not appear in argv
        assert "--final-pruning" not in argv
        assert "--threshold" not in argv

    def test_rulebased_pruner_file_remapped(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        params = {**minimal_pruner_train_params, "rulebased_pruner_file": "gsap_rulebased.json"}
        argv = pruner_config_to_argv(minimal_pruner_cfg, params, "gsap")
        assert "--rulebased-pruner-file" in argv
        assert argv[argv.index("--rulebased-pruner-file") + 1] == "gsap_rulebased.json"

    def test_train_params_override_shared(
        self, minimal_pruner_cfg: dict, minimal_pruner_train_params: dict
    ) -> None:
        # learning_rate from train_params should be used
        argv = pruner_config_to_argv(minimal_pruner_cfg, minimal_pruner_train_params, "gsap")
        assert "--learning-rate" in argv
        assert argv[argv.index("--learning-rate") + 1] == str(1e-6)


# ---------------------------------------------------------------------------
# hgere_config_to_argv
# ---------------------------------------------------------------------------


class TestHgereConfigToArgv:
    def test_model_dir_remapped_to_output_dir(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--output-dir" in argv
        assert argv[argv.index("--output-dir") + 1] == "saves/hgere"
        assert "--model-dir" not in argv

    def test_base_model_remapped_to_model_name_or_path(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--model-name-or-path" in argv
        assert "--base-model-name-or-path" not in argv

    def test_label_set_added(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--label-set" in argv
        assert argv[argv.index("--label-set") + 1] == "gsap"

    def test_do_train_added_by_default(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--do-train" in argv

    def test_eval_dev_eval_test_added_by_default(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--eval-dev" in argv
        assert "--eval-test" in argv

    def test_layernorm_bool_flag(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--layernorm" in argv
        assert "--layernorm-1st" in argv

    def test_log_wandb_bool_flag(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--log-wandb" in argv

    def test_train_params_included(
        self, minimal_hgere_cfg: dict, minimal_hgere_train_params: dict
    ) -> None:
        argv = hgere_config_to_argv(minimal_hgere_cfg, minimal_hgere_train_params, "gsap")
        assert "--ner-prediction-dir" in argv
        assert "--loss-re-weight-alpha" in argv


# ---------------------------------------------------------------------------
# CLI end-to-end (subprocess mocked)
# ---------------------------------------------------------------------------


class TestTrainPrunerCli:
    def _write_config(self, tmp_path: Path, pruner_cfg: dict, train_params: dict) -> Path:
        cfg = {
            "label_set": "gsap",
            "pruner": {**pruner_cfg, "train_params": train_params},
            "hgere": {"model_dir": "x", "base_model_name_or_path": "x"},
        }
        p = tmp_path / "train.yaml"
        _write_yaml(p, cfg)
        return p

    def test_calls_train_span_classifier(
        self,
        tmp_path: Path,
        minimal_pruner_cfg: dict,
        minimal_pruner_train_params: dict,
    ) -> None:
        config_path = self._write_config(tmp_path, minimal_pruner_cfg, minimal_pruner_train_params)
        mock_result = MagicMock(returncode=0)
        with (
            patch("hgere.commands.train_pruner.subprocess.run", return_value=mock_result) as mock_run,
            patch.object(sys, "argv", ["train-pruner", "--config", str(config_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_pruner_cli()
        assert exc_info.value.code == 0
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "uv"
        assert "train-span-classifier" in cmd

    def test_missing_config_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(sys, "argv", ["train-pruner", "--config", str(tmp_path / "missing.yaml")]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_pruner_cli()
        assert exc_info.value.code != 0

    def test_propagates_subprocess_returncode(
        self,
        tmp_path: Path,
        minimal_pruner_cfg: dict,
        minimal_pruner_train_params: dict,
    ) -> None:
        config_path = self._write_config(tmp_path, minimal_pruner_cfg, minimal_pruner_train_params)
        mock_result = MagicMock(returncode=1)
        with (
            patch("hgere.commands.train_pruner.subprocess.run", return_value=mock_result),
            patch.object(sys, "argv", ["train-pruner", "--config", str(config_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_pruner_cli()
        assert exc_info.value.code == 1


class TestTrainHgereCli:
    def _write_config(self, tmp_path: Path, hgere_cfg: dict, train_params: dict) -> Path:
        cfg = {
            "label_set": "gsap",
            "pruner": {"model_dir": "x", "base_model_name_or_path": "x"},
            "hgere": {**hgere_cfg, "train_params": train_params},
        }
        p = tmp_path / "train.yaml"
        _write_yaml(p, cfg)
        return p

    def test_calls_run_hgnn(
        self,
        tmp_path: Path,
        minimal_hgere_cfg: dict,
        minimal_hgere_train_params: dict,
    ) -> None:
        config_path = self._write_config(tmp_path, minimal_hgere_cfg, minimal_hgere_train_params)
        mock_result = MagicMock(returncode=0)
        with (
            patch("hgere.commands.train_hgere.subprocess.run", return_value=mock_result) as mock_run,
            patch.object(sys, "argv", ["train-hgere", "--config", str(config_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_hgere_cli()
        assert exc_info.value.code == 0
        cmd = mock_run.call_args[0][0]
        assert "run_hgnn.py" in cmd

    def test_output_dir_in_command(
        self,
        tmp_path: Path,
        minimal_hgere_cfg: dict,
        minimal_hgere_train_params: dict,
    ) -> None:
        config_path = self._write_config(tmp_path, minimal_hgere_cfg, minimal_hgere_train_params)
        mock_result = MagicMock(returncode=0)
        with (
            patch("hgere.commands.train_hgere.subprocess.run", return_value=mock_result) as mock_run,
            patch.object(sys, "argv", ["train-hgere", "--config", str(config_path)]),
            pytest.raises(SystemExit),
        ):
            train_hgere_cli()
        cmd = mock_run.call_args[0][0]
        assert "--output-dir" in cmd
        assert cmd[cmd.index("--output-dir") + 1] == "saves/hgere"

    def test_missing_config_exits(self, tmp_path: Path) -> None:
        with (
            patch.object(sys, "argv", ["train-hgere", "--config", str(tmp_path / "missing.yaml")]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_hgere_cli()
        assert exc_info.value.code != 0

    def test_propagates_subprocess_returncode(
        self,
        tmp_path: Path,
        minimal_hgere_cfg: dict,
        minimal_hgere_train_params: dict,
    ) -> None:
        config_path = self._write_config(tmp_path, minimal_hgere_cfg, minimal_hgere_train_params)
        mock_result = MagicMock(returncode=2)
        with (
            patch("hgere.commands.train_hgere.subprocess.run", return_value=mock_result),
            patch.object(sys, "argv", ["train-hgere", "--config", str(config_path)]),
            pytest.raises(SystemExit) as exc_info,
        ):
            train_hgere_cli()
        assert exc_info.value.code == 2
