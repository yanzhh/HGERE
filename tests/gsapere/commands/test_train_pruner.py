"""Tests for train_pruner refactoring.

Covers:
- _config_to_namespace: model_dir/output_dir duplication, neg_inf, config_name
- main(): positional-arg shortcut routes to PrunerTrainConfig.from_yaml
- main(): --config flag routes through build_argparser + apply_config_and_cli
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from gsapere.commands.train_pruner import _config_to_namespace
from gsapere.pruner.config import PrunerTrainConfig


# ---------------------------------------------------------------------------
# Minimal valid config fixture
# ---------------------------------------------------------------------------

MINIMAL_TRAIN_PARAMS: dict[str, Any] = {
    "data_dir": "datasets/gsap",
    "learning_rate": 1e-6,
    "num_train_epochs": 8,
    "per_gpu_train_batch_size": 8,
}

MINIMAL_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "label_set": "gsap",
    "model_dir": "saves/pruner/my_run",
    "base_model_name_or_path": "pretrained_models/scibert",
    "train_params": MINIMAL_TRAIN_PARAMS,
}


def _make_config(**overrides: Any) -> PrunerTrainConfig:
    data = {**MINIMAL_CONFIG}
    if "train_params" in overrides:
        data["train_params"] = {**MINIMAL_TRAIN_PARAMS, **overrides.pop("train_params")}
    data.update(overrides)
    return PrunerTrainConfig.model_validate(data)


def _write_config_yaml(tmp_path: Path) -> str:
    """Write a config YAML using tmp_path as model_dir so file I/O succeeds."""
    config = {
        **MINIMAL_CONFIG,
        "model_dir": str(tmp_path / "model"),
    }
    p = tmp_path / "train.yaml"
    p.write_text(yaml.dump(config))
    return str(p)


# ---------------------------------------------------------------------------
# _config_to_namespace
# ---------------------------------------------------------------------------


class TestConfigToNamespace:
    def test_model_dir_present(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.model_dir == "saves/pruner/my_run"

    def test_output_dir_defaults_to_model_dir(self) -> None:
        """output_dir falls back to model_dir when not explicitly set."""
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.output_dir == ns.model_dir == "saves/pruner/my_run"

    def test_output_dir_explicit(self) -> None:
        """Explicitly configured output_dir takes precedence over model_dir."""
        config = _make_config(output_dir="saves/pruner/predictions")
        ns = _config_to_namespace(config)
        assert ns.output_dir == "saves/pruner/predictions"
        assert ns.model_dir == "saves/pruner/my_run"

    def test_base_model_name_or_path_present(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.base_model_name_or_path == "pretrained_models/scibert"

    def test_config_name_defaults_to_empty_string(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.config_name == ""

    def test_neg_inf_not_in_namespace(self) -> None:
        """neg_inf is no longer set by _config_to_namespace — computed in evaluate.py."""
        config = _make_config()
        ns = _config_to_namespace(config)
        assert not hasattr(ns, "neg_inf")

    def test_train_params_flattened_to_top_level(self) -> None:
        config = _make_config(train_params={"learning_rate": 2e-6})
        ns = _config_to_namespace(config)
        assert ns.learning_rate == 2e-6
        assert not hasattr(ns, "train_params")

    def test_top_level_fields_present(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.label_set == "gsap"
        assert ns.model_type == "bertspanmarkerpruner"

    def test_schema_version_excluded(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert not hasattr(ns, "schema_version")

    def test_bool_fields_preserved(self) -> None:
        config = _make_config(train_params={"fp16": True})
        ns = _config_to_namespace(config)
        assert ns.fp16 is True

    def test_do_train_defaults_true(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.do_train is True

    def test_alpha_present(self) -> None:
        config = _make_config(alpha=0.5)
        ns = _config_to_namespace(config)
        assert ns.alpha == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Shared patch context for main() routing tests
# ---------------------------------------------------------------------------


def _patch_main(fake_run: Any) -> Any:
    """Return a combined patch context that stubs training."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch(
            "gsapere.commands.train_pruner.run_train_pruner",
            side_effect=fake_run,
        )
    )
    return stack


# ---------------------------------------------------------------------------
# main() routing tests
# ---------------------------------------------------------------------------


class TestMainRouting:
    def test_positional_arg_loads_config(self, tmp_path: Path) -> None:
        """Single positional arg (no '--') triggers from_yaml loading."""
        config_path = _write_config_yaml(tmp_path)
        captured: list[Any] = []

        with _patch_main(lambda args: captured.append(args)):
            from gsapere.commands.train_pruner import main

            main([config_path])

        assert len(captured) == 1
        ns = captured[0]
        assert ns.model_dir.endswith("model")
        assert ns.output_dir == ns.model_dir
        assert ns.base_model_name_or_path == "pretrained_models/scibert"
        assert ns.label_set == "gsap"

    def test_pure_cli_params(self, tmp_path: Path) -> None:
        """All parameters supplied as CLI flags (no config file)."""
        captured: list[Any] = []

        with _patch_main(lambda args: captured.append(args)):
            from gsapere.commands.train_pruner import main

            main(
                [
                    "--model_dir",
                    str(tmp_path / "model"),
                    "--label_set",
                    "gsap",
                    "--base_model_name_or_path",
                    "pretrained_models/scibert",
                    "--train_params__data_dir",
                    "datasets/gsap",
                    "--train_params__learning_rate",
                    "2e-6",
                    "--train_params__num_train_epochs",
                    "8",
                    "--train_params__per_gpu_train_batch_size",
                    "8",
                ]
            )

        assert len(captured) == 1
        ns = captured[0]
        assert ns.model_dir.endswith("model")
        assert ns.output_dir == ns.model_dir
        assert ns.learning_rate == pytest.approx(2e-6)

    def test_no_args_prints_error_and_exits(self) -> None:
        """Calling main() with no args should exit non-zero (missing required fields)."""
        with pytest.raises(SystemExit):
            from gsapere.commands.train_pruner import main

            main([])
