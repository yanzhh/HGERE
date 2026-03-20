"""Tests for train_hgere refactoring.

Covers:
- _config_to_namespace: field mapping, key remaps, config_name default
- main(): positional-arg shortcut routes to HGERETrainConfig.from_yaml
- main(): --config flag routes through build_argparser + apply_config_and_cli
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hgere.commands.train_hgere import _KEY_REMAP, _config_to_namespace
from hgere.hgere.config import HGERETrainConfig


# ---------------------------------------------------------------------------
# Minimal valid config fixture
# ---------------------------------------------------------------------------

MINIMAL_TRAIN_PARAMS: dict[str, Any] = {
    "learning_rate": 1e-5,
    "num_train_epochs": 5,
    "per_gpu_train_batch_size": 8,
}

MINIMAL_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "label_set": "gsap",
    "model_dir": "saves/hgere/my_run",
    "base_model_name_or_path": "pretrained_models/scibert",
    "ner_prediction_dir": "data/pruner_output",
    "train_params": MINIMAL_TRAIN_PARAMS,
}


def _make_config(**overrides: Any) -> HGERETrainConfig:
    data = {**MINIMAL_CONFIG}
    if "train_params" in overrides:
        data["train_params"] = {**MINIMAL_TRAIN_PARAMS, **overrides.pop("train_params")}
    data.update(overrides)
    return HGERETrainConfig.model_validate(data)


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
    def test_output_dir_mapped_from_model_dir(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.output_dir == "saves/hgere/my_run"
        assert not hasattr(ns, "model_dir")

    def test_model_name_or_path_mapped_from_base_model_name_or_path(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.model_name_or_path == "pretrained_models/scibert"
        assert not hasattr(ns, "base_model_name_or_path")

    def test_iter_mapped_from_n_iter(self) -> None:
        config = _make_config(n_iter=5)
        ns = _config_to_namespace(config)
        assert ns.iter == 5
        assert not hasattr(ns, "n_iter")

    def test_config_name_defaults_to_empty_string(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.config_name == ""

    def test_train_params_flattened_to_top_level(self) -> None:
        config = _make_config(train_params={"learning_rate": 2e-5})
        ns = _config_to_namespace(config)
        assert ns.learning_rate == 2e-5
        assert not hasattr(ns, "train_params")

    def test_top_level_fields_present(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.label_set == "gsap"
        assert ns.ner_prediction_dir == "data/pruner_output"

    def test_schema_version_excluded(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert not hasattr(ns, "schema_version")

    def test_bool_fields_preserved(self) -> None:
        config = _make_config(train_params={"fp16": True})
        ns = _config_to_namespace(config)
        assert ns.fp16 is True

    def test_key_remap_covers_all_expected_mappings(self) -> None:
        assert _KEY_REMAP == {
            "model_dir": "output_dir",
            "base_model_name_or_path": "model_name_or_path",
            "n_iter": "iter",
        }


# ---------------------------------------------------------------------------
# Shared patch context for main() routing tests
# ---------------------------------------------------------------------------

_MAIN_PATCHES = [
    "hgere.commands.train_hgere.setup_training",
    "hgere.commands.train_hgere.get_logger",
    "os.path.exists",
    "os.makedirs",
    "os.mkdir",
    "os.listdir",
    "hgere.commands.train_hgere.get_last_checkpoint",
]


def _patch_main(fake_setup: Any) -> Any:
    """Return a combined patch context that stubs training but allows real file I/O."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("hgere.commands.train_hgere.setup_training", side_effect=fake_setup)
    )
    stack.enter_context(
        patch("hgere.commands.train_hgere.get_logger", return_value=MagicMock())
    )
    stack.enter_context(
        patch("hgere.commands.train_hgere.get_last_checkpoint", return_value=(None, 0))
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

        with _patch_main(lambda args, logger: captured.append(args)):
            from hgere.commands.train_hgere import main

            main([config_path])

        assert len(captured) == 1
        ns = captured[0]
        assert ns.output_dir.endswith("model")
        assert ns.model_name_or_path == "pretrained_models/scibert"
        assert ns.label_set == "gsap"

    def test_config_flag_loads_config(self, tmp_path: Path) -> None:
        """--config flag also loads config correctly."""
        config_path = _write_config_yaml(tmp_path)
        captured: list[Any] = []

        with _patch_main(lambda args, logger: captured.append(args)):
            from hgere.commands.train_hgere import main

            main(["--config", config_path])

        assert len(captured) == 1
        assert captured[0].output_dir.endswith("model")

    def test_config_flag_with_override(self, tmp_path: Path) -> None:
        """CLI override (--train_params__learning_rate) takes precedence over config file."""
        config_path = _write_config_yaml(tmp_path)
        captured: list[Any] = []

        with _patch_main(lambda args, logger: captured.append(args)):
            from hgere.commands.train_hgere import main

            main(["--config", config_path, "--train_params__learning_rate", "3e-5"])

        assert captured[0].learning_rate == pytest.approx(3e-5)

    def test_no_args_prints_error_and_exits(self) -> None:
        """Calling main() with no args should exit non-zero."""
        with pytest.raises(SystemExit):
            from hgere.commands.train_hgere import main

            main([])
