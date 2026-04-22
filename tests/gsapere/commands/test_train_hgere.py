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

from gsapere.commands.train_hgere import (
    _apply_seed_suffix,
    _config_to_namespace,
    _resolve_seeds,
)
from gsapere.hgere.config import HGERETrainConfig


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
    def test_model_dir_present(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.model_dir == "saves/hgere/my_run"

    def test_base_model_name_or_path_present(self) -> None:
        config = _make_config()
        ns = _config_to_namespace(config)
        assert ns.base_model_name_or_path == "pretrained_models/scibert"

    def test_n_iter_present(self) -> None:
        config = _make_config(n_iter=5)
        ns = _config_to_namespace(config)
        assert ns.n_iter == 5

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

    def test_run_name_defaults_to_model_dir_name(self) -> None:
        config = _make_config(model_dir="saves/hgere/my_experiment")
        ns = _config_to_namespace(config)
        assert ns.run_name == "my_experiment"

    def test_run_name_not_overridden_when_set(self) -> None:
        config = _make_config(train_params={"run_name": "custom_name"})
        ns = _config_to_namespace(config)
        assert ns.run_name == "custom_name"

    def test_wandb_group_defaults_to_run_name_single_seed(self) -> None:
        config = _make_config(model_dir="saves/hgere/my_experiment")
        ns = _config_to_namespace(config)
        assert ns.wandb_group == "my_experiment"

    def test_wandb_group_not_overridden_when_set(self) -> None:
        config = _make_config(train_params={"wandb_group": "my_group"})
        ns = _config_to_namespace(config)
        assert ns.wandb_group == "my_group"


# ---------------------------------------------------------------------------
# Shared patch context for main() routing tests
# ---------------------------------------------------------------------------

_MAIN_PATCHES = [
    "gsapere.commands.train_hgere.setup_training",
    "gsapere.commands.train_hgere.get_logger",
    "os.path.exists",
    "os.makedirs",
    "os.mkdir",
    "os.listdir",
    "gsapere.commands.train_hgere.get_last_checkpoint",
]


def _patch_main(fake_setup: Any) -> Any:
    """Return a combined patch context that stubs training but allows real file I/O."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("gsapere.commands.train_hgere.setup_training", side_effect=fake_setup)
    )
    stack.enter_context(
        patch("gsapere.commands.train_hgere.get_logger", return_value=MagicMock())
    )
    stack.enter_context(
        patch(
            "gsapere.commands.train_hgere.get_last_checkpoint", return_value=(None, 0)
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

        with _patch_main(lambda args, logger: captured.append(args)):
            from gsapere.commands.train_hgere import main

            main([config_path])

        assert len(captured) == 1
        ns = captured[0]
        assert ns.model_dir.endswith("model")
        assert ns.base_model_name_or_path == "pretrained_models/scibert"
        assert ns.label_set == "gsap"

    def test_pure_cli_params(self, tmp_path: Path) -> None:
        """All parameters supplied as CLI flags (no config file)."""
        captured: list[Any] = []

        with _patch_main(lambda args, logger: captured.append(args)):
            from gsapere.commands.train_hgere import main

            main(
                [
                    "--model_dir",
                    str(tmp_path / "model"),
                    "--label_set",
                    "gsap",
                    "--base_model_name_or_path",
                    "pretrained_models/scibert",
                    "--ner_prediction_dir",
                    "saves/pruner/output",
                    "--train_params__learning_rate",
                    "2e-5",
                    "--train_params__num_train_epochs",
                    "5",
                    "--train_params__per_gpu_train_batch_size",
                    "8",
                ]
            )

        assert len(captured) == 1
        ns = captured[0]
        assert ns.model_dir.endswith("model")
        assert ns.learning_rate == pytest.approx(2e-5)
        assert ns.label_set == "gsap"

    def test_no_args_prints_error_and_exits(self) -> None:
        """Calling main() with no args should exit non-zero (missing required fields)."""
        with pytest.raises(SystemExit):
            from gsapere.commands.train_hgere import main

            main([])


# ---------------------------------------------------------------------------
# _resolve_seeds
# ---------------------------------------------------------------------------


class TestResolveSeeds:
    def test_single_int(self) -> None:
        config = _make_config(seeds=42)
        assert _resolve_seeds(config) == [42]

    def test_list_of_ints(self) -> None:
        config = _make_config(seeds=[1, 2, 3])
        assert _resolve_seeds(config) == [1, 2, 3]

    def test_default_is_42(self) -> None:
        config = _make_config()
        assert _resolve_seeds(config) == [42]


# ---------------------------------------------------------------------------
# _apply_seed_suffix
# ---------------------------------------------------------------------------


class TestApplySeedSuffix:
    def test_model_dir_gets_suffix(self) -> None:
        config = _make_config()
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.model_dir == "saves/hgere/my_run_seed7"

    def test_seeds_resolved_to_single_int(self) -> None:
        config = _make_config(seeds=[42, 7])
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.seeds == 7

    def test_run_name_gets_suffix_when_set(self) -> None:
        config = _make_config(train_params={"run_name": "myrun"})
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.run_name == "myrun_seed7"

    def test_run_name_stays_none_when_unset(self) -> None:
        config = _make_config()
        assert config.train_params.run_name is None
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.run_name is None

    def test_output_dir_gets_suffix_when_set(self) -> None:
        config = _make_config(train_params={"output_dir": "saves/output"})
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.output_dir == "saves/output_seed7"

    def test_output_dir_stays_none_when_unset(self) -> None:
        config = _make_config()
        assert config.train_params.output_dir is None
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.output_dir is None

    def test_wandb_group_set_to_base_run_name(self) -> None:
        config = _make_config(train_params={"run_name": "myrun"})
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.wandb_group == "myrun"

    def test_wandb_group_not_overridden_when_already_set(self) -> None:
        config = _make_config(
            train_params={"run_name": "myrun", "wandb_group": "mygroup"}
        )
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.wandb_group == "mygroup"

    def test_wandb_group_falls_back_to_model_dir_name_when_run_name_unset(self) -> None:
        config = _make_config(model_dir="saves/hgere/my_experiment")
        assert config.train_params.run_name is None
        result = _apply_seed_suffix(config, 7, "_seed7")
        assert result.train_params.wandb_group == "my_experiment"


# ---------------------------------------------------------------------------
# _config_to_namespace – seed field
# ---------------------------------------------------------------------------


class TestConfigToNamespaceSeed:
    def test_seed_present_single(self) -> None:
        config = _make_config(seeds=99)
        ns = _config_to_namespace(config)
        assert ns.seed == 99

    def test_seed_resolved_from_list(self) -> None:
        # After _apply_seed_suffix the list collapses to a single int.
        config = _make_config(seeds=[10, 20])
        applied = _apply_seed_suffix(config, 10, "_seed10")
        ns = _config_to_namespace(applied)
        assert ns.seed == 10

    def test_seeds_field_not_in_namespace(self) -> None:
        config = _make_config(seeds=42)
        ns = _config_to_namespace(config)
        assert not hasattr(ns, "seeds")


# ---------------------------------------------------------------------------
# Deprecation error for train_params.seed
# ---------------------------------------------------------------------------


class TestDeprecatedSeedField:
    def test_train_params_seed_raises_deprecation(self) -> None:
        data = {
            **MINIMAL_CONFIG,
            "train_params": {**MINIMAL_TRAIN_PARAMS, "seed": 42},
        }
        with pytest.raises(Exception, match="'train_params.seed' has been moved"):
            HGERETrainConfig.model_validate(data)


# ---------------------------------------------------------------------------
# main() multi-seed integration
# ---------------------------------------------------------------------------


class TestMainMultiSeed:
    def test_two_seeds_call_setup_training_twice(self, tmp_path: Path) -> None:
        config_data = {
            **MINIMAL_CONFIG,
            "model_dir": str(tmp_path / "model"),
            "seeds": [10, 20],
        }
        p = tmp_path / "train.yaml"
        p.write_text(yaml.dump(config_data))

        captured: list[Any] = []

        with _patch_main(lambda args, logger: captured.append(args)):
            from gsapere.commands.train_hgere import main

            main([str(p)])

        assert len(captured) == 2
        dirs = [ns.model_dir for ns in captured]
        assert any("_seed10" in d for d in dirs)
        assert any("_seed20" in d for d in dirs)

    def test_single_seed_no_suffix(self, tmp_path: Path) -> None:
        model_dir = str(tmp_path / "model")
        config_data = {
            **MINIMAL_CONFIG,
            "model_dir": model_dir,
            "seeds": 42,
        }
        p = tmp_path / "train.yaml"
        p.write_text(yaml.dump(config_data))

        captured: list[Any] = []

        with _patch_main(lambda args, logger: captured.append(args)):
            from gsapere.commands.train_hgere import main

            main([str(p)])

        assert len(captured) == 1
        assert captured[0].model_dir == model_dir

    def test_multi_seed_correct_seed_value_per_run(self, tmp_path: Path) -> None:
        config_data = {
            **MINIMAL_CONFIG,
            "model_dir": str(tmp_path / "model"),
            "seeds": [7, 13],
        }
        p = tmp_path / "train.yaml"
        p.write_text(yaml.dump(config_data))

        captured: list[Any] = []

        with _patch_main(lambda args, logger: captured.append(args)):
            from gsapere.commands.train_hgere import main

            main([str(p)])

        seeds_seen = {ns.seed for ns in captured}
        assert seeds_seen == {7, 13}
