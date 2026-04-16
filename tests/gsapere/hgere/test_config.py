"""Tests for HGERETrainConfig multi-dataset extensions.

Covers:
- Single-dataset config still validates correctly (backward compat)
- MultiDatasetConfig / DatasetEntry parse and validate correctly
- HGERETrainConfig with multi_dataset set validates without label_set/ner_prediction_dir
- Exactly one of (label_set, multi_dataset) must be present
"""

from __future__ import annotations

from typing import Any

import pytest

from gsapere.hgere.config import HGERETrainConfig

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MINIMAL_TRAIN_PARAMS: dict[str, Any] = {
    "learning_rate": 1e-5,
    "num_train_epochs": 5,
    "per_gpu_train_batch_size": 8,
}

SINGLE_DS_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "label_set": "gsap",
    "model_dir": "saves/hgere/run1",
    "base_model_name_or_path": "pretrained_models/scibert",
    "ner_prediction_dir": "data/pruner_output",
    "train_params": MINIMAL_TRAIN_PARAMS,
}

MULTI_DS_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "model_dir": "saves/hgere/multi_run",
    "base_model_name_or_path": "pretrained_models/scibert",
    "multi_dataset": {
        "sampling_temperature": 0.5,
        "datasets": [
            {
                "name": "scier",
                "label_set": "scier",
                "ner_prediction_dir": "saves/scier/pruner/output",
            },
            {
                "name": "scinlp",
                "label_set": "scinlp",
                "ner_prediction_dir": "saves/scinlp/pruner/output",
            },
        ],
    },
    "train_params": MINIMAL_TRAIN_PARAMS,
}


# ---------------------------------------------------------------------------
# Single-dataset backward compat
# ---------------------------------------------------------------------------


class TestSingleDatasetBackwardCompat:
    def test_single_dataset_config_parses(self) -> None:
        config = HGERETrainConfig.model_validate(SINGLE_DS_CONFIG)
        assert config.label_set == "gsap"
        assert config.ner_prediction_dir == "data/pruner_output"
        assert config.multi_dataset is None

    def test_single_dataset_train_params_present(self) -> None:
        config = HGERETrainConfig.model_validate(SINGLE_DS_CONFIG)
        assert config.train_params.learning_rate == pytest.approx(1e-5)

    def test_single_dataset_model_dir(self) -> None:
        config = HGERETrainConfig.model_validate(SINGLE_DS_CONFIG)
        assert config.model_dir == "saves/hgere/run1"


# ---------------------------------------------------------------------------
# Multi-dataset config
# ---------------------------------------------------------------------------


class TestMultiDatasetConfig:
    def test_multi_dataset_config_parses(self) -> None:
        config = HGERETrainConfig.model_validate(MULTI_DS_CONFIG)
        assert config.multi_dataset is not None
        assert len(config.multi_dataset.datasets) == 2

    def test_multi_dataset_no_label_set_required(self) -> None:
        """label_set and ner_prediction_dir should not be required in multi-dataset mode."""
        config = HGERETrainConfig.model_validate(MULTI_DS_CONFIG)
        assert config.label_set is None
        assert config.ner_prediction_dir is None

    def test_dataset_entries_parsed(self) -> None:
        config = HGERETrainConfig.model_validate(MULTI_DS_CONFIG)
        ds = config.multi_dataset.datasets
        assert ds[0].name == "scier"
        assert ds[0].label_set == "scier"
        assert ds[0].ner_prediction_dir == "saves/scier/pruner/output"
        assert ds[1].name == "scinlp"

    def test_dataset_entry_defaults(self) -> None:
        config = HGERETrainConfig.model_validate(MULTI_DS_CONFIG)
        ds = config.multi_dataset.datasets[0]
        assert ds.train_file == "train.json"
        assert ds.dev_file == "dev.json"
        assert ds.test_file == "test.json"
        assert ds.sampling_weight == pytest.approx(1.0)

    def test_sampling_temperature_parsed(self) -> None:
        config = HGERETrainConfig.model_validate(MULTI_DS_CONFIG)
        assert config.multi_dataset.sampling_temperature == pytest.approx(0.5)

    def test_sampling_temperature_default(self) -> None:
        data = {**MULTI_DS_CONFIG, "multi_dataset": {**MULTI_DS_CONFIG["multi_dataset"]}}
        del data["multi_dataset"]["sampling_temperature"]
        config = HGERETrainConfig.model_validate(data)
        assert config.multi_dataset.sampling_temperature == pytest.approx(0.5)

    def test_multi_dataset_with_label_set_raises(self) -> None:
        """Providing both label_set and multi_dataset should fail."""
        data = {**MULTI_DS_CONFIG, "label_set": "gsap"}
        with pytest.raises(Exception):
            HGERETrainConfig.model_validate(data)


# ---------------------------------------------------------------------------
# Validation: neither label_set nor multi_dataset
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_neither_label_set_nor_multi_dataset_raises(self) -> None:
        data = {
            "schema_version": "1.0",
            "model_dir": "saves/hgere/run1",
            "base_model_name_or_path": "pretrained_models/scibert",
            "train_params": MINIMAL_TRAIN_PARAMS,
        }
        with pytest.raises(Exception):
            HGERETrainConfig.model_validate(data)

    def test_single_dataset_without_ner_prediction_dir_raises(self) -> None:
        data = {k: v for k, v in SINGLE_DS_CONFIG.items() if k != "ner_prediction_dir"}
        with pytest.raises(Exception):
            HGERETrainConfig.model_validate(data)
