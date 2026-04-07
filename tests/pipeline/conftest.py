"""Shared fixtures for pipeline tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------


def _make_doc(
    doc_key: str,
    sentences: list[list[str]],
    ner: list[list[list[Any]]] | None = None,
    relations: list[list[list[Any]]] | None = None,
    predicted_ner: list[list[list[Any]]] | None = None,
) -> dict[str, Any]:
    if ner is None:
        ner = [[] for _ in sentences]
    if relations is None:
        relations = [[] for _ in sentences]
    doc: dict[str, Any] = {
        "doc_key": doc_key,
        "sentences": sentences,
        "ner": ner,
        "relations": relations,
    }
    if predicted_ner is not None:
        doc["predicted_ner"] = predicted_ner
    return doc


@pytest.fixture
def tiny_doc() -> dict[str, Any]:
    """Minimal document dict for pipeline input."""
    return _make_doc(
        "test_doc_0",
        [["The", "neural", "model", "trains", "fast"]],
        ner=[[[1, 2, "Method"]]],
        relations=[[]],
    )


@pytest.fixture
def tiny_doc_with_candidates() -> dict[str, Any]:
    """Document dict that already has predicted_ner (from pruner stage)."""
    return _make_doc(
        "test_doc_0",
        [["The", "neural", "model", "trains", "fast"]],
        ner=[[[1, 2, "Method"]]],
        relations=[[]],
        predicted_ner=[[[1, 2, "Method"]]],
    )


@pytest.fixture
def tiny_docs() -> list[dict[str, Any]]:
    """Two-document list for batch pipeline input."""
    return [
        _make_doc(
            "test_doc_0",
            [["The", "neural", "model", "trains", "fast"]],
            ner=[[[1, 2, "Method"]]],
        ),
        _make_doc(
            "test_doc_1",
            [["Results", "show", "accuracy"]],
            ner=[[[2, 2, "Task"]]],
        ),
    ]


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Minimal valid pipeline config YAML."""
    config = {
        "label_set": "gsap",
        "pruner": {
            "model_dir": str(tmp_path / "pruner"),
            "base_model_name_or_path": str(tmp_path / "scibert"),
        },
        "hgere": {
            "model_dir": str(tmp_path / "hgere"),
            "base_model_name_or_path": str(tmp_path / "scibert"),
        },
    }
    path = tmp_path / "config.yaml"
    import yaml

    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def full_config_yaml(tmp_path: Path) -> Path:
    """Full pipeline config YAML matching pipeline/config.yaml format."""
    config = {
        "label_set": "gsap",
        "pruner": {
            "model_dir": str(tmp_path / "pruner"),
            "base_model_name_or_path": str(tmp_path / "scibert"),
            "model_type": "bertspanmarkerpruner",
            "per_gpu_eval_batch_size": 32,
            "max_seq_length": 256,
            "max_pair_length": 64,
            "max_mention_ori_length": 12,
            "final_pruning": {
                "method": "threshold",
                "threshold": 0.0005,
            },
        },
        "hgere": {
            "model_dir": str(tmp_path / "hgere"),
            "base_model_name_or_path": str(tmp_path / "scibert"),
            "model_type": "hyper",
            "per_gpu_eval_batch_size": 32,
            "max_seq_length": 512,
            "factor_type": "tersibcop",
            "factor_encoder": "biaf",
            "ent_dim": 400,
            "rel_dim": 400,
            "mem_dim": 400,
        },
    }
    path = tmp_path / "config_full.yaml"
    import yaml

    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def sentences_predictions_fixture() -> dict[tuple[int, int], list[tuple]]:
    """Fake sentences_predictions dict: (line_idx, sent_idx) -> [(s, e, prob, lbl)]."""
    return {
        (0, 0): [
            (1, 2, 0.95, "Method"),
            (0, 0, 0.001, "NIL"),
            (3, 4, 0.80, "Task"),
            (0, 2, 0.0001, "NIL"),
        ]
    }
