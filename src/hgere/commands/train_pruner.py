"""CLI command: train-pruner

Trains the span pruner from a YAML training config.

Usage
-----
    uv run train-pruner --config configs/train/gsap/train_gsap.yaml
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# Keys under pruner: that are inference-only and must not be forwarded to the trainer.
_PRUNER_SKIP_KEYS: frozenset[str] = frozenset({"final_pruning", "train_params"})

# Boolean store_true flags accepted by train-span-classifier.
_BOOL_FLAGS: frozenset[str] = frozenset(
    {
        "do_train",
        "do_eval",
        "do_test",
        "do_lower_case",
        "evaluate_during_training",
        "eval_all_checkpoints",
        "overwrite_model_dir",
        "fp16",
        "onedropout",
        "lminit",
        "nocross",
        "biaf_span",
        "biaf_factorize",
        "output_results",
        "no_cuda",
        "overwrite_cache",
        "norm_emb",
        "shuffle",
        "group_edge",
        "group_sort",
        "no_test",
    }
)

# YAML key → CLI flag name (only where they differ).
_KEY_REMAP: dict[str, str] = {
    "rulebased_pruner_file": "rulebased-pruner-file",
    "prune_config": "prune-config",
    "base_model_name_or_path": "base-model-name-or-path",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the span pruner from a YAML config.")
    p.add_argument(
        "--config", type=str, required=True, help="Path to training YAML config."
    )
    return p


def _config_to_argv(
    shared: dict[str, Any],
    train_params: dict[str, Any],
    label_set: str,
) -> list[str]:
    flat: dict[str, Any] = {
        k: v
        for k, v in shared.items()
        if k not in _PRUNER_SKIP_KEYS and not isinstance(v, dict)
    }
    flat.update(train_params)
    flat["label_set"] = label_set

    # pruner trainer expects both --model_dir and --output_dir; use model_dir for both.
    if "model_dir" in flat and "output_dir" not in flat:
        flat["output_dir"] = flat["model_dir"]

    # Always enable training + evaluation modes.
    flat.setdefault("do_train", True)
    flat.setdefault("do_eval", True)
    flat.setdefault("do_test", True)

    argv: list[str] = []
    for key, value in flat.items():
        if value is None:
            continue
        flag = _KEY_REMAP.get(key, key.replace("_", "-"))
        if key in _BOOL_FLAGS:
            if value:
                argv.append(f"--{flag}")
        else:
            argv.extend([f"--{flag}", str(value)])
    return argv


def cli() -> None:
    parsed = _build_parser().parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    config_path = Path(parsed.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    label_set: str = cfg.get("label_set", "")
    pruner_cfg: dict[str, Any] = cfg.get("pruner", {})
    train_params: dict[str, Any] = pruner_cfg.get("train_params", {})

    argv = _config_to_argv(pruner_cfg, train_params, label_set)
    cmd = ["uv", "run", "train-span-classifier"] + argv
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
