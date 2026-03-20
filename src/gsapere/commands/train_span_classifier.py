"""CLI entry point: train-span-classifier

``train-span-classifier`` (``main()``) accepts two modes:

1. **YAML config** — single positional argument::

       uv run train-span-classifier configs/train/gsap/train_gsap.yaml

2. **CLI parameters** — all fields passed directly as flags::

       uv run train-span-classifier \\
           --model_dir saves/pruner/my_run \\
           --label_set gsap \\
           --base_model_name_or_path pretrained_models/scibert \\
           --train_params__data_dir datasets/gsap \\
           --train_params__learning_rate 2e-6 \\
           --train_params__num_train_epochs 8

The argument parser is generated dynamically from
:class:`~hgere.span_classifier.config.PrunerTrainConfig` (the Pydantic model),
so there is no duplicate parameter definition.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from ..span_classifier.config import PrunerTrainConfig
from ..span_classifier.neural.train import run_train_span_classifier
from ._cli_utils import load_config_from_argv

# ---------------------------------------------------------------------------
# Config → flat namespace
# ---------------------------------------------------------------------------


def _config_to_namespace(config: PrunerTrainConfig) -> argparse.Namespace:
    """Convert a validated :class:`PrunerTrainConfig` to a flat :class:`argparse.Namespace`.

    1. Top-level fields (excluding ``schema_version`` and ``train_params``) are
       included as-is.
    2. ``train_params`` fields are flattened into the top level.
    3. ``output_dir`` falls back to ``model_dir`` when not explicitly configured.
    4. ``config_name`` (a legacy HuggingFace arg) is defaulted to ``""``.
    """
    flat: dict[str, Any] = config.model_dump(exclude={"schema_version", "train_params"})
    flat.update(config.train_params.model_dump())

    # output_dir is where prediction files are written; falls back to model_dir.
    if flat.get("output_dir") is None:
        flat["output_dir"] = flat["model_dir"]

    flat.setdefault("config_name", "")

    return argparse.Namespace(**flat)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    """Run the full pruner training pipeline.

    Parameters
    ----------
    argv:
        Argument list.  Pass ``None`` to read from ``sys.argv[1:]``.
        See module docstring for the two accepted modes (YAML file or CLI flags).
    """
    if argv is None:
        argv = sys.argv[1:]

    config = load_config_from_argv(
        argv,
        PrunerTrainConfig,
        description=(
            "Train the span pruner model. Pass a YAML config as a positional argument "
            "(train-span-classifier config.yaml) or supply all parameters directly as "
            "CLI flags (e.g. --model_dir saves/pruner --train_params__learning_rate 2e-6)."
        ),
    )

    args = _config_to_namespace(config)
    run_train_span_classifier(args)
