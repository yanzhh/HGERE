"""CLI entry point: train-span-classifier / train-span-classifier-by-config

``train-span-classifier`` (``main()``) — config-driven entry point.  Accepts either:

1. A single positional argument (config file path)::

       uv run train-span-classifier configs/train/gsap/train_gsap.yaml

2. A ``--config`` flag with optional per-field CLI overrides::

       uv run train-span-classifier --config configs/train/gsap/train_gsap.yaml
       uv run train-span-classifier --config configs/train/gsap/train_gsap.yaml \\
           --train_params__learning_rate 2e-6 \\
           --train_params__fp16


The argument parser is generated dynamically from
:class:`~hgere.span_classifier.config.PrunerTrainConfig` (the Pydantic model),
so there is no duplicate parameter definition.

Priority: CLI flags > config file > Pydantic defaults.
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
        A single element that does not start with ``-`` is treated as a config
        file path (shortcut form).  Otherwise a full argparse parser derived
        from :class:`~hgere.span_classifier.config.PrunerTrainConfig` is used.
    """
    if argv is None:
        argv = sys.argv[1:]

    config = load_config_from_argv(
        argv,
        PrunerTrainConfig,
        description=(
            "Train the span pruner model. Pass a config YAML as a positional argument "
            "(train-span-classifier config.yaml) or use --config PATH with optional "
            "per-field CLI overrides (e.g. --train_params__learning_rate 2e-6)."
        ),
    )

    args = _config_to_namespace(config)
    run_train_span_classifier(args)
