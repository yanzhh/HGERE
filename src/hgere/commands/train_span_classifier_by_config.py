"""CLI command: train-span-classifier-by-config

Trains the span pruner from a YAML training config.

The CLI is generated dynamically from :class:`~hgere.span_classifier.config.PrunerTrainConfig`
so that the Pydantic model is the single source of truth for all parameters.

Usage — config file only
------------------------
    uv run train-span-classifier-by-config --config configs/train/gsap/train_gsap.yaml

Usage — override individual values via CLI
------------------------------------------
    uv run train-span-classifier-by-config \\
        --config configs/train/gsap/train_gsap.yaml \\
        --train_params__learning_rate 2e-6 \\
        --train_params__fp16

Priority: CLI flags > config file > Pydantic defaults.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

from hgere.config.cli_gen import apply_config_and_cli, build_argparser
from hgere.span_classifier.config import PrunerTrainConfig

# ---------------------------------------------------------------------------
# Boolean store_true flags accepted by train-span-classifier
# ---------------------------------------------------------------------------

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
    }
)

# CLI flag remaps: field name → flag name used by train-span-classifier
_KEY_REMAP: dict[str, str] = {
    "rulebased_pruner_file": "rulebased-pruner-file",
}


# ---------------------------------------------------------------------------
# Model → subprocess argv for train-span-classifier
# ---------------------------------------------------------------------------


def model_to_argv(config: PrunerTrainConfig) -> list[str]:
    """Translate a validated :class:`PrunerTrainConfig` to a flat argv list for
    ``train-span-classifier``.

    The pruner trainer (``train_span_classifier.py``) still uses a flat argparse
    namespace.  This function bridges the two worlds by:

    1. Flattening ``train_params`` into the top-level dict (dropping the prefix).
    2. Duplicating ``model_dir`` as ``output_dir`` (the trainer expects both).
    3. Injecting the always-on training mode flags.
    4. Converting booleans to ``store_true`` style flags.
    """
    flat: dict[str, Any] = config.model_dump(exclude={"schema_version", "train_params"})
    flat.update(config.train_params.model_dump())

    # The trainer expects --output_dir as well as --model_dir.
    flat["output_dir"] = flat["model_dir"]

    # Always-on training flags.
    flat.setdefault("do_train", True)
    flat.setdefault("do_eval", True)
    flat.setdefault("do_test", True)
    flat.setdefault("output_results", True)

    argv: list[str] = []
    for key, value in flat.items():
        if value is None:
            continue
        flag = _KEY_REMAP.get(key, key)
        if key in _BOOL_FLAGS:
            if value:
                argv.append(f"--{flag}")
        else:
            argv.extend([f"--{flag}", str(value)])
    return argv


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cli() -> None:
    parser = build_argparser(
        PrunerTrainConfig,
        description="Train the span pruner. Parameters are loaded from a YAML config "
        "and can be overridden with individual CLI flags.",
    )
    namespace = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Require at least --config (individual flags alone are not enough to make
    # a valid config without all required fields).
    if namespace.config is None:
        parser.error("--config is required.")

    # Load + merge + validate.  Pydantic gives descriptive errors on bad input.
    try:
        config = apply_config_and_cli(namespace, PrunerTrainConfig)
    except Exception as exc:
        logger.error("Invalid config: %s", exc)
        sys.exit(1)

    argv = model_to_argv(config)
    cmd = ["uv", "run", "train-span-classifier"] + argv
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
