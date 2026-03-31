"""Shared utilities for config-driven CLI entry points.

Used by both ``train_hgere`` and ``train_pruner`` to avoid
duplicating the positional-arg shortcut pattern, JSON arg saving, and
serialisability checks.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Type, TypeVar

from ..config.cli_gen import apply_config_and_cli, build_argparser

M = TypeVar("M")


def load_config_from_argv(
    argv: list[str],
    model_cls: Type[M],
    *,
    description: str,
) -> M:
    """Load and validate a Pydantic config from *argv*.

    Two modes:

    1. **Positional shortcut** — a single argument that does not start with
       ``-`` is treated as a YAML config file path::

           train-hgere configs/train/gsap/train_gsap.yaml

    2. **CLI parameters** — all required and optional fields can be supplied
       directly as ``--flag value`` arguments generated from the Pydantic
       model::

           train-hgere --model_dir saves/hgere \\
               --label_set gsap \\
               --train_params__learning_rate 2e-5

    Parameters
    ----------
    argv:
        Argument list (already sliced from ``sys.argv[1:]``).
    model_cls:
        Pydantic model class with a ``from_yaml`` classmethod.
    description:
        Argparser description shown in ``--help``.

    Returns
    -------
    M
        A fully validated Pydantic config instance.

    Raises
    ------
    SystemExit
        On parse errors or validation failures.
    """
    # ── Mode 1: positional YAML shortcut ─────────────────────────────────────
    if len(argv) == 1 and not argv[0].startswith("-"):
        try:
            return model_cls.from_yaml(argv[0])
        except Exception as exc:
            print(f"Error loading config file '{argv[0]}': {exc}", file=sys.stderr)
            sys.exit(1)

    # ── Mode 2: pure CLI parameters ──────────────────────────────────────────
    parser = build_argparser(model_cls, description=description)
    namespace = parser.parse_args(argv)

    try:
        return apply_config_and_cli(namespace, model_cls)
    except Exception as exc:
        logging.error("Invalid config: %s", exc)
        sys.exit(1)


def save_args(args: Any, path: str, filename: str = "training_args.json") -> None:
    """Persist serialisable namespace fields to *path*/*filename* as JSON."""
    if not os.path.exists(path):
        os.makedirs(path)
    args_file = os.path.join(path, filename)
    with open(args_file, "w") as f:
        json.dump(
            {k: v for k, v in vars(args).items() if is_json_serializable(v)},
            f,
            indent=4,
        )


def is_json_serializable(value: Any) -> bool:
    """Return True if *value* can be serialised to JSON without error."""
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False
