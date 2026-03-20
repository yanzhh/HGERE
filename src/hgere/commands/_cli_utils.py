"""Shared utilities for config-driven CLI entry points.

Used by both ``train_hgere`` and ``train_span_classifier`` to avoid
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
    config_loader: Any = None,
) -> M:
    """Load and validate a Pydantic config from *argv*.

    Two modes:

    1. **Positional shortcut** — a single argument that does not start with
       ``-`` is treated as a config file path::

           train-hgere configs/train/gsap/train_gsap.yaml

    2. **Full argparser** — ``--config PATH`` with optional per-field
       overrides generated from the Pydantic model::

           train-hgere --config configs/train/gsap/train_gsap.yaml \\
               --train_params__learning_rate 2e-5

    Parameters
    ----------
    argv:
        Argument list (already sliced from ``sys.argv[1:]``).
    model_cls:
        Pydantic model class with a ``from_yaml`` classmethod.
    description:
        Argparser description shown in ``--help``.
    config_loader:
        Optional callable ``(path: str) -> dict`` used by
        :func:`apply_config_and_cli`.  Defaults to
        ``lambda p: model_cls.from_yaml(p).model_dump()``.

    Returns
    -------
    M
        A fully validated Pydantic config instance.

    Raises
    ------
    SystemExit
        On parse errors or validation failures.
    """
    if config_loader is None:
        config_loader = lambda p: model_cls.from_yaml(p).model_dump()  # noqa: E731

    # ── Positional shortcut ──────────────────────────────────────────────────
    if len(argv) == 1 and not argv[0].startswith("-"):
        try:
            return model_cls.from_yaml(argv[0])
        except Exception as exc:
            print(f"Error loading config file '{argv[0]}': {exc}", file=sys.stderr)
            sys.exit(1)

    # ── Full argparser from Pydantic model ───────────────────────────────────
    parser = build_argparser(model_cls, description=description)
    namespace = parser.parse_args(argv)

    if namespace.config is None:
        parser.error(
            "Provide a config file as a positional argument "
            "(<command> config.yaml) or with --config PATH."
        )

    try:
        return apply_config_and_cli(
            namespace,
            model_cls,
            config_loader=config_loader,
        )
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
