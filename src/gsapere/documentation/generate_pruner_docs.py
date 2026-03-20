"""Generate parameter documentation for the span pruner and HGERE model.

Reads PrunerTrainConfig and HGERETrainConfig (the single sources of truth) and
renders Markdown files with parameter tables.

Run this script whenever the Pydantic models change to keep documentation in
sync.

Usage
-----
    uv run generate-pruner-docs
    uv run generate-pruner-docs --output docs/custom.md
    uv run generate-pruner-docs --model hgere --output docs/hgere.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Union, get_args, get_origin


try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore[assignment]

from gsapere.config.cli_gen import SEP, FlatField, collect_flat_fields
from gsapere.hgere.config import (
    CURRENT_SCHEMA_VERSION as HGERE_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS as HGERE_SUPPORTED_VERSIONS,
    HGERETrainConfig,
)
from gsapere.span_classifier.config import (
    CURRENT_SCHEMA_VERSION as PRUNER_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS as PRUNER_SUPPORTED_VERSIONS,
    PrunerTrainConfig,
)

_DEFAULT_PRUNER_OUTPUT = Path("documentation/api/pruner.md")
_DEFAULT_HGERE_OUTPUT = Path("documentation/api/hgere.md")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_type(annotation: Any) -> str:
    """Convert a Python type annotation to a readable Markdown string."""
    origin = get_origin(annotation)

    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return f"{_format_type(non_none[0])} *(optional)*"
        return " | ".join(_format_type(a) for a in non_none)

    if origin is Literal:
        return " | ".join(f'`"{c}"`' for c in get_args(annotation))

    type_map = {str: "`string`", int: "`integer`", float: "`float`", bool: "`boolean`"}
    return type_map.get(
        annotation, f"`{getattr(annotation, '__name__', str(annotation))}`"
    )


def _format_default(field: FlatField) -> str:
    """Format a field's default value for Markdown display."""
    if field.is_required:
        return "**required**"
    v = field.default
    if v is None:
        return "`null`"
    if isinstance(v, bool):
        return f"`{'true' if v else 'false'}`"
    if isinstance(v, str):
        return f'`"{v}"`'
    return f"`{v}`"


def _md_escape(text: str) -> str:
    """Escape pipe characters so they don't break Markdown table cells."""
    return text.replace("|", "\\|")


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _render_table(fields: list[FlatField]) -> str:
    """Render a list of FlatFields as a Markdown parameter table."""
    header = "| Parameter | CLI flag | Type | Default | Description |"
    sep = "|-----------|----------|------|---------|-------------|"
    rows = [header, sep]
    for f in fields:
        param = f"`{f.path[-1]}`"
        flag = f"`--{f.flat_name}`"
        typ = _format_type(f.annotation)
        dflt = _format_default(f)
        desc = _md_escape(f.description or "")
        rows.append(f"| {param} | {flag} | {typ} | {dflt} | {desc} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------


def _split_fields(
    model_cls: type,
) -> tuple[list[FlatField], list[FlatField]]:
    """Return (top_level_fields, train_params_fields)."""
    all_fields = collect_flat_fields(model_cls)

    top: list[FlatField] = []
    train: list[FlatField] = []

    for f in all_fields:
        if f.path[0] == "train_params":
            train.append(f)
        else:
            top.append(f)

    return top, train


# ---------------------------------------------------------------------------
# Pruner document renderer
# ---------------------------------------------------------------------------


def render_pruner_document() -> str:
    """Return the full Markdown documentation string for the pruner."""
    top_fields, train_fields = _split_fields(PrunerTrainConfig)

    _TRAIN_GROUPS: list[tuple[str, list[str]]] = [
        (
            "Data",
            [
                "data_dir",
                "train_file",
                "dev_file",
                "test_file",
                "rulebased_pruner_file",
            ],
        ),
        (
            "Optimisation",
            [
                "seed",
                "learning_rate",
                "learning_rate_span",
                "num_train_epochs",
                "eval_epochs",
                "per_gpu_train_batch_size",
                "gradient_accumulation_steps",
                "adam_epsilon",
                "weight_decay",
                "max_grad_norm",
                "max_steps",
                "warmup_steps",
                "logging_steps",
                "save_steps",
                "save_total_limit",
                "fp16",
                "local_rank",
            ],
        ),
        (
            "Hardware",
            [
                "no_cuda",
                "server_ip",
                "server_port",
                "debug_overflow",
            ],
        ),
        ("Loss", ["pruner_loss", "focal_gamma", "focal_alpha"]),
        (
            "Candidate filtering (eval during training)",
            [
                "topk_ratio",
                "min_mentions_num",
                "max_mentions_num",
            ],
        ),
        (
            "Model flags",
            [
                "onedropout",
                "lminit",
                "nocross",
                "biaf_span",
                "biaf_mode",
                "biaf_factorize",
                "span_hidden_size",
                "rank",
                "span_size",
            ],
        ),
        (
            "Evaluation & checkpointing",
            [
                "evaluate_during_training",
                "eval_all_checkpoints",
                "overwrite_model_dir",
                "overwrite_cache",
            ],
        ),
        (
            "Run modes",
            [
                "do_train",
                "do_test",
                "output_results",
                "shuffle",
            ],
        ),
        (
            "Eval settings",
            [
                "target_recall_diff",
                "prune_config",
                "use_full_layer",
            ],
        ),
        ("Weights & Biases", ["project_name", "run_name"]),
    ]

    train_by_name: dict[str, FlatField] = {f.path[-1]: f for f in train_fields}

    lines: list[str] = []

    lines += [
        "# Pruner parameter reference",
        "",
        "> **Auto-generated** from `hgere.span_classifier.config.PrunerTrainConfig`.",
        "> Do not edit by hand — run `uv run generate-pruner-docs` to regenerate.",
        "",
        "## Schema versioning",
        "",
        "| Key | Value |",
        "|-----|-------|",
        f"| Current version | `{PRUNER_SCHEMA_VERSION}` |",
        f"| Supported versions | {', '.join(f'`{v}`' for v in sorted(PRUNER_SUPPORTED_VERSIONS))} |",
        "",
        'Add `schema_version: "1.0"` to your YAML config. '
        "An unsupported version raises a clear error at load time.",
        "",
    ]

    lines += [
        "## Shared parameters",
        "",
        "These fields live at the top level of the config and are used both at "
        "inference time (by the pipeline) and at training time.",
        "",
        _render_table(top_fields),
        "",
    ]

    lines += [
        "## Training parameters (`train_params`)",
        "",
        "These fields live under `train_params:` in the YAML and are ignored "
        "at inference time.  On the CLI they are prefixed with "
        f"`--train_params{SEP}` (e.g. `--train_params{SEP}learning_rate`).",
        "",
    ]

    rendered_names: set[str] = set()
    for group_title, names in _TRAIN_GROUPS:
        group_fields = [train_by_name[n] for n in names if n in train_by_name]
        if not group_fields:
            continue
        lines += [
            f"### {group_title}",
            "",
            _render_table(group_fields),
            "",
        ]
        rendered_names.update(f.path[-1] for f in group_fields)

    ungrouped = [f for f in train_fields if f.path[-1] not in rendered_names]
    if ungrouped:
        lines += [
            "### Other",
            "",
            _render_table(ungrouped),
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HGERE document renderer
# ---------------------------------------------------------------------------


def render_hgere_document() -> str:
    """Return the full Markdown documentation string for HGERE."""
    top_fields, train_fields = _split_fields(HGERETrainConfig)

    _TRAIN_GROUPS: list[tuple[str, list[str]]] = [
        (
            "Data",
            ["train_file", "dev_file", "test_file"],
        ),
        (
            "Optimisation",
            [
                "seed",
                "learning_rate",
                "learning_rate_cls",
                "num_train_epochs",
                "per_gpu_train_batch_size",
                "gradient_accumulation_steps",
                "adam_epsilon",
                "weight_decay",
                "max_grad_norm",
                "max_steps",
                "warmup_steps",
                "warmup_ratio",
                "logging_steps",
                "save_steps",
                "eval_epochs",
                "save_total_limit",
            ],
        ),
        (
            "Hardware",
            ["no_cuda", "fp16", "local_rank", "server_ip", "server_port"],
        ),
        (
            "Loss",
            [
                "loss_re_weight_alpha",
                "train_time_loss_weighting",
                "train_time_loss_turn",
                "train_time_loss_steepness",
            ],
        ),
        (
            "Evaluation & checkpointing",
            [
                "evaluate_during_training",
                "eval_all_checkpoints",
                "overwrite_output_dir",
                "overwrite_cache",
            ],
        ),
        (
            "Run modes",
            [
                "do_train",
                "eval_train",
                "eval_dev",
                "eval_test",
                "no_test",
                "save_results",
            ],
        ),
        (
            "Data loading",
            ["shuffle", "pre_filter_params", "batch_by_size", "preload_dataset"],
        ),
        ("Weights & Biases", ["project_name", "run_name", "log_wandb"]),
    ]

    train_by_name: dict[str, FlatField] = {f.path[-1]: f for f in train_fields}

    lines: list[str] = []

    lines += [
        "# HGERE parameter reference",
        "",
        "> **Auto-generated** from `hgere.hgere.config.HGERETrainConfig`.",
        "> Do not edit by hand — run `uv run generate-pruner-docs --model hgere` to regenerate.",
        "",
        "## Schema versioning",
        "",
        "| Key | Value |",
        "|-----|-------|",
        f"| Current version | `{HGERE_SCHEMA_VERSION}` |",
        f"| Supported versions | {', '.join(f'`{v}`' for v in sorted(HGERE_SUPPORTED_VERSIONS))} |",
        "",
        'Add `schema_version: "1.0"` to your YAML config. '
        "An unsupported version raises a clear error at load time.",
        "",
    ]

    lines += [
        "## Shared parameters",
        "",
        "These fields live at the top level of the config and are used both at "
        "inference time (by the pipeline) and at training time.",
        "",
        _render_table(top_fields),
        "",
    ]

    lines += [
        "## Training parameters (`train_params`)",
        "",
        "These fields live under `train_params:` in the YAML and are ignored "
        "at inference time.  On the CLI they are prefixed with "
        f"`--train_params{SEP}` (e.g. `--train_params{SEP}learning_rate`).",
        "",
    ]

    rendered_names: set[str] = set()
    for group_title, names in _TRAIN_GROUPS:
        group_fields = [train_by_name[n] for n in names if n in train_by_name]
        if not group_fields:
            continue
        lines += [
            f"### {group_title}",
            "",
            _render_table(group_fields),
            "",
        ]
        rendered_names.update(f.path[-1] for f in group_fields)

    ungrouped = [f for f in train_fields if f.path[-1] not in rendered_names]
    if ungrouped:
        lines += [
            "### Other",
            "",
            _render_table(ungrouped),
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown parameter docs for pruner and HGERE from Pydantic models."
    )
    parser.add_argument(
        "--model",
        choices=["pruner", "hgere", "all"],
        default="all",
        help="Which model to generate docs for (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output Markdown file. When --model=all this is ignored; "
            f"defaults are {_DEFAULT_PRUNER_OUTPUT} and {_DEFAULT_HGERE_OUTPUT}."
        ),
    )
    args = parser.parse_args()

    if args.model in ("pruner", "all"):
        out = (
            Path(args.output)
            if args.output and args.model != "all"
            else _DEFAULT_PRUNER_OUTPUT
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_pruner_document(), encoding="utf-8")
        print(f"Written to {out}", file=sys.stderr)

    if args.model in ("hgere", "all"):
        out = (
            Path(args.output)
            if args.output and args.model != "all"
            else _DEFAULT_HGERE_OUTPUT
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_hgere_document(), encoding="utf-8")
        print(f"Written to {out}", file=sys.stderr)


if __name__ == "__main__":
    cli()
