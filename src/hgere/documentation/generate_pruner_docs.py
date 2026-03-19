"""Generate parameter documentation for the span pruner.

Reads PrunerTrainConfig (the single source of truth) and renders a Markdown
file with parameter tables — one for shared inference/training fields and one
for training-only fields.

Run this script whenever the Pydantic models change to keep documentation in
sync.

Usage
-----
    uv run generate-pruner-docs
    uv run generate-pruner-docs --output docs/custom.md
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

from hgere.config.cli_gen import SEP, FlatField, collect_flat_fields
from hgere.span_classifier.config import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    PrunerTrainConfig,
)

_DEFAULT_OUTPUT = Path("documentation/api/pruner.md")

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
    """Return (top_level_fields, train_params_fields).

    Top-level fields are those whose path has length 1 (direct fields of
    PrunerTrainConfig, excluding schema_version which is internal).
    Train-params fields are those whose first path component is 'train_params'.
    """
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
# Document renderer
# ---------------------------------------------------------------------------


def render_document() -> str:
    """Return the full Markdown documentation string."""
    top_fields, train_fields = _split_fields(PrunerTrainConfig)

    # Collect section groups inside train_params by inspecting the model's
    # field ordering and the comment-delimited groups we know exist.
    # We expose the groups via a small explicit mapping so the doc stays
    # readable without polluting the Pydantic model with extra metadata.
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
                "num_train_epochs",
                "eval_epochs",
                "per_gpu_train_batch_size",
                "gradient_accumulation_steps",
                "adam_epsilon",
                "weight_decay",
                "save_steps",
                "fp16",
                "local_rank",
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
            ],
        ),
        ("Weights & Biases", ["project_name", "run_name"]),
    ]

    # Index train fields by leaf name for quick lookup.
    train_by_name: dict[str, FlatField] = {f.path[-1]: f for f in train_fields}

    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
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
        f"| Current version | `{CURRENT_SCHEMA_VERSION}` |",
        f"| Supported versions | {', '.join(f'`{v}`' for v in sorted(SUPPORTED_SCHEMA_VERSIONS))} |",
        "",
        'Add `schema_version: "1.0"` to your YAML config. '
        "An unsupported version raises a clear error at load time.",
        "",
    ]

    # ── Shared inference / training fields ───────────────────────────────────
    lines += [
        "## Shared parameters",
        "",
        "These fields live at the top level of the config and are used both at "
        "inference time (by the pipeline) and at training time.",
        "",
        _render_table(top_fields),
        "",
    ]

    # ── Training-only parameters ─────────────────────────────────────────────
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

    # Catch any fields not covered by the explicit groups (future-proofing).
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
        description="Generate Markdown parameter docs for the pruner from Pydantic models."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=f"Output Markdown file (default: {_DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = render_document()
    output_path.write_text(doc, encoding="utf-8")
    print(f"Written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    cli()
