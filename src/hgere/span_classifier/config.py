"""Pydantic configuration models for the span pruner.

Single source of truth for all pruner parameters (inference and training).
Used to:
  - Validate configs loaded from YAML
  - Generate CLI argument parsers dynamically (via hgere.config.cli_gen)
  - Bundle parameter knowledge in the installed package

Schema versioning
-----------------
Bump CURRENT_SCHEMA_VERSION when making breaking changes to field names or
semantics. Add the previous version to SUPPORTED_SCHEMA_VERSIONS together
with a note. Old configs that carry an unsupported version receive a clear
error pointing at the current version.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Annotated

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION: str = "1.0"

#: All versions this code can load.  When you drop support for an old version,
#: remove it from here so users get an actionable error instead of silent
#: mis-parsing.
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})

# --------------------------------------------------------------------------
# Pre-filtering parameters (nested under pruner.train_params in yaml and for evaluation)
# --------------------------------------------------------------------------


class ThresholdPreFilterParams(BaseModel):
    method: Literal["threshold"]
    value: float = Field(ge=0.0, le=1.0)


class TopKPreFilterParams(BaseModel):
    method: Literal["topk"]
    ratio: float = Field(ge=0.0, le=1.0)
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def check_min_max(self):
        if self.max < self.min:
            raise ValueError("max must be >= min")
        return self


PreFilterParams = Annotated[
    Union[ThresholdPreFilterParams, TopKPreFilterParams],
    Field(discriminator="method"),
]

# ---------------------------------------------------------------------------
# Training parameters (nested under pruner.train_params in YAML)
# ---------------------------------------------------------------------------


class PrunerTrainParams(BaseModel):
    """Parameters used only during training — not needed at inference time."""

    # ── Data ────────────────────────────────────────────────────────────────
    data_dir: str = Field(
        description="Directory containing train/dev/test split files."
    )
    train_file: str = Field(
        default="train.jsonl", description="Training split filename inside data_dir."
    )
    dev_file: str = Field(
        default="dev.jsonl", description="Dev split filename inside data_dir."
    )
    test_file: str = Field(
        default="test.jsonl", description="Test split filename inside data_dir."
    )
    rulebased_pruner_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a rule-based pruner JSON (from eval-rulebased-pruner). "
            "Spans matching its patterns are filtered before the neural pruner sees them."
        ),
    )

    # ── Optimisation ────────────────────────────────────────────────────────
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    learning_rate: float = Field(description="Learning rate for BERT layers.")
    num_train_epochs: int = Field(description="Total number of training epochs.")
    eval_epochs: int = Field(
        default=1,
        description="Evaluate every N epochs. Set to -1 to use save_steps instead.",
    )
    per_gpu_train_batch_size: int = Field(description="Training batch size per GPU.")
    gradient_accumulation_steps: int = Field(
        default=1, description="Gradient accumulation steps before a weight update."
    )
    adam_epsilon: float = Field(
        default=1e-8, description="Epsilon for the Adam optimiser."
    )
    weight_decay: float = Field(default=0.0, description="Weight decay coefficient.")
    save_steps: int = Field(
        default=1000, description="Save a checkpoint every N update steps."
    )
    fp16: bool = Field(
        default=False, description="Use mixed-precision (fp16) training."
    )
    local_rank: int = Field(
        default=-1, description="Local rank for distributed training (-1 = single GPU)."
    )

    # ── Loss ────────────────────────────────────────────────────────────────
    pruner_loss: Literal["bce", "focal"] = Field(
        default="bce",
        description="Loss function: 'bce' = BCEWithLogitsLoss, 'focal' = focal loss.",
    )
    focal_gamma: float = Field(
        default=2.0, description="Focusing parameter γ for focal loss."
    )
    focal_alpha: float = Field(
        default=0.25, description="Class-balance factor α for focal loss."
    )

    # ── Candidate filtering used during eval inside training ─────────────────
    topk_ratio: float = Field(
        default=0.5,
        description="K = clamp(topk_ratio × sent_len, min_mentions_num, max_mentions_num).",
    )
    min_mentions_num: int = Field(
        default=3, description="Minimum spans to keep per sentence."
    )
    max_mentions_num: int = Field(
        default=18, description="Maximum spans to keep per sentence."
    )

    # ── Model flags ──────────────────────────────────────────────────────────
    onedropout: bool = Field(
        default=False,
        description="Share a single dropout mask across the span encoder.",
    )
    lminit: bool = Field(
        default=False, description="Initialise span boundary embeddings from LM output."
    )
    nocross: bool = Field(
        default=False, description="Disable cross-sentence span candidates."
    )
    biaf_span: bool = Field(
        default=False, description="Use biaffine span representation."
    )
    biaf_mode: int = Field(default=3, description="Biaffine span repr mode.")
    biaf_factorize: bool = Field(
        default=False, description="Factorize the biaffine span matrix."
    )
    span_hidden_size: int = Field(
        default=768, description="Hidden size for the span encoder."
    )
    rank: int = Field(default=768, description="Rank for biaffine factorization.")
    span_size: int = Field(
        default=256, description="Output size of the span representation."
    )

    # ── Evaluation / checkpointing ────────────────────────────────────────
    evaluate_during_training: bool = Field(
        default=True, description="Run evaluation on dev set after every checkpoint."
    )
    eval_all_checkpoints: bool = Field(
        default=False,
        description="Evaluate all saved checkpoints at the end of training.",
    )
    overwrite_model_dir: bool = Field(
        default=False, description="Allow overwriting an existing model directory."
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    project_name: str = Field(
        default="hgere-pruner", description="Weights & Biases project name."
    )
    run_name: Optional[str] = Field(
        default=None, description="Weights & Biases run name."
    )


# ---------------------------------------------------------------------------
# Top-level training config
# ---------------------------------------------------------------------------


class PrunerTrainConfig(BaseModel):
    """Full pruner training configuration.

    The shared inference fields (model_dir, base_model_name_or_path, …) live
    at the top level of this model and mirror the inference config used in
    the pipeline.  Training-only parameters live under ``train_params``.

    YAML layout (standalone pruner training file)
    ---------------------------------------------
    .. code-block:: yaml

        schema_version: "1.0"
        label_set: gsap
        model_dir: saves/pruner/my_run
        base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
        ...
        train_params:
          learning_rate: 1.0e-6
          ...

    When loading from a *full* pipeline training YAML (which has a ``pruner:``
    section), use :py:meth:`from_training_yaml` — it hoists ``label_set`` and
    ``schema_version`` from the top level automatically.
    """

    # ── Versioning ────────────────────────────────────────────────────────────
    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description=f"Config schema version. Current supported: {CURRENT_SCHEMA_VERSION}.",
    )

    # ── Identity ──────────────────────────────────────────────────────────────
    label_set: str = Field(
        description="Label set for entity/relation types (e.g. gsap, scier, scinlp)."
    )

    # ── Shared inference fields ────────────────────────────────────────────────
    model_dir: str = Field(description="Directory where checkpoints will be written.")
    base_model_name_or_path: str = Field(
        description="Transformer model path or HuggingFace name."
    )
    model_type: str = Field(
        default="bertspanmarkerpruner",
        description="Pruner architecture key (bertspanmarkerpruner | modernbertspanmarkerpruner).",
    )
    do_lower_case: bool = Field(
        default=True, description="Lowercase input tokens (true for uncased models)."
    )
    per_gpu_eval_batch_size: int = Field(
        default=32,
        description="Evaluation batch size per GPU (used during training eval).",
    )
    max_seq_length: int = Field(
        default=256, description="Maximum tokenised sequence length."
    )
    max_pair_length: int = Field(
        default=64, description="Maximum number of span pairs per sequence."
    )
    max_mention_ori_length: int = Field(
        default=12, description="Maximum span width in tokens."
    )

    # ── Training parameters ────────────────────────────────────────────────────
    train_params: PrunerTrainParams

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def _check_schema_version(cls, data: dict) -> dict:
        version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version '{version}'. "
                f"Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}. "
                f"Please migrate your config to version {CURRENT_SCHEMA_VERSION}."
            )
        return data

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrunerTrainConfig":
        """Load from a YAML file.

        Accepts both formats:
        - A standalone pruner training YAML (flat, all fields at top level).
        - A full pipeline training YAML with a ``pruner:`` section; in that
          case ``label_set`` and ``schema_version`` are hoisted from the top
          level into the pruner data automatically.
        """
        with open(Path(path)) as f:
            data: dict = yaml.safe_load(f)

        if "pruner" in data:
            pruner_data: dict = dict(data["pruner"])
            pruner_data.setdefault("label_set", data.get("label_set", ""))
            pruner_data.setdefault(
                "schema_version", data.get("schema_version", CURRENT_SCHEMA_VERSION)
            )
            return cls.model_validate(pruner_data)

        return cls.model_validate(data)
