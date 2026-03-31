"""Pydantic configuration models for the HGERE model.

Single source of truth for all HGERE parameters (inference and training).
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
from typing import TYPE_CHECKING, Optional


from ..config import load_yaml_strict
from pydantic import BaseModel, Field, model_validator

from ..pre_filter.config import PreFilterParams

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


# ---------------------------------------------------------------------------
# Training parameters (nested under hgere.train_params in YAML)
# ---------------------------------------------------------------------------


class HGERETrainParams(BaseModel):
    """Parameters used only during training — not needed at inference time."""

    # ── Data ────────────────────────────────────────────────────────────────
    train_file: str = Field(
        default="train.json",
        description="Training split filename inside ner_prediction_dir.",
    )
    dev_file: str = Field(
        default="dev.json", description="Dev split filename inside ner_prediction_dir."
    )
    test_file: str = Field(
        default="test.json",
        description="Test split filename inside ner_prediction_dir.",
    )

    # ── Optimisation ────────────────────────────────────────────────────────
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    learning_rate: float = Field(description="Learning rate for BERT layers.")
    learning_rate_cls: float = Field(
        default=-1,
        description="Learning rate for layers beyond BERT. -1 = use learning_rate.",
    )
    num_train_epochs: float = Field(
        description="Total number of training epochs to perform."
    )
    per_gpu_train_batch_size: int = Field(description="Training batch size per GPU.")
    gradient_accumulation_steps: int = Field(
        default=1, description="Gradient accumulation steps before a weight update."
    )
    adam_epsilon: float = Field(
        default=1e-8, description="Epsilon for the Adam optimiser."
    )
    weight_decay: float = Field(default=0.0, description="Weight decay coefficient.")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm.")
    max_steps: int = Field(
        default=-1,
        description="If > 0: set total number of training steps. Overrides num_train_epochs.",
    )
    warmup_steps: int = Field(
        default=-1, description="Linear warmup over warmup_steps."
    )
    warmup_ratio: float = Field(
        default=0.1, description="Linear warmup ratio (used if warmup_steps=-1)."
    )
    logging_steps: int = Field(default=5, description="Log every N update steps.")
    save_steps: int = Field(
        default=1000, description="Save a checkpoint every N update steps."
    )
    eval_epochs: int = Field(
        default=-1,
        description="Evaluate every N epochs. Set to -1 to use save_steps instead.",
    )
    save_total_limit: int = Field(
        default=1, description="Limit total checkpoints; deletes older ones."
    )

    # ── Evaluation / checkpointing ─────────────────────────────────────────
    evaluate_during_training: bool = Field(
        default=False, description="Run evaluation on dev set during training."
    )
    eval_all_checkpoints: bool = Field(
        default=False,
        description="Evaluate all saved checkpoints at the end of training.",
    )
    overwrite_output_dir: bool = Field(
        default=False, description="Allow overwriting an existing output directory."
    )
    overwrite_cache: bool = Field(
        default=False, description="Overwrite the cached training and evaluation sets."
    )

    # ── Hardware ──────────────────────────────────────────────────────────────
    no_cuda: bool = Field(default=False, description="Avoid using CUDA when available.")
    fp16: bool = Field(
        default=False, description="Use mixed-precision (fp16) training."
    )
    local_rank: int = Field(
        default=-1, description="Local rank for distributed training (-1 = single GPU)."
    )
    server_ip: str = Field(default="", description="For distant debugging.")
    server_port: str = Field(default="", description="For distant debugging.")

    # ── Loss ────────────────────────────────────────────────────────────────
    loss_re_weight_alpha: float = Field(
        default=0.5,
        description=(
            "Weight the RE loss relative to NER loss. "
            "E.g. 0.7 => 0.7 RE loss + 0.3 NER loss."
        ),
    )
    train_time_loss_weighting: bool = Field(
        default=False,
        description=(
            "Enable dynamic NER→RE loss weighting over training. Alpha shifts via sigmoid."
        ),
    )
    train_time_loss_turn: float = Field(
        default=0.5,
        description=(
            "Fractional training progress [0, 1] at which the NER→RE weighting is at midpoint."
        ),
    )
    train_time_loss_steepness: float = Field(
        default=10.0,
        description="Steepness of the sigmoid phase transition for dynamic loss weighting.",
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    project_name: str = Field(
        default="hgere", description="Weights & Biases project name."
    )
    run_name: Optional[str] = Field(
        default=None, description="Weights & Biases run name."
    )
    log_wandb: bool = Field(
        default=False, description="Whether to log training in W&B."
    )

    # ── Data loading ─────────────────────────────────────────────────────────
    shuffle: bool = Field(default=False, description="Shuffle training data.")
    n_workers: int = Field(
        default=32,
        description="Number of DataLoader worker processes for data loading.",
    )
    pre_filter_params: Optional[PreFilterParams] = Field(
        default=None, description="Parameters for pre-filtering NER candidates."
    )
    batch_by_size: bool = Field(
        default=False,
        description=(
            "Sort sentences by entity count before batching. "
            "Mutually exclusive with shuffle."
        ),
    )
    preload_dataset: bool = Field(
        default=False, description="Preload dataset into memory."
    )

    # ── Inference modes ───────────────────────────────────────────────────────
    no_test: bool = Field(default=False, description="Skip test set evaluation.")
    save_results: bool = Field(
        default=True, description="Persist predictions to disk after evaluation."
    )
    do_train: bool = Field(default=True, description="Whether to run training.")
    eval_train: bool = Field(
        default=False, description="Run eval on the train set and save predictions."
    )
    eval_dev: bool = Field(
        default=True, description="Run eval on the dev set and save predictions."
    )
    eval_test: bool = Field(
        default=True, description="Run eval on the test set and save predictions."
    )


# ---------------------------------------------------------------------------
# Top-level training config
# ---------------------------------------------------------------------------


class HGERETrainConfig(BaseModel):
    """Full HGERE training configuration.

    The shared inference fields (model_dir, base_model_name_or_path, …) live
    at the top level of this model and mirror the inference config used in
    the pipeline.  Training-only parameters live under ``train_params``.

    YAML layout (standalone HGERE training file)
    ---------------------------------------------
    .. code-block:: yaml

        schema_version: "1.0"
        label_set: gsap
        model_dir: saves/hgere/my_run
        base_model_name_or_path: pretrained_models/scibert_scivocab_uncased
        ner_prediction_dir: pipeline/data/output/pruner
        ...
        train_params:
          learning_rate: 1.0e-5
          ...

    When loading from a *full* pipeline training YAML (which has an ``hgere:``
    section), use :py:meth:`from_yaml` — it hoists ``label_set`` and
    ``schema_version`` from the top level automatically.
    """

    # ── Versioning ────────────────────────────────────────────────────────────
    schema_version: str = Field(
        default=CURRENT_SCHEMA_VERSION,
        description=f"Config schema version. Current supported: {CURRENT_SCHEMA_VERSION}.",
        exclude=True,
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
    ner_prediction_dir: str = Field(
        description="Input data directory containing pruner output files for HGERE."
    )
    model_type: str = Field(
        default="hyper",
        description="HGERE architecture key (hyper | modernberthyper).",
    )
    do_lower_case: bool = Field(
        default=False, description="Lowercase input tokens (true for uncased models)."
    )
    per_gpu_eval_batch_size: int = Field(
        default=8,
        description="Evaluation batch size per GPU (used during training eval).",
    )
    max_seq_length: int = Field(
        default=384, description="Maximum tokenised sequence length."
    )
    max_pair_length: int = Field(
        default=64, description="Maximum number of span pairs per sequence."
    )
    alpha: float = Field(default=1.0, description="Loss scale alpha.")
    no_sym: bool = Field(
        default=False, description="Disable symmetric relation labels."
    )

    # ── Entity encoder ────────────────────────────────────────────────────────
    ent_repr: str = Field(
        default="mix",
        description="Entity representation source: sub | obj | mix.",
    )
    ent_enc: str = Field(default="cat", description="Entity encoder type.")
    ner_cls: str = Field(default="cat", description="NER classifier type.")
    rel_enc: str = Field(default="cat", description="Relation encoder type.")
    ent_dim: int = Field(default=200, description="Entity dimension.")
    rel_dim: int = Field(default=200, description="Relation dimension.")
    rel_rank: int = Field(default=200, description="Rank for biaffine factorisation.")
    rel_factorize: bool = Field(
        default=False, description="Use biaffine relation factorisation."
    )
    factor_type: str = Field(default="ternary", description="Factor type for HyperGNN.")
    mem_dim: int = Field(default=200, description="Memory dimension for HyperGNN.")

    # ── HyperGNN ──────────────────────────────────────────────────────────────
    n_iter: int = Field(
        default=3,
        description="Number of HyperGNN iterations. Maps to --iter in run_hgnn.py.",
    )
    factor_encoder: str = Field(default="cat", description="Factor encoder type.")
    iter1: int = Field(default=1, description="Number of first-order iterations.")
    lminit: bool = Field(
        default=False, description="Initialise span boundary embeddings from LM output."
    )
    nocross: bool = Field(
        default=False, description="Disable cross-sentence span candidates."
    )

    # ── Attention ─────────────────────────────────────────────────────────────
    att_left: bool = Field(default=False, description="Use left attention.")
    att_right: bool = Field(default=False, description="Use right attention.")
    use_ner_results: bool = Field(
        default=False, description="Use NER results during relation extraction."
    )
    use_typemarker: bool = Field(
        default=False, description="Use type markers in the input."
    )
    layernorm: bool = Field(default=False, description="Apply layer normalisation.")
    layernorm_1st: bool = Field(
        default=False, description="Apply layer normalisation for first-order."
    )
    attn_self: bool = Field(default=False, description="Use self-attention.")
    aggregate_type: str = Field(
        default="attn", description="Aggregation type: attn or test."
    )
    aggregate_func: str = Field(
        default="max", description="Aggregation function: max or sum."
    )
    agg_with_self: bool = Field(default=False, description="Aggregate with self node.")
    fix_obj: bool = Field(default=False, description="Fix object representation.")
    edgetype: str = Field(default="sib", description="Edge type for HTNN.")

    # ── Attention scorer ──────────────────────────────────────────────────────
    attn_scorer: str = Field(default="biaf", description="Attention scorer type.")
    attn_res: bool = Field(
        default=False, description="Use residual in attention scorer."
    )
    n_head: int = Field(default=8, description="Number of attention heads.")
    d_head: int = Field(default=32, description="Dimension per attention head.")

    # ── Focal loss ────────────────────────────────────────────────────────────
    re_focal_loss: bool = Field(
        default=False,
        description="Use focal loss for relation classification.",
    )
    re_focal_gamma: float = Field(
        default=2.0, description="Focusing parameter γ for RE focal loss."
    )
    ner_focal_loss: bool = Field(
        default=False,
        description="Use focal loss for NER classification.",
    )
    ner_focal_gamma: float = Field(
        default=2.0, description="Focusing parameter γ for NER focal loss."
    )

    # ── Evaluation flags ──────────────────────────────────────────────────────
    uni_ent: bool = Field(
        default=False,
        description="Use uniform entity representation (same repr for sub/obj).",
    )
    pred_sub: bool = Field(default=False, description="Predict subject.")
    eval_logits: bool = Field(
        default=False, description="Decode with non-normalised logits."
    )
    eval_logsoftmax: bool = Field(default=False, description="Decode with log-softmax.")
    eval_softmax: bool = Field(default=False, description="Decode with softmax.")
    eval_unidirect: bool = Field(
        default=False, description="Evaluate with unidirectional relations."
    )
    baseline: str = Field(default="firstorder", description="Baseline method.")

    # ── Training parameters ────────────────────────────────────────────────────
    train_params: HGERETrainParams

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

    # ── Factories ─────────────────────────────────────────────────────────────

    def to_relation_dataset_params(
        self,
        doc_limit: Optional[int] = None,
        preload: bool = False,
    ) -> "RelationDatasetParams":  # noqa: F821
        """Build a :class:`~gsapere.data.config.RelationDatasetParams` from this config.

        Parameters
        ----------
        doc_limit:
            Override the number of documents to load (default: all).
        preload:
            Preload the entire dataset into memory.
        """
        from ..data.config import RelationDatasetParams

        return RelationDatasetParams(
            max_seq_length=self.max_seq_length,
            max_pair_length=self.max_pair_length,
            model_type=self.model_type,
            use_typemarker=self.use_typemarker,
            no_sym=self.no_sym,
            nocross=self.nocross,
            local_rank=self.train_params.local_rank,
            doc_limit=doc_limit,
            preload=preload,
        )

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "HGERETrainConfig":
        """Load from a YAML file.

        Accepts both formats:
        - A standalone HGERE training YAML (flat, all fields at top level).
        - A full pipeline training YAML with an ``hgere:`` section; in that
          case ``label_set`` and ``schema_version`` are hoisted from the top
          level into the hgere data automatically.
        """
        data: dict = load_yaml_strict(path)

        if "hgere" in data:
            hgere_data: dict = dict(data["hgere"])
            hgere_data.setdefault("label_set", data.get("label_set", ""))
            hgere_data.setdefault(
                "schema_version", data.get("schema_version", CURRENT_SCHEMA_VERSION)
            )
            return cls.model_validate(hgere_data)

        return cls.model_validate(data)
