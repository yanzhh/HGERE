"""Pydantic configuration models for HGERE data loading.

Single source of truth for all RelationDataset construction parameters.
Replaces the loose ``args`` namespace that was previously threaded through
``RelationDataset.__init__``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RelationDatasetParams(BaseModel):
    """Parameters for constructing a :class:`~gsapere.data.relation_dataset.RelationDataset`.

    All fields have sensible defaults so callers only need to set what
    differs from those defaults.
    """

    model_config = ConfigDict(extra="forbid")

    # ── Sequence lengths ────────────────────────────────────────────────────
    max_seq_length: int = Field(
        default=384, description="Maximum tokenised sequence length."
    )
    max_ents: int = Field(
        default=18, description="Maximum entity candidates per sentence."
    )

    # ── Model behaviour ─────────────────────────────────────────────────────
    model_type: str = Field(
        default="hyper",
        description=(
            "Model architecture key. Affects token-id insertion "
            "(e.g. 'albert' uses different special-token ids)."
        ),
    )
    use_typemarker: bool = Field(
        default=False,
        description="Insert entity-type markers around subjects in the input sequence.",
    )
    no_sym: bool = Field(
        default=False, description="Disable symmetric relation label handling."
    )
    nocross: bool = Field(
        default=False, description="Disable cross-sentence context window."
    )

    # ── Distributed / runtime ───────────────────────────────────────────────
    local_rank: int = Field(
        default=-1,
        description=(
            "Local rank for distributed training (-1 = single GPU or non-distributed)."
        ),
    )

    # ── Dataset split ───────────────────────────────────────────────────────
    split: Literal["train", "dev", "test", "inference"] = Field(
        default="train",
        description=(
            "Dataset split role.  Training-specific warnings (dropped NER labels, "
            "self-relations, relations absent from candidate set) are only emitted "
            "for 'train'.  Use 'inference' for pipeline runs where no gold labels "
            "are required."
        ),
    )

    # ── Data loading options ────────────────────────────────────────────────
    doc_limit: Optional[int] = Field(
        default=None,
        description="Limit number of documents loaded from the file. None = load all.",
    )
    preload: bool = Field(
        default=False,
        description="Preload all items into memory at construction time.",
    )
    pre_filter_params: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Pre-filtering parameters for NER candidates (PreFilterParams dict: "
            "{'method': 'topk'|'threshold', ...}). When set, requires "
            "'predicted_ner_proba' in each doc."
        ),
    )
    use_gold_ner: bool = Field(
        default=False,
        description=(
            "Force gold NER annotations as entity candidates, ignoring predicted_ner "
            "and predicted_ner_proba. Gives an oracle upper bound for the HGERE stage."
        ),
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description=(
            "Dataset identifier for multi-head training (e.g. 'scier', 'scinlp', 'gsap'). "
            "When set, batches include this string so the model selects the correct heads. "
            "Leave None for single-dataset training."
        ),
    )
    dataset_cls_token_id: Optional[int] = Field(
        default=None,
        description=(
            "Pre-resolved token ID of the [unusedX] slot to substitute at position 0 "
            "instead of [CLS]. Set by the training/inference orchestration code from "
            "the model config's dataset_cls_token_ids mapping. When None, no substitution "
            "is performed. Requires dataset_id to be set (for head routing)."
        ),
    )
