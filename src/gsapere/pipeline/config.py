"""Pipeline configuration models.

Loaded from a YAML file via PipelineConfig.from_yaml().
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal, Optional, Union  # noqa: F401

import transformers

from ..config import load_yaml_strict
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..pre_filter.config import PreFilterParams


@contextmanager
def suppress_transformers_warnings() -> Generator[None, None, None]:
    """Suppress HuggingFace weight-mismatch warnings unless DEBUG logging is active."""
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        yield
    else:
        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            yield
        finally:
            transformers.logging.set_verbosity(prev)


class FinalPruningConfig(BaseModel):
    """Post-inference candidate selection strategy.

    method="threshold": keep all spans with prob >= threshold.
    method="topk":      keep top-K spans per sentence, where
                        K = clamp(topk_ratio * sent_len, min, max).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["topk", "threshold"] = "threshold"
    topk_ratio: float = 0.5
    min_mentions_by_sentence: int = 3
    max_mentions_by_sentence: int = 18
    threshold: float | None = 0.0005

    @model_validator(mode="after")
    def _check_method_params(self) -> "FinalPruningConfig":
        if self.method == "threshold" and self.threshold is None:
            raise ValueError("threshold must be set when method='threshold'")
        return self


class PrunerConfig(BaseModel):
    """Configuration for the pruner stage."""

    model_config = ConfigDict(extra="forbid")

    model_dir: str
    base_model_name_or_path: str
    do_lower_case: bool = True
    model_type: str = "bertspanmarkerpruner"
    per_gpu_eval_batch_size: int = 32
    max_seq_length: int = 256
    max_pair_length: int = 64
    max_mention_ori_length: int = 12
    # Topk pre-filter params passed to run_pruner_inference (applied before final_pruning).
    # Override defaults by setting these directly in the config.
    # If prune_config is also set, it takes precedence.
    topk_ratio: float = 0.5
    min_mentions_num: int = 3
    max_mentions_num: int = 18
    # Path to a best_config.json (from gsapere-train-prefilter) for pre-filtering
    # topk params used inside run_pruner_inference.  Overrides topk_ratio/min/max above.
    prune_config: str | None = None
    # Path to a RuleBasedPruner pattern file (.json) produced by gsapere-fit-rulebased-pruner.
    # When set, spans matching rulebased patterns are removed before the neural pruner runs.
    rulebased_pruner_file: str | None = None
    final_pruning: FinalPruningConfig = FinalPruningConfig()


class HGEREConfig(BaseModel):
    """Configuration for the HGERE stage."""

    model_config = ConfigDict(extra="forbid")

    model_dir: str
    base_model_name_or_path: str
    model_type: str = "hyper"
    per_gpu_eval_batch_size: int = 32
    max_seq_length: int = 512
    factor_type: str = "tersibcop"
    factor_encoder: str = "biaf"
    ent_dim: int = 400
    rel_dim: int = 400
    mem_dim: int = 400
    # Entity representation: "mix" (sub+obj), "sub", or "obj"
    ent_repr: str = "mix"
    # GNN iterations
    n_iter: int = 3
    layernorm: bool = False
    layernorm_1st: bool = False
    attn_self: bool = False
    attn_res: bool = False
    unirel: bool = False
    do_lower_case: bool = False
    use_typemarker: bool = False
    no_sym: bool = True
    nocross: bool = False
    shuffle: bool = False
    local_rank: int = -1
    pre_filter_params: Optional[PreFilterParams] = None
    use_gold_ner: bool = False
    n_workers: int = 4
    use_dataset_id_token_as_cls: bool = False
    seeds: Union[int, list[int]] = Field(
        default=42,
        description=(
            "Seed(s) for HGERE model selection. Pass a plain integer to run a single "
            "un-suffixed model. Pass a list to iterate over seed-specific model "
            "directories (e.g. model_dir + '_seed42') and write outputs into "
            "a '/{seed}/' subfolder."
        ),
    )


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    Exactly one of ``label_set`` or ``label_sets`` must be provided:

    - ``label_set: scier`` — single-dataset mode (backward compatible).
    - ``label_sets: [scier, scinlp]`` — multi-head mode: HGERE runs once per
      label set and writes separate output files.

    After construction, ``label_sets`` is always a non-empty list regardless of
    which form was used.
    """

    model_config = ConfigDict(extra="forbid")

    label_set: Optional[str] = Field(
        default=None,
        description="Single dataset label set (backward-compatible shorthand).",
    )
    label_sets: Optional[list[str]] = Field(
        default=None,
        description=(
            "One or more dataset label sets for multi-head inference. "
            "HGERE runs once per entry and writes separate output files."
        ),
    )
    pruner: PrunerConfig
    hgere: HGEREConfig

    @model_validator(mode="after")
    def _normalize_label_sets(self) -> "PipelineConfig":
        has_single = self.label_set is not None
        has_multi = self.label_sets is not None
        if has_single and has_multi:
            raise ValueError("Set either 'label_set' or 'label_sets', not both.")
        if not has_single and not has_multi:
            raise ValueError("One of 'label_set' or 'label_sets' is required.")
        if not has_multi:
            # Normalize single → list so callers can always use label_sets.
            self.label_sets = [self.label_set]
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file."""
        data = load_yaml_strict(path)
        return cls.model_validate(data)
