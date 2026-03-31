"""Pipeline configuration models.

Loaded from a YAML file via PipelineConfig.from_yaml().
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal, Optional

import transformers

from ..config import load_yaml_strict
from pydantic import BaseModel, model_validator

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

    model_dir: str
    base_model_name_or_path: str
    model_type: str = "hyper"
    per_gpu_eval_batch_size: int = 32
    max_seq_length: int = 512
    max_pair_length: int = 18
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
    local_rank: int = -1
    # NER decoding: if False (default), standard argmax — NIL predictions are
    # dropped and only high-confidence non-NIL entities appear in output.
    # Set to True only for gold-span / cross-dataset evaluation where every
    # candidate must receive a non-NIL label.
    force_non_nil: bool = False
    pre_filter_params: Optional[PreFilterParams] = None
    n_workers: int = 4


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    label_set: str
    pruner: PrunerConfig
    hgere: HGEREConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a PipelineConfig from a YAML file."""
        data = load_yaml_strict(path)
        return cls.model_validate(data)
