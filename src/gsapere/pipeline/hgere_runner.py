"""HGERE inference runner for the pipeline.

Loads the HGERE model once and predicts entity types and relations
on pruner-filtered candidate spans.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import types
from pathlib import Path
from typing import Any

import torch

from gsapere.data.config import RelationDatasetParams
from gsapere.data.relation_dataset import RelationDataset
from gsapere.hgere.inference import LabelMeta, infer_hgere, infer_hgere_multi
from gsapere.hgere.train_setup import resolve_checkpoint
from gsapere.labels import LABELS
from gsapere.models.hgere import MODEL_CLASSES
from gsapere.pipeline.config import HGEREConfig, suppress_transformers_warnings

logger = logging.getLogger(__name__)

# Architecture parameters that must match the trained model.
# These are loaded from training_args.bin when available and take precedence
# over the pipeline YAML config.
_ARCH_PARAMS: tuple[str, ...] = (
    "ent_dim",
    "rel_dim",
    "mem_dim",
    "factor_type",
    "factor_encoder",
    "ent_repr",
    "n_iter",
    "layernorm",
    "layernorm_1st",
    "attn_self",
    "attn_res",
    "unirel",
    "head_hidden_dim",
)


class HGERERunner:
    """Wraps the HGERE model for on-demand entity and relation extraction.

    The model is loaded once at construction. For multi-head models, call
    :meth:`run` with an explicit *label_set* to select the appropriate head,
    or call :meth:`run_multi` to run all heads in a single forward pass per
    batch (encoder and HyperGNN run once; each head is applied separately).

    Args:
        config: HGERE stage configuration.
        label_sets: One or more label set names. The first entry is used as the
            baseline for HuggingFace config initialisation.
    """

    def __init__(self, config: HGEREConfig, label_sets: list[str]) -> None:
        self._config = config
        self._label_sets = label_sets
        self._load_model()

    def _load_model(self) -> None:
        """Load HGERE model and tokenizer from model_dir."""
        cfg = self._config
        config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg.model_type]

        model_path = Path(cfg.model_dir).absolute()
        if not (model_path / "config.json").exists():
            resolved = resolve_checkpoint(model_path)
            if resolved is not None:
                model_path = resolved
                logger.info("HGERE: using checkpoint %s", model_path)

        # Load training_args.bin to recover architecture parameters.
        training_args_path = model_path / "training_args.bin"
        if training_args_path.exists():
            model_args = torch.load(
                training_args_path, map_location="cpu", weights_only=False
            )
            logger.info("HGERE: loaded training_args from %s", training_args_path)
            for param in _ARCH_PARAMS:
                bin_val = getattr(model_args, param, None)
                cfg_val = getattr(cfg, param)
                if bin_val is not None and bin_val != cfg_val:
                    logger.warning(
                        "HGERE: training_args.bin overrides %s: config=%r → bin=%r",
                        param,
                        cfg_val,
                        bin_val,
                    )
        else:
            model_args = None
            logger.info(
                "HGERE: training_args.bin not found at %s, using pipeline config for arch params",
                model_path,
            )

        def _arch(param: str) -> Any:
            """Return arch param from training_args.bin if present, else config."""
            if model_args is not None:
                bin_val = getattr(model_args, param, None)
                if bin_val is not None:
                    return bin_val
            return getattr(cfg, param)

        labels = LABELS[self._label_sets[0]]
        num_rel_labels = labels.num_rel_labels(cfg.no_sym)

        with suppress_transformers_warnings():
            bert_config = config_class.from_pretrained(
                str(model_path), num_labels=num_rel_labels
            )
        bert_config.max_seq_length = cfg.max_seq_length
        # For multi-head checkpoints, bert_config already has dataset_heads from
        # config.json, and the model will build nn.ModuleDicts instead.  These
        # single-head fallback values are only used when dataset_heads is absent.
        bert_config.num_ner_labels = labels.num_ner_labels
        bert_config.alpha = 1.0

        with suppress_transformers_warnings():
            self._tokenizer = tokenizer_class.from_pretrained(
                cfg.base_model_name_or_path,
                do_lower_case=cfg.do_lower_case,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        infer_args = types.SimpleNamespace(
            do_train=False,
            lminit=False,
            ent_dim=_arch("ent_dim"),
            rel_dim=_arch("rel_dim"),
            mem_dim=_arch("mem_dim"),
            factor_type=_arch("factor_type"),
            factor_encoder=_arch("factor_encoder"),
            ent_repr=_arch("ent_repr"),
            n_iter=_arch("n_iter"),
            layernorm=_arch("layernorm"),
            layernorm_1st=_arch("layernorm_1st"),
            attn_self=_arch("attn_self"),
            attn_res=_arch("attn_res"),
            unirel=_arch("unirel"),
            head_hidden_dim=_arch("head_hidden_dim"),
            device=device,
        )
        with suppress_transformers_warnings():
            self._model = model_class.from_pretrained(
                str(model_path), config=bert_config, args=infer_args
            )
        self._model.to(device)
        self._model.eval()
        self._bert_config = bert_config
        logger.info("HGERE model loaded from %s", model_path)
        if cfg.pre_filter_params is not None:
            logger.info(
                "HGERE pre_filter_params: %s", cfg.pre_filter_params.model_dump()
            )
        else:
            logger.info("HGERE pre_filter_params: None (using predicted_ner as-is)")

    def _make_args(self, n_gpu: int, label_set: str) -> object:
        cfg = self._config
        return types.SimpleNamespace(
            model_type=cfg.model_type,
            label_set=label_set,
            max_seq_length=cfg.max_seq_length,
            per_gpu_eval_batch_size=cfg.per_gpu_eval_batch_size,
            train_batch_size=cfg.per_gpu_eval_batch_size,
            eval_batch_size=cfg.per_gpu_eval_batch_size * max(1, n_gpu),
            n_gpu=n_gpu,
            device=self._device,
            use_typemarker=cfg.use_typemarker,
            no_sym=cfg.no_sym,
            nocross=cfg.nocross,
            local_rank=cfg.local_rank,
            do_lower_case=cfg.do_lower_case,
            do_train=False,
            lminit=False,
            shuffle=cfg.shuffle,
            batch_by_size=False,
            preload_dataset=False,
            pre_filter_params=(
                cfg.pre_filter_params.model_dump() if cfg.pre_filter_params else None
            ),
        )

    def _run_inference(
        self,
        docs: list[dict[str, Any]],
        label_set: str,
        show_progress: bool = False,
        debug_break_on_first_rel: bool = False,
        debug_log_rel_probs: bool = False,
    ) -> list[dict[str, Any]]:
        """Write docs to tempfiles, run HGERE with *label_set* head, read results back."""
        n_gpu = torch.cuda.device_count()
        args = self._make_args(n_gpu, label_set)
        labels = LABELS[label_set]

        tmp_input = None
        tmp_output = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f_in:
                tmp_input = f_in.name
                for doc in docs:
                    f_in.write(json.dumps(doc) + "\n")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f_out:
                tmp_output = f_out.name

            cfg = self._config
            dataset_cls_token_id: int | None = None
            if cfg.use_dataset_id_token_as_cls:
                cls_ids: dict[str, int] = (
                    getattr(self._bert_config, "dataset_cls_token_ids", None) or {}
                )
                dataset_cls_token_id = cls_ids.get(label_set)
                if dataset_cls_token_id is None:
                    logger.warning(
                        "use_dataset_id_token_as_cls=True but no token ID found for "
                        "'%s' in model config.dataset_cls_token_ids=%r",
                        label_set,
                        cls_ids,
                    )
            dataset_params = RelationDatasetParams(
                max_seq_length=cfg.max_seq_length,
                model_type=cfg.model_type,
                use_typemarker=cfg.use_typemarker,
                no_sym=cfg.no_sym,
                nocross=cfg.nocross,
                local_rank=cfg.local_rank,
                split="inference",
                pre_filter_params=(
                    cfg.pre_filter_params.model_dump()
                    if cfg.pre_filter_params
                    else None
                ),
                use_gold_ner=cfg.use_gold_ner,
                # Pass label_set as dataset_id so multi-head models route to
                # the correct NER and relation head in forward().
                dataset_id=label_set,
                dataset_cls_token_id=dataset_cls_token_id,
            )
            dataset = RelationDataset(
                logger=logger,
                tokenizer=self._tokenizer,
                labels=labels,
                file_path=tmp_input,
                params=dataset_params,
            )
            dataset.build(
                batch_size=args.eval_batch_size,
                shuffle=False,
                batch_by_size=False,
                n_workers=cfg.n_workers,
                pin_memory=True,
            )

            infer_hgere(
                model=self._model,
                eval_dataset=dataset,
                args=args,
                logger=logger,
                source_file_path=tmp_input,
                output_path=tmp_output,
                gold_only=False,
                disable_progress=not show_progress,
                debug_break_on_first_rel=debug_break_on_first_rel,
                debug_log_rel_probs=debug_log_rel_probs,
            )

            result = []
            with open(tmp_output) as f:
                for line in f:
                    result.append(json.loads(line))
            return result

        finally:
            for p in (tmp_input, tmp_output):
                if p and os.path.exists(p):
                    os.unlink(p)

    def run(
        self,
        docs: list[dict[str, Any]],
        label_set: str | None = None,
        show_progress: bool = False,
        debug_break_on_first_rel: bool = False,
        debug_log_rel_probs: bool = False,
    ) -> list[dict[str, Any]]:
        """Run HGERE on docs that already have ``predicted_ner`` candidates.

        Args:
            docs: Documents with ``predicted_ner`` populated by PrunerRunner.
            label_set: Label set / head to activate. Defaults to the first
                entry in ``label_sets`` passed at construction.
            show_progress: Show a tqdm progress bar over inference batches.
            debug_break_on_first_rel: If True, log and raise on the first
                predicted relation — for diagnosing zero-relation issues.

        Returns:
            Same docs enriched with ``predicted_ner``, ``predicted_ner_proba``,
            ``predicted_rel``, and ``predicted_rel_proba``.
        """
        if not docs:
            return []
        return self._run_inference(
            docs,
            label_set=label_set or self._label_sets[0],
            show_progress=show_progress,
            debug_break_on_first_rel=debug_break_on_first_rel,
            debug_log_rel_probs=debug_log_rel_probs,
        )

    def _run_inference_multi(
        self,
        docs: list[dict[str, Any]],
        show_progress: bool = False,
        debug_break_on_first_rel: bool = False,
        debug_log_rel_probs: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run all heads in one forward pass per batch."""
        n_gpu = torch.cuda.device_count()
        cfg = self._config
        primary_ls = self._label_sets[0]
        args = self._make_args(n_gpu, primary_ls)

        tmp_input = None
        tmp_outputs: dict[str, str] = {}
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f_in:
                tmp_input = f_in.name
                for doc in docs:
                    f_in.write(json.dumps(doc) + "\n")

            for ls in self._label_sets:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".jsonl", delete=False
                ) as f_out:
                    tmp_outputs[ls] = f_out.name

            dataset_cls_token_id: int | None = None
            if cfg.use_dataset_id_token_as_cls:
                cls_ids: dict[str, int] = (
                    getattr(self._bert_config, "dataset_cls_token_ids", None) or {}
                )
                dataset_cls_token_id = cls_ids.get(primary_ls)

            dataset_params = RelationDatasetParams(
                max_seq_length=cfg.max_seq_length,
                model_type=cfg.model_type,
                use_typemarker=cfg.use_typemarker,
                no_sym=cfg.no_sym,
                nocross=cfg.nocross,
                local_rank=cfg.local_rank,
                split="inference",
                pre_filter_params=(
                    cfg.pre_filter_params.model_dump()
                    if cfg.pre_filter_params
                    else None
                ),
                use_gold_ner=cfg.use_gold_ner,
                dataset_id=primary_ls,
                dataset_cls_token_id=dataset_cls_token_id,
            )
            dataset = RelationDataset(
                logger=logger,
                tokenizer=self._tokenizer,
                labels=LABELS[primary_ls],
                file_path=tmp_input,
                params=dataset_params,
            )
            dataset.build(
                batch_size=args.eval_batch_size,
                shuffle=False,
                batch_by_size=False,
                n_workers=cfg.n_workers,
                pin_memory=True,
            )

            label_metas = {
                ls: LabelMeta(
                    ner_label_list=LABELS[ls].ner,
                    sym_labels=LABELS[ls].rel.symmetric(only_nil=cfg.no_sym),
                    label_list=LABELS[ls].rel.all,
                )
                for ls in self._label_sets
            }

            infer_hgere_multi(
                model=self._model,
                eval_dataset=dataset,
                label_metas=label_metas,
                args=args,
                logger=logger,
                source_file_path=tmp_input,
                output_paths=tmp_outputs,
                gold_only=False,
                disable_progress=not show_progress,
                debug_break_on_first_rel=debug_break_on_first_rel,
                debug_log_rel_probs=debug_log_rel_probs,
            )

            results: dict[str, list[dict[str, Any]]] = {}
            for ls, tmp_out in tmp_outputs.items():
                with open(tmp_out) as f:
                    results[ls] = [json.loads(line) for line in f]
            return results

        finally:
            for p in [tmp_input, *tmp_outputs.values()]:
                if p and os.path.exists(p):
                    os.unlink(p)

    def run_multi(
        self,
        docs: list[dict[str, Any]],
        show_progress: bool = False,
        debug_break_on_first_rel: bool = False,
        debug_log_rel_probs: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run all HGERE heads in a single forward pass per batch.

        The encoder and HyperGNN run once; every configured head is applied to
        the shared representations.

        Args:
            docs: Documents with ``predicted_ner`` already populated.
            show_progress: Show tqdm progress bars.
            debug_break_on_first_rel: Raise after first predicted relation.

        Returns:
            Mapping ``{label_set: enriched_docs}`` — one entry per label set.
        """
        if not docs:
            return {ls: [] for ls in self._label_sets}
        return self._run_inference_multi(
            docs,
            show_progress=show_progress,
            debug_break_on_first_rel=debug_break_on_first_rel,
            debug_log_rel_probs=debug_log_rel_probs,
        )
