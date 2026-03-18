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
from transformers import AutoTokenizer, BertConfig

from hgere.data.relation_dataset import RelationDataset
from hgere.hgere.infer_fixed_spans import infer_fixed_spans
from hgere.labels import LABELS
from hgere.models.hgere import BertForHyperGNN
from hgere.pipeline.config import HGEREConfig

logger = logging.getLogger(__name__)

MODEL_CLASSES: dict[str, tuple] = {
    "hyper": (BertConfig, BertForHyperGNN, AutoTokenizer),
}


class HGERERunner:
    """Wraps the HGERE model for on-demand entity and relation extraction.

    Models are loaded once at construction. Call run() repeatedly.
    """

    def __init__(self, config: HGEREConfig, label_set: str) -> None:
        self._config = config
        self._label_set = label_set
        self._load_model()

    def _load_model(self) -> None:
        """Load HGERE model and tokenizer from model_dir."""
        cfg = self._config
        config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg.model_type]

        model_path = Path(cfg.model_dir).absolute()
        if not (model_path / "config.json").exists():
            checkpoints = sorted(
                [
                    p
                    for p in model_path.iterdir()
                    if p.is_dir() and p.name.startswith("checkpoint-")
                ],
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if checkpoints:
                model_path = checkpoints[-1]
                logger.info("HGERE: using checkpoint %s", model_path)

        labels = LABELS[self._label_set]
        num_rel_labels = labels.num_rel_labels(cfg.no_sym)

        bert_config = config_class.from_pretrained(
            str(model_path), num_labels=num_rel_labels
        )
        bert_config.max_seq_length = cfg.max_seq_length
        bert_config.num_ner_labels = labels.num_ner_labels
        bert_config.alpha = 1.0

        self._tokenizer = tokenizer_class.from_pretrained(
            cfg.base_model_name_or_path,
            do_lower_case=cfg.do_lower_case,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        infer_args = types.SimpleNamespace(
            do_train=False,
            lminit=False,
            ent_dim=cfg.ent_dim,
            rel_dim=cfg.rel_dim,
            mem_dim=cfg.mem_dim,
            factor_type=cfg.factor_type,
            factor_encoder=cfg.factor_encoder,
            ent_repr=cfg.ent_repr,
            iter=cfg.iter,
            layernorm=cfg.layernorm,
            layernorm_1st=cfg.layernorm_1st,
            attn_self=cfg.attn_self,
            attn_res=cfg.attn_res,
            unirel=cfg.unirel,
            device=device,
        )
        self._model = model_class.from_pretrained(
            str(model_path), config=bert_config, args=infer_args
        )
        self._model.to(device)
        self._model.eval()
        logger.info("HGERE model loaded from %s", model_path)

    def _make_args(self, n_gpu: int) -> object:
        cfg = self._config
        return types.SimpleNamespace(
            model_type=cfg.model_type,
            label_set=self._label_set,
            max_seq_length=cfg.max_seq_length,
            max_pair_length=cfg.max_pair_length,
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
            shuffle=False,
            batch_by_size=False,
            preload_dataset=False,
        )

    def _run_inference(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Write docs to tempfiles, run HGERE, read results back."""
        n_gpu = torch.cuda.device_count()
        args = self._make_args(n_gpu)
        labels = LABELS[self._label_set]

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

            dataset = RelationDataset(
                logger=logger,
                tokenizer=self._tokenizer,
                labels=labels,
                file_path=tmp_input,
                args=args,
                max_pair_length=args.max_pair_length,
                preload=False,
            )
            dataset.build(
                batch_size=args.eval_batch_size,
                shuffle=False,
                batch_by_size=False,
                n_workers=4,
                pin_memory=True,
            )

            infer_fixed_spans(
                model=self._model,
                eval_dataset=dataset,
                args=args,
                logger=logger,
                source_file_path=tmp_input,
                output_path=tmp_output,
                gold_only=False,
                disable_progress=True,
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

    def run(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run HGERE on docs that already have ``predicted_ner`` candidates.

        Args:
            docs: Documents with ``predicted_ner`` populated by PrunerRunner.

        Returns:
            Same docs enriched with ``predicted_ner``, ``predicted_ner_proba``,
            ``predicted_rel``, and ``predicted_rel_proba``.
        """
        if not docs:
            return []
        return self._run_inference(docs)
