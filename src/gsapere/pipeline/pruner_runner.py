"""Pruner inference runner for the pipeline.

Loads the pruner model once and scores all n-gram spans in a document.
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

from gsapere.pipeline.config import (
    FinalPruningConfig,
    PrunerConfig,
    suppress_transformers_warnings,
)
from gsapere.pruner.evaluate import run_pruner_inference
from gsapere.pruner.train import MODEL_CLASSES

logger = logging.getLogger(__name__)

# Default pre-filtering topk params used when no prune_config file is provided.
_DEFAULT_PREFILTER = {"topk_ratio": 0.5, "min_mentions_num": 3, "max_mentions_num": 18}


def apply_threshold(
    pool: dict[tuple[int, int], list[tuple[int, int, float, str]]],
    threshold: float,
) -> dict[tuple[int, int], list[tuple[int, int, float, str]]]:
    """Filter span pool to spans with prob >= threshold.

    Args:
        pool: Mapping (line_idx, sent_idx) -> [(start, end, prob, label), ...].
        threshold: Minimum probability to keep a span.

    Returns:
        Same structure with low-probability spans removed.
    """
    return {
        key: [(s, e, p, lbl) for s, e, p, lbl in spans if p >= threshold]
        for key, spans in pool.items()
    }


def apply_topk(
    pool: dict[tuple[int, int], list[tuple[int, int, float, str]]],
    topk_ratio: float,
    min_mentions_by_sentence: int,
    max_mentions_by_sentence: int,
) -> dict[tuple[int, int], list[tuple[int, int, float, str]]]:
    """Keep top-K spans per sentence ranked by probability.

    K = clamp(topk_ratio * sentence_length, min_mentions_by_sentence, max_mentions_by_sentence).
    Sentence length is estimated as max(end_index) + 1 across the sentence's spans.

    Args:
        pool: Mapping (line_idx, sent_idx) -> [(start, end, prob, label), ...].
        topk_ratio: Ratio of sentence length to determine K.
        min_mentions_by_sentence: Minimum K per sentence.
        max_mentions_by_sentence: Maximum K per sentence.

    Returns:
        Same structure with only the top-K spans per sentence.
    """
    result: dict[tuple[int, int], list[tuple[int, int, float, str]]] = {}
    for key, spans in pool.items():
        if not spans:
            result[key] = []
            continue
        sent_len = max(e for _, e, _, _ in spans) + 1
        k = max(
            min_mentions_by_sentence,
            min(int(topk_ratio * sent_len), max_mentions_by_sentence),
        )
        result[key] = sorted(spans, key=lambda x: -x[2])[:k]
    return result


def apply_final_pruning(
    pool: dict[tuple[int, int], list[tuple[int, int, float, str]]],
    config: FinalPruningConfig,
) -> dict[tuple[int, int], list[tuple[int, int, float, str]]]:
    """Apply the configured final pruning strategy to the span pool.

    Args:
        pool: Full scored span pool from inference.
        config: FinalPruningConfig specifying method and parameters.

    Returns:
        Filtered pool to be used as predicted_ner candidates.
    """
    if config.method == "threshold":
        assert config.threshold is not None
        return apply_threshold(pool, config.threshold)
    return apply_topk(
        pool,
        topk_ratio=config.topk_ratio,
        min_mentions_by_sentence=config.min_mentions_by_sentence,
        max_mentions_by_sentence=config.max_mentions_by_sentence,
    )


def _load_prefilter_params(prune_config_path: str) -> dict[str, Any]:
    """Load pre-filtering topk params from a best_config.json file.

    Expects the Pydantic PreFilterParams format:
      topk:      {"method": "topk", "ratio": float, "min": int, "max": int}
      threshold: {"method": "threshold", "value": float}

    Falls back to _DEFAULT_PREFILTER for threshold-method configs (which have
    no topk parameters).
    """
    data = json.loads(Path(prune_config_path).read_text())
    if data.get("method") == "topk":
        return {
            "topk_ratio": float(data["ratio"]),
            "min_mentions_num": int(data["min"]),
            "max_mentions_num": int(data["max"]),
        }
    # threshold method — no topk params available; use defaults
    return dict(_DEFAULT_PREFILTER)


def build_predicted_ner_from_pool(
    pool: dict[tuple[int, int], list[tuple[int, int, float, str]]],
    n_docs: int,
    n_sents_per_doc: list[int],
) -> list[list[list[list[Any]]]]:
    """Convert span pool dict to nested list structure for JSONL output.

    Args:
        pool: Mapping (line_idx, sent_idx) -> [(start, end, prob, label), ...].
        n_docs: Number of documents.
        n_sents_per_doc: Number of sentences per document.

    Returns:
        Nested list: [doc_idx][sent_idx] -> [[start, end, label], ...]
    """
    result: list[list[list[list[Any]]]] = []
    for doc_idx in range(n_docs):
        doc_sents: list[list[list[Any]]] = []
        for sent_idx in range(n_sents_per_doc[doc_idx]):
            spans = pool.get((doc_idx, sent_idx), [])
            doc_sents.append([[s, e, lbl] for s, e, _p, lbl in spans])
        result.append(doc_sents)
    return result


def _build_ner_candidates_proba(
    pool: dict[tuple[int, int], list[tuple[int, int, float, str]]],
    n_docs: int,
    n_sents_per_doc: list[int],
) -> list[list[list[list[Any]]]]:
    """Convert span pool to ner_candidates_proba format (includes probability).

    Returns:
        Nested list: [doc_idx][sent_idx] -> [[start, end, prob, label], ...]
    """
    result: list[list[list[list[Any]]]] = []
    for doc_idx in range(n_docs):
        doc_sents: list[list[list[Any]]] = []
        for sent_idx in range(n_sents_per_doc[doc_idx]):
            spans = pool.get((doc_idx, sent_idx), [])
            doc_sents.append([[s, e, p, lbl] for s, e, p, lbl in spans])
        result.append(doc_sents)
    return result


class PrunerRunner:
    """Wraps the pruner model for on-demand document scoring.

    Models are loaded once at construction. Call run() repeatedly.
    """

    def __init__(self, config: PrunerConfig, label_set: str) -> None:
        self._config = config
        self._label_set = label_set
        if config.prune_config is not None:
            self._prefilter_params = _load_prefilter_params(config.prune_config)
        else:
            self._prefilter_params = {
                "topk_ratio": config.topk_ratio,
                "min_mentions_num": config.min_mentions_num,
                "max_mentions_num": config.max_mentions_num,
            }
        logger.info("Pruner prefilter params: %s", self._prefilter_params)
        self._load_model()

    def _load_model(self) -> None:
        """Load pruner model and tokenizer from model_dir."""
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
                logger.info("Pruner: using checkpoint %s", model_path)

        with suppress_transformers_warnings():
            bert_config = config_class.from_pretrained(str(model_path))
        with suppress_transformers_warnings():
            self._tokenizer = tokenizer_class.from_pretrained(
                cfg.base_model_name_or_path,
                do_lower_case=cfg.do_lower_case,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        training_args_path = model_path / "training_args.bin"
        if training_args_path.exists():
            model_args = torch.load(
                training_args_path, map_location="cpu", weights_only=False
            )
            logger.info("Pruner: loaded training_args from %s", training_args_path)
        else:
            model_args = types.SimpleNamespace(
                do_train=False,
                lminit=False,
                biaf_span=False,
                biaf_mode=2,
                biaf_factorize=True,
                span_hidden_size=768,
                rank=768,
                span_size=256,
                extra_repr="",
                onedropout=False,
            )
            logger.warning(
                "Pruner: training_args.bin not found at %s, using defaults",
                model_path,
            )
        self._model_args = model_args
        with suppress_transformers_warnings():
            self._model = model_class.from_pretrained(
                str(model_path), config=bert_config, args=model_args
            )
        self._model.to(device)
        self._model.eval()
        logger.info("Pruner model loaded from %s", model_path)

    def _make_args(self, n_gpu: int) -> object:
        cfg = self._config
        p = self._prefilter_params
        return types.SimpleNamespace(
            model_type=cfg.model_type,
            label_set=self._label_set,
            max_seq_length=cfg.max_seq_length,
            max_pair_length=cfg.max_pair_length,
            max_mention_ori_length=cfg.max_mention_ori_length,
            topk_ratio=p["topk_ratio"],
            max_mentions_num=p["max_mentions_num"],
            min_mentions_num=p["min_mentions_num"],
            per_gpu_eval_batch_size=cfg.per_gpu_eval_batch_size,
            n_gpu=n_gpu,
            device=self._device,
            nocross=getattr(self._model_args, "nocross", False),
            rulebased_pruner_file=self._config.rulebased_pruner_file,
        )

    def run(
        self, docs: list[dict[str, Any]], show_progress: bool = False
    ) -> list[dict[str, Any]]:
        """Score all spans in docs and annotate with candidates.

        Args:
            docs: List of document dicts with at least 'sentences' and 'doc_key'.
            show_progress: Show a tqdm progress bar over inference batches.

        Returns:
            Same docs enriched with:
            - ``ner_candidates_proba``: full scored span pool per sentence,
              each entry [start, end, prob, gold_label].
            - ``predicted_ner``: final-pruning-filtered candidates per sentence,
              each entry [start, end, label].
        """
        if not docs:
            return []

        n_sents_per_doc = [len(d["sentences"]) for d in docs]
        n_gpu = torch.cuda.device_count()
        args = self._make_args(n_gpu)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as tmp_f:
                tmp_path = tmp_f.name
                for doc in docs:
                    tmp_f.write(json.dumps(doc) + "\n")

            pool = run_pruner_inference(
                args,
                self._model,
                self._tokenizer,
                tmp_path,
                disable_progress=not show_progress,
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        filtered = apply_final_pruning(pool, self._config.final_pruning)

        ner_candidates_proba = _build_ner_candidates_proba(
            pool, len(docs), n_sents_per_doc
        )
        predicted_ner = build_predicted_ner_from_pool(
            filtered, len(docs), n_sents_per_doc
        )

        result = []
        for i, doc in enumerate(docs):
            out = dict(doc)
            out["ner_candidates_proba"] = ner_candidates_proba[i]
            out["predicted_ner_proba"] = ner_candidates_proba[i]
            out["predicted_ner"] = predicted_ner[i]
            result.append(out)
        return result
