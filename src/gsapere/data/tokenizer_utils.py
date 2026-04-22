"""Shared tokenizer adjustment utilities for pruner and HGERE training.

Both the pruner and HGERE add special marker tokens to the tokenizer and
optionally initialise their embeddings from existing vocabulary entries
(LM-init). This module provides a single implementation used by both.
"""

from __future__ import annotations

from typing import Any, List


def _get_word_embeddings(model: Any, model_type: str) -> Any:
    """Return the word-embedding weight tensor for *model*."""
    if model_type.startswith("albert"):
        return model.albert.embeddings.word_embeddings.weight.data
    elif model_type.startswith("roberta"):
        return model.roberta.embeddings.word_embeddings.weight.data
    elif model_type.startswith("modernbert"):
        return model.bert.embeddings.tok_embeddings.weight.data
    else:  # BERT / SciBERT
        return model.bert.embeddings.word_embeddings.weight.data


def _get_marker_ids(tokenizer: Any, model_type: str, n: int) -> List[int]:
    """Return the first *n* marker token IDs for *model_type*.

    For BERT/SciBERT ``[unusedX]`` sits at vocabulary index ``X + 1``.
    For Albert/ModernBERT the tokens are added explicitly by
    :func:`adjust_tokenizer` so the tokenizer lookup is authoritative.
    For RoBERTa the marker slots are pre-existing BPE positions.
    """
    if model_type.startswith("roberta"):
        return [50261 + i for i in range(n)]
    elif model_type.startswith("modernbert"):
        return [tokenizer.convert_tokens_to_ids(f"[unused{i}]") for i in range(n)]
    elif model_type.startswith("albert"):
        return [30000 + i for i in range(n)]
    else:  # BERT / SciBERT
        return [i + 1 for i in range(n)]  # [unused0]=1, [unused1]=2, …


def adjust_tokenizer(
    tokenizer: Any,
    model: Any,
    model_type: str,
    n_special_tokens: int,
    lminit: bool,
    init_tokens: List[str],
    logger: Any,
) -> None:
    """Add marker tokens to *tokenizer* and optionally initialise their embeddings.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer.
    model
        HuggingFace model whose embedding table may be resized.
    model_type
        Model-type string (e.g. ``"bert"``, ``"albert"``, ``"modernbert"``).
    n_special_tokens
        Number of ``[unusedX]`` tokens to add (for albert/modernbert only;
        ignored for BERT/RoBERTa which have them pre-existing in vocabulary).
    lminit
        When ``True``, copy existing embeddings into the new marker slots.
        Callers should combine ``do_train`` and ``lminit`` flags before passing.
    init_tokens
        Seed-token names for LM init, one per marker *pair*.
        E.g. ``["entity"]`` (pruner) or ``["subject", "object"]`` (HGERE).
        Pair *i* → ``[unused2i]`` ← ``[MASK]``,
                    ``[unused2i+1]`` ← ``init_tokens[i]``.
    logger
        Python logger.
    """
    # --- Step 1: add special tokens and resize embeddings ---
    if model_type.startswith("albert"):
        special_tokens_dict = {
            "additional_special_tokens": [
                f"[unused{x}]" for x in range(n_special_tokens)
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.albert.resize_token_embeddings(len(tokenizer))
    elif model_type.startswith("modernbert"):
        special_tokens_dict = {
            "additional_special_tokens": [
                f"[unused{x}]" for x in range(n_special_tokens)
            ]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.bert.resize_token_embeddings(len(tokenizer))

    if not lminit:
        return

    # --- Step 2: resolve [MASK] ID ---
    if model_type.startswith("roberta"):
        mask_id = 50264  # hardcoded RoBERTa vocab position
    else:
        mask_ids = tokenizer.encode("[MASK]", add_special_tokens=False)
        assert len(mask_ids) == 1
        mask_id = mask_ids[0]

    # --- Step 3: initialise marker embeddings ---
    word_embeddings = _get_word_embeddings(model, model_type)
    marker_ids = _get_marker_ids(tokenizer, model_type, len(init_tokens) * 2)

    for i, init_token_name in enumerate(init_tokens):
        start_id = marker_ids[2 * i]
        end_id = marker_ids[2 * i + 1]

        if model_type.startswith("modernbert"):
            # BPE tokenizer: token may be multi-token; fall back to [MASK] if so.
            token_ids = tokenizer.encode(init_token_name, add_special_tokens=False)
            init_id = token_ids[0] if len(token_ids) == 1 else mask_id
            logger.info("mask_id: %d, %s_id: %d", mask_id, init_token_name, init_id)
        elif model_type.startswith("roberta"):
            init_id = 10014  # hardcoded RoBERTa vocab position for "entity"
        else:  # BERT / SciBERT / Albert
            token_ids = tokenizer.encode(init_token_name, add_special_tokens=False)
            assert len(token_ids) == 1, (
                f"Expected single token for '{init_token_name}', got {token_ids}"
            )
            init_id = token_ids[0]
            logger.info("%s_id: %d, mask_id: %d", init_token_name, init_id, mask_id)

        word_embeddings[start_id].copy_(word_embeddings[mask_id])
        word_embeddings[end_id].copy_(word_embeddings[init_id])


def add_dataset_cls_tokens(
    tokenizer: Any,
    model: Any,
    model_type: str,
    dataset_names: List[str],
    start_unused_idx: int,
    logger: Any,
) -> dict[str, int]:
    """Reserve [unusedX] slots for per-dataset CLS tokens; init from [CLS]+noise.

    Uses pre-existing ``[unusedX]`` vocabulary entries (same mechanism as the
    subject/object span markers) rather than adding brand-new special tokens.
    This avoids any tokenizer-persistence issues — the IDs are stable positions
    in the original BERT/SciBERT vocabulary.

    For albert/modernbert the required ``[unusedX]`` tokens are explicitly added
    to the tokenizer and the embedding table is resized (same as adjust_tokenizer).
    For BERT/SciBERT the tokens already exist at vocabulary position ``X + 1``.

    Parameters
    ----------
    start_unused_idx:
        Index of the first free ``[unusedX]`` slot — typically ``n_special_tokens``
        (the number of slots already consumed by subject/object markers).

    Returns
    -------
    dict[str, int]
        Mapping ``{dataset_name: token_id}`` for storage in the HF model config.
    """
    import torch

    n = len(dataset_names)
    end_idx = start_unused_idx + n

    # albert / modernbert: add the new [unusedX] tokens and resize
    if model_type.startswith("albert"):
        new_tokens = [f"[unused{x}]" for x in range(start_unused_idx, end_idx)]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.albert.resize_token_embeddings(len(tokenizer))
    elif model_type.startswith("modernbert"):
        new_tokens = [f"[unused{x}]" for x in range(start_unused_idx, end_idx)]
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.bert.resize_token_embeddings(len(tokenizer))
    # BERT / SciBERT: [unusedX] already exists in vocabulary at position X+1

    all_ids = _get_marker_ids(tokenizer, model_type, end_idx)
    dataset_token_ids = all_ids[start_unused_idx:]

    word_embeddings = _get_word_embeddings(model, model_type)
    cls_id = tokenizer.cls_token_id
    cls_emb = word_embeddings[cls_id].clone()

    mapping: dict[str, int] = {}
    for i, dataset_name in enumerate(dataset_names):
        tok_id = dataset_token_ids[i]
        noise = torch.randn_like(cls_emb) * 0.02
        word_embeddings[tok_id] = cls_emb + noise
        logger.info(
            "Dataset CLS token '%s': [unused%d] (id=%d), init from [CLS]+noise(σ=0.02)",
            dataset_name,
            start_unused_idx + i,
            tok_id,
        )
        mapping[dataset_name] = tok_id

    return mapping
