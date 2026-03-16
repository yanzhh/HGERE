"""CLI command: infer-pruner-augmented

Runs HGERE inference using the existing pruner/pipeline ``predicted_ner``
candidates augmented with any missing gold NER spans.  Only the gold-span
predictions are kept in the output.

Workflow
--------
1. Input file is an ``ent_pred_*.json`` file that already contains:
   - ``predicted_ner``: spans detected by the pruner+HGERE pipeline
   - ``ner``: gold NER annotations
2. The candidate set for each sentence is extended with any gold spans not
   already present in ``predicted_ner``.
3. HGERE is re-run on the augmented candidates with NIL masked (every span
   receives the best non-NIL label).
4. The output retains only predictions whose span endpoints are gold spans,
   together with relations between gold spans.

This gives more realistic confidence scores than pure fixed-span inference
(where no pipeline candidates are present) while still covering all gold spans.

Usage
-----
    uv run infer-pruner-augmented \\
        --model_type hyper \\
        --model_name_or_path saves/my_checkpoint \\
        --label_set gsap \\
        --input_file saves/.../ent_pred_tes.json \\
        --output_file out/pruner_augmented_scinlp_test.json
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import torch

from hgere.data.relation_dataset import RelationDataset
from hgere.hgere.infer_fixed_spans import infer_fixed_spans, make_augmented_candidate_file
from hgere.labels import LABELS
from hgere.utils import set_seed
from transformers import AutoTokenizer, BertConfig

from hgere.models.hgere import BertForHyperGNN

MODEL_CLASSES = {
    "hyper": (BertConfig, BertForHyperGNN, AutoTokenizer),
}


def _build_parser():
    p = argparse.ArgumentParser(
        description="HGERE inference with pruner candidates augmented by gold spans."
    )

    # --- required ---
    p.add_argument("--model_type", type=str, default="hyper",
                   choices=list(MODEL_CLASSES.keys()))
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--label_set", type=str, required=True)
    p.add_argument("--input_file", type=str, required=True,
                   help="ent_pred_*.json file with predicted_ner (pipeline) and ner (gold).")
    p.add_argument("--output_file", type=str, required=True)

    # --- tokenizer / sequences ---
    p.add_argument("--do_lower_case", action="store_true")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_seq_length", type=int, default=384)
    p.add_argument("--max_pair_length", type=int, default=64)

    # --- dataset behaviour ---
    p.add_argument("--use_typemarker", action="store_true")
    p.add_argument("--no_sym", action="store_true")
    p.add_argument("--nocross", action="store_true")
    p.add_argument("--per_gpu_eval_batch_size", type=int, default=8)

    # --- hardware ---
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # --- model architecture (must match the checkpoint) ---
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--ent_repr", type=str, default="mix")
    p.add_argument("--ent_enc", type=str, default="cat")
    p.add_argument("--uni_ent", action="store_true")
    p.add_argument("--pred_sub", action="store_true")
    p.add_argument("--ner_cls", type=str, default="cat")
    p.add_argument("--rel_enc", type=str, default="cat")
    p.add_argument("--ent_dim", type=int, default=200)
    p.add_argument("--rel_dim", type=int, default=200)
    p.add_argument("--rel_rank", type=int, default=200)
    p.add_argument("--rel_factorize", action="store_true")
    p.add_argument("--baseline", type=str, default="firstorder")
    p.add_argument("--factor_type", type=str, default="ternary")
    p.add_argument("--mem_dim", type=int, default=200)
    p.add_argument("--iter", type=int, default=3)
    p.add_argument("--iter1", type=int, default=1)
    p.add_argument("--layernorm", action="store_true")
    p.add_argument("--layernorm_1st", action="store_true")
    p.add_argument("--attn_self", action="store_true")
    p.add_argument("--aggregate_type", type=str, default="attn")
    p.add_argument("--aggregate_func", type=str, default="max")
    p.add_argument("--agg_with_self", action="store_true")
    p.add_argument("--fix_obj", action="store_true")
    p.add_argument("--edgetype", type=str, default="sib")
    p.add_argument("--attn_scorer", type=str, default="biaf")
    p.add_argument("--attn_res", action="store_true")
    p.add_argument("--att_left", action="store_true")
    p.add_argument("--att_right", action="store_true")
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_head", type=int, default=32)
    p.add_argument("--factor_encoder", type=str, default="cat")
    p.add_argument("--re_focal_loss", action="store_true")
    p.add_argument("--re_focal_gamma", type=float, default=2.0)
    p.add_argument("--ner_focal_loss", action="store_true")
    p.add_argument("--ner_focal_gamma", type=float, default=2.0)
    return p


def cli():
    args = _build_parser().parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # --- device ---
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    set_seed(args)

    # --- labels ---
    assert args.label_set in LABELS, f"Unknown label_set '{args.label_set}'."
    labels = LABELS[args.label_set]
    args.num_ner_labels = labels.num_ner_labels
    args.num_rel_labels = labels.num_rel_labels(args.no_sym)

    # --- model / tokenizer ---
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    model_path = Path(args.model_name_or_path).absolute()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not (model_path / "config.json").exists():
        checkpoints = sorted(
            [p for p in model_path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No config.json and no checkpoint-* subdirs in {model_path}")
        model_path = checkpoints[-1]
        logger.info("Using checkpoint: %s", model_path)
    args.model_name_or_path = str(model_path)

    config = config_class.from_pretrained(args.model_name_or_path, num_labels=args.num_rel_labels)
    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = args.num_ner_labels

    tokenizer_path = Path(
        args.tokenizer_path
        or getattr(config, "_name_or_path", None)
        or args.model_name_or_path
    )
    if not (tokenizer_path / "vocab.txt").exists():
        raise FileNotFoundError(
            f"No vocab.txt in '{tokenizer_path}'. Pass --tokenizer_path."
        )
    tokenizer = tokenizer_class.from_pretrained(str(tokenizer_path), do_lower_case=args.do_lower_case)

    args.do_train = False
    args.lminit = False

    _tf_logger = logging.getLogger("transformers.modeling_utils")
    _prev = _tf_logger.level
    _tf_logger.setLevel(logging.ERROR)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, args=args)
    _tf_logger.setLevel(_prev)

    model.to(device)
    model.eval()

    # --- dataset ---
    args.input_file = str(Path(args.input_file).resolve())
    args.output_file = str(Path(args.output_file).resolve())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_f:
        tmp_path = tmp_f.name
    try:
        make_augmented_candidate_file(args.input_file, tmp_path)

        args.train_batch_size = args.per_gpu_eval_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.shuffle = False
        args.batch_by_size = False
        args.preload_dataset = False
        args.model_type = args.model_type.lower()

        dataset = RelationDataset(
            logger=logger,
            tokenizer=tokenizer,
            labels=labels,
            file_path=tmp_path,
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
        logger.info("Loaded %d examples from %s", len(dataset), args.input_file)

        infer_fixed_spans(
            model=model,
            eval_dataset=dataset,
            args=args,
            logger=logger,
            source_file_path=args.input_file,
            output_path=args.output_file,
            gold_only=True,
        )
        logger.info("Predictions written to %s", args.output_file)
    finally:
        os.unlink(tmp_path)
