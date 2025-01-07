# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

import argparse
import glob
import json
import logging
import os
import random
import re
import shutil
import socket
import sys
from pathlib import Path

from tqdm import tqdm, trange
import numpy as np
from tensorboardX import SummaryWriter

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    BertConfig,
    BertTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup,
    AdamW,
    BertForSpanMarkerNerPruner,
    #AlbertConfig,
    #AlbertTokenizer,
    #AlbertForSpanMarkerNerPruner,
)

from wolf_data_utils import ACEDatasetNER
from wolf_eval_pruner import evaluate

"""
from transformers import (
    WEIGHTS_NAME,
    AutoTokenizer,
    RobertaTokenizer,
    AlbertForNER,
    AlbertForSpanNER,
    AlbertForSpanMarkerNER,
    BertForNER,
    BertForSpanNER,
    BertForLeftLMNER,
    BertForSpanMarkerNER,
    BertForSpanMarkerBiNER,
    RobertaForNER,
    RobertaForSpanNER,
    RobertaForSpanMarkerNER,
)
"""
# Logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bertspanmarkerpruner": (BertConfig, BertForSpanMarkerNerPruner, BertTokenizer),
    #"albertspanmarkerpruner": (
    #    AlbertConfig,
    #    AlbertForSpanMarkerNerPruner,
    #    AlbertTokenizer,
    #),
}

# NEG_INF = -1e30


def train(logger, args, model, tokenizer):
    train_sampler, train_data_loader = load_data(args, tokenizer, logger)
    len_train = len(train_data_loader)
    model, optimizer, scheduler, tb_writer = setup_training(args, model, len_train, logger)
    


    # Train!

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_result = -1

    for epoch_num in train_iterator:
        logger.info(f"------------------Epoch {epoch_num} starts.-----------------")
        logging_loss_steps_0, logging_loss_epoch = global_step, 0.0

        # if _ > 0 and (args.shuffle or args.group_edge or args.group_sort):
        #     train_dataset.initialize()
        #     if args.group_edge:
        #         train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        #         train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=2*int(args.output_dir.find('test')==-1))

        epoch_iterator = tqdm(
            train_data_loader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "position_ids": batch[2],
                "labels": batch[3],
            }

            if args.model_type.find("span") != -1:
                inputs["mention_pos"] = batch[4]
            if args.use_full_layer != -1:
                inputs["full_attention_mask"] = batch[5]

            outputs = model(**inputs)
            loss = outputs[
                0
            ]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                raise Exception("Not supported")
                #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            logging_loss_epoch += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        raise Exception("Not supported")
                        #torch.nn.utils.clip_grad_norm_(
                        #    amp.master_params(optimizer), args.max_grad_norm
                        #)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                optimizer.step()
                if args.fp16:
                    raise Exception("Not supported")
                    #if (
                    #    amp._amp_state.loss_scalers[0]._unskipped != 0
                    #):  # assuming you are using a single optimizer
                    #    scheduler.step()
                else:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    # Log metrics
                    tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step,
                    )
                    logging_loss = tr_loss

                if (
                    args.local_rank in [-1, 0]
                    and args.eval_steps > 0
                    and global_step % args.eval_steps == 0
                ):
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        dev_file = Path(args.data_dir) / args.dev_file
                        results = evaluate(
                            logger, args, model, tokenizer, file_path=dev_file
                        )
                        ent_recall = results["r_overlap"]

                        if ent_recall >= best_result:
                            best_result = ent_recall
                            logger.info(
                                f"Best recall overlap:{best_result:.4f} in step:{global_step}"
                            )
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = "checkpoint"
                        output_dir = (
                            Path(args.output_dir) / f"{checkpoint_prefix}-{global_step}"
                        )
                        if not output_dir.exists():
                            output_dir.mkdir(parents=True, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(logger, args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        logger.info(f">>> lr of epoch{epoch_num}: {scheduler.get_last_lr()[0]:.4e}")
        dstep = global_step - logging_loss_steps_0
        avg_loss = logging_loss_epoch / dstep
        logger.info(f">>> Average loss of epoch{epoch_num}: {avg_loss:.4f}")

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_result

def load_data(args, tokenizer, logger):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_file = Path(args.data_dir) / args.train_file
    logger.info(f"Train file: {train_file.name}")
    train_dataset = ACEDatasetNER(
        logger=logger, tokenizer=tokenizer, file_path=train_file, args=args
    )
    train_sampler = (RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_data_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=1,
    )
    return train_sampler, train_data_loader

def setup_training(args, model, len_train, logger):
    """Train the model"""
    tb_writer = None
    if args.local_rank in [-1, 0]:
        # tb_writer = SummaryWriter("logs/ace_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])
        tb_writer = SummaryWriter(
            "logs/"
            + args.data_dir[max(args.data_dir.rfind("/"), 0) :]
            + "_ner_logs/"
            + args.output_dir[args.output_dir.rfind("/") :]
        )


    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len_train // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len_train
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    if args.eval_epochs > 0:
        args.eval_steps = (
            len_train // args.gradient_accumulation_steps * args.eval_epochs
        )
    else:
        args.eval_steps = args.save_steps


    # ---------for span encoder---------
    optimizer = get_span_optimizer(model.named_parameters(), args)
    num_warmup_steps = args.warmup_steps if args.warmup_steps != -1 else int(0.1 * t_total)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len_train)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Eval steps = %d", args.eval_steps)

    return model, optimizer, scheduler, tb_writer

def get_span_optimizer(model_named_parameters, args):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    is_span_model = args.learning_rate_span > 0
    params_span_no_decay = []
    params_span = []
    params_bert_no_decay = []
    params_bert = []
    params_default_no_decay = []
    params_default = []
    for n, p in model_named_parameters:
        is_no_decay = any(nd in n for nd in no_decay)
        is_span_encoder_param = "span_encoder" in n or "ner_classifier" in n
        if is_span_model and is_span_encoder_param and is_no_decay:
            params_span_no_decay.append(p)
        elif is_span_model and is_span_encoder_param:
            params_span.append(p)
        elif is_span_model and is_no_decay:
            params_bert_no_decay.append(p)
        elif is_span_model:
            params_bert.append(p)
        elif is_no_decay:
            params_default_no_decay.append(p)
        else:
            params_default.append(p)
    grouped_params = []
    if is_span_model:
        grouped_params = [
                dict(
                    params=params_span,
                    weight_decay=args.weight_decay,
                    lr=args.learning_rate),
                dict(
                    params=params_span_no_decay,
                    weight_decay=0.,
                    lr=args.learning_rate),
                dict(
                    params=params_bert,
                    weight_decay=args.weight_decay,
                    lr=args.learning_rate),
                dict(
                    params=params_bert_no_decay,
                    weight_decay=0.,
                    lr=args.learning_rate),
                ]
    else:
        grouped_params = [
                dict(
                    params=params_default,
                    weight_decay=args.weight_decay,
                    lr=args.learning_rate),
                dict(
                    params=params_default_no_decay,
                    weight_decay=0.,
                    lr=args.learning_rate),
                ]
    optimizer = AdamW(
            grouped_params, lr=args.learning_rate, eps=args.adam_epsilon)
    return optimizer



def main():
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    if (
        not args.overwrite_output_dir
        and output_dir.exists()
        and args.do_train
        and list(output_dir.iterdir())
    ):
        raise ValueError(
            f"Output directory ({output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # if args.do_train and args.local_rank in [-1, 0]:
    exp_path = create_exp_dir(
        args.output_dir,
        scripts_to_save=[
            "run_pruner.py",
            "transformers/src/transformers/modules.py",
            "transformers/src/transformers/modeling_bert.py",
            "transformers/src/transformers/modeling_albert.py",
        ],
    )

    logger = set_logger(args, exp_path, args.do_test)

    if not args.do_test:
        args_file = output_dir / "training_args.txt"
        with args_file.open("w") as json_file:
            json.dump(vars(args), json_file, indent=4)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logger.info(
        "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.data_dir.find("ace") != -1:
        num_labels = 8
    elif args.data_dir.find("scierc") != -1:
        num_labels = 7
    elif args.data_dir.find("ontonotes") != -1:
        num_labels = 19
    elif args.data_dir.find("gsap") != -1:
        num_labels = 11
    else:
        assert False

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.onedropout = args.onedropout
    config.use_full_layer = args.use_full_layer

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        args=args,
    )

    add_special_tokens(model, args, tokenizer, logger)


    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_result = 0
    # Training
    if args.do_train:
        global_step, tr_loss, best_result = train(logger, args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        output_dir = Path(args.output_dir)
        if not output_dir.exists() and args.local_rank in [-1, 0]:
            output_dir.mkdir(parents=True)

        update = True
        if args.evaluate_during_training:
            dev_file = output_dir / args.dev_file
            results = evaluate(logger, args, model, tokenizer, file_path=dev_file)
            # logger.info(f"Epoch: {epoch_num}, F1: {results['f1']}, recall: {results['recall']}")
            # logger.info(f"Epoch: {epoch_num}, ent_recall: {results['recall_score']}, R_overlap: {results['r_overlap']}")
            ent_recall = results["r_overlap"]
            if ent_recall >= best_result:
                best_result = ent_recall
                logger.info(f"Best recall overlap:{best_result:.4f}")
            else:
                update = False

        if update:
            checkpoint_prefix = "checkpoint"
            output_dir_checkpoint = output_dir / f"{checkpoint_prefix}-{global_step}"
            output_dir_checkpoint.mkdir(parents=True, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir_checkpoint)

            torch.save(args, output_dir_checkpoint / "training_args.bin")
            logger.info(f"Saving model checkpoint to {output_dir_checkpoint}")
            _rotate_checkpoints(logger, args, checkpoint_prefix)

        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, output_dir / "training_args.bin")

    # Evaluation test file
    # results = {'dev_best_result':best_result}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoints = [args.output_dir]

    #     WEIGHTS_NAME = 'pytorch_model.bin'

    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

    #     logger.info("Evaluate on test set")

    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1]

    #         model = model_class.from_pretrained(checkpoint, config=config, args=args)

    #         model.to(args.device)
    #         test_file = os.path.join(args.data_dir, args.test_file)

    #         result = evaluate(logger, args, model, tokenizer, file_path=test_file, prefix=global_step, do_test=not args.no_test)

    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)

    # if args.do_train and args.local_rank in [-1, 0]:
    #     output_eval_file = os.path.join(args.output_dir, "results.json")
    #     json.dump(results, open(output_eval_file, "w"))
    #     # logger.info("Result in checkpoint: %s", json.dumps(results))
    #     res = {k:f'{v:.4f}' for k,v in results.items()}
    #     logger.info(f"Test Results in last checkpoint: {res}")

    # -------------------------------------------------------------
    # test all files
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]

        WEIGHTS_NAME = "pytorch_model.bin"
        # pdb.set_trace()
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(f"{args.output_dir}/**/{WEIGHTS_NAME}", recursive=True)
                )
            )

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            output_test_file = Path(args.output_dir) / "test_results.txt"

            model = model_class.from_pretrained(checkpoint, config=config, args=args)
            model.to(args.device)

            results = {}
            for file_name in (
                args.train_file,
                args.dev_file,
                args.test_file,
            ):
                # for file_name in (args.test_file,):
                test_file = Path(args.data_dir) / file_name
                file_name = test_file.name
                if test_file.exists():
                    logger.info(f"Evaluate on {file_name}")
                    result = evaluate(
                        logger,
                        args,
                        model,
                        tokenizer,
                        file_path=test_file,
                        prefix=global_step,
                        do_test=True,
                    )
                    results[file_name] = result
                else:
                    logger.info(f"{test_file} does not exist!")
            with open(output_test_file, "w") as f:
                json.dump(results, f, indent=4)


def add_special_tokens(model, args, tokenizer, logger):
    # Add special tokens to tokenizer
    if args.model_type.startswith("albert"):
        special_tokens_dict = {
            "additional_special_tokens": [f"[unused{x}]" for x in range(4)]
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.lminit:
        # not roberta: BERT or ALBERTA (or SciBERT)
        if args.model_type.find("roberta") == -1: 
            entity_id = tokenizer.encode("entity", add_special_tokens=False)
            assert len(entity_id) == 1
            entity_id = entity_id[0]
            mask_id = tokenizer.encode("[MASK]", add_special_tokens=False)
            assert len(mask_id) == 1
            mask_id = mask_id[0]
        else: # Roberta: Hard Coded
            entity_id = 10014
            mask_id = 50264

        logger.info("entity_id: %d", entity_id)
        logger.info("mask_id: %d", mask_id)

        if args.model_type.startswith("albert"):
            word_embeddings = model.albert.embeddings.word_embeddings.weight.data
            word_embeddings[30000].copy_(word_embeddings[mask_id])
            word_embeddings[30001].copy_(word_embeddings[entity_id])
        elif args.model_type.startswith("roberta"):
            word_embeddings = model.roberta.embeddings.word_embeddings.weight.data
            word_embeddings[50261].copy_(word_embeddings[mask_id])  # entity
            word_embeddings[50262].data.copy_(word_embeddings[entity_id])
        else:
            word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            word_embeddings[1].copy_(word_embeddings[mask_id])
            word_embeddings[2].copy_(word_embeddings[entity_id])  # entity


def set_logger(args, log_path, test):
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)
    if test:
        log_file = f"test_{args.hostname}.log"
    else:
        log_file = f"all_{args.hostname}.log"
    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    # Setup logging ???
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    return logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(logger, args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(
        os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix))
    )
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append(
                    (int(regex_match.groups()[0]), path)
                )

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(
        0, len(checkpoints_sorted) - args.save_total_limit
    )
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            "Deleting older checkpoint [{}] due to args.save_total_limit".format(
                checkpoint
            )
        )
        shutil.rmtree(checkpoint)


def create_exp_dir(path, scripts_to_save=None):
    """
    Create path and copy scripts
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    print(f"Experiment dir : {path}")
    if scripts_to_save is not None:
        script_path = path / "scripts"
        if not script_path.exists():
            script_path.mkdir(parents=True, exist_ok=True)
        for script in scripts_to_save:
            dst_file = script_path / Path(script).name
            shutil.copyfile(script, dst_file)
    return path


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="ace_data",
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run test on the dev set."
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--learning_rate_span",
        default=-1,
        type=float,
        help="The initial learning rate for span encoder.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=-1, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=5, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--eval_epochs", type=int, default=-1, help="Save checkpoint every eval_epochs."
    )

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=1,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )

    parser.add_argument("--train_file", default="train.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)

    parser.add_argument("--alpha", type=float, default=1, help="")
    parser.add_argument("--max_pair_length", type=int, default=256, help="")
    parser.add_argument("--max_mention_ori_length", type=int, default=8, help="")
    parser.add_argument("--lminit", action="store_true")
    parser.add_argument("--norm_emb", action="store_true")
    parser.add_argument("--output_results", action="store_true")
    parser.add_argument("--onedropout", action="store_true")
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--use_full_layer", type=int, default=-1, help="")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--group_edge", action="store_true")
    parser.add_argument("--group_axis", type=int, default=-1, help="")
    parser.add_argument("--group_sort", action="store_true")

    # no cross-sentence
    parser.add_argument("--nocross", action="store_true")

    # for pruner
    parser.add_argument(
        "--topk_ratio",
        default=0.5,
        type=float,
        help="Topk ratio, candidate entity number divide sentence length.",
    )
    parser.add_argument(
        "--min_mentions_num",
        type=int,
        default=3,
        help="max mentions number feed in pruner",
    )
    parser.add_argument(
        "--max_mentions_num",
        type=int,
        default=18,
        help="min mentions number feed in pruner",
    )
    parser.add_argument(
        "--extra_repr", type=str, default=None, help="use extra span repr"
    )

    # for biaf span repr
    parser.add_argument("--biaf_span", action="store_true", help="use BiaffineSpanRepr")
    parser.add_argument(
        "--biaf_factorize", action="store_true", help="use BiaffineSpanRepr"
    )
    parser.add_argument("--biaf_mode", type=int, default=3, help="for BiaffineSpanRepr")
    parser.add_argument("--rank", type=int, default=768, help="for BiaffineSpanRepr")
    parser.add_argument(
        "--span_hidden_size", type=int, default=768, help="for BiaffineSpanRepr"
    )
    parser.add_argument(
        "--span_size", type=int, default=256, help="for BiaffineSpanRepr"
    )

    args = parser.parse_args()

    # get hostname
    args.hostname = socket.gethostname()

    args.neg_inf = -1e4 if args.fp16 else -1e30

    return args
if __name__ == "__main__":
    main()
