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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
from multiprocessing.sharedctypes import Value
import os
import random
from collections import defaultdict
import re
import shutil
import sys


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import time
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForACEBothOneDropoutSub,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  AlbertForACEBothOneDropoutSub,
                                  AlBertForBaselines,
                                  BertForACEBothOneDropoutSubNoNer,
                                  BertForBaselines,
                                #   BertForAttnHyperGNN,
                                  BertForHyperGNN,
                                #   BertForHyperGNNPlus,
                                #   BertForHyperGNNwithUnifyEntity,
                                AlbertForHyperGNN
                                  )
# from transformers.modeling_albert import 
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset
import json
import pickle
import numpy as np
import unicodedata
import itertools
import timeit

from utils.data import Dataset


from tqdm import tqdm
import socket 

import pdb

WEIGHTS_NAME = 'pytorch_model.bin'
# logger = logging.getLogger(__name__)
TRAIN_KEYS = ['input_ids', 'attention_mask', 'position_ids', 'sub_positions', 
                    'ner_labels','rel_labels', 'ent_numbers']
EVAL_KEYS = ['input_ids', 'attention_mask', 'position_ids', 'sub_positions',  'ent_numbers']

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  AlbertConfig)), ())

MODEL_CLASSES = {
    'baseline' : (BertConfig, BertForBaselines, BertTokenizer),
    'albertbaseline': (AlbertConfig, AlBertForBaselines, AlbertTokenizer),
    'hyper' : (BertConfig, BertForHyperGNN, BertTokenizer),
    'alberthyper' : (AlbertConfig, AlbertForHyperGNN, AlbertTokenizer)
    }

task_ner_labels = {
    'ace04': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'ace05': ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER'],
    'scierc': ['Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric'],
}

task_rel_labels = {
    'ace04': ['PER-SOC', 'OTHER-AFF', 'ART', 'GPE-AFF', 'EMP-ORG', 'PHYS'],
    'ace05': ['PER-SOC', 'ART', 'ORG-AFF', 'GEN-AFF', 'PHYS', 'PART-WHOLE'],
    'scierc': ['PART-OF', 'USED-FOR', 'FEATURE-OF', 'CONJUNCTION', 'EVALUATE-FOR', 'HYPONYM-OF', 'COMPARE'],
}


def set_logger(args, log_path, test):

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)
    if test:
        log_file = f'test_{args.hostname}.log'
    else:
        log_file = f'all_{args.hostname}.log'
    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_saved_checkpoint(args, checkpoint_prefix):
    """search for saved checkpoint fold"""
    checkpoints = os.listdir(args.output_dir)
    global_steps = []
    for checkpoint_fold in checkpoints:
        if checkpoint_fold.startswith(checkpoint_prefix):
            global_step = int(checkpoint_fold.split('-')[-1])
            global_steps.append(global_step)
    if len(global_steps)>0:
        max_global_step = max(global_steps)
        saved_checkpoint = os.path.join(args.output_dir, f'{checkpoint_prefix}-{max_global_step}')
    else:
        saved_checkpoint = None
        max_global_step = 0
    return saved_checkpoint, max_global_step
 

    
def _rotate_checkpoints(logger, args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(logger, args, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter("logs/"+args.ner_prediction_dir[max(args.ner_prediction_dir.rfind('/'),0):]+"_re_logs/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_file = os.path.join(args.ner_prediction_dir, args.train_file)
    # train_dataset = ACEDataset(logger=logger, tokenizer=tokenizer, file_path=train_file, args=args, max_pair_length=args.max_pair_length)
    train_dataset = Dataset(logger=logger, tokenizer=tokenizer, file_path=train_file, args=args, max_pair_length=args.max_pair_length)
    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.train_batch_size, num_workers=4, collate_fn=list)
    train_dataloader = train_dataset.build(batch_size=args.train_batch_size, shuffle=args.shuffle, n_workers=0, pin_memory=True)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.eval_epochs > 0:
        eval_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.eval_epochs
    else:
        eval_steps = args.save_steps



    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    #---------for span encoder---------
    if args.learning_rate_cls>0:
        opt_grouped_bert = [{'params': [], 'weight_decay': args.weight_decay, 'lr':args.learning_rate}, {'params':[], 'weight_decay': 0.0, 'lr':args.learning_rate}]
        opt_grouped_cls = [{'params': [], 'weight_decay': args.weight_decay, 'lr':args.learning_rate_cls}, {'params':[], 'weight_decay': 0.0, 'lr':args.learning_rate_cls}]
        for n, p in model.named_parameters():
            if 'bert' not in n:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_cls[0]['params'].append(p)
                else:
                    opt_grouped_cls[1]['params'].append(p)
            else:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_bert[0]['params'].append(p)
                else:
                    opt_grouped_bert[1]['params'].append(p)
        optimizer = AdamW(opt_grouped_bert+opt_grouped_cls, eps=args.adam_epsilon)
    else:
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    
    

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_ratio*t_total), num_training_steps=t_total
        )

    if args.continue_training:
        train_states_checkpoint = torch.load(os.path.join(args.model_path, 'train_states.bin'))
        past_epoch = train_states_checkpoint['epoch']
        optimizer.load_state_dict(train_states_checkpoint['optimizer']) 
        scheduler.load_state_dict(train_states_checkpoint['scheduler'])
    else:
        past_epoch = -1
    start_epoch = past_epoch + 1

    # ori_model = model
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Eval steps = %d", eval_steps)


    def _save_model(args, model, logger, global_step, current_epoch):
        checkpoint_prefix = 'checkpoint'
        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

        model_to_save.save_pretrained(output_dir)
        # for continue training
        train_states_checkpoint = {'epoch': current_epoch, 
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(),
                    'args':args}
        train_status_name = os.path.join(output_dir, 'train_states.bin')
        torch.save(train_states_checkpoint, train_status_name)

        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

        _rotate_checkpoints(logger, args, checkpoint_prefix)
        args_file = os.path.join(output_dir, 'training_args.txt')
        with open(args_file, 'w') as f:
            f.write(str(vars(args)))

        return



    def _eval_and_save(args, model, logger, best_f1, best_result, epoch_num, global_step, current_epoch):
        if args.local_rank in [-1, 0]:
            update = False
            # Save model checkpoint
            if args.evaluate_during_training and ((epoch_num+1) % args.eval_epochs == 0 or epoch_num+1==args.num_train_epochs):  # Only evaluate when single GPU otherwise metrics may not average well
                dev_file = os.path.join(args.ner_prediction_dir, args.dev_file)
                logger.info(f'evaluate dev file.')
                results = evaluate(logger, args, model, tokenizer, file_path=dev_file)
                f1 = results['f1_with_ner']
                tb_writer.add_scalar('f1_with_ner', f1, global_step)

                if f1 > best_f1:
                    best_f1 = f1
                    best_result = results
                    print ('Best F1', best_f1)
                    update = True
                else:
                    update = False

            if update:
                _save_model(args, model, logger, global_step, current_epoch)
            else:
                checkpoint_prefix = 'checkpoint'
                if epoch_num+1==args.num_train_epochs and len(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True))==0:
                        _save_model(args, model, logger, global_step, current_epoch)

                # checkpoint_prefix = 'checkpoint'
                # output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                # if not os.path.exists(output_dir):
                #     os.makedirs(output_dir)
                # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                # model_to_save.save_pretrained(output_dir)

                # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                # logger.info("Saving model checkpoint to %s", output_dir)

                # _rotate_checkpoints(logger, args, checkpoint_prefix)
                # args_file = os.path.join(output_dir, 'training_args.txt')
                # with open(args_file, 'w') as f:
                #     f.write(str(vars(args)))

        return best_f1, best_result

    

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs-start_epoch), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
   
    global_step = args.global_step
    tr_loss, logging_loss = 0.0, 0.0
    tr_ner_loss, logging_ner_loss = 0.0, 0.0
    tr_re_loss, logging_re_loss = 0.0, 0.0

    best_f1 = -1
    best_result = None
    for epoch_num in train_iterator:
        epoch_num += start_epoch
        logger.info(f">>> Epoch {epoch_num} starts.")
        logging_loss_steps, logging_reloss, logging_nerloss = global_step, 0.0, 0.0

        # if args.shuffle and _ > 0:
        #     train_dataset.initialize()
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
         
        for step, batch in enumerate(epoch_iterator):
        # for step, batch in enumerate(train_dataloader):

            model.train()
                 
            inputs = {}
            input_keys = TRAIN_KEYS
            
            for k,v in batch.items():
                if k in input_keys:
                    v = v.to(args.device)
                    inputs[k] = v
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            re_loss = outputs[1]
            ner_loss = outputs[2]

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                re_loss = re_loss / args.gradient_accumulation_steps
                ner_loss = ner_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # t3 = timeit.default_timer()
            # logger.info(f"time for loss backward: {t3-t2}s")

            tr_loss += loss.item()
            if re_loss > 0:
                tr_re_loss += re_loss.item()
            if ner_loss > 0:
                tr_ner_loss += ner_loss.item()

            logging_reloss += re_loss.item()
            logging_nerloss += ner_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if args.fp16:
                    if amp._amp_state.loss_scalers[0]._unskipped != 0: # assuming you are using a single optimizer
                        scheduler.step()
                else:
                    scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                    tb_writer.add_scalar('RE_loss', (tr_re_loss - logging_re_loss)/args.logging_steps, global_step)
                    logging_re_loss = tr_re_loss

                    tb_writer.add_scalar('NER_loss', (tr_ner_loss - logging_ner_loss)/args.logging_steps, global_step)
                    logging_ner_loss = tr_ner_loss


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            
        best_f1, best_result = _eval_and_save(args, model, logger, best_f1, best_result, epoch_num, global_step, epoch_num)


        logger.info(f'>>> current global steps: {global_step}')
        logger.info(f'>>> lr of epoch {epoch_num}: {scheduler.get_last_lr()[0]:.4e}')
        dstep = global_step - logging_loss_steps
        avg_reloss = logging_reloss/dstep
        avg_nerloss = logging_nerloss/dstep
        logger.info(f'>>> Average loss of epoch{epoch_num}: ner_{avg_nerloss:.6f}, re_{avg_reloss:.6f}')

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_f1, best_result


def to_list(tensor):
    return tensor.detach().cpu().tolist()

def get_gold_ner_with_nolabel(ner_golden_labels):
    ner_golden_nolabels = set()
    for ner in ner_golden_labels:
        ner_nolabel = (ner[0], ner[1])
        ner_golden_nolabels.add(ner_nolabel)
    return ner_golden_nolabels


def evaluate(logger, args, model, tokenizer, file_path, prefix="", do_test=False):

    eval_output_dir = args.output_dir

    
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)   

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()

    eval_dataset = Dataset(logger=logger, tokenizer=tokenizer, file_path=file_path, args=args, evaluate=True, max_pair_length=args.max_pair_length)
    eval_dataloader = eval_dataset.build(batch_size=args.eval_batch_size, shuffle=args.shuffle, n_workers=0, pin_memory=True)
    
    # eval_sampler = SequentialSampler(eval_dataset) 
    # eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.eval_batch_size,  collate_fn=list, num_workers=4)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))

    start_time = timeit.default_timer() 
      

    #---------------------------------------------------            

    
    scores = defaultdict(dict)
    pred_ners = defaultdict(dict)
    ner_predictions = defaultdict(set)

    with torch.no_grad():
        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in tqdm(eval_dataloader, desc="Evaluating"):

            indexs = batch['indexs']
            obj_mentions = batch['obj_token_pos']
            subs = batch['sub']

            rel_labels = batch['rel_labels']
            ner_labels = batch['ner_labels']
            ent_numbers = batch['ent_numbers']
            inputs = {}
            input_keys = EVAL_KEYS

            for k,v in batch.items():
                if k in input_keys:
                    v = v.to(args.device)
                    inputs[k] = v
           
            outputs = model(**inputs)

            rel_logits = outputs[0]             # bs * n_ent * n_ent * num_rel_labels
            ner_logits = outputs[1] 
                    
            rel_logits = torch.nn.functional.log_softmax(rel_logits, dim=-1)

            ner_preds = torch.argmax(ner_logits, dim=-1)            
            rel_logits = rel_logits.cpu().numpy()                   # for plmk, n_ent_total * max_n_ent * num_rel_labels
            ner_preds = ner_preds.cpu().numpy()                     # for plmk, n_ent_total * num_ner_labels
            # print(f'indexs:{indexs}')
            # print(f'ner_labels: {ner_labels}')
            # if args.baseline not in {'firstorder', 'mfvi', 'gnn'}:
            #     rel_logits_split = torch.split(rel_logits, ent_numbers)
            #     rel_logits = pad_sequence(rel_logits_split, batch_first=True, padding_value=0)

            for i, index in enumerate(indexs):
                n_ent = ent_numbers[i]
                
                index = tuple(index)
                obj_mentions_i = obj_mentions[i]
                if n_ent>0:
                    for j, obj_mention in enumerate(obj_mentions_i):
                        obj_mention = tuple(obj_mention)
                        ner_label = eval_dataset.ner_label_list[ner_preds[i, j]] 
                        if ner_label != 'NIL':
                            pred_ners[index][obj_mention] = ner_label
                            ner_predictions[index].add(obj_mention + (ner_label,))
                else:
                    pred_ners[index] = {}
                    ner_predictions[index] = {}

  
            for i, index in enumerate(indexs):
                n_ent = ent_numbers[i]
                index = tuple(index)
                if n_ent>0:
                    for j, sub in enumerate(subs[i]):                   # obj tokens, e.g.: [(2, 3), (3, 4), (6, 6), (6, 7), (10, 10), (10, 11), (13, 14)] 
                        sub_mention = (sub[0], sub[1])
                        for k, obj_mention in enumerate(obj_mentions[i]):
                            obj_mention = tuple(obj_mention)
                            rel_scores = rel_logits[i, j, k].tolist()
                            scores[index][(sub_mention, obj_mention)] = rel_scores
                else:
                    scores[index] = {}
    #---------------------------------------------------
    # decode
    global_predicted_ners = eval_dataset.global_predicted_ners
    gold_rels = set(eval_dataset.golden_labels)
    gold_rels_with_ner = set(eval_dataset.golden_labels_withner) 
    gold_ners = eval_dataset.ner_golden_labels

    label_list = list(eval_dataset.label_list)
    sym_labels = list(eval_dataset.sym_labels)
    num_label = len(label_list)
    n_syms = len(sym_labels)
    n_unsyms = num_label - n_syms

    tot_recall = eval_dataset.tot_recall
    n_pred_ner = 0
    n_tp_ner = 0
    n_pred_rel = 0
    n_tp_rel = 0
    n_tp_rel_with_ner = 0

    tot_predicted_relations = {}
    tot_predicted_ners = {}

    for example_index, scores_i in sorted(scores.items(), key=lambda x:x):  # example_index: (doc_id, sent_id), pair_dict: {((sub_h, sub_t), (obj_h, obj_t)):([scores], predicted_ner_label)

        if len(scores_i)>0:
            visited  = set([])
            sentence_results = []               # predictions
            # print(f'processing index:{example_index}')
            for relation, rel_scores in scores_i.items():
                sub_mention, obj_mention = relation
                if relation in visited:
                    continue
                    
                if sub_mention==obj_mention:
                    continue
                
                sub_label = pred_ners[example_index].get(sub_mention, 'NIL')
                obj_label = pred_ners[example_index].get(obj_mention, 'NIL')

                counter_relation = (obj_mention, sub_mention)

                visited.add(relation)
                visited.add(counter_relation)

                rel_scores_c = scores_i.get(counter_relation, None)
                assert rel_scores_c is not None, pdb.set_trace()
                # if m2a_ner_label=='NIL' or m1a_ner_label=='NIL':
                #     continue

                rel_scores = np.array(rel_scores)
                rel_scores_c = np.array(rel_scores_c)

                rel_scores_c = np.concatenate([rel_scores_c[:n_syms], rel_scores_c[num_label:], rel_scores_c[n_syms:num_label]])
                rel_scores = rel_scores + rel_scores_c



                pred_rel_label = np.argmax(rel_scores)
                pred_score = rel_scores[pred_rel_label]
                if 0< pred_rel_label < num_label:   # relation1 exists
                    pred_rel_label = label_list[pred_rel_label]
                    sentence_results.append((pred_score, sub_mention, obj_mention, pred_rel_label, sub_label, obj_label))
                elif pred_rel_label >= num_label:                               # pred_rel_label >= num_label:
                    pred_rel_label = pred_rel_label - n_unsyms
                    pred_rel_label = label_list[pred_rel_label]
                    sentence_results.append((pred_score, obj_mention, sub_mention, pred_rel_label, obj_label, sub_label))


            sentence_results.sort(key=lambda x: -x[0]) 

            rel_predictions = set()
            rel_with_ner_predictions = set()

            output_pred_rels = []
            output_pred_ners = []

            for item in sentence_results:
                _, sub_mention, obj_mention, rel_label, sub_label, obj_label = item
                rel_predictions.add((example_index, sub_mention, obj_mention, rel_label))
                rel_with_ner_predictions.add((example_index, (sub_mention[0], sub_mention[1], sub_label), (obj_mention[0], obj_mention[1], obj_label), rel_label))
                
                output_pred_rels.append((sub_mention[0], sub_mention[1], obj_mention[0], obj_mention[1], rel_label))

            output_pred_ners = list(ner_predictions[example_index])
            pred_ners_cur = set()
            for pred_ner in output_pred_ners:
                pred_ners_cur.add((example_index, (pred_ner[0], pred_ner[1]), pred_ner[2]))

            
            tp_rel = rel_predictions & gold_rels
            tp_rel_with_ner = rel_with_ner_predictions & gold_rels_with_ner
            tp_ner = pred_ners_cur & gold_ners
            
            n_pred_rel_i = len(sentence_results)
            n_tp_rel_i = len(tp_rel)
            n_tp_rel_with_ner_i = len(tp_rel_with_ner)
            n_pred_ner_i = len(pred_ners_cur)
            n_tp_ner_i = len(tp_ner)

            n_pred_ner += n_pred_ner_i
            n_tp_ner += n_tp_ner_i
            n_pred_rel += n_pred_rel_i
            n_tp_rel += n_tp_rel_i
            n_tp_rel_with_ner += n_tp_rel_with_ner_i

            
        else:
            output_pred_rels = []
            output_pred_ners = []

        if do_test:
            tot_predicted_relations[example_index] = output_pred_rels
            tot_predicted_ners[example_index] = output_pred_ners

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(global_predicted_ners) / evalTime)

    

    ner_p = n_tp_ner / n_pred_ner if n_pred_ner > 0 else 0 
    ner_r = n_tp_ner / len(gold_ners) 
    ner_f1 = 2 * (ner_p * ner_r) / (ner_p + ner_r) if n_tp_ner > 0 else 0.0

    p = n_tp_rel / n_pred_rel if n_pred_rel > 0 else 0 
    r = n_tp_rel / tot_recall 
    f1 = 2 * (p * r) / (p + r) if n_tp_rel > 0 else 0.0
    
    # assert(tot_recall==len(golden_labels))

    p_with_ner = n_tp_rel_with_ner / n_pred_rel if n_pred_rel > 0 else 0 
    r_with_ner = n_tp_rel_with_ner / tot_recall
    # assert(tot_recall==len(golden_labels_withner))
    f1_with_ner = 2 * (p_with_ner * r_with_ner) / (p_with_ner + r_with_ner) if n_tp_rel_with_ner > 0 else 0.0
    
    results = {'ner_p': ner_p, 'ner_r': ner_r, 'ner_f1': ner_f1, 'p':p, 'r':r, 'f1':  f1,  'p+':p_with_ner, 'r+':r_with_ner, 'f1_with_ner': f1_with_ner}

    log_results = f"ner_p:{ner_p:.4f}, ner_r:{ner_r:.4f}, ner_f1:{ner_f1:.4f}; rel_p:{p:.4f}, rel_r:{r:.4f}, rel_f1:{f1:.4f}; rel_p+:{p_with_ner:.4f}, rel_r+:{r_with_ner:.4f}, rel_f1+:{f1_with_ner:.4f}"

    # logger.info("Result: %s", json.dumps(results))
    logger.info(f"Result:{log_results}")

    # dump predictions
    if do_test:
        file_name = os.path.split(file_path)[-1]
        dump_name = 'pred_' + file_name

        f = open(eval_dataset.file_path) 
        output_w = open(os.path.join(args.output_dir, dump_name), 'w')  
        for l_idx, line in enumerate(f):
            data = json.loads(line)
            num_sents = len(data['sentences'])
            predicted_ner = []
            predicted_rel = []
            for n in range(num_sents):
                ner_item = tot_predicted_ners.get((l_idx, n), [])
                ner_item.sort()
                predicted_ner.append(ner_item)
                rel_item = tot_predicted_relations.get((l_idx, n), [])
                rel_item.sort()
                predicted_rel.append(rel_item)
            data['predicted_ner'] = predicted_ner
            data['predicted_rel'] = predicted_rel
            # pdb.set_trace()
            output_w.write(json.dumps(data)+'\n')
            # json.dump(tot_output_results, output_w)


    return results



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")


    parser.add_argument("--ner_prediction_dir", default='', type=str, required=True,
                        help="NER prediction dir. Should contain the .json files (or other data files) for the task.")




    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="want to test")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--learning_rate_cls", default=-1, type=float,
                        help="The initial learning rate for layers beyond bert.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=5,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument('--eval_epochs', type=int, default=-1,
                        help="Save checkpoint every eval_scale*total_steps.")


    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--save_total_limit', type=int, default=1,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')

    parser.add_argument("--train_file",  default="train.json", type=str)
    parser.add_argument("--dev_file",  default="dev.json", type=str)
    parser.add_argument("--test_file",  default="test.json", type=str)
    parser.add_argument('--max_pair_length', type=int, default=64,  help="")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--eval_logsoftmax', action='store_true')
    parser.add_argument('--eval_softmax', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--no_sym', action='store_true')
    parser.add_argument('--att_left', action='store_true')
    parser.add_argument('--att_right', action='store_true')
    parser.add_argument('--use_ner_results', action='store_true')
    parser.add_argument('--use_typemarker', action='store_true')
    parser.add_argument('--eval_unidirect', action='store_true')

    # no cross-sentence
    parser.add_argument('--nocross', action='store_true')
    # fix evaluation

    # change warmup
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")
    # # add for continue training
    # parser.add_argument("--load_save", action='store_true', help="load lastest saved checkpoint")

    # parser.add_argument("--use_bucket_data", action='store_true', help="sent with similar ent number in a bucket")
    # exp decoding
    parser.add_argument('--eval_logits', action='store_true', help='decoding with non-normalized logits')

    # add loss scale
    # parser.add_argument('--rel_loss_scale', type=float, default=-1.0, help="for BiaffineRelationCls")
    
    # encoder
    parser.add_argument('--ent_repr', type=str, default='mix', help="option: sub, obj, mix. choose the source of entity representations")
    parser.add_argument('--uni_ent', action='store_true', help="if True, sub/obj use the same repr from bert; else bert encode sub/obj respectively")
    parser.add_argument('--ent_enc', type=str, default='cat', help="entity encoder")
    parser.add_argument('--pred_sub', action='store_true', help="")
    parser.add_argument('--ner_cls', type=str, default='cat', help="")

    parser.add_argument('--rel_enc', type=str, default='cat', help="entity encoder")
    parser.add_argument('--ent_dim', type=int, default=200, help="for BiaffineRelationCls")
    parser.add_argument('--rel_dim', type=int, default=200, help="for BiaffineRelationCls")
    parser.add_argument('--rel_rank', type=int, default=200, help="for BiaffineRelationCls")
    parser.add_argument('--rel_factorize', action='store_true', help="use BiaffineRelationCls")
    # parser.add_argument('--rel_cls', type=str, default='biaf', help="")

    parser.add_argument('--baseline', type=str, default='firstorder', help="")

    
    # HyperGNN
    # parser.add_argument('--unirel', action='store_true', help="if true, treat two relations in hyperedge as unity node")
    # parser.add_argument('--noprojentrel', action='store_true', help="")
    parser.add_argument('--factor_type', type=str, default='ternary', help="")
    parser.add_argument('--mem_dim', type=int, default=200, help="for BiaffineRelationCls")
    parser.add_argument('--iter', type=int, default=3, help="for BiaffineRelationCls")
    parser.add_argument('--layernorm', action='store_true', help="")
    parser.add_argument('--layernorm_1st', action='store_true', help="layernorm for first order")
    # parser.add_argument('--attn_type', type=str, default='attn', help="")
    # parser.add_argument('--use_graph_output_number', type=int, default=1, help="for BiaffineRelationCls")
    # debug
    parser.add_argument('--attn_self', action='store_true', help="")
    # parser.add_argument('--ter_rel_aggregator', type=str, default='cat', help="")
    parser.add_argument('--aggregate_type', type=str, default='attn', help="attn or test")
    parser.add_argument('--aggregate_func', type=str, default='max', help="max or sum")
    parser.add_argument('--agg_with_self', action='store_true', help="")
    parser.add_argument('--fix_obj', action='store_true', help="")


    # HTNN
    # parser.add_argument('--sep_compose', action='store_true', help="")
    
    parser.add_argument('--edgetype', type=str, default='sib', help="")

    # AttnHTNN composer
    parser.add_argument('--attn_scorer', type=str, default='biaf', help="")
    parser.add_argument('--attn_res', action='store_true', help="")


    # Attn GNN

    parser.add_argument('--n_head', type=int, default=8, help="for BiaffineRelationCls")
    parser.add_argument('--d_head', type=int, default=32, help="for BiaffineRelationCls")

    # Factor graph
    parser.add_argument('--factor_encoder', type=str, default='cat', help="entity encoder")

    # HyperGNN plus
    parser.add_argument('--iter1', type=int, default=1, help="for BiaffineRelationCls")
    # parser.add_argument('--edge2nodefunc', action='store_true', help="")

    args = parser.parse_args()

    # get hostname
    args.hostname = socket.gethostname()


    def save_args(args, path, filename='training_args.txt'):
        if not os.path.exists(path):
            os.makedirs(path)
        args_file = os.path.join(path, filename)
        with open(args_file, 'w') as f:
            f.write(str(vars(args)))

    def create_exp_dir(path, scripts_to_save=None):
        if args.output_dir.endswith("test"):
            return
        
        if not os.path.exists(path):
            os.makedirs(path)
        

        print('Experiment dir : {}'.format(path))
        if scripts_to_save is not None:
            if not os.path.exists(os.path.join(path, 'scripts')):
                os.mkdir(os.path.join(path, 'scripts'))
            for script in scripts_to_save:
                dst_file = os.path.join(path, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        return path

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        exp_path = args.output_dir
        logger = set_logger(args, exp_path, args.do_test)
        # raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
        logger.warning(f'Output directory ({args.output_dir}) already exists and is not empty. It will continue training or use --overwrite_output_dir to overcome.')
    else:
        # if args.do_train and args.local_rank in [-1, 0]:
        exp_path = create_exp_dir(args.output_dir, scripts_to_save=[os.path.basename(__file__), 'utils/data.py','transformers/src/transformers/modules.py', 'transformers/src/transformers/modeling_bert.py', 'transformers/src/transformers/modeling_albert.py'])

        logger = set_logger(args, exp_path, args.do_test)

    if not args.do_test:
        save_args(args, os.path.join(exp_path, 'args'), 'training_args.txt')
    else:
        save_args(args, os.path.join(exp_path, 'args'), 'test_args.txt')

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    if args.ner_prediction_dir.find('ace')!=-1:
        num_ner_labels = 8

        if args.no_sym:
            num_labels = 7 + 7 - 1
        else:
            num_labels = 7 + 7 - 2
    elif args.ner_prediction_dir.find('scierc')!=-1:
        num_ner_labels = 7

        if args.no_sym:
            num_labels = 8 + 8 - 1
        else:
            num_labels = 8 + 8 - 3
    else:
        assert (False)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # for continue training
    if args.do_train and not args.overwrite_output_dir:
        saved_checkpoint, global_step = get_saved_checkpoint(args, 'checkpoint')
        args.model_path = saved_checkpoint if saved_checkpoint else args.model_name_or_path
        args.continue_training = True
    else:
        args.model_path = args.model_name_or_path
        global_step = 0
        args.continue_training = False
    args.global_step = global_step
    
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,  do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.num_ner_labels = num_ner_labels

    model = model_class.from_pretrained(args.model_path, from_tf=bool('.ckpt' in args.model_path), config=config, args=args)


    if args.model_type.startswith('albert'):
        if args.use_typemarker:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(num_ner_labels*4+2)]}
        else:
            special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train:
        subject_id = tokenizer.encode('subject', add_special_tokens=False)
        assert(len(subject_id)==1)
        subject_id = subject_id[0]
        object_id = tokenizer.encode('object', add_special_tokens=False)
        assert(len(object_id)==1)
        object_id = object_id[0]

        mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
        assert(len(mask_id)==1)
        mask_id = mask_id[0]

        logger.info(" subject_id = %s, object_id = %s, mask_id = %s", subject_id, object_id, mask_id)

        if args.lminit: 
            if args.model_type.startswith('albert'):
                word_embeddings = model.albert.embeddings.word_embeddings.weight.data
                subs = 30000
                sube = 30001
                objs = 30002
                obje = 30003
            else:
                word_embeddings = model.bert.embeddings.word_embeddings.weight.data
                subs = 1
                sube = 2
                objs = 3
                obje = 4

            word_embeddings[subs].copy_(word_embeddings[mask_id])     
            word_embeddings[sube].copy_(word_embeddings[subject_id])   

            word_embeddings[objs].copy_(word_embeddings[mask_id])      
            word_embeddings[obje].copy_(word_embeddings[object_id])     

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_f1 = 0
    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args,  tokenizer, evaluate=False)
        global_step, tr_loss, best_f1, best_result = train(logger, args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Evaluation
    results = {'dev_best_f1': best_f1}
    if args.do_eval and args.local_rank in [-1, 0]:

        checkpoints = [args.output_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        #----------test on all dataset----------------------
        logger.info("==========Evaluate the following checkpoints: %s", checkpoints)
        dumps = []
        if best_result is not None:
            dumps.append(f'--------------------------')
            dumps.append(f'Best result on dev {args.dev_file}')
            dumps.append(str(best_result))
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1]
            output_test_file = os.path.join(args.output_dir, "test_results.txt")
            model = model_class.from_pretrained(checkpoint, config=config, args=args)
            model.to(args.device)

            file_names  = (args.train_file, args.test_file)
            for file_name in file_names:
                test_file = os.path.join(args.ner_prediction_dir, file_name)
                if os.path.exists(test_file):
                    logger.info(f"Evaluate on {test_file}")
                    result = evaluate(logger, args, model, tokenizer, file_path=test_file, prefix=global_step, do_test=(file_name==args.test_file))
                    dumps.append(f'--------------------------')
                    dumps.append(f'Test result of {test_file}')
                    dumps.append(str(result))
            dumps = '\n'.join(dumps)
            with open(output_test_file, 'w') as f:
                f.write(dumps)



    # #---------FOR TESTING------------
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]

        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1]
            output_test_file = os.path.join(args.output_dir, f"test_results_{global_step}.txt")
            model = model_class.from_pretrained(checkpoint, config=config, args=args)
            model.to(args.device)

            dumps = []
            for file_name in (args.train_file, args.dev_file, args.test_file):
                test_file = os.path.join(args.ner_prediction_dir, file_name)
                if os.path.exists(test_file):
                    logger.info(f"Evaluate on {test_file}")
                    result = evaluate(logger, args, model, tokenizer, file_path=test_file, prefix=global_step, do_test=(file_name==args.test_file))
                    dumps.append(f'--------------------------')
                    dumps.append(f'Test result of {test_file}')
                    dumps.append(str(result))
            dumps = '\n'.join(dumps)
            with open(output_test_file, 'w') as f:
                f.write(dumps)

if __name__ == "__main__":
    main()


