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
import os
import random
from collections import defaultdict
import re
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaTokenizer,
                                  get_linear_schedule_with_warmup,
                                  AdamW,
                                  BertForNER,
                                  BertForSpanNER,
                                  BertForSpanMarkerNER,
                                  BertForSpanMarkerBiNER,
                                  BertForSpanMarkerNerPruner,
                                  AlbertForNER,
                                  AlbertConfig,
                                  AlbertTokenizer,
                                  AlbertForSpanMarkerNerPruner,
                                  BertForLeftLMNER,
                                  RobertaForNER,
                                  RobertaForSpanNER,
                                  RobertaForSpanMarkerNER,
                                  AlbertForSpanNER,
                                  AlbertForSpanMarkerNER,
                                  )

from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import pickle
import numpy as np
import unicodedata
import itertools
import math
from tqdm import tqdm
import re
import timeit
import socket 
import pdb

# Logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,  RobertaConfig)), ())

MODEL_CLASSES = {
    'bertspanmarkerpruner': (BertConfig, BertForSpanMarkerNerPruner, BertTokenizer),
    'albertspanmarkerpruner': (AlbertConfig, AlbertForSpanMarkerNerPruner, AlbertTokenizer),
}






# NEG_INF = -1e30

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

class ACEDatasetNER(Dataset):
    def __init__(self, logger, tokenizer, file_path=None, args=None, evaluate=False):
        # if not evaluate:
        #     file_path = os.path.join(args.data_dir, args.train_file)
        # else:
        #     if do_test:
        #         file_path = os.path.join(args.data_dir, args.test_file)
        #     else:
        #         file_path = os.path.join(args.data_dir, args.dev_file)
        
        
        self.logger = logger

        assert os.path.isfile(file_path)

        self.file_path = file_path
                
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        self.evaluate = evaluate
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        if args.data_dir.find('ace')!=-1:
            self.ner_label_list = ['NIL', 'FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
        elif args.data_dir.find('scierc')!=-1:
            self.ner_label_list = ['NIL', 'Method', 'OtherScientificTerm', 'Task', 'Generic', 'Material', 'Metric']
        else:
            self.ner_label_list = ['NIL', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

        self.max_pair_length = args.max_pair_length

        self.max_entity_length = args.max_pair_length * 2
        self.initialize()

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False 

    def get_original_token(self, token):
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in escape_to_original:
            token = escape_to_original[token]
        return token
        
    def initialize(self):
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 2

        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}               # e.g.: {'NIL': 0, 'Method': 1, 'OtherScientificTerm': 2, 'Task': 3, 'Generic': 4, 'Material': 5, 'Metric': 6}


        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        f = open(self.file_path, "r", encoding='utf-8')
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        maxL = 0
        maxR = 0
        for l_idx, line in enumerate(f):
            data = json.loads(line)
            # if len(self.data) > 5:
            #     break

            # if self.args.output_dir.find('test')!=-1:
            #     if len(self.data) > 5:
            #         break

            sentences = data['sentences']
            # doc_key = data['doc_key']
            # doc_id = self.doc2id[doc_key]

            for i in range(len(sentences)):
                for j in range(len(sentences[i])):
                    sentences[i][j] = self.get_original_token(sentences[i][j])
            
            ners = data['ner']                                          # ners: e.g. [[[4, 7, 'Task'], [13, 14, 'Method'], [17, 18, 'Material'], [22, 23, 'Material'], [34, 36, 'Method'], [38, 39, 'Method']], [[47, 48, 'Material']]]

            sentence_boundaries = [0]                                   # store the token numbers of sentences: e.g. [0, 41, 70], indicate the boudaries.
            words = []
            L = 0
            sent_lens = []
            for i in range(len(sentences)):
                L += len(sentences[i])
                sent_lens.append(len(sentences[i]))
                sentence_boundaries.append(L)   
                words += sentences[i]

            tokens = [tokenize_word(w) for w in words]
            subwords = [w for li in tokens for w in li]
            maxL = max(len(tokens), maxL)
            subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))
            token2subword = [0] + list(itertools.accumulate(len(li) for li in tokens))                          # e.g.: [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 24, 25, 26, 27, 28, 30, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 79, 80, 82, 83, 84, 85, 86]
            subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [sum(len(li) for li in tokens[:p]) for p in sentence_boundaries]      # store the subtoken numbers of sentences.

            for n in range(len(subword_sentence_boundaries) - 1):                                               # nth sentence
                sent_len = sent_lens[n]
                # sent_id = self.sent_ids[doc_id][n]
                
                sentence_ners = ners[n]
                self.tot_recall += len(sentence_ners)
                entity_labels = {}
                for start, end, label in sentence_ners:                                                         # start: start token idx; end: end token idx; 
                    entity_labels[(token2subword[start], token2subword[end+1])] = ner_label_map[label]    
                    self.ner_golden_labels.add( ((l_idx, n), (start, end), label) )

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[n : n + 2]

                left_length = doc_sent_start                                                                    # left_length | current sentence length | right_length; left_length: subtoken numbers before current sentence, right_length: subtoken numbers after current sentence.
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start

                # could half_context_length < 0?
                half_context_length = int((max_num_subwords - sentence_length) / 2)                             # 设置的最大subtokens长度允许在当前句子左或右能包含的context subtoken个数。         half_context_length | current sentence length | half_context_length

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(right_length, max_num_subwords - left_context_length - sentence_length)
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(left_length, max_num_subwords - right_context_length - sentence_length)
                if self.args.output_dir.find('ctx0')!=-1:
                    left_context_length = right_context_length = 0 # for debug
                if self.args.nocross:
                    left_context_length = right_context_length = 0
                # pdb.set_trace()
                doc_offset = doc_sent_start - left_context_length                                               # start idx of contexted sequence.
                target_tokens = subwords[doc_offset : doc_sent_end + right_context_length]                      # if max_num_subwords < sentence_length, choose subtokens from the center of current sentence.
                
                maxR = max(maxR, len(target_tokens))  
                
                assert(len(target_tokens)<=max_num_subwords),pdb.set_trace()
                target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                
                entity_infos = []

                # 遍历所有的entities
                for entity_start in range(left_context_length, left_context_length + sentence_length):                                          # entity start idx in input sequence. entity在当前输入的序列中的起始位置
                    doc_entity_start = entity_start + doc_offset                                                                                # doc_entity_start: start idx in total text; entity在当前doc (包含许多句子)的起始位置
                    if doc_entity_start not in subword_start_positions:
                        continue
                    for entity_end in range(entity_start + 1, left_context_length + sentence_length + 1):
                        doc_entity_end = entity_end + doc_offset
                        if doc_entity_end not in subword_start_positions:
                            continue

                        if subword2token[doc_entity_end - 1] - subword2token[doc_entity_start] + 1 > self.args.max_mention_ori_length:          # if current entity length > max entity length, continue
                            continue

                        label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                        entity_labels.pop((doc_entity_start, doc_entity_end), None)                                                            
                        # entity_start+1: target_tokens[0] is [CLS], entity_start should be right shifted by 1. 
                        entity_infos.append(((entity_start+1, entity_end), label, (subword2token[doc_entity_start], subword2token[doc_entity_end - 1] )))       # for x in entity_infos, x[0]:(start subtoken, end subtoken) x[2]: (start token, end token)
                
               
                # if len(entity_labels):
                #     print ((entity_labels))
                # assert(len(entity_labels)==0)
                    
                # dL = self.max_pair_length 
                # maxR = max(maxR, len(entity_infos))
                # for i in range(0, len(entity_infos), dL):
                #     examples = entity_infos[i : i + dL]
                #     item = {
                #         'sentence': target_tokens,
                #         'examples': examples,
                #         'example_index': (l_idx, n),
                #         'example_L': len(entity_infos)
                #     }                
                    
                #     self.data.append(item)      

                                                                           # maxR: max candidate entities numbers among sentences
                dL = self.max_pair_length 
                if self.args.shuffle:
                    random.shuffle(entity_infos)
                if self.args.group_sort:
                    group_axis = np.random.randint(2)
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1-group_axis]), reverse=sort_dir)

                if not self.args.group_edge:
                    for i in range(0, len(entity_infos), dL):
                        # pdb.set_trace()
                        examples = entity_infos[i : i + dL]
                        item = {
                            'sentence': target_tokens,          
                            'examples': examples,
                            'example_index': (l_idx, n),
                            'example_L': len(entity_infos),
                            'sent_length': sent_len
                        }        
                        self.data.append(item)
                else:
                    if self.args.group_axis==-1:
                        group_axis = np.random.randint(2)
                    else:
                        group_axis = self.args.group_axis
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(key=lambda x: (x[0][group_axis], x[0][1-group_axis]), reverse=sort_dir)
                    _start = 0 
                    while _start < len(entity_infos):
                        _end = _start+dL
                        if _end >= len(entity_infos):
                            _end = len(entity_infos)
                        else:
                            while  entity_infos[_end-1][0][group_axis]==entity_infos[_end][0][group_axis] and _end > _start:
                                _end -= 1
                            if _start == _end:
                                _end = _start+dL

                        examples = entity_infos[_start: _end]

                        item = {
                            'sentence': target_tokens,
                            'examples': examples,
                            'example_index': (l_idx, n),
                            'example_L': len(entity_infos),
                            'sent_length': sent_len
                        }  
                                       
                        self.data.append(item)   
                        _start = _end                 
                
                        

        self.logger.info('maxL: %d', maxL) # 334
        self.logger.info('maxR: %d', maxR) 

        # exit()  
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # for a sentence
        entry = self.data[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(entry['sentence'])                                                                 # turn input tokens into indices
        L = len(input_ids)

        input_ids += [0] * (self.max_seq_length - len(input_ids))                                                                           # padding with 0        # shape: max_seq_length
        position_plus_pad = int(self.model_type.find('roberta')!=-1) * 2                                                                    # pad 2 for 'Roberta', 0 for others


        # print(f'input_ids length: {len(input_ids)}')

        if self.model_type not in ['bertspan', 'robertaspan', 'albertspan']:

            if self.model_type.startswith('albert'):
                input_ids = input_ids + [30000] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))   
                input_ids = input_ids + [30001] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
            elif self.model_type.startswith('roberta'):
                input_ids = input_ids + [50261] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))   
                input_ids = input_ids + [50262] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))
            else:
                input_ids = input_ids + [1] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))              # 1 for entities start, 0 for padding.    shape: max_seq_length + max_pair_length
                input_ids = input_ids + [2] * (len(entry['examples'])) + [0] * (self.max_pair_length - len(entry['examples']))              # 2 for entities end, 0 for padding.    shape: max_seq_length + 2*max_pair_length

            attention_mask = torch.zeros((self.max_entity_length + self.max_seq_length, self.max_entity_length + self.max_seq_length), dtype=torch.int64)           # max_entity_length = max_pair_length * 2
            attention_mask[:L, :L] = 1
            position_ids = list(range(position_plus_pad, position_plus_pad+self.max_seq_length)) + [0] * self.max_entity_length             

        else:
            attention_mask = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
            position_ids = list(range(position_plus_pad, position_plus_pad+self.max_seq_length)) + [0] * self.max_entity_length 

        # print(f'input_ids length: {len(input_ids)}')

        labels = []
        mentions = []
        mention_pos = []
        num_pair = self.max_pair_length


        full_attention_mask = [1] * L + [0] * (self.max_seq_length - L) + [0] * (self.max_pair_length)*2            # max_seq_length | max_pair_length | max_pair_length;   full sequence: length L for input subtokens and (max_seq_length - L) paddings; max_pair_length for entity starts; max_pair_length for entity ends 


        for x_idx, x in enumerate(entry['examples']):                               # x_idx: entities idx for a sentence, entry['examples']: entity_infos
            m1 = x[0]                                                               # entity positions: (start idx, end idx)                                 
            label = x[1]                                    
            mentions.append(x[2])
            mention_pos.append((m1[0], m1[1]))
            labels.append(label)


            if self.model_type in ['bertspan', 'robertaspan', 'albertspan']:
                continue
            
            w1 = x_idx                                                              # for start subtoken of an entity
            w2 = w1 + num_pair                                                      # for end subtoken of an entity

            w1 += self.max_seq_length                                               # max_seq_length|w1(0), w1(1), ...|w2(0), w2(1), ...;                                  
            w2 += self.max_seq_length
            position_ids[w1] = m1[0]                                                
            position_ids[w2] = m1[1]

            for xx in [w1, w2]:
                full_attention_mask[xx] = 1
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1

        labels += [-1] * (num_pair - len(labels))                                   # padding with -1
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))                     # padding with (0, 0)



        item = [torch.tensor(input_ids),                                            
                attention_mask,
                torch.tensor(position_ids),
                torch.tensor(labels, dtype=torch.int64),
                torch.tensor(mention_pos),
                torch.tensor(full_attention_mask)
        ]       

        if self.evaluate:
            #------add sent lengths-------
            item.append(torch.tensor(entry['sent_length']))
            #-----------------------------
            item.append(entry['example_index'])
            item.append(mentions)
        return item

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 2
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        return stacked_fields

 

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
        # tb_writer = SummaryWriter("logs/ace_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])
        tb_writer = SummaryWriter("logs/"+args.data_dir[max(args.data_dir.rfind('/'),0):]+"_ner_logs/"+args.output_dir[args.output_dir.rfind('/'):])

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_file = os.path.join(args.data_dir, args.train_file)
    train_dataset = ACEDatasetNER(logger=logger, tokenizer=tokenizer, file_path=train_file, args=args)
                            
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1)

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
    if args.learning_rate_span>0:
        opt_grouped_bert = [{'params': [], 'weight_decay': args.weight_decay, 'lr':args.learning_rate}, {'params':[], 'weight_decay': 0.0, 'lr':args.learning_rate}]
        opt_grouped_span = [{'params': [], 'weight_decay': args.weight_decay, 'lr':args.learning_rate_span}, {'params':[], 'weight_decay': 0.0, 'lr':args.learning_rate_span}]
        for n, p in model.named_parameters():
            if 'span_encoder' in n or 'ner_classifier' in n:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_span[0]['params'].append(p)
                else:
                    opt_grouped_span[1]['params'].append(p)
            else:
                if not any(nd in n for nd in no_decay):
                    opt_grouped_bert[0]['params'].append(p)
                else:
                    opt_grouped_bert[1]['params'].append(p)
        optimizer = AdamW(opt_grouped_bert+opt_grouped_span, eps=args.adam_epsilon)
    else:
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps==-1:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1*t_total), num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
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

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
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

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'position_ids':   batch[2],
                      'labels':         batch[3],
                      }

            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer!=-1:
                inputs['full_attention_mask']= batch[5]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
  
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            logging_loss_epoch += loss.item()

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
                    
                if args.local_rank in [-1, 0] and eval_steps > 0 and global_step % eval_steps == 0:
                    update = True
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        dev_file = os.path.join(args.data_dir, args.dev_file)
                        results = evaluate(logger, args, model, tokenizer, file_path=dev_file)
                        ent_recall = results['r_overlap']

                        if ent_recall >= best_result:
                            best_result = ent_recall
                            logger.info(f'Best recall overlap:{best_result:.4f} in step:{global_step}')
                        else:
                            update = False

                    if update:
                        checkpoint_prefix = 'checkpoint'
                        output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(logger, args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        logger.info(f'>>> lr of epoch{epoch_num}: {scheduler.get_last_lr()[0]:.4e}')
        dstep = global_step - logging_loss_steps_0
        avg_loss = logging_loss_epoch/dstep
        logger.info(f'>>> Average loss of epoch{epoch_num}: {avg_loss:.4f}')

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()


    return global_step, tr_loss / global_step, best_result


def _exact_boundaries(indexs, sent_lens, batch_m2s):
    sent_lens_simple = []
    indexs_simple = []
    batch_m2s_simple = []
    ranges = []
    current_len = 1
    current_sent_index = indexs[0]
    sent_lens_simple.append(int(sent_lens[0]))
    indexs_simple.append(indexs[0])
    batch_m2s_simple.append(batch_m2s[0])
    for i in range(1, len(indexs)):
        index = indexs[i]
        if current_sent_index == index:
            current_len += 1
            batch_m2s_simple[-1] += batch_m2s[i]
        else:
            current_sent_index = index
            ranges.append(current_len)
            current_len = 1
            sent_lens_simple.append(int(sent_lens[i]))
            indexs_simple.append(indexs[i])
            batch_m2s_simple.append(batch_m2s[i])
    ranges.append(current_len)

    return ranges, sent_lens_simple, indexs_simple, batch_m2s_simple


def _pad_tensors(tensor_list, pad=0):
    assert len(tensor_list)>=2,pdb.set_trace()
    assert len(list(tensor_list[0].shape))==1
    max_shape = 0
    for t in tensor_list:
        shape_t = list(t.shape)[0]
        if shape_t > max_shape:
            max_shape = shape_t
    for i in range(len(tensor_list)):
        tensor_i = tensor_list[i]
        shape_i = list(tensor_list[i].shape)[0]
        ext_shape = max_shape - shape_i
        ext_tensor = torch.empty(ext_shape).fill_(pad).to(tensor_i.device)
        tensor_list[i] = torch.cat((tensor_i, ext_tensor))
    return tensor_list


def _remove_redundancy(infos):
    """ remove redundent dimensions according ent_masks """
    if infos['probs'] is None:
        return infos
    else:
        split_probs = infos['probs']
        indexs = infos['indexs']
        sent_lens = infos['sent_lens']
        ent_masks = infos['ent_masks']
        gold_labels = infos['gold_labels']
        mentions = infos['mentions']

        if len(indexs)>1:
            split_probs = _pad_tensors(split_probs, pad=0)
            ent_masks = _pad_tensors(ent_masks, pad=0)
            gold_labels = _pad_tensors(gold_labels, pad=0)

        max_n_ents = torch.stack(ent_masks).int().sum(-1).max().item()
        for i in range(len(indexs)):
            split_probs[i] = split_probs[i][:max_n_ents]
            ent_masks[i] = ent_masks[i][:max_n_ents]
            gold_labels[i] = gold_labels[i][:max_n_ents]

        return dict(probs=split_probs, indexs=indexs, sent_lens=sent_lens, ent_masks=ent_masks, gold_labels=gold_labels, mentions=mentions)


def _extent_tensor(last_infos, current_infos):
    


    if last_infos['probs'] is not None:
        last_probs = last_infos['probs'][0]
        last_index = last_infos['indexs'][0]
        last_sent_len = last_infos['sent_lens'][0]
        last_ent_mask = last_infos['ent_masks'][0]
        last_gold_labels = last_infos['gold_labels'][0]
        last_mention = last_infos['mentions'][0]
    else:
        last_probs = None

    cur_split_probs = current_infos['probs']
    cur_indexs = current_infos['indexs']
    cur_sent_lens = current_infos['sent_lens']
    cur_ent_masks = current_infos['ent_masks']
    cur_gold_labels = current_infos['gold_labels']
    cur_mentions = current_infos['mentions']

    if cur_split_probs is not None:
        if len(cur_indexs)==1:
            # update last infos
            if last_infos['probs'] is not None:
                # last_shape = list(last_probs.shape)
                # current_shape = list(cur_split_probs[0].shape)
                if last_index==cur_indexs[0]:
                    assert last_sent_len==cur_sent_lens[0]
                    # update probs
                    tensor0 = cur_split_probs[0]
                    last_probs = torch.cat((last_probs, tensor0))
                    # update ent mask
                    ent_mask0 = cur_ent_masks[0]
                    last_ent_mask = torch.cat((last_ent_mask, ent_mask0))
                    # update gold labels
                    gold_labels0 = cur_gold_labels[0]
                    last_gold_labels = torch.cat((last_gold_labels, gold_labels0))
                    # update mentions
                    last_mention = last_mention + cur_mentions[0]
                    last_infos = {'probs':[last_probs], 'indexs':[last_index], 'sent_lens':[last_sent_len], 'ent_masks':[last_ent_mask], 'gold_labels':[last_gold_labels], 'mentions':[last_mention]}
                    previous_infos = {'probs':None, 'indexs':None, 'sent_lens':None, 'ent_masks':None, 'gold_labels':None, 'mentions':None}
                else:
                    # last tensor and first tensor in current is from different sentence, generate previous infos, new last infos are None
                    # last infos --> previous infos
                    previous_indexs = [last_index] 
                    previous_sent_lens = [last_sent_len] 
                    previous_mentions = [last_mention] 

                    previous_split_probs = [last_probs] 
                    previous_ent_masks = [last_ent_mask] 
                    previous_gold_labels = [last_gold_labels] 

                    previous_infos = dict(probs=previous_split_probs, indexs=previous_indexs, sent_lens=previous_sent_lens, ent_masks=previous_ent_masks, gold_labels=previous_gold_labels, mentions=previous_mentions)
                    last_infos = {'probs':cur_split_probs, 'indexs':cur_indexs, 'sent_lens':cur_sent_lens, 'ent_masks':cur_ent_masks, 'gold_labels':cur_gold_labels, 'mentions':cur_mentions}
            else:
                previous_infos = {'probs':None, 'indexs':None, 'sent_lens':None, 'ent_masks':None, 'gold_labels':None, 'mentions':None}
                last_infos = {'probs':cur_split_probs, 'indexs':cur_indexs, 'sent_lens':cur_sent_lens, 'ent_masks':cur_ent_masks, 'gold_labels':cur_gold_labels, 'mentions':cur_mentions}
        else:
            if last_infos['probs'] is not None:
                # last_shape = list(last_probs.shape)
                # current_shape = list(cur_split_probs[0].shape)
                if last_index==cur_indexs[0]:
                    # last tensor and first tensor in current is entities of the same sentence.
                    assert last_sent_len==cur_sent_lens[0]
                    # update probs
                    tensor0 = cur_split_probs[0]
                    tensor0 = torch.cat((last_probs, tensor0))
                    cur_split_probs[0] = tensor0
                    # update ent mask
                    ent_mask0 = cur_ent_masks[0]
                    ent_mask0 = torch.cat((last_ent_mask, ent_mask0))
                    cur_ent_masks[0] = ent_mask0
                    # update gold labels
                    gold_labels0 = cur_gold_labels[0]
                    gold_labels0 = torch.cat((last_gold_labels, gold_labels0))
                    cur_gold_labels[0] = gold_labels0
                    # update mentions
                    cur_mentions[0] = last_mention + cur_mentions[0]
                        
                else:
                    # last tensor and first tensor in current is from different sentence
                    cur_indexs = [last_index] + cur_indexs
                    cur_sent_lens = [last_sent_len] + cur_sent_lens
                    cur_mentions = [last_mention] + cur_mentions

                    cur_split_probs = [last_probs] + cur_split_probs
                    cur_ent_masks = [last_ent_mask] + cur_ent_masks
                    cur_gold_labels = [last_gold_labels] + cur_gold_labels

            previous_infos = dict(probs=cur_split_probs[:-1], indexs=cur_indexs[:-1], sent_lens=cur_sent_lens[:-1], ent_masks=cur_ent_masks[:-1], gold_labels=cur_gold_labels[:-1], mentions=cur_mentions[:-1])
            last_infos = dict(probs=[cur_split_probs[-1]], indexs=[cur_indexs[-1]], sent_lens=[cur_sent_lens[-1]], ent_masks=[cur_ent_masks[-1]], gold_labels=[cur_gold_labels[-1]], mentions=[cur_mentions[-1]])
    else:   # after last batch
        if last_probs is not None:
            previous_infos = last_infos
            last_infos = {'probs':None, 'indexs':None, 'sent_lens':None, 'ent_masks':None, 'gold_labels':None, 'mentions':None}
        else:
            raise ValueError()
    
    previous_infos = _remove_redundancy(previous_infos)
    last_infos = _remove_redundancy(last_infos)
    return previous_infos, last_infos




def _decode_pruner_topk(topk_infos, previous_infos):

    previous_probs = previous_infos['probs']
    previous_indexs = previous_infos['indexs']
    previous_sent_lens = previous_infos['sent_lens']
    previous_ent_masks = previous_infos['ent_masks']
    previous_gold_labels = previous_infos['gold_labels']
    previous_mentions = previous_infos['mentions']
    
    topk_ratio, min_mentions_num, max_mentions_num= topk_infos

    previous_mentions = _get_batch_mentions(previous_mentions, previous_probs[0].device)

    ner_probs = torch.stack(previous_probs)
    sent_lens = torch.tensor(previous_sent_lens, device=ner_probs.device).reshape(-1,1)            # shape: (bs,)
    ent_masks = torch.stack(previous_ent_masks).bool()
    gold_labels = torch.stack(previous_gold_labels)

    bs = len(previous_probs)

    ner_probs_1 = ner_probs.masked_fill(ent_masks==False, 0)
    assert torch.equal(ner_probs, ner_probs_1),pdb.set_trace()
    _, indices = torch.sort(ner_probs, dim=1, descending=True)
    _, n_entity = ner_probs.shape

    # gold_entities_id_flat = gold_labels.masked_select(ent_masks)
    n_ent_topk = torch.ceil(sent_lens.float()*topk_ratio)

    min_topk_mask = n_ent_topk<min_mentions_num
    max_topk_mask = n_ent_topk>max_mentions_num

    n_ent_topk = min_topk_mask*min_mentions_num + (~min_topk_mask)*n_ent_topk			# if min_topk_mask=True, n_ent_topk=ent_mintopk, else: n_ent_topk=n_ent_topk
    n_ent_topk = max_topk_mask*max_mentions_num + (~max_topk_mask)*n_ent_topk

    # prune spans 
    entity_starts = previous_mentions[:,:,0]
    entity_ends = previous_mentions[:,:,1]
    max_ne = int(max(n_ent_topk))
    # b x ne
    pruned_ent_starts = torch.gather(entity_starts, dim=1, index=indices)[:, :max_ne]
    pruned_ent_ends = torch.gather(entity_ends, dim=1, index=indices)[:, :max_ne]
    pruned_ent_probs = torch.gather(ner_probs, dim=-1, index=indices)[:, :max_ne]
    # b x ne x 3
    pruned_ent_spans = torch.stack((pruned_ent_starts, pruned_ent_ends, pruned_ent_probs)).permute(1,2,0)

    # if len(pruned_ent_spans)!= len(previous_indexs):
    #     pdb.set_trace()

    # evaluate for metrics
    topk_mask = torch.arange(n_entity, device=n_ent_topk.device).repeat(bs).reshape(bs, -1)
    # topk_mask = (topk_mask<n_ent_topk).to(torch.long)
    topk_mask = (topk_mask<max_ne).to(torch.long)

    pred_ents_idx = torch.zeros((bs, n_entity), device=topk_mask.device, dtype=topk_mask.dtype).scatter_(dim=1, index=indices, src=topk_mask)
    predicted_entities_id_flat = pred_ents_idx.masked_select(ent_masks)
    gold_entities_id_flat = gold_labels.masked_select(ent_masks)
            
    # predicted_entities_id = pred_ents_idx*ent_masks
    
    return pruned_ent_spans, predicted_entities_id_flat, gold_entities_id_flat



def _get_batch_mentions(batch_m2s, model_device):
    bs = len(batch_m2s)

    batch_mentions = []
    for i in range(bs):
        batch_mentions.append(torch.tensor(batch_m2s[i], device=model_device))
    batch_mentions = pad_sequence(batch_mentions, padding_value=-1).permute(1,0,2)

    return batch_mentions


def _update_prediction(pruned_ent_spans, previous_infos, scores, goldspan2label):
    indexs = previous_infos['indexs']
    for i, index in enumerate(indexs):
        index = (int(index[0]), int(index[1]))
        for span in pruned_ent_spans[i]:
            span, prob = (int(span[0]), int(span[1])), float(span[2])
            if span[0]>=0:
                scores[index][span] = (prob, goldspan2label[index].get(span, 'NIL'))
    return scores

def _span2label(ner_golden_labels):
    label_dict = defaultdict(dict)
    for index, mention, label in ner_golden_labels:
        label_dict[index][mention] = label
    return label_dict

def evaluate(logger, args, model, tokenizer, file_path, prefix="",do_test=False):
    
    global NEG_INF 
    NEG_INF = args.neg_inf

    eval_output_dir = args.output_dir

    results = {}
    eval_dataset = ACEDatasetNER(logger=logger, tokenizer=tokenizer, file_path=file_path, args=args, evaluate=True)
    ner_golden_labels = set(eval_dataset.ner_golden_labels)
    ner_tot_recall = eval_dataset.tot_recall

    goldspan2label = _span2label(ner_golden_labels)
    # print(goldspan2label)
    #--------for pruner-----------------
    topk_ratio = args.topk_ratio
    max_mentions_num = args.max_mentions_num
    min_mentions_num = args.min_mentions_num

    topk_infos = (topk_ratio, min_mentions_num, max_mentions_num)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,  collate_fn=ACEDatasetNER.collate_fn, num_workers=1)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    scores = defaultdict(dict)
    predict_ners = defaultdict(list)
    predict_ners_overlap = defaultdict(list)

    model.eval()

    start_time = timeit.default_timer() 
       
    
    previous_infos = {'probs':None}             # from last batch
    last_infos = {'probs':None}                 # for next batch

    tp = 0
    n_pos = 0
    tot_pred = 0

    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):

        indexs = batch[-2]                                              # example_index, (doc_id, sent_id)
        batch_m2s = batch[-1]                                           # mentions
        
        #-------for pruner------------
        sent_lens = batch[6]

        split_ranges, sent_lens_simple, indexs_simple, batch_m2s_simple = _exact_boundaries(indexs, sent_lens, batch_m2s)           # in this batch

        # batch_mentions = _get_batch_mentions(batch_m2s_simple, args.device)
        
        batch = tuple(t.to(args.device) for t in batch[:6])

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'position_ids':   batch[2],
                      'labels':         batch[3]
                      }
            
            if args.model_type.find('span')!=-1:
                inputs['mention_pos'] = batch[4]
            if args.use_full_layer!=-1:
                inputs['full_attention_mask']= batch[5]

            outputs = model(**inputs)
            gold_labels = (inputs['labels']>0).int()
            ent_masks = (inputs['labels']>-1).int()
            ner_logits = outputs[1].squeeze(-1)
            ner_probs = ner_logits.sigmoid()
            # NEG_INF = NEG_INF if ner_logits.dtype==torch.float32 else -1e4
            ner_probs = ner_probs.masked_fill(ent_masks==False, 0)
            split_probs = torch.split(ner_probs, split_ranges, dim=0)
            split_mask = torch.split(ent_masks, split_ranges, dim=0)
            split_gold_labels = torch.split(gold_labels, split_ranges, dim=0)

            split_probs_cat = []
            split_mask_cat = []
            split_gold_labels_cat = []

            for i in range(len(split_ranges)):
                split_probs_cat.append(split_probs[i].reshape(-1))
                split_mask_cat.append(split_mask[i].reshape(-1))
                split_gold_labels_cat.append(split_gold_labels[i].reshape(-1))
            current_infos = {'probs':split_probs_cat, 'indexs':indexs_simple, 'sent_lens':sent_lens_simple, 'ent_masks':split_mask_cat, 'gold_labels':split_gold_labels_cat, 'mentions': batch_m2s_simple}

            # process previous infos from last batch
            if previous_infos['probs'] is not None:
                pruned_ent_spans, pred_entities, gold_entities = _decode_pruner_topk(topk_infos, previous_infos)

                scores = _update_prediction(pruned_ent_spans, previous_infos, scores, goldspan2label)

            previous_infos, last_infos = _extent_tensor(last_infos, current_infos)
                        
    if previous_infos['probs'] is not None:
        pruned_ent_spans, pred_entities, gold_entities = _decode_pruner_topk(topk_infos, previous_infos)

        scores = _update_prediction(pruned_ent_spans, previous_infos, scores, goldspan2label)

    if last_infos['probs'] is not None:
        # last_infos = {'probs':last_probs, 'indexs':last_index, 'sent_lens':last_sent_len, 'ent_masks':last_ent_mask, 'gold_labels':last_gold_labels, 'mentions':last_mention}
        current_infos = {'probs':None, 'indexs':None, 'sent_lens':None, 'ent_masks':None, 'gold_labels':None, 'mentions': None}

        previous_infos, last_infos = _extent_tensor(last_infos, current_infos)
        pruned_ent_spans, pred_entities, gold_entities = _decode_pruner_topk(topk_infos, previous_infos)

        scores = _update_prediction(pruned_ent_spans, previous_infos, scores, goldspan2label)


    # p = tp / tot_pred if tot_pred > 0 else 0 
    # r = tp / n_pos 
    # f1_tot = 2 * (p * r) / (p + r) if tp > 0 else 0.0

    # print(f'f1_tot:{f1_tot}')



    cor = 0 
    tot_pred = 0
    cor_tot = 0
    tot_pred_tot = 0

    for example_index, pair_dict in scores.items():  

        sentence_results = []
        for k1, (v2_score, v2_ner_label) in pair_dict.items():                  # k1: entity head/tail position
            # if v2_ner_label!='NIL':
            sentence_results.append((v2_score, k1, v2_ner_label))

        sentence_results.sort(key=lambda x: -x[0])
        no_overlap = []
        def is_overlap(m1, m2):
            if m2[0]<=m1[0] and m1[0]<=m2[1]:
                return True
            if m1[0]<=m2[0] and m2[0]<=m1[1]:
                return True
            return False

        for item in sentence_results:
            m2 = item[1]
            overlap = False
            for x in no_overlap:
                _m2 = x[1]
                if (is_overlap(m2, _m2)):
                    if args.data_dir.find('ontonotes')!=-1:
                        overlap = True
                        break
                    else:
                    
                        if item[2]==x[2]:
                            overlap = True
                            break

            if not overlap:
                no_overlap.append(item)

            pred_ner_label = item[2]
            tot_pred_tot += 1
            if args.output_results:
                predict_ners_overlap[example_index].append( (m2[0], m2[1], pred_ner_label) )
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor_tot += 1

        for item in no_overlap:
            m2 = item[1]
            pred_ner_label = item[2]
            tot_pred += 1
            if args.output_results:
                predict_ners[example_index].append( (m2[0], m2[1], pred_ner_label) )
            if (example_index, m2, pred_ner_label) in ner_golden_labels:
                cor += 1        

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f example per second)", evalTime,  len(eval_dataset) / evalTime)


    precision_score = p = cor / tot_pred if tot_pred > 0 else 0 
    recall_score = r = cor / ner_tot_recall 
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0 
    r = cor_tot / ner_tot_recall 
    f1_tot = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    
    results = {'f1':  f1, 'precision': precision_score, 'recall': recall_score, 'f1_overlap': f1_tot, 'p_overlap':p, 'r_overlap':r}


    res = {k:f'{v:.4f}' for k,v in results.items()}
    logger.info(f"Result: {res}")
    
    if args.output_results and (do_test or not args.do_train):
        # pdb.set_trace()
        if 'ace2004' in file_path or 'ace04' in file_path:
            file_name = os.path.split(file_path)[-1]
            if 'train' in file_path:
                file_name = 'train_' + file_name
            elif 'dev' in file_path:
                file_name = 'dev_' + file_name
            elif 'test' in file_path:
                file_name = 'test_' + file_name
            else:
                raise ValueError()
        else:
            file_name = os.path.split(file_path)[-1]
        for pred_ners in (predict_ners, predict_ners_overlap): 
            f = open(eval_dataset.file_path)                    # file_path is gold file
            dump_name = 'ent_pred_' + file_name.split('.json')[0]

            if pred_ners==predict_ners_overlap:
                dump_name += f'_overlap.json'
            else:
                dump_name += f'.json'
                
            output_w = open(os.path.join(args.output_dir, dump_name), 'w')  

            for l_idx, line in enumerate(f):
                data = json.loads(line)
                num_sents = len(data['sentences'])
                predicted_ner = []
                for n in range(num_sents):
                    item = pred_ners.get((l_idx, n), [])
                    item.sort()
                    predicted_ner.append( item )

                data['predicted_ner'] = predicted_ner
                output_w.write(json.dumps(data)+'\n')
            logger.info(f'evaluate test file into {dump_name} done.')


    return results



          





def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default='ace_data', type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

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
                        help="Whether to run test on the dev set.")

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
    parser.add_argument("--learning_rate_span", default=-1, type=float,
                        help="The initial learning rate for span encoder.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
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
                        help="Save checkpoint every eval_epochs.")

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

    parser.add_argument('--alpha', type=float, default=1,  help="")
    parser.add_argument('--max_pair_length', type=int, default=256,  help="")
    parser.add_argument('--max_mention_ori_length', type=int, default=8,  help="")
    parser.add_argument('--lminit', action='store_true')
    parser.add_argument('--norm_emb', action='store_true')
    parser.add_argument('--output_results', action='store_true')
    parser.add_argument('--onedropout', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--use_full_layer', type=int, default=-1,  help="")
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--group_edge', action='store_true')
    parser.add_argument('--group_axis', type=int, default=-1,  help="")
    parser.add_argument('--group_sort', action='store_true')
    
    # no cross-sentence
    parser.add_argument('--nocross', action='store_true')

    # for pruner
    parser.add_argument("--topk_ratio", default=0.5, type=float, help="Topk ratio, candidate entity number divide sentence length.")
    parser.add_argument('--min_mentions_num', type=int, default=3,  help="max mentions number feed in pruner")
    parser.add_argument('--max_mentions_num', type=int, default=18,  help="min mentions number feed in pruner")
    parser.add_argument('--extra_repr', type=str, default=None, help="use extra span repr")

    
    # for biaf span repr
    parser.add_argument('--biaf_span', action='store_true', help="use BiaffineSpanRepr")
    parser.add_argument('--biaf_factorize', action='store_true', help="use BiaffineSpanRepr")
    parser.add_argument('--biaf_mode', type=int, default=3, help="for BiaffineSpanRepr")
    parser.add_argument('--rank', type=int, default=768, help="for BiaffineSpanRepr")
    parser.add_argument('--span_hidden_size', type=int, default=768, help="for BiaffineSpanRepr")
    parser.add_argument('--span_size', type=int, default=256, help="for BiaffineSpanRepr")


    args = parser.parse_args()

    # get hostname
    args.hostname = socket.gethostname()

    args.neg_inf = -1e4 if args.fp16 else -1e30




    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    def create_exp_dir(path, scripts_to_save=None):

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

    # if args.do_train and args.local_rank in [-1, 0]:
    exp_path = create_exp_dir(args.output_dir, scripts_to_save=['run_pruner.py', 'transformers/src/transformers/modules.py', 'transformers/src/transformers/modeling_bert.py',  'transformers/src/transformers/modeling_albert.py'])

    logger = set_logger(args, exp_path, args.do_test)

    if not args.do_test:
        args_file = os.path.join(args.output_dir, 'training_args.txt')
        with open(args_file, 'w') as f:
            f.write(str(vars(args)))

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
    logger.info('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    if args.data_dir.find('ace')!=-1:
        num_labels = 8
    elif args.data_dir.find('scierc')!=-1:
        num_labels = 7
    elif args.data_dir.find('ontonotes')!=-1:
        num_labels = 19
    else:
        assert (False)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    args.model_type = args.model_type.lower()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,  do_lower_case=args.do_lower_case)

    config.max_seq_length = args.max_seq_length
    config.alpha = args.alpha
    config.onedropout = args.onedropout
    config.use_full_layer= args.use_full_layer

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, args=args)

    if args.model_type.startswith('albert'):
        special_tokens_dict = {'additional_special_tokens': ['[unused' + str(x) + ']' for x in range(4)]}
        tokenizer.add_special_tokens(special_tokens_dict)
        # print ('add tokens:', tokenizer.additional_special_tokens)
        # print ('add ids:', tokenizer.additional_special_tokens_ids)
        model.albert.resize_token_embeddings(len(tokenizer))

    if args.do_train and args.lminit: 
        if args.model_type.find('roberta')==-1:
            entity_id = tokenizer.encode('entity', add_special_tokens=False)
            assert(len(entity_id)==1)
            entity_id = entity_id[0]
            mask_id = tokenizer.encode('[MASK]', add_special_tokens=False)
            assert(len(mask_id)==1)
            mask_id = mask_id[0]
        else:
            entity_id = 10014
            mask_id = 50264

        logger.info('entity_id: %d', entity_id)
        logger.info('mask_id: %d', mask_id)


        if args.model_type.startswith('albert'):
            word_embeddings = model.albert.embeddings.word_embeddings.weight.data
            word_embeddings[30000].copy_(word_embeddings[mask_id])   
            word_embeddings[30001].copy_(word_embeddings[entity_id])   
        elif args.model_type.startswith('roberta'):
            word_embeddings = model.roberta.embeddings.word_embeddings.weight.data
            word_embeddings[50261].copy_(word_embeddings[mask_id])   # entity
            word_embeddings[50262].data.copy_(word_embeddings[entity_id]) 
        else:
            word_embeddings = model.bert.embeddings.word_embeddings.weight.data
            word_embeddings[1].copy_(word_embeddings[mask_id])  
            word_embeddings[2].copy_(word_embeddings[entity_id])     # entity

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_result = 0
    # Training
    if args.do_train:
        global_step, tr_loss, best_result = train(logger, args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            
        update = True
        if args.evaluate_during_training:
            dev_file = os.path.join(args.data_dir, args.dev_file)
            results = evaluate(logger, args, model, tokenizer, file_path=dev_file)
            # logger.info(f"Epoch: {epoch_num}, F1: {results['f1']}, recall: {results['recall']}")
            # logger.info(f"Epoch: {epoch_num}, ent_recall: {results['recall_score']}, R_overlap: {results['r_overlap']}")
            ent_recall = results['r_overlap']
            if ent_recall >= best_result:
                best_result = ent_recall
                logger.info(f'Best recall overlap:{best_result:.4f}')
            else:
                update = False

        if update:
            checkpoint_prefix = 'checkpoint'
            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
            _rotate_checkpoints(logger, args, checkpoint_prefix)

        
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


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
    


    #-------------------------------------------------------------
    # test all files
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]


        WEIGHTS_NAME = 'pytorch_model.bin'
        # pdb.set_trace()
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1]
            output_test_file = os.path.join(args.output_dir, "test_results.txt")
            
            model = model_class.from_pretrained(checkpoint, config=config, args=args)
            model.to(args.device)

            dumps = []
            for file_name in (args.train_file, args.dev_file, args.test_file,):
            # for file_name in (args.test_file,):
                test_file = os.path.join(args.data_dir, file_name)
                if os.path.exists(test_file):
                    logger.info(f"Evaluate on {file_name}")
                    result = evaluate(logger, args, model, tokenizer, file_path=test_file, prefix=global_step, do_test=True)
                    dumps.append(f'--------------------------')
                    dumps.append(f'Test result of {test_file}')
                    dumps.append(str(result))
            dumps = '\n'.join(dumps)
            with open(output_test_file, 'w') as f:
                f.write(dumps)



if __name__ == "__main__":
    main()
