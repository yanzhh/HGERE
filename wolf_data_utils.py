import pdb
import random
import itertools
import json
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import unicodedata
from pathlib import Path

import numpy as np

NER_LABEL_LISTS = dict(
    ace=[
        "NIL",
        "FAC",
        "WEA",
        "LOC",
        "VEH",
        "GPE",
        "ORG",
        "PER",
    ],
    scierc=[
        "NIL",
        "Method",
        "OtherScientificTerm",
        "Task",
        "Generic",
        "Material",
        "Metric",
    ],
    gsap=[
        "NIL",
        "MLModel",
        "MLModelGeneric",
        "ModelArchitecture",
        "Method",
        "Task",
        "Dataset",
        "Datasource",
        "DatasetGeneric",
        "URL",
        "ReferenceLink",
    ],
    default=[
        "NIL",
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ],
)


class ACEDatasetNER(Dataset):
    def __init__(self, logger, tokenizer, file_path, args, evaluate=False):
        # if not evaluate:
        #     file_path = os.path.join(args.data_dir, args.train_file)
        # else:
        #     if do_test:
        #         file_path = os.path.join(args.data_dir, args.test_file)
        #     else:
        #         file_path = os.path.join(args.data_dir, args.dev_file)

        self.logger = logger

        file_path = Path(file_path)
        if not file_path.is_file():
            logger.warn(f"File does not exist: {file_path}")
        assert file_path.is_file()

        self.file_path = file_path

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

        self.evaluate = evaluate
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        if args.data_dir.find("ace") != -1:
            self.ner_label_list = NER_LABEL_LISTS["ace"]
        elif args.data_dir.find("scierc") != -1:
            self.ner_label_list = NER_LABEL_LISTS["scierc"]
        elif args.data_dir.find("gsap") != -1:
            self.ner_label_list = NER_LABEL_LISTS["gsap"]
        else:
            self.ner_label_list = NER_LABEL_LISTS["default"]

        self.max_pair_length = args.max_pair_length

        self.max_entity_length = args.max_pair_length * 2
        self.initialize()

    def initialize(self):
        tokenizer = self.tokenizer
        max_num_subwords = self.max_seq_length - 2

        ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}
        # e.g.: {'NIL': 0, 'Method': 1, 'OtherScientificTerm': 2, 'Task': 3, 'Generic': 4, 'Material': 5, 'Metric': 6}

        def tokenize_word(text):
            ## add_prefix_space treats words like words, not like beginning of sequence
            ## https://huggingface.co/docs/tokenizers/python/latest/api/reference.html
            ## @todo: Why only for RoBERTa?
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        # read file
        f = open(self.file_path, "r", encoding="utf-8")
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        maxL = 0  # maximum number of words per document(not sub_words)
        maxR = 0  # maximum number of subword tokens incl. context per document
        # For each document
        for line_idx, line in enumerate(f):
            document = json.loads(line)
            # if len(self.data) > 5:
            #     break

            # if self.args.output_dir.find('test')!=-1:
            #     if len(self.data) > 5:
            #         break

            sentences = document["sentences"]
            # doc_key = data['doc_key']
            # doc_id = self.doc2id[doc_key]

            ## make sure, there are no replacements like "-LRB-" for "("
            for sent_idx in range(len(sentences)):
                for word_idx in range(len(sentences[sent_idx])):
                    sentences[sent_idx][word_idx] = self.get_original_token(
                        sentences[sent_idx][word_idx]
                    )

            # ners: e.g. [[[4, 7, 'Task'], [13, 14, 'Method'], [17, 18, 'Material'], [22, 23, 'Material'], [34, 36, 'Method'], [38, 39, 'Method']], [[47, 48, 'Material']]]
            ners = document["ner"]

            tokens = []
            L = 0  # total number of tokens
            # store the token numbers of sentences: e.g. [0, 41, 70], indicate the boudaries.
            sentence_boundaries = [0]
            sent_lens = []  # lengths of each sentence in token
            for sent_idx in range(len(sentences)):
                L += len(sentences[sent_idx])
                sent_lens.append(len(sentences[sent_idx]))
                sentence_boundaries.append(L)
                tokens += sentences[sent_idx]

            tokens_splitted = [tokenize_word(token) for token in tokens]
            # print("n token:", len(tokens))
            # print(len(tokens_splitted))
            subword_tokens = [
                subword_token
                for token_splitted in tokens_splitted
                for subword_token in token_splitted
            ]
            # print("n subwords:", len(subword_tokens))
            maxL = max(len(tokens), maxL)
            # print("maxL:", maxL)

            ## first this function creates list: [1,1,1, 2,2,2,2, ...]
            ## repetitions are the number of subword_tokens for each token
            subword2token = list(
                itertools.chain(
                    *[
                        [token_idx] * len(token_splitted)
                        for token_idx, token_splitted in enumerate(tokens_splitted)
                    ]
                )
            )
            #  mapping from token to first! subword_token!
            ## fyi: itertools.accumulate([1, 2, 3, 4, 5]) → 1 3 6 10 15
            token2subword = [0] + list(
                itertools.accumulate(
                    len(token_splitted) for token_splitted in tokens_splitted
                )
            )
            assert len(token2subword) == len(tokens_splitted) + 1, pdb.set_trace()
            assert len(tokens_splitted) == len(tokens), pdb.set_trace()
            # now check consistency
            for sentence_ners in ners:
                for start, end, label in sentence_ners:
                    assert start < len(tokens), pdb.set_trace()
                    assert end < len(tokens), pdb.set_trace()

            # e.g.: [
            #   0,   1,  2,  3,  4,  5,  6,  7,
            #   10, 11, 12, 13, 14, 15, 18, 19,
            #   20, 21, 24, 25, 26, 27, 28,
            #   30, 33, 34, 35, 36, 37,
            #   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            #   50, 51, 52, 53, 54, 55, 56, 57, 58,
            #   61, 62, 63, 64, 65, 66, 67, 68, 69,
            #   70, 71, 72, 73, 74, 75, 76, 79,
            #   80, 82, 83, 84, 85, 86]
            subword_start_positions = frozenset(token2subword)
            # @todo this should be the same as
            # subword_sentence_boundaries = [token2subword[sentence_start_token_idx]
            #                                 for sentence_start_token_idx
            #                                 in sentence_coundaries]
            subword_sentence_boundaries = [
                sum(
                    len(subword)
                    for subword in tokens_splitted[:sentence_start_token_idx]
                )
                for sentence_start_token_idx in sentence_boundaries
            ]
            ## @todo shouldn't the number of sentences be equal to the number of sentence_boundaries?
            ## I think we have a boundary at the very end is this wanted? Yes, it is the ending of the last sentence
            assert (
                len(subword_sentence_boundaries) == len(sentences) + 1
            ), pdb.set_trace()
            assert len(sentence_boundaries) == len(
                subword_sentence_boundaries
            ), pdb.set_trace()

            # iterate through the sentences by sentence_idx
            for sentence_idx in range(len(subword_sentence_boundaries) - 1):
                # nth sentence
                # length of sentence in token (not subword_token)
                sent_len = sent_lens[sentence_idx]
                # sent_id = self.sent_ids[doc_id][sentence_idx]

                sentence_ners = ners[sentence_idx]
                self.tot_recall += len(sentence_ners)
                # a mapping from subword_token_start, subword_token_end to label
                entity_labels = {}
                # start: start token idx; end: end token idx;
                for start, end, label in sentence_ners:
                    # if start >= len(token2subword) or end+1 >= len(token2subword):
                    #    print(start, end, label)
                    #    print(token2subword[7000:])
                    key = token2subword[start], token2subword[end + 1]
                    entity_labels[key] = ner_label_map[label]
                    self.ner_golden_labels.add(
                        ((line_idx, sentence_idx), (start, end), label)
                    )

                # get subword of sentence start and sentence end
                doc_sent_start, doc_sent_end = subword_sentence_boundaries[
                    sentence_idx : sentence_idx + 2
                ]

                left_length = doc_sent_start  # subword_tokens to the left

                # left_length | current sentence length | right_length;
                # left_length: subtoken numbers before current sentence, right_length: subtoken numbers after current sentence.
                right_length = (
                    len(subword_tokens) - doc_sent_end
                )  # subword_tokens to the right
                sentence_length = (
                    doc_sent_end - doc_sent_start
                )  # subword_tokens length of sentence

                # could half_context_length < 0?
                ## could occure, when sentence is longer than allowed max_num_subwords
                ## happens in gsap-rel
                if sentence_length > max_num_subwords:
                    # What to do now?
                    # problem with evaluation when just cut?
                    print(
                        f'Sentence to long ({sentence_length}): {" ".join(sentences[sentence_idx])}'
                    )
                    doc_sent_end -= sentence_length - max_num_subwords
                    sentence_length = doc_sent_end - doc_sent_start
                assert sentence_length <= max_num_subwords, pdb.set_trace()
                half_context_length = int((max_num_subwords - sentence_length) / 2)
                # 设置的最大subtokens长度允许在当前句子左或右能包含的context subtoken个数。
                # half_context_length | current sentence length | half_context_length

                if left_length < right_length:
                    left_context_length = min(left_length, half_context_length)
                    right_context_length = min(
                        right_length,
                        max_num_subwords - left_context_length - sentence_length,
                    )
                else:
                    right_context_length = min(right_length, half_context_length)
                    left_context_length = min(
                        left_length,
                        max_num_subwords - right_context_length - sentence_length,
                    )
                if self.args.output_dir.find("ctx0") != -1:
                    left_context_length = right_context_length = 0  # for debug
                if self.args.nocross:
                    left_context_length = right_context_length = 0
                # pdb.set_trace()
                doc_offset = doc_sent_start - left_context_length
                # start idx of contexted sequence.
                target_tokens = subword_tokens[
                    doc_offset : doc_sent_end + right_context_length
                ]
                # if max_num_subwords < sentence_length, choose subtokens from the center of current sentence.

                maxR = max(maxR, len(target_tokens))

                assert len(target_tokens) <= max_num_subwords, pdb.set_trace()
                target_tokens = (
                    [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
                )
                ## now collect entity infos!
                entity_infos = []

                ## Search entities in sentence
                sentence_begin_in_context = left_context_length
                sentence_end_in_context = left_context_length + sentence_length
                for context_entity_start in range(
                    sentence_begin_in_context, sentence_end_in_context
                ):
                    # entity start idx in input sequence. entity在当前输入的序列中的起始位置
                    doc_entity_start = doc_offset + context_entity_start
                    # doc_entity_start: start idx in total text; entity在当前doc (包含许多句子)的起始位置
                    if doc_entity_start not in subword_start_positions:
                        # let candidate entities not begin inside tokens!
                        continue
                    for context_entity_end in range(
                        context_entity_start + 1,
                        left_context_length + sentence_length + 1,
                    ):
                        doc_entity_end = doc_offset + context_entity_end
                        if doc_entity_end not in subword_start_positions:
                            # let candidate enities not end inside tokens!
                            continue

                        if (
                            subword2token[doc_entity_end - 1]
                            - subword2token[doc_entity_start]
                            + 1
                            > self.args.max_mention_ori_length
                        ):
                            # if current entity length > max entity length, continue
                            # @todo: why not break? next end will lead to a to long mention as well!
                            #   I try it
                            # continue
                            break
                        # try to get label. if not existing: add 0 as label
                        label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                        # every entity only onces
                        entity_labels.pop((doc_entity_start, doc_entity_end), None)
                        # entity_start+1: target_tokens[0] is [CLS], entity_start should be right shifted by 1.
                        entity_infos.append(
                            (
                                (context_entity_start + 1, context_entity_end),
                                label,
                                (
                                    subword2token[doc_entity_start],
                                    subword2token[doc_entity_end - 1],
                                ),
                            )
                        )
                        # for x in entity_infos, x[0]:(start subtoken, end subtoken) x[2]: (start token, end token)

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
                #         'example_index': (line_idx, n),
                #         'example_L': len(entity_infos)
                #     }

                #     self.data.append(item)

                # maxR: max candidate entities numbers among sentences
                dL = self.max_pair_length
                if self.args.shuffle:
                    random.shuffle(entity_infos)
                if self.args.group_sort:
                    # sort by begin or by end and sort ascending or descending
                    group_axis = np.random.randint(2)
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(
                        key=lambda x: (x[0][group_axis], x[0][1 - group_axis]),
                        reverse=sort_dir,
                    )

                if not self.args.group_edge:
                    for i in range(0, len(entity_infos), dL):
                        # pdb.set_trace()
                        examples = entity_infos[i : i + dL]
                        item = {
                            "sentence": target_tokens,
                            "examples": examples,
                            "example_index": (line_idx, sentence_idx),
                            "example_L": len(entity_infos),
                            "sent_length": sent_len,
                        }
                        self.data.append(item)
                else:
                    if self.args.group_axis == -1:
                        group_axis = np.random.randint(2)
                    else:
                        group_axis = self.args.group_axis
                    sort_dir = bool(np.random.randint(2))
                    entity_infos.sort(
                        key=lambda x: (x[0][group_axis], x[0][1 - group_axis]),
                        reverse=sort_dir,
                    )
                    _start = 0
                    while _start < len(entity_infos):
                        _end = _start + dL
                        if _end >= len(entity_infos):
                            _end = len(entity_infos)
                        else:
                            while (
                                entity_infos[_end - 1][0][group_axis]
                                == entity_infos[_end][0][group_axis]
                                and _end > _start
                            ):
                                _end -= 1
                            if _start == _end:
                                _end = _start + dL

                        examples = entity_infos[_start:_end]

                        item = {
                            "sentence": target_tokens,
                            "examples": examples,
                            "example_index": (line_idx, n),
                            "example_L": len(entity_infos),
                            "sent_length": sent_len,
                        }

                        self.data.append(item)
                        _start = _end

        self.logger.info("maxL: %d", maxL)  # 334
        self.logger.info("maxR: %d", maxR)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # for a sentence
        entry = self.data[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(
            entry["sentence"]
        )  # turn input tokens into indices
        L = len(input_ids)

        input_ids += [0] * (
            self.max_seq_length - len(input_ids)
        )  # padding with 0        # shape: max_seq_length
        position_plus_pad = (
            int(self.model_type.find("roberta") != -1) * 2
        )  # pad 2 for 'Roberta', 0 for others

        # print(f'input_ids length: {len(input_ids)}')

        if self.model_type not in ["bertspan", "robertaspan", "albertspan"]:
            if self.model_type.startswith("albert"):
                input_ids = (
                    input_ids
                    + [30000] * (len(entry["examples"]))
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )
                input_ids = (
                    input_ids
                    + [30001] * (len(entry["examples"]))
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )
            elif self.model_type.startswith("roberta"):
                input_ids = (
                    input_ids
                    + [50261] * (len(entry["examples"]))  # token based on "[MASK]"
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )
                input_ids = (
                    input_ids
                    + [50262] * (len(entry["examples"]))  # token based on "entity"
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )
            else:  # BERT
                input_ids = (
                    input_ids
                    + [1] * (len(entry["examples"]))
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )  # 1 for entities start, 0 for padding.    shape: max_seq_length + max_pair_length
                input_ids = (
                    input_ids
                    + [2] * (len(entry["examples"]))
                    + [0] * (self.max_pair_length - len(entry["examples"]))
                )  # 2 for entities end, 0 for padding.    shape: max_seq_length + 2*max_pair_length

            attention_mask = torch.zeros(
                (
                    self.max_entity_length + self.max_seq_length,
                    self.max_entity_length + self.max_seq_length,
                ),
                dtype=torch.int64,
            )
            # max_entity_length = max_pair_length * 2
            attention_mask[:L, :L] = 1
            position_ids = (
                list(range(position_plus_pad, position_plus_pad + self.max_seq_length))
                + [0] * self.max_entity_length
            )

        else:
            attention_mask = [1] * L + [0] * (self.max_seq_length - L)
            attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
            position_ids = (
                list(range(position_plus_pad, position_plus_pad + self.max_seq_length))
                + [0] * self.max_entity_length
            )

        # print(f'input_ids length: {len(input_ids)}')

        labels = []
        mentions = []
        mention_pos = []
        num_pair = self.max_pair_length

        full_attention_mask = (
            [1] * L + [0] * (self.max_seq_length - L) + [0] * (self.max_pair_length) * 2
        )  # max_seq_length | max_pair_length | max_pair_length;   full sequence: length L for input subtokens and (max_seq_length - L) paddings; max_pair_length for entity starts; max_pair_length for entity ends

        for x_idx, x in enumerate(entry["examples"]):
            # x_idx: entities idx for a sentence, entry['examples']: entity_infos
            m1 = x[0]  # entity positions: (start idx, end idx)
            label = x[1]
            mentions.append(x[2])
            mention_pos.append((m1[0], m1[1]))
            labels.append(label)

            # Why not in BERT, ROBERTA, ALBERT?
            # +++++++++++++++++++++++++++++++++
            if self.model_type in ["bertspan", "robertaspan", "albertspan"]:
                continue

            w1 = x_idx  # for start subtoken of an entity
            w2 = w1 + num_pair  # for end subtoken of an entity

            w1 += (
                self.max_seq_length
            )  # max_seq_length|w1(0), w1(1), ...|w2(0), w2(1), ...;
            w2 += self.max_seq_length
            position_ids[w1] = m1[0]
            position_ids[w2] = m1[1]

            for xx in [w1, w2]:
                full_attention_mask[xx] = 1
                for yy in [w1, w2]:
                    attention_mask[xx, yy] = 1
                attention_mask[xx, :L] = 1

        labels += [-1] * (num_pair - len(labels))  # padding with -1
        mention_pos += [(0, 0)] * (num_pair - len(mention_pos))  # padding with (0, 0)

        item = [
            torch.tensor(input_ids),
            attention_mask,
            torch.tensor(position_ids),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(mention_pos),
            torch.tensor(full_attention_mask),
        ]

        if self.evaluate:
            # ------add sent lengths-------
            item.append(torch.tensor(entry["sent_length"]))
            # -----------------------------
            item.append(entry["example_index"])
            item.append(mentions)
        return item

    def is_punctuation(self, char):
        # obtained from:
        # https://github.com/huggingface/transformers/blob/5f25a5f367497278bf19c9994569db43f96d5278/transformers/tokenization_bert.py#L489
        cp = ord(char)
        if (
            (cp >= 33 and cp <= 47)
            or (cp >= 58 and cp <= 64)
            or (cp >= 91 and cp <= 96)
            or (cp >= 123 and cp <= 126)
        ):
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
        return escape_to_original.get(token, token)

    @staticmethod
    def collate_fn(batch):
        fields = [x for x in zip(*batch)]

        num_metadata_fields = 2
        stacked_fields = [
            torch.stack(field) for field in fields[:-num_metadata_fields]
        ]  # don't stack metadata fields
        stacked_fields.extend(
            fields[-num_metadata_fields:]
        )  # add them as lists not torch tensors

        return stacked_fields
