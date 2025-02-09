import itertools
import json
import math
import os
import pdb

# from multiprocessing.sharedctypes import Value
# from re import I
from typing import List

import numpy as np

# from run_hgnn import get_saved_checkpoint
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset  # , TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import RobertaTokenizer

WEIGHTS_NAME = "pytorch_model.bin"


class SentenceSubjectCandidate:
    # for each sentence with m entities, there are m subjects, objects and m**2 relations.
    # a sent_item corresponds to one subject, it records m relations related to this subject
    # a sentence contains m sent_items
    def __init__(self, id_tuple, sub_tokens, examples, sub, rel_labels):
        self.index = id_tuple  # e.g.: (0,0)
        self.sub_tokens = sub_tokens
        self.relations = examples
        # e.g.: [((2, 2, 5), 0, (0, 0)), ((17, 18, 2), 0, (10, 10)),
        #  ((25, 30, 2), 0, (17, 20))],  formation: ((left, right, ner_label),
        #  rel_label, obj_token_index_tuple);
        #  left/right: obj subtoken index; obj token index tuples: [(0,0), (10,10), (17,20)]
        self.sub = sub  # e.g: ([0, 0, 'Material'], (1, 3), 5)
        self.rel_labels = rel_labels  # e.g.: [0,0,0]
        if len(examples) > 0:
            self.build()
        else:
            self.n_ent = 0
            (
                self.subject,
                self.sub_subtoken_pos,
                self.sub_label,
                self.obj_subtoken_pos,
                self.obj_token_pos,
            ) = [], [], [], [], []

    def build(self):
        self.subject, self.sub_subtoken_pos, self.sub_label = self.sub
        self.obj_subtoken_pos = []  # e.g.: [(2, 2), (17, 18), (25, 30)]
        self.obj_token_pos = []  # e.g.: [(0, 0), (10, 10), (17, 20)]
        for entry in self.relations:
            (obj_sub_head, obj_sub_tail, _), _, (obj_head, obj_tail) = entry
            self.obj_subtoken_pos.append((obj_sub_head, obj_sub_tail))
            self.obj_token_pos.append((obj_head, obj_tail))
        self.n_ent = len(self.relations)  # e.g.: 3

    def __repr__(self):
        if self.n_ent > 0:
            repr_str = f"<{self.n_ent} relations; sub: {self.subject}; objs: obj_subtokens-{self.obj_subtoken_pos}, obj_tokens-{self.obj_token_pos}>"
        else:
            repr_str = "<0 relations; ...>"
        return repr_str


class Sentence:
    def __init__(self, id_tuple, item_list, ner_labels, words):
        self.index = id_tuple
        self.items = item_list
        self.n_items = len(item_list)
        self.ner_labels = ner_labels
        self.words = words
        if self.n_items == 0:
            self.obj_token_pos = []
        else:
            self.obj_token_pos = self.items[0].obj_token_pos

    @property
    # number of entities
    def size(self):
        return len(self.items)


class Sampler(torch.utils.data.Sampler):
    def __init__(
        self,
        buckets: List,
        shuffle: bool = False,
    ):
        self.shuffle = shuffle
        self.buckets = buckets
        self.n_samples = len(buckets)

        # self.epoch = 1

    def __iter__(self):
        # indices = [self.buckets[i] for i in torch.arange(self.n_samples).tolist()]
        return iter(self.buckets[i] for i in torch.arange(self.n_samples).tolist())

    def __len__(self):
        return self.n_samples

    # def set_epoch(self, epoch: int) -> None:
    #     self.epoch = epoch


class DistrSampler(DistributedSampler):
    def __init__(
        self,
        buckets: List,
        shuffle: bool = False,
    ):
        self.n_samples = len(buckets)
        # super().__init__(self, num_replicas=None,rank=None, drop_last=False)
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(
                    rank, num_replicas - 1
                )
            )
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = False
        if self.drop_last and len(self) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        self.shuffle = shuffle
        self.buckets = buckets

    def __iter__(self):
        # batches = []
        # for i, bucket in enumerate(self.buckets):
        #     batches.append([bucket[j] for j in range_fn(len(bucket)).tolist()])
        # self.epoch += 1

        indices = [self.buckets[i] for i in torch.arange(self.n_samples).tolist()]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.n_samples


class RelationDataset(Dataset):
    def __init__(
        self,
        logger,
        tokenizer,
        labels,
        file_path=None,
        args=None,
        evaluate=False,
        max_pair_length=None,
    ):
        self.logger = logger

        assert os.path.isfile(file_path)

        self.file_path = file_path

        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.max_pair_length = max_pair_length
        self.max_entity_length = self.max_pair_length * 2

        self.evaluate = evaluate
        self.use_typemarker = args.use_typemarker
        self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type
        self.no_sym = args.no_sym

        self.ner_label_list = labels.ner
        self.sym_labels = labels.rel.symmetric(only_nil=args.no_sym)
        self.label_list = labels.rel.all

        self.global_predicted_ners = {}
        self.initialize()

    def initialize(self):
        tokenizer = self.tokenizer
        # @todo unused?: vocab_size = tokenizer.vocab_size
        max_num_subwords = self.max_seq_length - 4  # for two marker
        label_map = {
            label: i for i, label in enumerate(self.label_list)
        }  # read from pre-defined list, scierc: {'NIL': 0, 'CONJUNCTION': 1, 'COMPARE': 2, 'PART-OF': 3, 'USED-FOR': 4, 'FEATURE-OF': 5, 'EVALUATE-FOR': 6, 'HYPONYM-OF': 7}
        ner_label_map = {
            label: i for i, label in enumerate(self.ner_label_list)
        }  # read from pre-defined list,scierc: {'NIL': 0, 'Method': 1, 'OtherScientificTerm': 2, 'Task': 3, 'Generic': 4, 'Material': 5, 'Metric': 6}
        print(label_map)
        print(ner_label_map)

        def tokenize_word(text):
            if (
                isinstance(tokenizer, RobertaTokenizer)
                and (text[0] != "'")
                and (len(text) != 1 or not self.is_punctuation(text))
            ):
                return tokenizer.tokenize(text, add_prefix_space=True)
            return tokenizer.tokenize(text)

        self.logger.info(f"loading from {self.file_path}")
        f = open(self.file_path, "r", encoding="utf-8")
        self.ner_tot_recall = 0
        self.tot_recall = 0
        self.data = []
        self.sizes = []
        self.ner_golden_labels = set([])
        self.golden_labels = set([])
        self.golden_labels_with_ner = set([])
        n_docs = 0
        n_sentences = 0
        n_sents_used = 0
        n_ents = 0
        n_ents_used = 0
        n_rels = 0
        n_rels_used = 0
        maxR = 0
        maxL = 0

        for doc_idx, line in enumerate(f):
            data = json.loads(line)
            n_docs += 1
            # if self.args.output_dir.find('test')!=-1:
            #     if len(self.data) > 100:
            #         break

            sentences = data["sentences"]
            if "predicted_ner" in data:  # e2e predict
                ner_candidates = data["predicted_ner"]
            else:
                ner_candidates = data["ner"]

            ner_gold = data["ner"]  # gold entities

            relations = data["relations"]

            for sentence_relation in relations:  # e.g.: sentence_relation: list of relations [[13, 14, 4, 7, 'USED-FOR'], [13, 14, 17, 18, 'USED-FOR'], [34, 36, 22, 23, 'USED-FOR'], [34, 36, 38, 39, 'CONJUNCTION'], [38, 39, 22, 23, 'USED-FOR']]
                for x in sentence_relation:  # x: a relation
                    self.tot_recall += 1

            # Flatten sentencs to one big document
            sentence_boundaries = [0]
            words = []  # @todo: rename to tokens_doc
            doc_length = 0
            for sentence_idx, sentence in enumerate(sentences):
                n_sentences += 1
                doc_length += len(sentence)
                sentence_boundaries.append(doc_length)
                # e.g.: [0, 41] recording the start token indices of sentences, e.g. tokens of sent0 is tokens[0:41]
                words += sentence

            # tokens example: [['towards'], ['deep'], ['analysis'], ['of'], ['compositional'], ['classes'], ['of'], ['parap', '##hr', '##ases'], [','], ['we'], ['have'], ['examined'], ['a'], ['class', '-', 'oriented'], ['framework'], ['for'], ['collecting'], ['parap', '##hr', '##ase'], ['examples'], [','], ['in'], ['which'], ['sent', '##ential'], ['parap', '##hr', '##ases'], ['are'], ['collected'], ['for'], ['each'], ['parap', '##hr', '##ase'], ['class'], ['separately'], ['by'], ['means'], ['of'], ['automatic'], ['candidate'], ['generation'], ['and'], ['manual'], ['judgement'], ['.']]
            # all subtokens of a token is in a list
            tokens = [tokenize_word(w) for w in words]
            # ['towards', 'deep', 'analysis', 'of', 'compositional', 'classes', 'of', 'parap', '##hr', '##ases', ',', 'we', 'have', 'examined', 'a', 'class', '-', 'oriented', 'framework', 'for', 'collecting', 'parap', '##hr', '##ase', 'examples', ',', 'in', 'which', 'sent', '##ential', 'parap', '##hr', '##ases', 'are', 'collected', 'for', 'each', 'parap', '##hr', '##ase', 'class', 'separately', 'by', 'means', 'of', 'automatic', 'candidate', 'generation', 'and', 'manual', 'judgement', '.']
            subwords = [w for li in tokens for w in li]
            # max length of subtokens from each sentence.
            maxL = max(maxL, len(subwords))
            # @todo unused?: subword2token = list(itertools.chain(*[[i] * len(li) for i, li in enumerate(tokens)]))                  # token_id = subword2token[subtoken_id]
            token2subword = [0] + list(
                itertools.accumulate(len(li) for li in tokens)
            )  # start_subtoken_id = token2subword[token_id]
            # @todo unused?: subword_start_positions = frozenset(token2subword)
            subword_sentence_boundaries = [
                sum(len(li) for li in tokens[:p]) for p in sentence_boundaries
            ]  # [0, 52] recording the start subtoken indices of sentences

            #for sentence_idx in range(len(subword_sentence_boundaries) - 1):
            for sentence_idx, sentence in enumerate(sentences):
                ner_candidates_sent = ner_candidates[sentence_idx]  # predicted ner
                sentence_relations = relations[sentence_idx]
                n_rels += len(sentence_relations)
                # relation list of nth sentence,  [subj_head, subj_tail, obj_head, obj_tail, label_string]
                ner_sent_gold = ner_gold[sentence_idx]
                n_ents += len(ner_sent_gold)

                entity_labels_gold = {}
                self.ner_tot_recall += len(ner_sent_gold)

                for start, end, label in ner_sent_gold:
                    # Default labe to NIL if label is not in ner_label_map
                    label = "NIL" if label not in ner_label_map else label
                    if label != "NIL":
                        entity_labels_gold[(start, end)] = label
                    # entity_labels_gold: record gold entities of current sentence,
                    # e.g.: {(4, 7): 'Task', (13, 14): 'Method',
                    #        (17, 18): 'Material', (22, 23): 'Material',
                    #        (34, 36): 'Method', (38, 39): 'Method'}
                        self.ner_golden_labels.add(
                            ((doc_idx, sentence_idx), (start, end), label)
                        )
                    # add (doc_id, sent_id) into gold entities.
                    #       e.g.: {((0, 0), (13, 14), 'Method'),
                    #              ((0, 0), (22, 23), 'Material'),
                    #              ((0, 0), (17, 18), 'Material'),
                    #              ((0, 0), (34, 36), 'Method'),
                    #              ((0, 0), (38, 39), 'Method'),
                    #              ((0, 0), (4, 7), 'Task')}

                self.global_predicted_ners[(doc_idx, sentence_idx)] = list(
                    ner_candidates_sent
                )  # search predicted entity list by (doc_id, sent_id)

                doc_sent_start, doc_sent_end = subword_sentence_boundaries[
                    sentence_idx : sentence_idx + 2
                ]

                left_length = doc_sent_start
                right_length = len(subwords) - doc_sent_end
                sentence_length = doc_sent_end - doc_sent_start
                half_context_length = int((max_num_subwords - sentence_length) / 2)
                maxR = max(maxR, sentence_length)
                if sentence_length < max_num_subwords:
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
                else:
                    left_context_length = right_context_length = 0

                if self.args.nocross:
                    left_context_length = right_context_length = 0

                doc_offset = doc_sent_start - left_context_length
                target_tokens = subwords[
                    doc_offset : doc_sent_end + right_context_length
                ]
                target_tokens = (
                    [tokenizer.cls_token]
                    + target_tokens[: self.max_seq_length - 4]
                    + [tokenizer.sep_token]
                )  # [CLS] ... [SEP]
                assert (
                    len(target_tokens) <= self.max_seq_length - 2
                )  # [CLS] and [SEP] for 2 positions

                pos2label = {}  # sub/obj tokens to gold relation type
                for (
                    subj_begin,
                    subj_end,
                    obj_begin,
                    obj_end,
                    relation_label,
                ) in sentence_relations:
                    pos_key = subj_begin, subj_end, obj_begin, obj_end
                    # Default to "NIL" if label is not in label_map
                    relation_label = (
                        "NIL" if relation_label not in label_map else relation_label
                    )

                    pos2label[pos_key] = label_map[relation_label]
                    self.golden_labels.add(
                        (
                            (doc_idx, sentence_idx),
                            (subj_begin, subj_end),
                            (obj_begin, obj_end),
                            relation_label,
                        )
                    )
                    self.golden_labels_with_ner.add(
                        (
                            (doc_idx, sentence_idx),
                            (
                                subj_begin,
                                subj_end,
                                entity_labels_gold[(subj_begin, subj_end)],
                            ),
                            (
                                obj_begin,
                                obj_end,
                                entity_labels_gold[(obj_begin, obj_end)],
                            ),
                            relation_label,
                        )
                    )
                    if relation_label in self.sym_labels[1:]:
                        self.golden_labels.add(
                            (
                                (doc_idx, sentence_idx),
                                (obj_begin, obj_end),
                                (subj_begin, subj_end),
                                relation_label,
                            )
                        )  # add reversed relations by symmetric
                        self.golden_labels_with_ner.add(
                            (
                                (doc_idx, sentence_idx),
                                (
                                    obj_begin,
                                    obj_end,
                                    entity_labels_gold[(obj_begin, obj_end)],
                                ),
                                (
                                    subj_begin,
                                    subj_end,
                                    entity_labels_gold[(subj_begin, subj_end)],
                                ),
                                relation_label,
                            )
                        )

                # add reverse relations to pos2label.
                for (
                    subj_begin,
                    subj_end,
                    obj_begin,
                    obj_end,
                    relation_label,
                ) in sentence_relations:  # example of x: [11, 13, 5, 6, 'USED-FOR']
                    # Default to "NIL" if label is not in label_map
                    relation_label = (
                        "NIL" if relation_label not in label_map else relation_label
                    )
                    rel_pos_key = (
                        obj_begin,
                        obj_end,
                        subj_begin,
                        subj_end,
                    )  # example of w: (5, 6, 11, 13), it is the reverse relation of x.
                    if (
                        rel_pos_key not in pos2label
                    ):  # no reverse relation of x in pos2label.
                        # if not self.args.no_pseudo_label:
                        if (
                            relation_label in self.sym_labels[1:]
                        ):  # if relation of x is symmetric, add reverse relation of x into pos2label.
                            pos2label[rel_pos_key] = label_map[relation_label]
                            # pos2label[rel_pos_key] get the label_id for the reverse relation of x;
                        else:
                            # if not self.args.no_pseudo_label:
                            pos2label[rel_pos_key] = (
                                label_map[relation_label]
                                + len(label_map)
                                - len(self.sym_labels)
                            )
                            # add label id for non-symmetric labels (there is no relation exists for reverse of x, so we call the reverse of x the pseudo label)

                # if not self.evaluate:
                #     entities.append((10000, 10000, 'NIL')) # only for NER
                # [[4, 5, 'NIL'], [4, 7, 'Task'], [13, 13, 'NIL'], [13, 14, 'Method'], (10000, 10000, 'NIL')]
                subject_candidates = []
                for subj_begin, subj_end, subj_label in ner_candidates_sent:
                    # iterated entity of a sentence  as subject.  e.g.: [48, 48, 'Generic']
                    # only one subject for an input sequence
                    # doc not handle entities with begin att text end
                    # (@todo @wolf: Does this fix an index error?)
                    if subj_begin == len(token2subword):
                        print("subj_begin is last token in document")
                        continue
                    subj_begin_sent = token2subword[subj_begin] - doc_offset + 1
                    # entity head id in input sequence
                    subj_end_sent = token2subword[subj_end + 1] - doc_offset
                    # entity tail id in input sequence
                    # sub_label = ner_label_map[sub[2]]
                    # predicted subject label
                    subj_label_gold = ner_label_map.get(
                        entity_labels_gold.get((subj_begin, subj_end), "NIL")
                    )
                    # assert sub_label==subj_label_gold, pdb.set_trace()
                    # we are using pruner with gold entity types

                    if self.use_typemarker:
                        type_marker_begin_idx = 2 + subj_label_gold
                        subj_marker_begin = f"[unused{type_marker_begin_idx}]"
                        type_marker_end_idx = (
                            2 + subj_label_gold + len(self.ner_label_list)
                        )
                        subj_marker_end = f"[unused{type_marker_end_idx}]"
                    else:
                        subj_marker_begin = "[unused0]"  # solid marker
                        subj_marker_end = "[unused1]"

                    # add typemarker into input subtokens (befor subject span)
                    token_before_subj = target_tokens[:subj_begin_sent]
                    token_subj = target_tokens[subj_begin_sent : subj_end_sent + 1]
                    token_after_subj = target_tokens[subj_end_sent + 1 :]
                    sub_tokens = (
                        token_before_subj
                        + [subj_marker_begin]
                        + token_subj
                        + [subj_marker_end]
                        + token_after_subj
                    )
                    subj_end_sent += 2
                    # else:
                    #     subj_begin_sent = len(target_tokens)
                    #     subj_end_sent = len(target_tokens)+1
                    #     sub_tokens = target_tokens + ['[unused0]',  '[unused1]']
                    #     sub_label = -1

                    # if subject tail exceed the max_seq_length, we skip this subject.
                    if subj_end_sent >= self.max_seq_length - 1:
                        print("subj_begin after 512 token")
                        continue

                    object_candidates = []
                    rel_labels = []
                    ner_labels = []
                    # assert(subj_end_sent < self.max_seq_length)
                    # add objects from predicted ner_candidates
                    for obj_begin, obj_end, obj_label in ner_candidates_sent:
                        if obj_begin == len(token2subword):
                            print("obj_begin is last token in document")
                            continue
                        # if obj_begin >= len(token2subword):
                        #    print(obj_begin, obj_end, obj_label)
                        #    print(words[obj_begin:obj_end + 1])
                        #    print(len(token2subword))
                        #    print(words[doc_sent_start:doc_entity_end])
                        # default ner label to "NIL" if label is not in label mapping
                        obj_label = (
                            "NIL" if obj_label not in ner_label_map else obj_label
                        )
                        obj_label_gold = entity_labels_gold.get(
                            (obj_begin, obj_end), "NIL"
                        )
                        # assert obj_label==obj_label_gold
                        # if self.model_type.endswith('nersub'):
                        # if obj_begin==sub[0] and obj_end==sub[1]:
                        # to make sure objects of different subs are equal.
                        #     continue

                        obj_begin_doc = token2subword[obj_begin]
                        obj_end_doc = token2subword[obj_end + 1]
                        obj_begin_sent = obj_begin_doc - doc_offset + 1
                        obj_end_sent = obj_end_doc - doc_offset

                        if obj_begin >= subj_begin:
                            obj_begin_sent += 1
                            if obj_begin > subj_end:
                                obj_begin_sent += 1  # left, right: object head/tail subtokens positions

                        if obj_end >= subj_begin:
                            obj_end_sent += 1
                            if obj_begin > subj_end:
                                obj_end_sent += 1

                        relation_label = pos2label.get(
                            (subj_begin, subj_end, obj_begin, obj_end), 0
                        )

                        if obj_end_sent >= self.max_seq_length - 1:
                            print("obj_end is after 512 token")
                            continue
                        obj_label_idx = ner_label_map[obj_label]
                        obj_in_sentence = (obj_begin_sent, obj_end_sent, obj_label_idx)
                        if relation_label != 0:
                            n_rels_used += 1
                        object_candidates.append(
                            (obj_in_sentence, relation_label, (obj_begin, obj_end))
                        )
                        # left/right: subtoken indices; ner_label:predicted ner label; label: gold relation label; obj: token indices
                        rel_labels.append(relation_label)
                        ner_labels.append(ner_label_map[obj_label_gold])


                    if subj_label_gold != 0: n_ents_used += 1
                    subject_candidate = SentenceSubjectCandidate(
                        id_tuple=(doc_idx, sentence_idx),
                        sub_tokens=sub_tokens,  # context text (incl. sentence with subject marker, but no object Marker)
                        examples=object_candidates,  # for each obj candidates finaly a
                        sub=(
                            (subj_begin, subj_end, subj_label),
                            (subj_begin_sent, subj_end_sent),
                            subj_label_gold,
                        ),
                        rel_labels=rel_labels,
                    )
                    subject_candidates.append(subject_candidate)
                    # if sentence_length>self.max_seq_length:
                    #     pdb.set_trace()
                # if sentence_length>self.max_seq_length:
                #         pdb.set_trace()
                if len(subject_candidates) == 0:
                    # self.logger.info(f'sent id:{(doc_idx, sentence_idx)}, no predicted entities!')
                    ner_labels = []
                new_sent = Sentence(
                    id_tuple=(doc_idx, sentence_idx),
                    item_list=subject_candidates,
                    ner_labels=ner_labels,
                    words=target_tokens,
                )
                n_sents_used += 1
                self.data.append(new_sent)
                self.sizes.append(new_sent.size)
                # input for a subject and all possible objects are done, may be divided into several sequences in a batch.

        print("n_docs:", n_docs)
        print("n_sents:", n_sentences)
        print("n_sents_used:", n_sents_used)
        print("n_ents:", n_ents)
        print("n_ents_used:", n_ents_used)
        print("pruner entity recall:", n_ents_used / n_ents)
        print("n_rels:", n_rels)
        print("n_rels_used:", n_rels_used)
        self.logger.info("maxR: %s", maxR)
        self.logger.info("maxL: %s", maxL)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = []
        sent = self.data[idx]
        sent_index = sent.index
        ner_labels = sent.ner_labels
        obj_token_pos = sent.obj_token_pos

        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        sub_subtoken_pos_list = []
        rel_labels_list = []
        # obj_token_pos_list = []
        subs = []

        for i, entry in enumerate(sent.items):
            rel_labels = entry.rel_labels
            sub = entry.subject
            sub_subtoken_pos = entry.sub_subtoken_pos
            # @todo not used?: sub_label = entry.sub_label

            # if self.args.use_type_mem:
            #     type_tokens = ['[unused20]'] + [f'[unused{x+20}]' for x in range(self.num_ner_labels)] + ['[unused50]'] + [f'[unused{x+50}]' for x in range(self.num_labels)]
            #     input_tokens = entry.sub_tokens + type_tokens
            #     self.rel_seq_length = len(entry.sub_tokens)
            #     self.max_seq_length = self.args.max_seq_length
            # else:
            input_tokens = entry.sub_tokens

            input_ids = self.tokenizer.convert_tokens_to_ids(
                input_tokens
            )  # input of subtokens

            L = len(input_ids)
            input_ids += (
                [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
            )  # add padding of subtokens, max length = max_seq_length, include subject markers

            n_ent = entry.n_ent

            if self.model_type.startswith("albert"):
                input_ids = (
                    input_ids + [30002] * n_ent
                )  # we need self.max_pair_length > 18 (max ent number)
                input_ids = input_ids + [30003] * n_ent
            else:
                input_ids = input_ids + [3] * n_ent
                input_ids = input_ids + [4] * n_ent

            tot_seq_len = len(input_ids)
            attention_mask = torch.zeros((tot_seq_len, tot_seq_len), dtype=torch.int64)
            # if self.args.use_type_mem:
            #     attention_mask[:self.rel_seq_length, :self.rel_seq_length] = 1
            #     # solid mark for subject could see mem tokens
            #     subj_begin_sent, subj_end_sent = entry.sub[1]
            #     attention_mask[subj_begin_sent, self.rel_seq_length:L] = 1
            #     attention_mask[subj_end_sent, self.rel_seq_length:L] = 1
            #     # only mem tokens could see input tokens
            #     attention_mask[self.rel_seq_length:L, :L] = 1
            # else:
            attention_mask[:L, :L] = 1

            obj_subtoken_pos = (
                entry.obj_subtoken_pos
            )  # object subtoken head/tail positions
            # obj_token_pos = entry.obj_token_pos                # object token head/tail positions

            position_ids = list(range(self.max_seq_length)) + [0] * (
                tot_seq_len - self.max_seq_length
            )

            # if self.args.use_type_mem:
            #     position_ids[self.rel_seq_length:L] = [0]*(L-self.rel_seq_length)
            num_pair = n_ent

            for x_idx in range(n_ent):
                obj_sub_head, obj_sub_tail = obj_subtoken_pos[x_idx]

                w1 = x_idx
                w2 = w1 + num_pair

                w1 += self.max_seq_length  # obj_head
                w2 += self.max_seq_length  # obj_tail

                position_ids[w1] = obj_sub_head
                position_ids[w2] = obj_sub_tail

                # if self.args.visible_in_ents:
                for xx in [w1, w2]:
                    for yy in [w1, w2]:
                        attention_mask[xx, yy] = 1  # start/end could see each other
                    attention_mask[xx, :L] = 1  # starts could see all tokens

            # obj_subtoken_pos += [(0, 0)] * (num_pair - len(obj_subtoken_pos))       # sub_subtoken_pos 不会有(0,0), [CLS]使用了0

            # rel_labels += [-1] * (num_pair - len(rel_labels))
            # ner_labels += [-1] * (num_pair - len(ner_labels))

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            position_ids_list.append(position_ids)
            sub_subtoken_pos_list.append(sub_subtoken_pos)
            # obj_token_pos_list.append(obj_token_pos)
            rel_labels_list.append(rel_labels)
            subs.append(sub)
            # sub_labels_list.append(sub_label)

        n_ent = sent.size
        if n_ent == 0:
            input_ids = torch.tensor(
                []
            )  # n_ent x n_ids (e.g. 12 x 324, 12 is the accumulate entity numbers of each sentences)
            attention_mask = torch.tensor([])  # n_ent x n_ids x n_ids
            position_ids = torch.tensor([])  # n_ent x n_ids
            sub_subtoken_pos = torch.tensor([])  # n_ent x 2
            rel_labels = torch.tensor([], dtype=torch.int64)  # n_ent x n_ent
            ner_labels = torch.tensor([], dtype=torch.int64)
        else:
            input_ids = torch.tensor(
                input_ids_list
            )  # n_ent x n_ids (e.g. 12 x 324, 12 is the accumulate entity numbers of each sentences)
            attention_mask = torch.stack(attention_mask_list)  # n_ent x n_ids x n_ids
            position_ids = torch.tensor(position_ids_list)  # n_ent x n_ids
            sub_subtoken_pos = torch.tensor(sub_subtoken_pos_list)  # n_ent x 2
            rel_labels = torch.tensor(
                rel_labels_list, dtype=torch.int64
            )  # n_ent x n_ent
            ner_labels = torch.tensor(ner_labels, dtype=torch.int64)  # shape: n_ent

        items = dict(
            indexs=sent_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            sub_positions=sub_subtoken_pos,
            sub=subs,
            rel_labels=rel_labels,
            ner_labels=ner_labels,
            obj_token_pos=obj_token_pos,
            n_ent=n_ent,
            subtoken_len=self.max_seq_length,
        )
        return items

    def build(
        self, batch_size, shuffle=False, n_workers=0, pin_memory=True, sort=False
    ):
        if sort:
            sort_index = np.argsort(self.sizes)  # sentence index
            sorted_size = sorted(self.sizes)
        else:
            sort_index = np.arange(len(self.sizes))
            sorted_size = self.sizes
        sizes = []
        buckets = []
        new_sizes = []
        new_buck = []
        for i, size_i in enumerate(sorted_size):
            if (sum(new_sizes) + size_i <= batch_size) or (
                len(new_sizes) == 0 and size_i >= batch_size
            ):
                new_sizes.append(size_i)
                new_buck.append(sort_index[i])
            else:
                sizes.append(new_sizes)
                buckets.append(new_buck)
                new_sizes = [size_i]
                new_buck = [sort_index[i]]
        if len(new_sizes) > 0:
            sizes.append(new_sizes)
            buckets.append(new_buck)
        self.buckets = buckets
        sampler = Sampler if self.args.local_rank == -1 else DistrSampler
        # e.g. buckets: [[0, 1], [2, 3], [4, 5], [6, 7], [8]] indicate which sents are in one batch.
        self.loader = DataLoader(
            dataset=self,
            batch_sampler=sampler(buckets=self.buckets),
            num_workers=n_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
        )
        return self.loader

    def collate_fn(self, batch):
        """
        batch: list of items got by Dataset.__getitem__()
        """
        ent_numbers = torch.tensor([sent["n_ent"] for sent in batch])
        n_sent = len(batch)
        if len(ent_numbers) == 0:
            pdb.set_trace()
        if len(ent_numbers) == 1:  # only one sentence
            items = batch[0]
            # print(f'index:{items["indexs"]}')
            # print(f'ner_labels:{items["ner_labels"]}')
            items.pop("n_ent")
            items.pop("subtoken_len")
            items.update({"ent_numbers": ent_numbers})
            batch = items
            for k in ["rel_labels", "ner_labels"]:
                batch[k] = batch[k].unsqueeze(0)
            for k in ["indexs", "sub", "obj_token_pos"]:
                batch[k] = [batch[k]]
            return batch
        else:
            subs = []
            obj_token_pos = []
            indexs = []
            max_ent_num = max(ent_numbers)
            seqlen = batch[0].pop("subtoken_len")
            bs = sum(
                ent_numbers
            )  # bs: number of input sequences for sentences in batch
            batch_input_ids = (
                torch.zeros((bs, seqlen + 2 * max_ent_num), dtype=torch.int)
                + self.tokenizer.pad_token_id
            )
            batch_pos_ids = torch.zeros((bs, seqlen + 2 * max_ent_num), dtype=torch.int)
            batch_sub_subtoken_pos = torch.zeros((bs, 2), dtype=torch.long)
            batch_attn_mask = torch.zeros(
                (bs, seqlen + 2 * max_ent_num, seqlen + 2 * max_ent_num),
                dtype=torch.int,
            )
            batch_rel_labels = (
                torch.zeros((n_sent, max_ent_num, max_ent_num), dtype=torch.int64) - 1
            )
            batch_ner_labels = torch.zeros((n_sent, max_ent_num), dtype=torch.int64) - 1
            n_ent_0 = 0
            for i, items_i in enumerate(batch):
                # print(f'index:{items_i["indexs"]}')
                # print(f'ner_labels:{items_i["ner_labels"]}')
                indexs.append(items_i["indexs"])
                n_ent = ent_numbers[i]
                if n_ent > 0:
                    n_ent_1 = n_ent_0 + n_ent
                    input_ids_split = items_i["input_ids"].split(
                        [seqlen, n_ent, n_ent], dim=-1
                    )
                    batch_input_ids[n_ent_0:n_ent_1, :seqlen] = input_ids_split[0]
                    for j in range(2):
                        batch_input_ids[
                            n_ent_0:n_ent_1,
                            seqlen + j * max_ent_num : seqlen + j * max_ent_num + n_ent,
                        ] = input_ids_split[j + 1]
                    pos_ids_split = items_i["position_ids"].split(
                        [seqlen, n_ent, n_ent], dim=-1
                    )
                    batch_pos_ids[n_ent_0:n_ent_1, :seqlen] = pos_ids_split[0]
                    for j in range(2):
                        batch_pos_ids[
                            n_ent_0:n_ent_1,
                            seqlen + j * max_ent_num : seqlen + j * max_ent_num + n_ent,
                        ] = pos_ids_split[j + 1]
                    # batch_pos_ids[n_ent_0:n_ent_1, seqlen:seqlen+max_ent_num] = pos_ids_split[1]
                    # batch_pos_ids[n_ent_0:n_ent_1, seqlen+max_ent_num:seqlen+2*max_ent_num] = pos_ids_split[2]

                    sub_subtoken_pos_i = items_i["sub_positions"]
                    batch_sub_subtoken_pos[n_ent_0:n_ent_1, :] = sub_subtoken_pos_i

                    attn_mask_i = items_i["attention_mask"]
                    for j in range(2):
                        batch_attn_mask[
                            n_ent_0:n_ent_1,
                            seqlen + j * max_ent_num : seqlen + n_ent + j * max_ent_num,
                            :seqlen,
                        ] = attn_mask_i[
                            :, seqlen + j * n_ent : seqlen + (j + 1) * n_ent, :seqlen
                        ]
                        for k in range(2):
                            batch_attn_mask[
                                n_ent_0:n_ent_1,
                                seqlen + j * max_ent_num : seqlen
                                + n_ent
                                + j * max_ent_num,
                                seqlen + k * max_ent_num : seqlen
                                + n_ent
                                + k * max_ent_num,
                            ] = attn_mask_i[
                                :,
                                seqlen + j * n_ent : seqlen + (j + 1) * n_ent,
                                seqlen + k * n_ent : seqlen + (k + 1) * n_ent,
                            ]

                    rel_labels_i = items_i["rel_labels"]
                    batch_rel_labels[i, :n_ent, :n_ent] = rel_labels_i

                    ner_labels_i = items_i["ner_labels"]
                    batch_ner_labels[i, :n_ent] = ner_labels_i

                    subs.append(items_i["sub"])
                    obj_token_pos.append(items_i["obj_token_pos"])

                    n_ent_0 = n_ent_1
                else:
                    obj_token_pos.append([])
                    subs.append([])

                #     print(f'sent id: {items_i["indexs"]}')
            # if 0 in ent_numbers:
            #     pdb.set_trace()
            batch = dict(
                indexs=indexs,
                input_ids=batch_input_ids,
                attention_mask=batch_attn_mask,
                position_ids=batch_pos_ids,
                sub_positions=batch_sub_subtoken_pos,
                rel_labels=batch_rel_labels,
                ner_labels=batch_ner_labels,
                sub=subs,
                obj_token_pos=obj_token_pos,
                ent_numbers=ent_numbers,
            )
        return batch
