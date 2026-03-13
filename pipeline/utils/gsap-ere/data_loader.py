import pdb
import random
import itertools
import json
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import unicodedata
from pathlib import Path
from dataclasses import dataclass
import numpy as np


@dataclass
class Span:
    begin: int
    end: int
    begin_subtoken: int
    end_subtoken: int
    label: int


@dataclass
class Sample:
    context_tokens: list[str]
    spans: list[Span]
    sentence_id: tuple[int, int]  # doc_idx, sentence_idx
    n_span_sentence: int
    len_sent: int


class PrunerDataArguments:
    label_set: str = "gsap"
    model_type: str = "bertspanmarkerpruner"
    max_sequence_length: int = 256
    max_entities: int = 64  # former max_pair_length # this * 2 + max_sequence_length should sum up to 512!
    max_n_gram_size: int = 12
    nocross: bool = True  # no context expansion for the sentences
    group_axis: int = -1  # {-1, 0, 1}
    group_edge: bool = False
    group_sort: bool = False
    group_shuffle: bool = False  # orginal "shuffle"
    # local_rank: int = -1


@dataclass
class DocumentSubTokenIndex:
    # list of all subtokens of a document
    sub_tokens: list[int]  # texts of subtokens (n=len(sentences))
    input_ids: list[int]  # ids of subtokens (n=len(sentences))
    # map from token to first subtoken (len: number of token)
    token_to_sub: np.ndarray  # dim (<number of token>, 2) i.e., begin and end

    def __init__(self, doc, tokenizer):
        tokens = doc.tokens
        ## make sure, there are no replacements like "-LRB-" for "("
        tokens = _normalize_brackets(tokens)
        sub_tokens = []
        token_to_subtoken_begin = []
        token_to_subtoken_end = []
        tokens_splitted = [_tokenize_word(token, tokenizer) for token in tokens]
        token_position = 0
        for token_idx, token_splitted in enumerate(tokens_splitted):
            token_to_subtoken_begin.append(token_position)
            sub_tokens.extend(token_splitted)
            token_position += len(token_splitted)
            token_to_subtoken_end.append(token_position - 1)
        self.token_to_sub = np.array([token_to_subtoken_begin, token_to_subtoken_end]).T
        self.sub_tokens = np.array(sub_tokens)
        self.input_ids = np.array(
            tokenizer.convert_tokens_to_ids(self.sub_tokens), dtype=int
        )


@dataclass
class Document_:
    doc_id: int
    tokens: list[str]
    sentences_begin: list[int]  # dim: len(sentences)
    sentences_end: list[int]  # dim: len(sentences)
    ner: list[list[tuple[int, int, int]]]
    relations: list[list[tuple[int, int, int, int, int]]]
    ner_candidates: list[list[tuple[int, int, int]]]

    @classmethod
    def from_dict(cls, doc):
        tokens = []
        sentences_begin = []
        sentences_end = []
        current_begin = 0
        for sent in doc["sentences"]:
            sentences_begin.append(current_begin)
            tokens.extend(sent)
            current_begin += len(sent)
            sentences_end.append(current_begin - 1)
        use_from_dict_doc = {"doc_id", "ner", "ner_candidates", "relations"}
        # not_used = [key for key in doc.keys() if key not in use_from_dict_doc]
        doc = {k: v for k, v in doc.items() if k in use_from_dict_doc}
        doc["tokens"] = tokens
        doc["sentences_begin"] = sentences_begin
        doc["sentences_end"] = sentences_end
        doc = cls(**doc)
        return doc


@dataclass
class Document:
    """
    Interesting questions:
    Maximum number of token, subtoken in document
    """

    doc_id: int
    tokens: list[str]
    sentence_boundaries: list[int]  # dim: len(sentences) + 1
    sent_lens = list[int]  # dim: len(sentences)
    subword_tokens: list[str]  #
    ners: list[list[tuple[int, int, int]]]
    # mappings
    subword_to_token: list[int]  # dim: len(subword_tokens)
    token_to_first_subword: list[int]  # dim: len(tokens)
    subword_start_positions: set[int]  # dim: len(tokens)

    def __init__(self, document, tokenizer):
        sentences = document["sentences"]
        self.doc_id = document["doc_id"]

        self.ners = document["ner"]
        self.tokens = []
        n_all_tokens = 0  # total number of tokens
        # store the token numbers of sentences: e.g. [0, 41, 70], indicate the boudaries.
        self.sentence_boundaries = [0]
        self.sent_lens = []  # lengths of each sentence in token
        for sentence in sentences:
            ## make sure, there are no replacements like "-LRB-" for "("
            sentence = _normalize_brackets(sentence)
            n_all_tokens += len(sentence)
            self.sent_lens.append(len(sentence))
            self.sentence_boundaries.append(n_all_tokens)
            self.tokens += sentence

        tokens_splitted = [_tokenize_word(token, tokenizer) for token in self.tokens]
        # print("n token:", len(tokens))
        # print(len(tokens_splitted))
        self.subword_tokens = [
            subword_token
            for token_splitted in tokens_splitted
            for subword_token in token_splitted
        ]
        # print("n subwords:", len(subword_tokens))

        ## first this function creates list: [1,1,1, 2,2,2,2, ...]
        ## repetitions are the number of subword_tokens for each token
        self.subword_to_token = list(
            itertools.chain(
                *[
                    [token_idx] * len(token_splitted)
                    for token_idx, token_splitted in enumerate(tokens_splitted)
                ]
            )
        )
        #  mapping from token to first! subword_token!
        ## fyi: itertools.accumulate([1, 2, 3, 4, 5]) → 1 3 6 10 15
        self.token_to_first_subword = [0] + list(
            itertools.accumulate(
                len(token_splitted) for token_splitted in tokens_splitted
            )
        )
        assert (
            len(self.token_to_first_subword) == len(tokens_splitted) + 1
        ), pdb.set_trace()
        assert len(tokens_splitted) == len(self.tokens), pdb.set_trace()
        # now check consistency
        for sentence_ners in self.ners:
            for start, end, label in sentence_ners:
                assert start < len(self.tokens), pdb.set_trace()
                assert end < len(self.tokens), pdb.set_trace()

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
        self.subword_start_positions = frozenset(self.token_to_first_subword)
        # @todo this should be the same as
        # subword_sentence_boundaries = [token_to_first_subword[sentence_start_token_idx]
        #                                 for sentence_start_token_idx
        #                                 in sentence_coundaries]
        self.subword_sentence_boundaries = [
            sum(len(subword) for subword in tokens_splitted[:sentence_start_token_idx])
            for sentence_start_token_idx in self.sentence_boundaries
        ]
        ## @todo shouldn't the number of sentences be equal to the number of sentence_boundaries?
        ## I think we have a boundary at the very end is this wanted? Yes, it is the ending of the last sentence
        assert (
            len(self.subword_sentence_boundaries) == len(sentences) + 1
        ), pdb.set_trace()
        assert len(self.sentence_boundaries) == len(
            self.subword_sentence_boundaries
        ), pdb.set_trace()


class PrunerDataset(Dataset):
    def __init__(
        self, logger, tokenizer, args: PrunerDataArguments, evaluate: bool = False
    ):
        # if not evaluate:
        #     file_path = os.path.join(args.data_dir, args.train_file)
        # else:
        #     if do_test:
        #         file_path = os.path.join(args.data_dir, args.test_file)
        #     else:
        #         file_path = os.path.join(args.data_dir, args.dev_file)

        self.logger = logger
        self.tokenizer = tokenizer
        self.max_sequence_length = args.max_sequence_length

        self.evaluate = evaluate
        # self.local_rank = args.local_rank
        self.args = args
        self.model_type = args.model_type

        self.ner_label_list = NER_LABEL_LISTS.get(args.label_set)
        if self.ner_label_list is None:
            raise Exception(f"No valid --label_set parameter '{args.label_set}'")
        self.ner_label_map = {label: i for i, label in enumerate(self.ner_label_list)}

        print(self.ner_label_list)
        self.max_entities = args.max_entities
        self.max_n_gram_size = args.max_n_gram_size

    def initialize_from_document(self, document_raw):
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        document = Document(document_raw, self.tokenizer)
        self.logger.info(f"document len: {len(document.sentence_boundaries)}")
        samples = self._document_to_samples(document)
        self.data.extend(samples)

    def initialize_from_filename(self, filename):
        # e.g.: {'NIL': 0, 'Method': 1, 'OtherScientificTerm': 2, 'Task': 3, 'Generic': 4, 'Material': 5, 'Metric': 6}

        # read file
        file_path = Path(filename)
        if not file_path.is_file():
            self.logger.warn(f"File does not exist: {file_path}")
        assert file_path.is_file()

        f = open(filename, "r", encoding="utf-8")
        self.data = []
        self.tot_recall = 0
        self.ner_golden_labels = set([])
        # For each document
        for line_idx, line in enumerate(f):
            document_raw = json.loads(line)
            document_raw["doc_id"] = line_idx
            document = Document(document_raw, self.tokenizer)
            samples = self._document_to_samples(document)
            self.data.extend(samples)

    def _document_to_samples(self, doc: Document):
        samples = []

        # iterate through the sentences by sentence_idx
        for sentence_idx in range(len(doc.subword_sentence_boundaries) - 1):
            new_samples = self._sentence_to_samples(sentence_idx, doc)
            samples.extend(new_samples)
            self.logger.info(
                f"document_to_samples, sentence_idx: {sentence_idx}, {len(new_samples)}"
            )
        return samples

    def _sentence_to_samples(self, sentence_idx: int, doc: Document):
        # get subword of sentence start and sentence end
        sent_start, sent_end = doc.subword_sentence_boundaries[
            sentence_idx : sentence_idx + 2
        ]
        sentence_ners = doc.ners[sentence_idx]
        sent_len = doc.sent_lens[sentence_idx]
        # nth sentence
        # length of sentence in token (not subword_token)
        # sent_id = self.sent_ids[doc_id][sentence_idx]

        self.tot_recall += len(sentence_ners)
        # a mapping from subword_token_start, subword_token_end to label
        entity_labels = {}
        # start: start token idx; end: end token idx;
        for start, end, label in sentence_ners:
            # if start >= len(token_to_first_subword) or end+1 >= len(token_to_first_subword):
            #    print(start, end, label)
            #    print(token_to_first_subword[7000:])
            key = doc.token_to_first_subword[start], doc.token_to_first_subword[end + 1]

            entity_labels[key] = self.ner_label_map[label]
            self.ner_golden_labels.add(
                ((doc.doc_id, sentence_idx), (start, end), label)
            )
        len_doc = len(doc.subword_tokens)
        (
            context_begin,
            context_end,
            sentence_begin_in_context,
            sentence_end_in_context,
        ) = self._extend_context(sent_start, sent_end, len_doc)

        ## now collect entity infos!
        entity_infos = self._collect_entity_infos(
            context_begin,
            sentence_begin_in_context,
            sentence_end_in_context,
            doc,
            entity_labels,
        )
        self.logger.info(f"entity_infos: {len(entity_infos)}")
        # start idx of contexted sequence.
        context_tokens = doc.subword_tokens[context_begin:context_end]

        context_tokens = (
            [self.tokenizer.cls_token] + context_tokens + [self.tokenizer.sep_token]
        )

        span_groups = self._grouped_spans(
            entity_infos,
            self.args.group_shuffle,
            self.args.group_sort,
            self.args.group_edge,
            self.args.group_axis,
        )
        span_groups = list(span_groups)
        self.logger.info(f"{len(span_groups)}")
        samples = []
        for span_group in span_groups:
            sample = {
                "sentence": context_tokens,
                "spans": span_group,
                "example_index": (doc.doc_id, sentence_idx),
                "example_L": len(entity_infos),
                "sent_length": sent_len,
            }
            samples.append(sample)
        return samples

    def _extend_context(self, sent_start, sent_end, len_doc):
        max_num_subwords = self.max_sequence_length - 2
        # subword_tokens length of sentence
        sentence_length = sent_end - sent_start
        if sentence_length > max_num_subwords:
            # sentence to long
            # What to do now? Shorten
            # problem with evaluation when just cut?
            # ???
            # ?? if max_num_subwords < sentence_length, choose subtokens from the center of current sentence.
            print(f"Sentence to long ({sentence_length})")
            context_begin = sent_start
            context_end = sent_end - sentence_length - max_num_subwords
            sent_in_context_begin = 0
            sent_in_context_end = context_end
        elif self.args.nocross:
            # use no context
            context_begin, context_end = sent_start, sent_end
            sent_in_context_begin = 0
            sent_in_context_end = context_end
        else:
            # left_length: subtoken numbers before current sentenc
            left_length = sent_start  # subword_tokens to the left

            # right_length: subtoken numbers after current sentence.
            right_length = len_doc - sent_end

            # half_context_length | current sentence length | half_context_length
            half_context_length = int((max_num_subwords - sentence_length) / 2)

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
            # pdb.set_trace()
            context_begin = sent_start - left_context_length
            context_end = sent_end + right_context_length
            sent_in_context_begin = left_context_length
            sent_in_context_end = sent_in_context_begin + sentence_length
        return context_begin, context_end, sent_in_context_begin, sent_in_context_end

    def _collect_entity_infos(
        self,
        context_begin,
        sentence_begin_in_context,
        sentence_end_in_context,
        doc,
        entity_labels,
    ):
        entity_infos = []

        ## Search entities in sentence
        for context_entity_start in range(
            sentence_begin_in_context, sentence_end_in_context
        ):
            # entity start idx in input sequence.
            doc_entity_start = context_begin + context_entity_start
            # doc_entity_start: start idx in total text; entity在当前doc (包含许多句子)的起始位置
            if doc_entity_start not in doc.subword_start_positions:
                # let candidate entities not begin inside tokens!
                continue
            for context_entity_end in range(
                context_entity_start + 1,
                sentence_end_in_context + 1,
            ):
                doc_entity_end = context_begin + context_entity_end
                if doc_entity_end not in doc.subword_start_positions:
                    # let candidate enities not end inside tokens!
                    continue

                if (
                    doc.subword_to_token[doc_entity_end - 1]
                    - doc.subword_to_token[doc_entity_start]
                    + 1
                    > self.max_n_gram_size
                ):
                    # if current entity length > max entity length (measured in subword token), break
                    break
                # try to get label. if not existing: add 0 as label
                label = entity_labels.get((doc_entity_start, doc_entity_end), 0)
                # every entity only onces
                entity_labels.pop((doc_entity_start, doc_entity_end), None)
                # entity_start+1: context_tokens[0] is [CLS], entity_start should be right shifted by 1.
                # entity_info: ( (start_in_context, end_in_context), label_id, (span_begin, span_end))
                entity_infos.append(
                    (
                        (context_entity_start + 1, context_entity_end),
                        label,
                        (
                            doc.subword_to_token[doc_entity_start],
                            doc.subword_to_token[doc_entity_end - 1],
                        ),
                    )
                )

        return entity_infos

    # "shuffle", "group_sort", "group_edge"
    def _grouped_spans(
        self, spans, shuffle: bool, group_sort: bool, group_edge: bool, group_axis: int
    ):
        assert group_axis in {-1, 0, 1}
        if shuffle:
            random.shuffle(spans)
        if group_sort:
            # sort by begin or by end and sort ascending or descending
            group_axis = np.random.randint(2)
            sort_direction = bool(np.random.randint(2))
            spans.sort(
                key=lambda x: (x[0][group_axis], x[0][1 - group_axis]),
                reverse=sort_direction,
            )

        if group_edge:
            if group_axis == -1:
                group_axis = np.random.randint(2)
            sort_direction = bool(np.random.randint(2))
            spans.sort(
                key=lambda x: (x[0][group_axis], x[0][1 - group_axis]),
                reverse=sort_direction,
            )
            _start = 0
            while _start < len(spans):
                _end = _start + self.max_entities
                if _end >= len(spans):
                    _end = len(spans)
                else:
                    while (
                        spans[_end - 1][0][group_axis] == spans[_end][0][group_axis]
                        and _end > _start
                    ):
                        _end -= 1
                    if _start == _end:
                        _end = _start + self.max_entities

                spans_group = spans[_start:_end]
                yield spans_group  # ??? or spans and change
                _start = _end
        else:
            for i in range(0, len(spans), self.max_entities):
                # pdb.set_trace()
                spans_group = spans[i : i + self.max_entities]
                yield spans_group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # for a sentence
        sample = self.data[idx]

        input_ids = self.tokenizer.convert_tokens_to_ids(
            sample["sentence"]
        )  # turn input tokens into indices
        text_length = len(input_ids)

        input_ids += [0] * (
            self.max_sequence_length - text_length
        )  # padding with 0        # shape: max_sequence_length

        input_ids = self._add_span_marker(input_ids, len(sample["spans"]))

        # print(f'input_ids length: {len(input_ids)}')
        position_ids = self._generate_marker_position_ids(sample["spans"])

        # attent only from marker begin to end and to full text span
        attention_mask = self._generate_attention_mask(sample["spans"], text_length)

        labels = []
        mentions = []
        mention_pos = []
        # spans: [(start_idx, end_idx), label, mention]
        for (start_idx, end_idx), label, mention in sample["spans"]:
            # x_idx: entities idx for a sentence, sample['spans']: entity_infos
            mentions.append(mention)
            mention_pos.append((start_idx, end_idx))
            labels.append(label)

        labels += [-1] * (self.max_entities - len(labels))  # padding with -1
        mention_pos += [(0, 0)] * (
            self.max_entities - len(mention_pos)
        )  # padding with (0, 0)

        item = [
            torch.tensor(input_ids),
            attention_mask,
            torch.tensor(position_ids),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(mention_pos),
        ]

        if self.evaluate:
            # ------add sent lengths-------
            item.append(torch.tensor(sample["sent_length"]))
            # -----------------------------
            item.append(sample["example_index"])
            item.append(mentions)
        return item

    def _generate_marker_position_ids(self, spans):
        """Add marker position ids from spans in context (anchors)"""
        start_idx = 2 if "roberta" in self.model_type else 0
        # copy begin and end position ids from spans to markers
        position_ids = list(range(start_idx, start_idx + self.max_sequence_length)) + [
            0
        ] * (self.max_entities * 2)
        for span_idx, ((start_idx, end_idx), label, mention) in enumerate(spans):
            start_subtoken_idx = (
                self.max_sequence_length + span_idx
            )  # for start subtoken of an entity
            end_subtoken_idx = (
                self.max_sequence_length + self.max_entities + span_idx
            )  # for end subtoken of an entity
            # copy position_id from anchor spans begin and end
            position_ids[start_subtoken_idx] = start_idx
            position_ids[end_subtoken_idx] = end_idx
        return position_ids

    def _add_span_marker(self, input_ids, len_spans):
        model_type = self.model_type
        span_begin_token_id = None
        span_end_token_id = None
        if model_type.startswith("bert"):
            span_begin_token_id = 1
            span_end_token_id = 2
        elif model_type.startswith("albert"):
            span_begin_token_id = 30000
            span_end_token_id = 30001
        elif model_type.startswith("roberta"):
            span_begin_token_id = 50261  # token based on "[MASK]"
            span_end_token_id = 30001  # token based on token "entity"
        else:
            # modelspanmarkerpruner
            raise Exception(f"{model_type} is not supported in DataLoader.")

        len_pad = self.max_entities - len_spans
        # shape: max_sequence_length + 2 * max_spans_per_sample
        input_ids = (
            input_ids
            + [span_begin_token_id] * len_spans  # spans begin
            + [0] * len_pad
            + [span_end_token_id] * len_spans  # spans end
            + [0] * len_pad
        )
        return input_ids

    def _generate_attention_mask(self, spans, text_length):
        attention_mask = torch.zeros(
            (
                self.max_sequence_length + (self.max_entities * 2),
                self.max_sequence_length + (self.max_entities * 2),
            ),
            dtype=torch.int64,
        )
        attention_mask[:text_length, :text_length] = 1
        for span_idx, ((start_idx, end_idx), label, mention) in enumerate(spans):
            start_subtoken_idx = (
                self.max_sequence_length + span_idx
            )  # for start subtoken of an entity
            end_subtoken_idx = (
                self.max_sequence_length + self.max_entities + span_idx
            )  # for end subtoken of an entity
            # attent from subtokens to full text span
            attention_mask[start_subtoken_idx, :text_length] = 1
            attention_mask[end_subtoken_idx, :text_length] = 1
            # attent from begin_subtoken to end_subtoken and back
            for x in [start_subtoken_idx, end_subtoken_idx]:
                for y in [start_subtoken_idx, end_subtoken_idx]:
                    attention_mask[x, y] = 1
        return attention_mask

    def iter_batches(self, batch_size):
        for start in range(0, len(self.data), batch_size):
            end = start + batch_size
            self.logger.info(f"batch: {start}, {end}")
            batch_items = self.data[start:end]
            batch = _collate_fn(batch_items)
            yield batch

    @staticmethod
    def collate_fn(batch):
        return _collate_fn(batch)


def _collate_fn(batch):
    fields = [x for x in zip(*batch)]

    num_metadata_fields = 2
    stacked_fields = [
        torch.stack(field) for field in fields[:-num_metadata_fields]
    ]  # don't stack metadata fields
    stacked_fields.extend(
        fields[-num_metadata_fields:]
    )  # add them as lists not torch tensors

    return stacked_fields


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
        "DataSource",
        "DatasetGeneric",
        "URL",
        "ReferenceLink",
    ],
    somd=[
        "Unknown",
        "Abbreviation",
        "AlternativeName",
        "Application",
        "Citation",
        "Developer",
        "Extension",
        "License",
        "OperatingSystem",
        "PlugIn",
        "ProgrammingEnvironment",
        "Release",
        "SoftwareCoreference",
        "URL",
        "Version",
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


def _tokenize_word(text, tokenizer):
    ## add_prefix_space treats words like words, not like beginning of sequence
    ## https://huggingface.co/docs/tokenizers/python/latest/api/reference.html
    ## @todo: Why only for RoBERTa?
    if (
        isinstance(tokenizer, RobertaTokenizer)
        and (text[0] != "'")
        and (len(text) != 1 or not is_punctuation(text))
    ):
        return tokenizer.tokenize(text, add_prefix_space=True)
    return tokenizer.tokenize(text)


def is_punctuation(char):
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


def _normalize_brackets(token):
    escape_to_original = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    return [escape_to_original.get(t, t) for t in token]


def _sentences_to_sub_token_idxs(doc, sub_token_index, len_context, expand=False):
    """
    Map sentences and spans based on document token to subtoken (e.g., BERT)
    """
    len_context -= 2  # -2 for <cls> and <sep> token at begin and end
    doc_sti = sub_token_index
    sents_sub_tokens_with_contexts = []
    for sent_idx, (sent_begin, sent_end, sent_spans) in enumerate(
        zip(doc.sentences_begin, doc.sentences_end, doc.ner_candidates)
    ):
        if not sent_spans:
            continue
        # workon sub token idxs (e.g. BERT subword token)
        sentence_sub_begin = doc_sti.token_to_sub[sent_begin, 0]
        sentence_sub_end = doc_sti.token_to_sub[sent_end, 1]
        len_sent = sentence_sub_end - sentence_sub_begin
        if len_context < len_sent:
            raise Exception(
                f"Sentence to long ({len_sent} more than {len_context} token)"
            )
        if expand:
            len_context_before_after = (len_context - len_sent) // 2
            context_sub_begin: int = max(
                0, sentence_sub_begin - len_context_before_after
            )
            context_sub_end: int = min(
                len(doc.tokens), sentence_sub_end + len_context_before_after
            )
        else:
            context_sub_begin: int = sentence_sub_begin
            context_sub_end: int = sentence_sub_end

        begins, ends, labels = zip(*sent_spans)
        begins_sub_token_idxs = doc_sti.token_to_sub[begins, 0]
        ends_sub_token_idxs = doc_sti.token_to_sub[ends, 1]
        pos_in_context_begins = (
            begins_sub_token_idxs - context_sub_begin + 1
        )  # +1 for <cls> token at begin
        pos_in_context_ends = (
            ends_sub_token_idxs - context_sub_begin + 1
        )  # +1 for <cls> token at begin
        # labels = np.array(labels) + 1 # label 1 for false and 2 for True 0 is used for padding
        spans = np.stack(
            [
                begins_sub_token_idxs,
                ends_sub_token_idxs,
                pos_in_context_begins,
                pos_in_context_ends,
                labels,
                # doc_ids,
                # sentence_ids,
                # char_begin_idx,
                # char_end_idx
            ]
        ).T
        # print(spans)
        sents_sub_tokens_with_contexts.append(
            ([context_sub_begin, context_sub_end], spans)
        )
    return sents_sub_tokens_with_contexts


def sentence_to_samples(sent_sub_tokens_with_spans, max_spans=10):
    """
    packs max_spans with the connected context as sample
    """
    context, spans = sent_sub_tokens_with_spans
    context_begin, context_end = context
    context_token_idxs = np.arange(context_begin, context_end + 1)
    # add -1 before and after for begin and end token
    context_token_idxs = np.pad(
        context_token_idxs, (1, 1), mode="constant", constant_values=-1
    )

    # print(context_token_idxs.shape)
    # print(np.arange(len(context_token_idxs)))
    context_with_pos = np.stack(
        [
            context_token_idxs,  # token_idx    [ 45, 46, ...]
            np.arange(len(context_token_idxs)),
        ]
    ).T  # position ids [  0,  1, ...]
    for span_pack in batch_iter(spans, max_spans, 0):
        yield context_with_pos.copy(), span_pack


def batch_samples(
    sub_token_index,
    sents,
    batch_size,
    len_context=256,
    n_spans_max=3,
    padding_constant=0,
    token_id_span_begin=1,
    token_id_span_end=2,
    token_id_cls=103,
    token_id_sep=104,
    document_id=0,
):
    """
    batches samples (i.e. context, span_list pairs) incl. padding and mask creation
      yields batch_size, embedding with pos vector + batch_size, list_of labels
      output dims: (batch_size, (len_context + 2 * n_spans_max), 2), # input_ids + pos
            (batch_size, n_spans_max) #labels
    """
    contexts, spans = zip(*sents)
    contexts = list(
        contexts
    )  # list of arrays of shape (<len_context>, 2) i.e., index of subtoken and pos_in_context
    spans = list(spans)  # list of span packs
    # Pad spans before batching
    spans_padded = []
    span_infos_padded = []
    # @todo: Only masks are missing
    for span in spans:
        span_info = np.stack(
            [
                span[:, 0],  # position_doc_begin
                span[:, 1],  # position_doc_end
                span[:, 2],  # position_in_context_begin
                span[:, 3],  # position_in_context_begin
                span[:, 4],  # label
                np.repeat(document_id, len(span)),  # document_id
            ]
        ).T
        span_info_padded = np.pad(
            span_info,
            ((0, n_spans_max - span_info.shape[0]), (0, 0)),
            mode="constant",
            constant_values=-1,
        )
        # print(span_info_padded)
        span_infos_padded.append(span_info_padded)
        # document id
        # replace indexes with span_begin and span_end token
        span = np.stack(
            [
                np.full(len(span), token_id_span_begin),
                np.full(len(span), token_id_span_end),
                span[:, 2],  # pos in context begin
                span[:, 3],  # pos in context end
                np.full(len(span), 1),  # full mask
            ]
        ).T
        span_padded = np.pad(
            span,
            ((0, n_spans_max - span.shape[0]), (0, 0)),
            mode="constant",
            constant_values=padding_constant,
        )
        # pad label with -1
        # label are in 5th value of the span vectors
        # labels: [label_1, ..., label_n, -1, -1] (len = n_spans_max)
        # rearrange spans (index and position vectors:
        # [[span_begin_1, ..., span_begin_n], [span_end_1, ... span_end_n]] =>
        # [span_begin_1, ..., span_begin_n, span_end_1, ... span_end_n]
        span_padded = np.stack(
            [
                np.concatenate(
                    [span_padded[:, 0], span_padded[:, 1]]
                ),  # begin and end token
                np.concatenate([span_padded[:, 2], span_padded[:, 3]]),  # position
                np.concatenate([span_padded[:, 4], span_padded[:, 4]]),  # full_mask
            ]
        ).T
        spans_padded.append(span_padded)
    # Create batches
    for batch_begin in range(0, len(contexts), batch_size):
        # masks_square = []
        batch_end = batch_begin + batch_size
        # if batch_end - batch_begin <= 0:
        #    continue
        batch_contexts = contexts[batch_begin:batch_end]
        # Add mask for each batch
        mask_size = len_context + 2 * n_spans_max
        mask_batch = np.zeros((len(batch_contexts), mask_size, mask_size))
        # print(spans)
        batch_spans = np.stack(spans_padded[batch_begin:batch_end])
        batch_span_infos = np.stack(span_infos_padded[batch_begin:batch_end])
        # Pad context to len_context
        if len_context is None:
            max_sent_len_batch = max([len(s) for s in batch_contexts])
        else:
            max_sent_len_batch = len_context
        batch_contexts_padded = []
        for context_idx, context in enumerate(batch_contexts):
            # add full mask to token infos
            #  (token_idx, pos_in_context, mask (i.e., 1 for every entry))
            mask_full = np.ones((len(context), 1), dtype=np.int64)
            # replace token_indexes with input_ids
            try:
                input_ids = sub_token_index.input_ids[context[:, 0]]
            except Exception as e:
                print(e)
                pdb.set_trace()
                raise e
            input_ids[0] = token_id_cls
            input_ids[-1] = token_id_sep
            context[:, 0] = input_ids

            context = np.hstack([context, mask_full])
            # ???? pad_len = batch_size - len(context)
            context_padded = np.pad(
                context,
                ((0, max_sent_len_batch - context.shape[0]), (0, 0)),
                mode="constant",
                constant_values=padding_constant,
            )
            batch_contexts_padded.append(context_padded)
            # set mask:
            mask_batch[context_idx, : len(context), : len(context)] = 1
            n_spans = len(spans[context_idx])
            begins_start = max_sent_len_batch
            begins_end = begins_start + n_spans
            ends_start = max_sent_len_batch + n_spans_max
            ends_end = ends_start + n_spans
            mask_batch[context_idx, begins_start:begins_end, : len(context)] = 1
            mask_batch[context_idx, ends_start:ends_end, : len(context)] = 1
            span_begins = np.arange(begins_start, begins_end)
            span_ends = np.arange(ends_start, ends_end)
            # begin/end-span attention (only diagonals (attent to context + begin + corresponding end and context + end + correstponding begin)
            mask_batch[context_idx, span_begins, span_begins] = 1
            mask_batch[context_idx, span_begins, span_ends] = 1
            mask_batch[context_idx, span_ends, span_begins] = 1
            mask_batch[context_idx, span_ends, span_ends] = 1
            # mask_batch[context_idx, begins_start:begins_end, ends_start:ends_end] = 1
            # mask_batch[context_idx, ends_start:ends_end, ends_start:ends_end] = 1
            # mask_batch[context_idx, begins_start:begins_end, begins_start:begins_end] = 1
            # mask_batch[context_idx, ends_start:ends_end, begins_start:begins_end] = 1
            # mask = mask_batch[context_idx]
            # print(mask)
            # for line in mask:
            #    print()
            #    for pos in line:
            #        print(int(pos), end=" ")
            # raise Exception()

        batch_contexts_padded = np.stack(batch_contexts_padded)
        context_and_spans = np.concatenate([batch_contexts_padded, batch_spans], axis=1)
        # print(context_padded.shape)
        # batch_contexts_padded.append(context_and_spans)
        yield context_and_spans, mask_batch, batch_span_infos


def batch_iter(x, batch_size, pad=True, padding_constant=0):
    n = x.shape[0]
    for batch_begin in range(0, n, batch_size):
        batch_end = batch_begin + batch_size
        chunk = x[batch_begin:batch_end]
        if pad and len(chunk) < batch_size:
            # @todo: really not usefull? # pad_len = batch_size - len(chunk)
            chunk = np.pad(
                chunk,
                ((0, batch_size - chunk.shape[0]), (0, 0)),
                mode="constant",
                constant_values=padding_constant,
            )
            # chunk = np.pad(chunk, (0, pad_len)) # only one dim
        yield chunk


def batch_sub_token_sents(self, sents_sub_token, batch_size):
    batch = []
    batch_max_context_size = 0
    for sent, sent_spans in sents_sub_token:
        context_start, context_end = sent
        batch_max_context_size = max(
            batch_max_context_size, context_end - context_start
        )
        batch_sent_spans = []
        for span_idx in range(len(sent_spans)):
            if len(batch) == batch_size:
                yield batch_max_context_size, batch
            batch_sent_spans.append(context_start, context_end)

    # shuffle sen
    # sentence_context_begin
    # sentence_context_end
    # spans:
    #   begins, begins_pos, ends, ends_pos, labels


"""
Arguments needed from args

outputdir # not necessary here (used for error logging)
label_set # name of label set (e.g., gsap)

local_rank # gpu id -1 if cpu (not used here)

model_type: bertspanmarkerpruner (alternative albertspanmarkerpruner not implemented)

max_sequence_length # int default 384  # (128 missing to 512)
    (gsap-ere: 256)
    "The maximum total input sequence length after tokenization.
     Sequences longer than this will be truncated, sequences shorter will be padded."
max_pair_length # int default 256 # 
    (gsap-ere: 64)
max_n_gram_size # int default 8 # max_n_gram_size
    (gsap-ere: 12) 
(former: max_mention_ori_length) 

nocross   # action="store_true" # no left or right sentence context
    (gsap-ere: True)

## Grouping of entities
group_axis # int default -1
    (gsap-ere: -1)
group_edge # action="store_true"
    (gsap-ere: False)
group_sort # action="store_true"
    (gsap-ere: False)

shuffle    # shuffle data
    (gsap-ere: False)
"""
