from data_loader import (
    PrunerDataArguments,
    PrunerDataset,
    Document_,
    DocumentSubTokenIndex,
    _sentences_to_sub_token_idxs,
    sentence_to_samples,
    batch_samples,
)
from ngram_filter import generate_doc_ner_candidates
from utils import get_logger
import json
from transformers import BertTokenizer
from copy import deepcopy
import numpy as np
from dataclasses import dataclass

@dataclass
class LoggerArgs:
    hostname:str = "no_hostname"
    local_rank:int = -1


def main():
    """
    Try out data loader using a jsonl file containing documents
    """
    fn_data = "../../data/input/docs.jsonl"
    model_name_or_path = "../../../pretrained_models/scibert_scivocab_uncased"
    batches, docs = batches_from_path(fn_data, model_name_or_path)
    print(len(batches))
    return batches, docs


def batches_from_path(
    fn_data,
    model_name_or_path,
    len_context=256,
    n_spans_max=128,  # or 128?
    batch_size=32,
    max_ngram=12,  # 12
    do_prefilter=False,
):
    args = PrunerDataArguments()
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path, do_lower_case=do_lower_case
    )
    # evaluate = False
    document = None

    # confusion: tp, fp, tn, fn, max_ngrams, n_token
    confusion = np.zeros(6, dtype=int)
    all_batches = []
    n_docs = 0
    n_sentences = 0
    n_samples = 0
    docs = []
    with open("../../data/input/docs.jsonl") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            n_docs += 1
            sentences_span_packs = []
            document = json.loads(line)
            document_raw = deepcopy(document)
            confusion += generate_doc_ner_candidates(
                document_raw, n=max_ngram, do_filter=do_prefilter
            )
            # logger.info(f'raw document len: {len(document["sentences"])}')
            document = Document_.from_dict(document)
            docs.append(document)
            n_sentences += len(document.sentences_begin)
            sub_token_index = DocumentSubTokenIndex(document, tokenizer)
            sentences = _sentences_to_sub_token_idxs(
                document, sub_token_index, len_context=256, expand=False
            )
            # print(f"n_sentences: {len(sentences)}")
            for sentence in sentences:
                (context_begin, context_end), spans = sentence
                # for begin, end, _, _, label in zip(*spans.T):
                #    print()
                span_packs = sentence_to_samples(sentence, n_spans_max)
                span_packs = list(span_packs)
                sentences_span_packs.extend(span_packs)
                n_samples += len(span_packs)
            batches = batch_samples(
                sub_token_index,
                sentences_span_packs,
                batch_size=batch_size,
                len_context=len_context,
                n_spans_max=n_spans_max,
                document_id=line_idx,
            )
            all_batches += list(batches)
            break
    print(confusion)
    print(f"n_docs: {n_docs}")
    print(f"n_sentences: {n_sentences}")
    print(f"n_samples: {n_samples}")
    # Create Batches from samples
    print(f"n_batches: {len(all_batches)}")
    batches = []
    for context, mask_square, label_info in all_batches:
        # print(context[0,:])# [103, 0, 1]
        # print(context[:,:,0].shape)# [batchsize, len_context + 2 * n_spans_max]
        inputs = dict(
            input_ids=context[:, :, 0],
            attention_mask=mask_square,
            position_ids=context[:, :, 1],
            labels=label_info[:, :, 4],
        )
        batch_size, sample_size, _ = label_info.shape
        from itertools import product

        span_info_dicts = [[] for _ in range(batch_size)]
        for sample_idx, span_idx in product(range(batch_size), range(sample_size)):
            begin_in_doc, end_in_doc, _, _, label, document_id = label_info[
                sample_idx, span_idx
            ]
            if label != -1:
                span_info_dicts[sample_idx].append(
                    dict(
                        document_id=document_id,
                        begin_in_doc=begin_in_doc,
                        end_in_doc=end_in_doc,
                        label=label,
                    )
                )
        # span_info = dict(
        #    document_ids=label_info[:,:,5],
        #    begin_in_doc=label_info[:,:,0],
        #    end_in_doc=label_info[:,:,1],
        #    labels=label_info[:,:,4],
        # )
        use_full_layer = -1
        if use_full_layer != -1:
            inputs["full_attention_mask"] = context[:, :, 2]
        # @todo: missing: mention_pos!
        if True or args.model_type.find("span") != -1:
            inputs["mention_pos"] = label_info[:, :, 2:4]
        batches.append((inputs, span_info_dicts))
    return batches, docs


def main_():
    args = PrunerDataArguments()
    logger = get_logger(LoggerArgs(), "test_log")
    model_name_or_path = "../../../pretrained_models/scibert_scivocab_uncased"
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path, do_lower_case=do_lower_case
    )
    evaluate = False
    document = None
    with open("../../data/input/docs.jsonl") as f:
        for line in f:
            document = json.loads(line)
            document_raw = deepcopy(document)
            tp, fp, tn, fn = generate_doc_ner_candidates(document, n=12, do_filter=True)
            print(tp, fn, fp, tn)
            break
    logger.info(f'raw document len: {len(document["sentences"])}')
    document = Document_.from_dict(document)
    sub_token_index = DocumentSubTokenIndex(document, tokenizer)
    sentences = _sentences_to_sub_token_idxs(document, sub_token_index, 110)
    print(f"n_sentences: {len(sentences)}")
    sentences_span_packs = []
    for sentence in sentences:
        (context_begin, context_end), spans = sentence
        # for begin, end, _, _, label in zip(*spans.T):
        #    print()
        len_context = 130
        n_spans_max = 10
        batch_size = 3
        span_packs = sentence_to_samples(sentence, n_spans_max)
        span_packs = list(span_packs)
        sentences_span_packs.extend(span_packs)
    print(f"n_samples: {len(sentences_span_packs)}")
    batches = batch_samples(
        sub_token_index,
        sentences_span_packs,
        batch_size=batch_size,
        len_context=len_context,
        n_spans_max=n_spans_max,
    )
    batches = list(batches)
    for context, mask_square, labels in batches:
        # print(context[0,:])# [103, 0, 1]
        # print(context[:,:,0].shape)# [batchsize, len_context + 2 * n_spans_max]
        inputs = dict(
            input_ids=context[:, :, 0],
            attention_mask=mask_square,
            position_ids=context[:, :, 1],
            labels=labels,
        )
        use_full_layer = -1
        if use_full_layer != -1:
            inputs["full_attention_mask"] = context[:, :, 2]
        # @todo: missing: mention_pos!

    print(f"n_batchs: {len(batches)}")
    return
    context, mask_square, label = batches[0]
    print(context.shape, label.shape)
    # print_batches(sub_token_index, batches, len_context, n_spans_max)
    dataset = PrunerDataset(logger, tokenizer, args, evaluate)
    logger.info("start init doc")
    # print(document_raw)
    document_raw["sentences"] = document_raw["sentences"][:3]
    document_raw["ner"] = document_raw["ner"][:3]
    dataset.initialize_from_document(document_raw)
    logger.info("end init doc")
    logger.info(len(dataset.data))
    for item_idx, item in enumerate(dataset):
        if item_idx < 30:
            continue
        """
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
        """
        print(item[3].shape)
        print(item[2][:3])
        print(item[4][:3])
        break
        m = item[1]
        for line in m:
            print()
            for e in line:
                print(int(e), end="")
        break


def print_batches(sub_token_index, batches, len_context, n_spans_max):
    for batch_idx, (context_span, label) in enumerate(batches):
        print(f"batch: {batch_idx}")
        for sample_idx in range(len(label)):
            print(f"\tsample: {sample_idx}")
            for context_idx in range(len_context):
                token_idx, pos = context_span[sample_idx][context_idx]
                if token_idx != 0:
                    print(
                        token_idx,
                        pos,
                        context_idx,
                        sub_token_index.sub_tokens[token_idx],
                    )
            print("\tNow the spans")
            for span_idx in range(n_spans_max):
                label_sample = label[sample_idx][span_idx]
                in_sample_idx_begin = len_context + span_idx
                in_sample_idx_end = len_context + n_spans_max + span_idx
                token_id_begin, pos_in_context_begin = context_span[sample_idx][
                    in_sample_idx_begin
                ]
                token_idx_begin = context_span[sample_idx][pos_in_context_begin][0]
                token_id_end, pos_in_context_end = context_span[sample_idx][
                    in_sample_idx_end
                ]
                token_idx_end = context_span[sample_idx][pos_in_context_end][0]
                if token_id_begin != 0:
                    print(f"\t\tspan: {span_idx}")
                    print(
                        "\t\t",
                        token_id_begin,
                        pos_in_context_begin,
                        sub_token_index.sub_tokens[token_idx_begin],
                        token_id_end,
                        pos_in_context_end,
                        sub_token_index.sub_tokens[token_idx_end],
                        label_sample,
                    )


class Predictor:
    """a wrapper arround a model doing all data pre- and postprocessing"""

    def __init__(self, data_handler, model, config):
        self.data_handler = data_handler
        pass

    def predict(self, document):
        # doc_as_dataset = self.data_handler.initialize_from_document(document)
        # predictions_raw = []
        # for batch in doc_as_dataset.iter_batches:
        # model.predict(batch)
        # preditions_raw.extend(batch)
        # predictions = data_handler.postprocess(predictions_raw)
        # return predictions
        pass


if __name__ == "__main__":
    main()
