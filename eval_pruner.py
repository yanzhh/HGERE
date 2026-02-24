import timeit
import pdb
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from wolf_data_utils import ACEDatasetNER


def evaluate(logger, args, model, tokenizer, file_path, prefix="", do_test=False):
    is_ontonotes = args.data_dir.find("ontonotes") != -1

    global NEG_INF
    NEG_INF = args.neg_inf

    eval_output_dir = Path(args.output_dir)
    if args.local_rank in [-1, 0]:
        eval_output_dir.mkdir(parents=True, exist_ok=True)

    # -- load dataset and labels --
    eval_dataset = ACEDatasetNER(
        logger=logger,
        tokenizer=tokenizer,
        file_path=file_path,
        args=args,
        evaluate=True,
    )
    ner_golden_labels = set(eval_dataset.ner_golden_labels)

    goldspan2label = _span2label(ner_golden_labels)
    eval_sampler = SequentialSampler(eval_dataset)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=ACEDatasetNER.collate_fn,
        num_workers=1,
    )

    # -- parameter for pruner --
    topk_ratio = args.topk_ratio
    max_mentions_num = args.max_mentions_num
    min_mentions_num = args.min_mentions_num
    topk_infos = (topk_ratio, min_mentions_num, max_mentions_num)


    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size   = {args.eval_batch_size}")

    sentences_predictions = defaultdict(list)

    model.eval()

    start_time = timeit.default_timer()

    closed_sentences = {"probs": None}  # from last batch
    open_sentence = {"probs": None}  # for next batch

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        indexs = batch[-2]
        # example_index, (doc_id, sent_id)
        batch_m2s = batch[-1]
        # mentions

        # -------for pruner------------
        sent_lens = batch[6]

        split_ranges, sent_lens_simple, indexs_simple, batch_m2s_simple = (
            _exact_boundaries(indexs, sent_lens, batch_m2s)
        )
        # in this batch

        # batch_mentions = _get_batch_mentions(batch_m2s_simple, args.device)

        batch = tuple(t.to(args.device) for t in batch[:6])

        with torch.no_grad():
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

            gold_labels = (
                inputs["labels"] > 0
            ).int()  # target: gold label exist (classes: 0, 1)
            ent_masks = (inputs["labels"] > -1).bool()  # entity mask without -1 labels
            ner_logits = outputs[1].squeeze(-1)
            ner_probs = ner_logits.sigmoid()
            # NEG_INF = NEG_INF if ner_logits.dtype==torch.float32 else -1e4
            ner_probs = ner_probs.masked_fill(~ent_masks, 0)
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

            current_sentences = dict(
                probs=split_probs_cat,
                indexs=indexs_simple,
                sent_lens=sent_lens_simple,
                ent_masks=split_mask_cat,
                gold_labels=split_gold_labels_cat,
                mentions=batch_m2s_simple,
            )

            closed_sentences, open_sentence = _extent_tensor(
                open_sentence, current_sentences
            )
            # process closed_sentences
            if closed_sentences["probs"] is not None:
                pruned_ent_spans, pred_entities, gold_entities = _decode_pruner_topk(
                    topk_infos, closed_sentences
                )

                _update_sentences_predictions(
                    sentences_predictions,
                    closed_sentences,
                    pruned_ent_spans,
                    goldspan2label,
                )

    if open_sentence["probs"] is not None:
        closed_sentences = open_sentence
        pruned_ent_spans, pred_entities, gold_entities = _decode_pruner_topk(
            topk_infos, closed_sentences
        )
        _update_sentences_predictions(
            sentences_predictions, closed_sentences, pruned_ent_spans, goldspan2label
        )

    force_same_label = not is_ontonotes
    ner_total_recall = eval_dataset.tot_recall
    predict_ners, predict_ners_overlap, results = postprocess_predictions(
        sentences_predictions, ner_golden_labels, force_same_label, ner_total_recall
    )
    res = {k: f"{v:.4f}" for k, v in results.items()}
    logger.info(f"Result: {res}")
    eval_time = timeit.default_timer() - start_time
    logger.info(
        f"  Evaluation done in total {eval_time} secs ({len(eval_dataset) / eval_time}) example per second)"
    )

    if args.output_results and (do_test or not args.do_train):
        eval_filename = eval_dataset.file_path
        save_results(logger, results, eval_filename, args, file_path, predict_ners)
        save_results(
            logger, results, eval_filename, args, file_path, predict_ners_overlap, overlap=True
        )
    return results


def postprocess_predictions(
    sentences_predictions, ner_golden_labels, force_same_label, ner_total_recall
):
    # p = tp / tot_pred if tot_pred > 0 else 0
    # r = tp / n_pos
    # f1_tot = 2 * (p * r) / (p + r) if tp > 0 else 0.0

    # print(f'f1_tot:{f1_tot}')
    predict_ners = defaultdict(list)
    predict_ners_overlap = defaultdict(list)
    cor = 0
    tot_pred = 0
    cor_tot = 0
    tot_pred_tot = 0
    for sentence_id, sentence_spans in sentences_predictions.items():
        # sort by probability (prefer highest probability)
        sentence_spans.sort(key=lambda x: -x[2])
        non_overlapping_spans = []

        for start, end, prob, label_gold in sentence_spans:
            is_overlap = _overlapping_span_exist(
                start, end, label_gold, non_overlapping_spans, force_same_label
            )
            if not is_overlap:
                non_overlapping_spans.append((start, end, prob, label_gold))

            tot_pred_tot += 1
            if not is_overlap:
                tot_pred += 1
                predict_ners[sentence_id].append((start, end, prob, label_gold))
            predict_ners_overlap[sentence_id].append((start, end, prob, label_gold))
            if (sentence_id, (start, end), label_gold) in ner_golden_labels:
                cor_tot += 1
                if not is_overlap:
                    cor += 1


    precision_score = p = cor / tot_pred if tot_pred > 0 else 0
    recall_score = r = cor / ner_total_recall
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    p = cor_tot / tot_pred_tot if tot_pred_tot > 0 else 0
    r = cor_tot / ner_total_recall
    f1_tot = 2 * (p * r) / (p + r) if cor > 0 else 0.0

    results = {
        "f1": f1,
        "precision": precision_score,
        "recall": recall_score,
        "f1_overlap": f1_tot,
        "p_overlap": p,
        "r_overlap": r,
    }

    return predict_ners, predict_ners_overlap, results


def save_results(
    logger, results, eval_filename, args, file_path, predicted_ners, overlap=False
):
    # pdb.set_trace()
    file_path = Path(file_path)
    file_name = file_path.name
    if "ace2004" in str(file_path) or "ace04" in str(file_path):
        if "train" in str(file_path):
            file_name = "train_" + file_name
        elif "dev" in str(file_path):
            file_name = "dev_" + file_name
        elif "test" in str(file_path):
            file_name = "test_" + file_name
        else:
            raise ValueError()

    # file_path is gold file
    input_lines = open(eval_filename)
    filename_wo_json = file_name[:-6]
    overlap_info = "_overlap" if overlap else ""
    target_filename = f"ent_pred_{filename_wo_json}{overlap_info}.json"
    target_path = Path(args.output_dir) / target_filename
    output_w = target_path.open("w")

    for line_idx, line in enumerate(input_lines):
        doc = json.loads(line)
        num_sents = len(doc["sentences"])
        predicted_ner = []
        predicted_ner_proba = []
        for sentence_idx in range(num_sents):
            sentence_ner = predicted_ners.get((line_idx, sentence_idx), [])
            #sentence_ner.sort(key=lambda x: -x[2])
            predicted_ner_proba.append(sentence_ner)
            spans_wo_prob = [
                (start, end, label) for start, end, prob, label in sentence_ner
            ]
            predicted_ner.append(spans_wo_prob)

        doc["predicted_ner"] = predicted_ner
        doc["predicted_ner_proba"] = predicted_ner_proba

        output_w.write(json.dumps(doc) + "\n")
    logger.info(f"evaluate test file into {target_filename} done.")


# Helper for checking existing overlapping spans:
def _overlapping_span_exist(start, end, label_gold, known_spans, force_same_label):
    for start_known, end_known, _, label_known in known_spans:
        is_overlap = _is_overlapping_span(start, end, start_known, end_known)
        is_same_label = label_gold == label_known
        # check gold label (Why?) but only if not ontonotes (Why?)
        if is_overlap and (is_same_label or not force_same_label):
            return True
    return False


def _is_overlapping_span(start, end, start_other, end_other):
    overlapping = False
    # ...| span |...
    # ....| other |....
    if start_other <= start and start <= end_other:
        overlapping = True
    # ...| span |...
    # ..| other |....
    elif start <= start_other and start_other <= end:
        overlapping = True
    return overlapping


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
    assert len(tensor_list) >= 2, pdb.set_trace()
    assert len(list(tensor_list[0].shape)) == 1
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
    """remove redundent dimensions according ent_masks"""
    if infos["probs"] is None:
        return infos
    else:
        split_probs = infos["probs"]
        indexs = infos["indexs"]
        sent_lens = infos["sent_lens"]
        ent_masks = infos["ent_masks"]
        gold_labels = infos["gold_labels"]
        mentions = infos["mentions"]

        if len(indexs) > 1:
            split_probs = _pad_tensors(split_probs, pad=0)
            ent_masks = _pad_tensors(ent_masks, pad=0)
            gold_labels = _pad_tensors(gold_labels, pad=0)

        max_n_ents = torch.stack(ent_masks).int().sum(-1).max().item()
        for i in range(len(indexs)):
            split_probs[i] = split_probs[i][:max_n_ents]
            ent_masks[i] = ent_masks[i][:max_n_ents]
            gold_labels[i] = gold_labels[i][:max_n_ents]

        return dict(
            probs=split_probs,
            indexs=indexs,
            sent_lens=sent_lens,
            ent_masks=ent_masks,
            gold_labels=gold_labels,
            mentions=mentions,
        )


def _combine_sentences(open_sentence, current_sentences):
    """@todo"""
    raise Exception("Not implemented yet")
    # info_fields = ["probs", "indexs", "sent_lens",
    #               "ent_masks", "gold_labels", "mentions"]
    # closed_sentences = {}
    # new_open_sentence = {}
    # open_sentence_exist = open_sentence["probs"] is not None
    # current_sentences_exist = current_sentences["probs"] is not None
    # open_sencente_is_first_current_sentence = open_sentence_exist and current_sentences_exist and open_sentence["indexs"][0] == current_sentences["indexs"][0]


def _extent_tensor(last_infos, current_infos):
    if last_infos["probs"] is not None:
        last_probs = last_infos["probs"][0]
        last_index = last_infos["indexs"][0]
        last_sent_len = last_infos["sent_lens"][0]
        last_ent_mask = last_infos["ent_masks"][0]
        last_gold_labels = last_infos["gold_labels"][0]
        last_mention = last_infos["mentions"][0]
    else:
        last_probs = None

    cur_split_probs = current_infos["probs"]
    cur_indexs = current_infos["indexs"]
    cur_sent_lens = current_infos["sent_lens"]
    cur_ent_masks = current_infos["ent_masks"]
    cur_gold_labels = current_infos["gold_labels"]
    cur_mentions = current_infos["mentions"]

    if cur_split_probs is not None:
        if len(cur_indexs) == 1:
            # update last infos
            if last_infos["probs"] is not None:
                # last_shape = list(last_probs.shape)
                # current_shape = list(cur_split_probs[0].shape)
                if last_index == cur_indexs[0]:
                    assert last_sent_len == cur_sent_lens[0]
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
                    last_infos = {
                        "probs": [last_probs],
                        "indexs": [last_index],
                        "sent_lens": [last_sent_len],
                        "ent_masks": [last_ent_mask],
                        "gold_labels": [last_gold_labels],
                        "mentions": [last_mention],
                    }
                    previous_infos = {
                        "probs": None,
                        "indexs": None,
                        "sent_lens": None,
                        "ent_masks": None,
                        "gold_labels": None,
                        "mentions": None,
                    }
                else:
                    # last tensor and first tensor in current is from different sentence, generate previous infos, new last infos are None
                    # last infos --> previous infos
                    previous_indexs = [last_index]
                    previous_sent_lens = [last_sent_len]
                    previous_mentions = [last_mention]

                    previous_split_probs = [last_probs]
                    previous_ent_masks = [last_ent_mask]
                    previous_gold_labels = [last_gold_labels]

                    previous_infos = dict(
                        probs=previous_split_probs,
                        indexs=previous_indexs,
                        sent_lens=previous_sent_lens,
                        ent_masks=previous_ent_masks,
                        gold_labels=previous_gold_labels,
                        mentions=previous_mentions,
                    )
                    last_infos = {
                        "probs": cur_split_probs,
                        "indexs": cur_indexs,
                        "sent_lens": cur_sent_lens,
                        "ent_masks": cur_ent_masks,
                        "gold_labels": cur_gold_labels,
                        "mentions": cur_mentions,
                    }
            else:
                previous_infos = {
                    "probs": None,
                    "indexs": None,
                    "sent_lens": None,
                    "ent_masks": None,
                    "gold_labels": None,
                    "mentions": None,
                }
                last_infos = {
                    "probs": cur_split_probs,
                    "indexs": cur_indexs,
                    "sent_lens": cur_sent_lens,
                    "ent_masks": cur_ent_masks,
                    "gold_labels": cur_gold_labels,
                    "mentions": cur_mentions,
                }
        else:
            if last_infos["probs"] is not None:
                # last_shape = list(last_probs.shape)
                # current_shape = list(cur_split_probs[0].shape)
                if last_index == cur_indexs[0]:
                    # last tensor and first tensor in current is entities of the same sentence.
                    assert last_sent_len == cur_sent_lens[0]
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

            previous_infos = dict(
                probs=cur_split_probs[:-1],
                indexs=cur_indexs[:-1],
                sent_lens=cur_sent_lens[:-1],
                ent_masks=cur_ent_masks[:-1],
                gold_labels=cur_gold_labels[:-1],
                mentions=cur_mentions[:-1],
            )
            last_infos = dict(
                probs=[cur_split_probs[-1]],
                indexs=[cur_indexs[-1]],
                sent_lens=[cur_sent_lens[-1]],
                ent_masks=[cur_ent_masks[-1]],
                gold_labels=[cur_gold_labels[-1]],
                mentions=[cur_mentions[-1]],
            )
    else:  # after last batch
        if last_probs is not None:
            previous_infos = last_infos
            last_infos = {
                "probs": None,
                "indexs": None,
                "sent_lens": None,
                "ent_masks": None,
                "gold_labels": None,
                "mentions": None,
            }
        else:
            raise ValueError()

    previous_infos = _remove_redundancy(previous_infos)
    last_infos = _remove_redundancy(last_infos)
    return previous_infos, last_infos


def _decode_pruner_topk(topk_infos, previous_infos):
    previous_probs = previous_infos["probs"]
    # previous_indexs = previous_infos['indexs']
    previous_sent_lens = previous_infos["sent_lens"]
    previous_ent_masks = previous_infos["ent_masks"]
    previous_gold_labels = previous_infos["gold_labels"]
    previous_mentions = previous_infos["mentions"]

    topk_ratio, min_mentions_num, max_mentions_num = topk_infos

    previous_mentions = _get_batch_mentions(previous_mentions, previous_probs[0].device)

    ner_probs = torch.stack(previous_probs)
    sent_lens = torch.tensor(previous_sent_lens, device=ner_probs.device).reshape(
        -1, 1
    )  # shape: (bs,)
    ent_masks = torch.stack(previous_ent_masks).bool()
    gold_labels = torch.stack(previous_gold_labels)

    bs = len(previous_probs)

    # ner_probs_1 = ner_probs.masked_fill(ent_masks==False, 0)
    # assert torch.equal(ner_probs, ner_probs_1),pdb.set_trace()
    _, indices = torch.sort(ner_probs, dim=1, descending=True)
    _, n_entity = ner_probs.shape
    
    # Determine max entities for each sentence

    # gold_entities_id_flat = gold_labels.masked_select(ent_masks)
    ## topk_ratio in hgere paper: 0.5
    ### 1. define the maximum number of candidates based on topk ratio
    n_ent_topk = torch.ceil(sent_lens.float() * topk_ratio)

    ### 2. select all sentences, where the initial number of candidates is lower than the defined minimum (e.g. sentence length 4: after topk-ratio: expect 2 candidates)
    #  * min_topk_masked sentences need to be set to minimum
    min_topk_mask = n_ent_topk < min_mentions_num
    #  * max_topk_masked sentences need to be reduced to max_mention_num value
    max_topk_mask = n_ent_topk > max_mentions_num
    ### 3. select right value
    #  * select min_mention_num if min_menition_num is higher than min_topk else n_ent_topk
    n_ent_topk = (
        min_topk_mask * min_mentions_num + (~min_topk_mask) * n_ent_topk
    )
    #  * select max_mention_num if max_menition_num is higher than max_topk else n_ent_topk
    n_ent_topk = max_topk_mask * max_mentions_num + (~max_topk_mask) * n_ent_topk

    # prune spans
    entity_starts = previous_mentions[:, :, 0]
    entity_ends = previous_mentions[:, :, 1]
    max_ne = int(max(n_ent_topk))
    # b x ne
    pruned_ent_starts = torch.gather(entity_starts, dim=1, index=indices)[:, :max_ne]
    pruned_ent_ends = torch.gather(entity_ends, dim=1, index=indices)[:, :max_ne]
    pruned_ent_probs = torch.gather(ner_probs, dim=-1, index=indices)[:, :max_ne]
    # b x ne x 3
    pruned_ent_spans = torch.stack(
        (pruned_ent_starts, pruned_ent_ends, pruned_ent_probs)
    ).permute(1, 2, 0)

    # if len(pruned_ent_spans)!= len(previous_indexs):
    #     pdb.set_trace()

    # evaluate for metrics
    topk_mask = (
        torch.arange(n_entity, device=n_ent_topk.device).repeat(bs).reshape(bs, -1)
    )
    # topk_mask = (topk_mask<n_ent_topk).to(torch.long)
    topk_mask = (topk_mask < max_ne).to(torch.long)

    pred_ents_idx = torch.zeros(
        (bs, n_entity), device=topk_mask.device, dtype=topk_mask.dtype
    ).scatter_(dim=1, index=indices, src=topk_mask)
    predicted_entities_id_flat = pred_ents_idx.masked_select(ent_masks)
    gold_entities_id_flat = gold_labels.masked_select(ent_masks)

    # predicted_entities_id = pred_ents_idx*ent_masks
    return pruned_ent_spans, predicted_entities_id_flat, gold_entities_id_flat


def _get_batch_mentions(batch_m2s, model_device):
    batch_size = len(batch_m2s)
    batch_mentions = []
    for batch_idx in range(batch_size):
        batch_mentions.append(torch.tensor(batch_m2s[batch_idx], device=model_device))
    batch_mentions = pad_sequence(batch_mentions, padding_value=-1).permute(1, 0, 2)

    return batch_mentions


def _update_sentences_predictions(
    sentences_predictions, closed_sentences, sentence_spans, goldspan2label
):
    sentence_ids = closed_sentences["indexs"]
    for sentence_id, sentence_spans in zip(sentence_ids, sentence_spans):
        line_idx, sentence_idx = sentence_id
        sentence_id = int(line_idx), int(sentence_idx)
        for start, end, prob in sentence_spans:
            start, end = int(start), int(end)
            prob = float(prob)
            if start >= 0:  # Why?
                gold_label = goldspan2label[sentence_id].get((start, end), "NIL")
                sentences_predictions[sentence_id].append(
                    (start, end, prob, gold_label)
                )


def _span2label(ner_golden_labels):
    label_dict = defaultdict(dict)
    for index, mention, label in ner_golden_labels:
        label_dict[index][mention] = label
    return label_dict
