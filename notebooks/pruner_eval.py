import json
from itertools import chain
import pandas as pd


def load_docs(fn):
    docs = []
    with open(fn) as f:
        for line in f.readlines():
            doc = json.loads(line)
            if "doc_tokens" not in doc:
                doc["doc_tokens"] = list(chain(*doc["sentences"]))
            docs.append(doc)
    return docs

def save_docs(docs, fn):
    docs_json = [json.dumps(doc) for doc in docs]
    docs_json = "\n".join(docs_json)
    with open(fn, "w") as f:
        f.write(docs_json)

def filter_ent_preds(docs, threshold = 0.00015, pred_max = 18):
    for doc in docs:
        new_predicted_ner = []
        for sent_idx, sent_ner_proba in enumerate(doc["predicted_ner_proba"]):
            #assert len(sent_ner_proba)
            sent_ner = [[begin, end, label]
                         for idx, (begin, end, proba, label)
                         in enumerate(sent_ner_proba)
                         if proba >= threshold or idx == 0][:pred_max]
            if len(sent_ner) == 0:
                sent = doc["sentences"][sent_idx]
                print(f"Caution: one sent without ner prediction: '{sent}'")
                print(doc["doc_id"])
            new_predicted_ner.append(sent_ner)
        doc["predicted_ner"] = new_predicted_ner


def sentence_wise_metrics(docs, threshold_score, threshold_len_mult=None, pred_max=None, mini=1):
    sent_infos = []
    for doc in docs:
        doc_id = doc.get("doc_id")
        if doc_id is None:
            doc_id = doc.get("doc_key")
        sent_idxs = list(range(len(doc["sentences"])))
        for sent_idx, sentence in enumerate(doc["sentences"]):
            sent_pred_max = pred_max
            n_token = len(sentence)
            if threshold_len_mult is not None:
                pred_max_len_mult = int(n_token * threshold_len_mult)
                if sent_pred_max is not None:
                    sent_pred_max = min(sent_pred_max, pred_max_len_mult)
                else:
                    sent_pred_max = pred_max_len_mult
            if "predicted_ner_proba" in doc:
                ner_pred = doc["predicted_ner_proba"][sent_idx]
                ner_pred.sort(key=lambda x: -x[2])
                #print(ner_pred)
                #print()
                if sent_pred_max is not None:
                    #print("y", sent_pred_max)    
                    sent_pred_max = max(mini, sent_pred_max)
                    ner_pred = ner_pred[:sent_pred_max]
                #print(sent_pred_max)
                #print("z", sent_pred_max)
                #print(ner_pred)
                #print()
                ner_pred_set = {(begin, end)
                                for begin, end, proba, _
                                in ner_pred
                                if proba >= threshold_score}
                #print(ner_pred_set)
                #raise Exception()
            else:
                ner_pred = doc["predicted_ner"][sent_idx]
                ner_pred_set = {(begin, end)
                                for begin, end, _
                                in ner_pred
                                }
            ner = doc["ner"][sent_idx]
            ner_set = {(begin, end) for begin, end, _ in ner}

            metrics = precision_recall_f1_set(ner_set, ner_pred_set)
            metrics |= dict(
                n_token=n_token,
                n_pred_max = len(ner_pred),    
                sent_id=(doc_id, sent_idx),
                doc_id=doc_id
            )
            sent_infos.append(metrics)
    sent_infos = pd.DataFrame(sent_infos)
    return sent_infos


def precision_recall_f1_set(set_one, set_compare):
    tp = len(set_one & set_compare)
    fn = len(set_one - set_compare)
    fp = len(set_compare - set_one)
    support = len(set_one)
    n_predicted = len(set_compare)
    recall = 1. # if no support we define recall as 1. (all possible items are found)
    if support:
        recall = tp / support
    precision = 1. # if no items are predicted we define precision as 1. (no wrong items are predicted)
    if n_predicted:
        precision = tp / n_predicted
    f_score = 0.
    if precision + recall > 0:
        f_score = 2 * (recall * precision) / (precision + recall)
    return dict(
        precision=precision,
        recall=recall,
        f1=f_score,
        tp=tp,
        fn=fn,
        fp=fp,
        support=support,
        n_predicted=n_predicted,
        diff = n_predicted - support,
        )
    
  
    
