"""
Apply top-k filtering to pruner prediction files.
factor=0.5, min=1, max=18: for each sentence, keep top clip(factor*sent_len, min, max) spans.
Reads predicted_ner_proba (sorted by prob desc), writes filtered predicted_ner.
"""
import json
import math
import os

FACTOR = 0.5
MIN_K = 1
MAX_K = 18

SRC = "/home/ottowg/projects/gsap/gsap-rel-related/HGERE/pruner_predictions/scierc/focal-prefilter-2e-5"
DST = "/home/ottowg/projects/gsap/gsap-rel-related/HGERE/pruner_predictions/scierc/topk-factor0.5-min1-max18"

FILES = ["ent_pred_trai.json", "ent_pred_de.json", "ent_pred_tes.json",
         "ent_pred_trai_overlap.json", "ent_pred_de_overlap.json", "ent_pred_tes_overlap.json"]


def apply_topk(doc):
    new_predicted_ner = []
    for i, sent in enumerate(doc["sentences"]):
        sent_len = len(sent)
        k = int(max(MIN_K, min(MAX_K, round(FACTOR * sent_len))))
        proba_list = doc["predicted_ner_proba"][i]  # already sorted by prob desc
        top_k = proba_list[:k]
        new_predicted_ner.append([[start, end, gold_type] for start, end, prob, gold_type in top_k])
    doc["predicted_ner"] = new_predicted_ner
    return doc


for fname in FILES:
    src_path = os.path.join(SRC, fname)
    dst_path = os.path.join(DST, fname)
    if not os.path.exists(src_path):
        print(f"Skipping {fname} (not found)")
        continue
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc = apply_topk(doc)
            fout.write(json.dumps(doc) + "\n")
    print(f"Written: {fname}")

print("Done.")
