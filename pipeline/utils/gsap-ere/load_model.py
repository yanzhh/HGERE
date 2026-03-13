import time
print("start import")
from dataclasses import dataclass
from pathlib import Path
from test_data_loader import main as load_batches
import torch





from transformers import (
    #AdamW,
    BertConfig,
    BertForSpanMarkerNerPruner,
    # AlbertConfig,
    # AlbertTokenizer,
    # AlbertForSpanMarkerNerPruner,
    BertTokenizer,
    #RobertaConfig,
    #get_linear_schedule_with_warmup,
)
print("finished import")


MODEL_CLASSES = {
    "bertspanmarkerpruner": (BertConfig, BertForSpanMarkerNerPruner, BertTokenizer),
    # "albertspanmarkerpruner": (
    #    AlbertConfig,
    #    AlbertForSpanMarkerNerPruner,
    #    AlbertTokenizer,
    # ),
}

@dataclass
class ModelConfig:
    model_name_or_path:str
    num_labels:int=11
    do_lower_case:bool=True
    max_seq_length:int=256
    alpha:int=1
    onedropout:bool=True
    use_full_layer:int=-1 # not used!

@dataclass
class Arguments:
    extra_repr:str=None # use extra span repr ("attn")
    biaf_span:bool=True # use BiaffineSpanRepr
    biaf_factorize:bool=True # from config
    span_hidden_size:int=768 # for BiaffineSpanRepr (2 * 383)
    span_size:int=256 # for BiaffineSpanRepr
    biaf_mode:int=2 # from config, default: 3
    rank:int=768 # also for biaffine config + default
    device:str="cuda:2" # "cuda:0"

def main():
    # load data!
    print("load data")
    batches, docs = load_batches()
    print("load model")
    # load model
    model_path = Path("../../..")
    model_path /= "saves/gsap/pruner/lr1e-5-epochs4/gsap_scibert_data_2025-04-07-bs16-44/checkpoint-11618"  
    model_path_scibert = "../../../pretrained_models/scibert_scivocab_uncased"
    cfg = ModelConfig(str(model_path))
    args = Arguments()
    model_class = BertForSpanMarkerNerPruner
    num_labels = 11 
    config = BertConfig.from_pretrained(
        cfg.model_name_or_path,
        num_labels=cfg.num_labels
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_path_scibert, do_lower_case=cfg.do_lower_case
    )
    config.max_seq_length = cfg.max_seq_length
    config.alpha = cfg.alpha
    config.onedropout = cfg.onedropout
    config.use_full_layer = cfg.use_full_layer

    model = BertForSpanMarkerNerPruner.from_pretrained(
        cfg.model_name_or_path,
        from_tf=bool(".ckpt" in cfg.model_name_or_path), # @todo needed? tensorflow?
        config=config,
        args=args
    )
    model.to(args.device)
    model.zero_grad()
    model.eval()
    print("start prediction")
    start = time.time()
    prediction = []
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_info) in enumerate(batches, start=1):
            # predict
            batch_inputs = {k:torch.tensor(v).to(args.device) for k, v in batch_inputs.items()}
            print(f"predict batch {batch_idx}/{len(batches)}")
            outputs = model(**batch_inputs)
            ent_masks = (batch_inputs["labels"] > -1).bool()  # entity mask without -1 labels
            ner_logits = outputs[1].squeeze(-1)
            ner_probs = ner_logits.sigmoid()
            ner_probs = ner_probs.masked_fill(~ent_masks, 0)
            probs = ner_probs.detach().cpu().numpy()
            # get prediction metadata
            for sample_idx, sample in enumerate(batch_info):
                for span_idx, span in enumerate(sample):
                    prob = probs[sample_idx, span_idx]
                    span["score"] = prob
                    prediction.append(span)
            """
            doc_ids = batch_info["document_ids"]
            for sample_idx in range(len(doc_ids)):
                for span_idx in range(len(doc_ids[0])):
                    fields = ["document_ids", "begin_in_doc", "end_in_doc", "labels"]
                    sample = {f:int(batch_info[f][sample_idx, span_idx]) for f in fields}
                    if sample["labels"] == -1:
                        continue
                    sample["score"] = float(probs[sample_idx, span_idx])
                    prediction.append(sample)
            print(len(prediction))
            """
    end = time.time()
    print(f"sec/doc: {(end - start) / 10}")
    tp, fp, fn, tn = 0, 0, 0, 0
    doc = docs[0]
    for sample in prediction:
        pred = sample["score"] > 0.0005
        if pred and sample["label"] == 1:
            tp += 1
        elif pred:
            fp += 1
        elif sample["label"] == 1:
            fn += 1
        else:
            tn += 1
    print(tp, fp, fn, tn)
    print(f"prec: {tp / (tp + fp)}")
    print(f"reca: {tp / (tp + fn)}")

            
            # should be: 726 is: 384
    






if __name__ == "__main__":
    main()




