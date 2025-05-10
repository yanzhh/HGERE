from importlib import reload

from pathlib import Path
import pandas as pd
import pruner_eval
reload(pruner_eval)
from pruner_eval import load_docs, sentence_wise_metrics, filter_ent_preds, save_docs

## Approach
# * load validation set predictions
#   * Test thresholds: 0. - 0.10
#   * Get threshold for given recall (e.g. > 0.89 recall)
# * prune train/validation/test set results as hgere input



date_model = "2025-04-07"
date_data = "2025-04-15"
annotator = "_Suyash"
annotator = ""
#date_data = date_model
pred_base_path = Path("../saves/gsap/pruner/entnum30-30-lr1e-5-epochs4/gsap_scibert_data_2025-03-06-bs16-1e-5-4-44")
pred_base_path = Path("../saves/somd/pruner/entnum/somd_phase1_scibert_data-bs16-1e-5-4-44/")
pred_base_path = Path(f"../saves/gsap/pruner/lr1e-5-epochs4/gsap_scibert_data_{date_model}-bs16-44/")
target_path = Path("../saves/somd/pruned_ner/phase1")
target_path = Path(f"../saves/gsap/pruned_ner/{date_data}")
target_path.mkdir(parents=True, exist_ok=True)


# find best parampeter

fn_dev = pred_base_path / "ent_pred_somd_phase1_dev.json"
#fn_dev = pred_base_path / "ent_pred_somd_phase2_all.json"
fn_dev = pred_base_path / "ent_pred_somd_phase2_test.json"
fn_dev = pred_base_path / f"ent_pred_{date_data}{annotator}_dev.json"
docs = load_docs(fn_dev)
len(docs)


steps = 20
pred_max = 18
threshold_len_mult = 5.
precision_recall_curve = []
min_threshold = None
max_threshold = None
for step in range(steps + 1):
    threshold = (step / steps) * 0.001# + 0.0007
    if step == 0:
        min_threshold = threshold
    max_threshold = threshold
    metrics = sentence_wise_metrics(docs, threshold, threshold_len_mult=threshold_len_mult, pred_max=pred_max)
    recall = metrics.tp.sum() / metrics.support.sum()
    precision = 1.
    if metrics.n_predicted.sum():
        precision = metrics.tp.sum() / metrics.n_predicted.sum()
    precision_recall_curve.append(dict(recall=recall, precision=precision, threshold=threshold, n_predicted=metrics.n_predicted.sum()))
print(min_threshold, max_threshold)
pr_rec = pd.DataFrame(precision_recall_curve)
pr_rec["n_predicted_share"] = pr_rec["n_predicted"] / pr_rec["n_predicted"].max()
pr_rec.set_index("threshold")[["recall", "n_predicted_share"]].plot();