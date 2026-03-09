# Pruner

## Role in the pipeline

The pruner is the first stage of the two-stage HGERE pipeline.  Its sole
purpose is to reduce the combinatorially large set of possible token spans to a
manageable subset that is then passed to the HGERE model for entity and relation
extraction.

Without pruning, a sentence of length *n* produces *O(n²)* candidate spans (all
contiguous token sub-sequences up to a maximum span length *L*).  For typical
scientific text this is far too many for the HGERE graph model to process
efficiently.  The pruner keeps only the top-*k* most promising spans.

## What the pruner does

* Enumerates **all possible n-grams** (contiguous token spans) up to a
  configured maximum span length.
* For each span it predicts a scalar **entity probability** — the probability
  that the span is *any* entity (binary classification: entity vs. non-entity).
* Spans are ranked by this score and the top-*k* are forwarded to HGERE.

The pruner does **not** predict entity types.  Type classification is the
responsibility of HGERE.

## Prediction file format (`ent_pred_*.json`)

Each line is a JSON object representing one document:

| Field | Type | Description |
|---|---|---|
| `doc_id` | str | Document identifier |
| `ner` | `list[list[[start, end, type]]]` | Gold entities per sentence |
| `predicted_ner` | `list[list[[start, end, type]]]` | Predicted spans (after pruning), with gold type for reference |
| `predicted_ner_proba` | `list[list[[start, end, prob, gold_type]]]` | All scored candidate spans |

### Important: `gold_type` is not a predicted label

The fourth element of each entry in `predicted_ner_proba` is the **gold label**
stored for analysis purposes only — it is not the pruner's output.  The pruner
outputs only the scalar `prob` (entity probability).

Correct threshold logic:

```python
# A span is a candidate entity if its score exceeds the threshold
is_candidate = prob >= threshold   # gold_type is NOT used
```

Using the gold type as a filter would leak ground-truth information and make
false-positive computation impossible (all non-NIL predictions would trivially
match gold).

## Evaluation

Evaluation is span-level (no types):

* **TP** — predicted span (prob ≥ threshold) matches a gold entity span
* **FP** — predicted span does not match any gold entity span (spurious candidate)
* **FN** — gold entity span not predicted (missed by pruner)

Per-type metrics report type-specific **recall** (how well each entity type is
covered) while FP/precision/FP-share are shared across all types because FP
spans carry no type assignment.

See `evaluation/pruner.py` for the implementation and
`reports/report_scripts/scier_dataset/threshold_analysis.py` for the threshold
analysis script.

## Candidate selection strategies

Two strategies are available for selecting which spans to pass to HGERE.

### Original approach: top-K selection (from the paper)

Spans are ranked by their pruner score and the top *K* are kept per sentence,
where *K* is proportional to sentence length *n*:

```
K = max(lmin, min(λ · n, lmax))
```

The bounds prevent *K* from being too small for very short sentences or too
large for very long ones.

| Parameter | Argument | Default | Description |
|---|---|---|---|
| λ | `--topk_ratio` | `0.5` | Ratio of candidates to sentence length |
| l_min | `--min_mentions_num` | `3` | Minimum candidates per sentence |
| l_max | `--max_mentions_num` | `18` | Maximum candidates per sentence |

### Alternative: score threshold

A fixed probability threshold can be applied instead of top-K via
`--filter_threshold` (default `None`, i.e. top-K is used).

* **Lower threshold** → more spans passed to HGERE → higher recall, more FPs
* **Higher threshold** → fewer spans → lower recall, fewer FPs

For the pipeline to work well, **recall should be kept very high** (≥ 98 %) so
that HGERE has access to all genuine entity spans.  The threshold analysis
script (`reports/report_scripts/scier_dataset/threshold_analysis.py`) performs
a binary search to find the highest threshold that still achieves the target
recall.
