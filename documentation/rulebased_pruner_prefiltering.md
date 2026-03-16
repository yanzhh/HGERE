# Rule-Based Pruner Pre-Filtering

## Method

### Motivation

The neural pruner is trained on all candidate spans up to a maximum span length.
For a sentence of length *n* this is *O(n²)* spans, the vast majority of which
are unambiguous non-entities: single punctuation marks, purely numeric tokens,
common function-word sequences such as *"of the"* or *"in a"*, etc.  Including
these in training adds noise-free but disproportionately many negative examples,
which can skew the loss and slow convergence.

The rule-based pruner is a lightweight pre-filter that removes spans that are
deterministically non-entities based on learned token n-gram patterns.  It is
applied **before** the neural pruner sees the training data, reducing the
candidate set while keeping recall on gold entities at or near 100 %.

### What the rule-based pruner is

The rule-based pruner is a set of token n-gram patterns learned from the
training data.  Each pattern is a tuple of tokens annotated with boundary
markers:

| Marker | Meaning |
|--------|---------|
| `<S>` | Span start boundary |
| `<E>` | Span end boundary |
| `<B>` | End of before-context (tokens in the same sentence, left of span) |
| `<A>` | Start of after-context (tokens in the same sentence, right of span) |

A candidate span is pruned if **any** of its n-grams — extracted from the
boundary-marked token sequence `[<S>] + words + [<E>]` — matches a pattern in
the set, or if a before/after context n-gram matches.

Patterns are selected from training-data statistics by two criteria:

1. **Purity** — the fraction of occurrences in which the pattern appears
   exclusively in NIL (non-entity) spans.  The default threshold is 1.0, meaning
   a pattern is only kept if it **never** co-occurs with a gold entity, giving a
   hard guarantee that no gold entity is pruned by a pattern that was seen during
   training.
2. **Frequency** — patterns must appear at least `min_count` times.  Low-count
   patterns are too rare to be reliable and are excluded.

A min-count sweep over the training data is run against a held-out dev split to
find the highest pruning rate that keeps false negatives (pruned gold entities)
at or below a configurable target.

### How it is applied during neural pruner training

When a path to a saved rule-based pruner JSON file is provided, `ACEDatasetNER`
loads it at dataset construction time.  During span enumeration in `initialize()`,
each candidate span is checked with `RuleBasedPruner.should_prune(words,
context_before, context_after)` before being added to the training set.  Spans
that match a prune pattern are silently dropped; gold entity labels are still
removed from the tracking dict so they do not leak.

Context passed to `should_prune` is sentence-bounded:

- `context_before` — all tokens in the current sentence that precede the span
- `context_after` — all tokens in the current sentence that follow the span

The rule-based pruner is applied **only during training**, not evaluation.
During evaluation the full candidate set is enumerated so that the neural
pruner's upper-bound recall is unaffected and metrics are comparable to a
baseline without pre-filtering.

---

## Usage

### Step 1 — Fit and save the rule-based pruner

```bash
uv run eval-rulebased-pruner \
    --train_file /path/to/train.jsonl \
    --dev_file   /path/to/dev.jsonl   \
    --save       saves/rulebased_pruner.json
```

The command collects n-gram statistics from the training file, runs a sweep over
`min_count` values to find the best pruning rate subject to a false-negative
budget, fits the final pruner, prints an evaluation report on the dev set, and
writes the pattern set to the JSON file given by `--save`.

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train_file` | required | Training JSONL |
| `--dev_file` | required | Dev JSONL (used for sweep and final evaluation) |
| `--save` | — | Output JSON file (required to persist the pruner) |
| `--target_fn` | 2 | Max false negatives allowed during the sweep |
| `--purity_threshold` | 1.0 | Minimum NIL purity; 1.0 = never prune a gold-seen pattern |
| `--min_count_ratio` | 0.01 | Fallback `min_count` as a fraction of total entity n-gram occurrences |
| `--pattern_types` | `prefix,suffix,full,infix` | Which n-gram types to learn |
| `--max_tokens` | 3 | Max word tokens per pattern (excluding boundary markers) |
| `--max_span_length` | 20 | Max span length considered during statistics collection |
| `--train_split` | 0.8 | Fraction of training docs used for statistics; remainder used for internal sweep dev split |

The sweep output shows recall, precision, pruning rate, and FN count for a range
of `min_count` values, and highlights the best configuration that satisfies
`--target_fn`.

### Step 2 — Train the neural pruner with pre-filtering

Pass the saved JSON file via `--rulebased-pruner-file`:

```bash
uv run run_pruner.py \
    --train_file  /path/to/train.jsonl \
    --dev_file    /path/to/dev.jsonl   \
    --rulebased-pruner-file saves/rulebased_pruner.json \
    ... # other training args
```

When this flag is set, the dataset class loads the pruner at startup (logged as
`Loaded rule-based pruner from ... (N patterns)`) and removes matching spans
from the training and evaluation candidate sets before the neural pruner sees
them.

Omitting `--rulebased-pruner-file` restores the original behaviour with no
pre-filtering.
