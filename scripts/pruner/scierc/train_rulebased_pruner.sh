#!/bin/bash
# Fit the rule-based span pruner on SciERC training data.
# Run from the HGERE repo root.
#
# Output: saves/scierc/pruner_rulebased/scierc_rulebased.json
#
# --target_fn 0  : do not allow any false negatives on the held-out split.
#                  SciERC annotates unusual spans like bare "." as entities,
#                  so any pattern that matches even one gold entity must be
#                  excluded.
# --purity_threshold 1.0  (default): only patterns that are 100% NIL in
#                  training are candidates at all.

DATADIR=/home/ottowg/projects/gsap/related_datasets/scierc/dataset/
OUTDIR=saves/scierc/pruner_rulebased
mkdir -p $OUTDIR

uv run eval-rulebased-pruner \
    --save $OUTDIR/scierc_rulebased.json \
    --train_file $DATADIR/train.json \
    --dev_file   $DATADIR/dev.json \
    --max_span_length 12 \
    --target_fn 0
