"""
Count n_candidates and n_samples per sentence for the pruner data reader.

Shows how many training samples a single sentence produces at different
max_pair_length settings, and what fraction of sentences spill into >1 sample.

Usage:
    uv run python scripts/analysis/count_pruner_samples_per_sent.py \
        --data_dir /home/ottowg/projects/gsap/related_datasets/scierc/dataset/ \
        --rulebased_pruner_file saves/scierc/pruner_rulebased/scierc_rulebased.json \
        --pair_lengths 64 96 128
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_rulebased_pruner(path):
    if path is None:
        return None
    # Import from the project so rule-based filtering is identical to training.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from hgere.span_classifier.rulebased import RuleBasedPruner
    return RuleBasedPruner.load(path)


def count_candidates(sentence_tokens, max_mention_len, rulebased_pruner):
    """
    Count n-gram candidates for one sentence using word-level spans
    (same logic as the data reader, but without subword alignment).
    """
    n = len(sentence_tokens)
    count = 0
    for start in range(n):
        for end in range(start + 1, min(n + 1, start + max_mention_len + 1)):
            if rulebased_pruner is not None:
                span_words = sentence_tokens[start:end]
                context_before = sentence_tokens[:start]
                context_after = sentence_tokens[end:]
                if rulebased_pruner.should_prune(span_words, context_before, context_after):
                    continue
            count += 1
    return count


def analyse(docs, pair_lengths, max_mention_len, rulebased_pruner):
    sent_lens = []
    cand_counts = []

    for doc in docs:
        for sentence in doc["sentences"]:
            n_cands = count_candidates(sentence, max_mention_len, rulebased_pruner)
            sent_lens.append(len(sentence))
            cand_counts.append(n_cands)

    sent_lens = np.array(sent_lens)
    cand_counts = np.array(cand_counts)

    print(f"\n{'─'*60}")
    print(f"  Sentences: {len(sent_lens)}")
    print(f"  Sentence length  — mean={sent_lens.mean():.1f}  "
          f"p50={np.percentile(sent_lens,50):.0f}  "
          f"p90={np.percentile(sent_lens,90):.0f}  "
          f"max={sent_lens.max()}")
    print(f"  Candidates/sent  — mean={cand_counts.mean():.1f}  "
          f"p50={np.percentile(cand_counts,50):.0f}  "
          f"p90={np.percentile(cand_counts,90):.0f}  "
          f"max={cand_counts.max()}")
    print(f"{'─'*60}")
    print(f"  {'max_pair_length':>16}  {'total_samples':>13}  "
          f"{'samples/sent mean':>18}  "
          f"{'% sents with >1 sample':>24}  "
          f"{'% sents with >2 samples':>24}")

    for pl in pair_lengths:
        n_samples = np.ceil(cand_counts / pl).astype(int)
        n_samples = np.maximum(n_samples, 1)  # even empty sentences get 1 sample
        total = n_samples.sum()
        mean_s = n_samples.mean()
        pct_gt1 = 100 * (n_samples > 1).mean()
        pct_gt2 = 100 * (n_samples > 2).mean()
        print(f"  {pl:>16}  {total:>13}  {mean_s:>18.2f}  "
              f"{pct_gt1:>23.1f}%  {pct_gt2:>23.1f}%")

    print(f"{'─'*60}\n")
    return sent_lens, cand_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",
                    default="/home/ottowg/projects/gsap/related_datasets/scierc/dataset/")
    ap.add_argument("--train_file", default="train.json")
    ap.add_argument("--dev_file", default="dev.json")
    ap.add_argument("--rulebased_pruner_file",
                    default="saves/scierc/pruner_rulebased/scierc_rulebased.json")
    ap.add_argument("--max_mention_len", type=int, default=12)
    ap.add_argument("--pair_lengths", type=int, nargs="+", default=[64, 96, 128])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rb = load_rulebased_pruner(args.rulebased_pruner_file)
    if rb is not None:
        print("Rule-based pruner loaded — candidates are pre-filtered.")
    else:
        print("No rule-based pruner — all n-grams counted.")

    for split, fname in [("train", args.train_file), ("dev", args.dev_file)]:
        path = data_dir / fname
        if not path.exists():
            print(f"  {path} not found, skipping.")
            continue
        docs = load_jsonl(path)
        print(f"\n=== {split.upper()} ({path.name}) ===")
        analyse(docs, args.pair_lengths, args.max_mention_len, rb)


if __name__ == "__main__":
    main()
