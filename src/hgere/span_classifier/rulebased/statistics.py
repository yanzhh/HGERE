"""Collect n-gram statistics from a training JSONL file for rule-based span pruning.

For each possible candidate span in the training data (all spans up to max_span_length),
we:
  1. Wrap the span's word list with boundary markers <S> and <E>
  2. Extract n-grams (mode-dependent, see collect_stats)
  3. Accumulate per n-gram: count_entity (span is a gold entity) / count_nil (span is not)

N-gram position types (determined by presence of boundary markers):
  prefix : starts with <S>  (e.g. ("<S>", ".", "This"))
  suffix : ends with <E>    (e.g. ("sentence", "<E>"))
  full   : starts with <S> and ends with <E>  (covers the entire span)
  infix  : neither          (e.g. (")", ".", "A"))
  before : ends with <B>    (e.g. ("Figure", "3", "<B>"))  — context left of span
  after  : starts with <A>  (e.g. ("<A>", "et", "al"))   — context right of span
"""

import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd

START = "<S>"
END = "<E>"
BEFORE = "<B>"
AFTER = "<A>"
SENT_START = "<SS>"
SENT_END = "<SE>"

_MARKERS = {START, END, BEFORE, AFTER, SENT_START, SENT_END}


def _iter_ngrams(tokens: List[str], max_n: int):
    """Yield all n-grams of length 1..max_n from tokens."""
    n = len(tokens)
    for size in range(1, min(max_n, n) + 1):
        for i in range(n - size + 1):
            yield tuple(tokens[i : i + size])


def _iter_boundary_ngrams(marked: List[str], max_prefix_tokens: int, max_suffix_tokens: int):
    """Yield only boundary n-grams from a boundary-marked token list.

    Generates:
      - Prefix n-grams: (<S>, w1, ..., wk) for k = 1..min(max_prefix_tokens, n_words)
      - Suffix n-grams: (wN-k+1, ..., wN, <E>) for k = 1..min(max_suffix_tokens, n_words)
      - Full span: (<S>, w1, ..., wN, <E>)

    No infix n-grams are produced.  Total output: at most
    max_prefix_tokens + max_suffix_tokens + 1 tuples per span.
    """
    n = len(marked)  # = n_words + 2  (START + words + END)
    n_words = n - 2
    for k in range(1, min(max_prefix_tokens, n_words) + 1):
        yield tuple(marked[: k + 1])
    for k in range(1, min(max_suffix_tokens, n_words) + 1):
        yield tuple(marked[n - 1 - k :])
    yield tuple(marked)


def _iter_infix_ngrams(words: List[str], max_n: int):
    """Yield infix n-grams (word tokens only, no boundary markers) of length 1..max_n."""
    n = len(words)
    for size in range(1, min(max_n, n) + 1):
        for i in range(n - size + 1):
            yield tuple(words[i : i + size])


def _iter_before_ngrams(words_before: List[str], max_n: int):
    """Yield before-context n-grams: (w_{-k}, ..., w_{-1}, <B>) for k=1..min(max_n, len).

    words_before should already be the (up to max_n) tokens immediately preceding the span
    within the same sentence.
    """
    n = len(words_before)
    for k in range(1, min(max_n, n) + 1):
        yield tuple(words_before[n - k :]) + (BEFORE,)


def _iter_after_ngrams(words_after: List[str], max_n: int):
    """Yield after-context n-grams: (<A>, w_{+1}, ..., w_{+k}) for k=1..min(max_n, len).

    words_after should already be the (up to max_n) tokens immediately following the span
    within the same sentence.
    """
    n = len(words_after)
    for k in range(1, min(max_n, n) + 1):
        yield (AFTER,) + tuple(words_after[:k])


def load_docs(file_path: str) -> List[dict]:
    """Load all documents from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def collect_stats(
    docs: List[dict],
    max_span_length: int = 20,
    max_ngram: int = 14,
    full_only: bool = False,
    max_prefix_tokens: int = None,
    max_suffix_tokens: int = None,
    max_infix_tokens: int = None,
    max_before_tokens: int = None,
    max_after_tokens: int = None,
) -> pd.DataFrame:
    """Compute n-gram statistics over all candidate spans in a list of documents.

    Parameters
    ----------
    docs : list of document dicts (SciERC/HGERE format, as returned by load_docs)
    max_span_length : maximum span length in tokens to consider as a candidate
    max_ngram : maximum n-gram length (including boundary markers <S>/<E>).
        Only used in general mode (no boundary/full_only flags set).
    full_only : if True, only store the complete span tuple (START, w1..wn, END).
        Ignored when max_prefix_tokens or max_suffix_tokens is set.
    max_prefix_tokens : if set, collect prefix n-grams with at most this many word
        tokens and suffix n-grams with at most max_suffix_tokens words, plus the full
        span. Generates at most max_prefix_tokens + max_suffix_tokens + 1 n-grams per
        span — much faster than general mode.
    max_before_tokens : if set, collect before-context n-grams up to this many tokens
        (sentence-bounded). Pattern form: (w_{-k}, ..., w_{-1}, <B>).
    max_after_tokens : if set, collect after-context n-grams up to this many tokens
        (sentence-bounded). Pattern form: (<A>, w_{+1}, ..., w_{+k}).

    Returns
    -------
    DataFrame with one row per unique n-gram, columns:
        ngram          : tuple of tokens (may include <S> / <E> / <B> / <A>)
        count_entity   : times this n-gram appears in a gold-entity span
        count_nil      : times this n-gram appears in a non-entity span
        frequency      : count_entity + count_nil
        purity         : count_nil / frequency  (1.0 = always NIL, never an entity)
        n_tokens       : word count (boundary markers excluded)
        has_start      : True if n-gram contains <S>  (prefix or full span)
        has_end        : True if n-gram contains <E>  (suffix or full span)
    """
    boundary_mode = max_prefix_tokens is not None or max_suffix_tokens is not None
    mp = max_prefix_tokens or 0
    ms = max_suffix_tokens or 0

    count_entity: Counter = Counter()
    count_nil: Counter = Counter()

    for data in docs:
        sentences: List[List[str]] = data["sentences"]
        ner_gold = data["ner"]  # list of lists of [start, end, label] (doc-level indices)

        # Build flat document-level word array and track sentence boundaries
        words: List[str] = []
        sentence_boundaries = [0]
        for sentence in sentences:
            words.extend(sentence)
            sentence_boundaries.append(len(words))

        # Collect gold entity spans as a set of (start, end) with doc-level indices
        entity_spans = set()
        for sent_ner in ner_gold:
            for start, end, label in sent_ner:
                if label != "NIL":
                    entity_spans.add((start, end))

        # Enumerate all candidate spans per sentence
        for sent_idx in range(len(sentences)):
            sent_start = sentence_boundaries[sent_idx]
            sent_end = sentence_boundaries[sent_idx + 1]

            for span_start in range(sent_start, sent_end):
                for span_end in range(
                    span_start,
                    min(sent_end, span_start + max_span_length),
                ):
                    is_entity = (span_start, span_end) in entity_spans

                    span_words = words[span_start : span_end + 1]

                    if boundary_mode:
                        marked = [START] + span_words + [END]
                        ngrams = list(_iter_boundary_ngrams(marked, mp, ms))
                        if max_infix_tokens is not None:
                            ngrams += list(_iter_infix_ngrams(span_words, max_infix_tokens))
                    elif full_only:
                        ngrams = [(START,) + tuple(span_words) + (END,)]
                    else:
                        marked = [START] + span_words + [END]
                        if max_infix_tokens is not None:
                            ngrams = [
                                ng for ng in _iter_ngrams(marked, max_ngram)
                                if ng[0] == START or ng[-1] == END or len(ng) <= max_infix_tokens
                            ]
                        else:
                            ngrams = list(_iter_ngrams(marked, max_ngram))

                    if max_before_tokens is not None:
                        ctx_before = [SENT_START] + words[sent_start:span_start]
                        ngrams += list(_iter_before_ngrams(ctx_before, max_before_tokens))
                    if max_after_tokens is not None:
                        ctx_after = words[span_end + 1 : sent_end] + [SENT_END]
                        ngrams += list(_iter_after_ngrams(ctx_after, max_after_tokens))

                    if is_entity:
                        for ng in ngrams:
                            count_entity[ng] += 1
                    else:
                        for ng in ngrams:
                            count_nil[ng] += 1

    all_ngrams = set(count_entity.keys()) | set(count_nil.keys())
    rows = []
    for ngram in all_ngrams:
        c_ent = count_entity[ngram]
        c_nil = count_nil[ngram]
        freq = c_ent + c_nil
        rows.append(
            {
                "ngram": ngram,
                "count_entity": c_ent,
                "count_nil": c_nil,
                "frequency": freq,
                "purity": c_nil / freq,
                "n_tokens": sum(1 for t in ngram if t not in _MARKERS),
                "has_start": ngram[0] == START,
                "has_end": ngram[-1] == END,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
