import json
from pathlib import Path
from collections import Counter

fn_lists = Path(__file__).parent / "word_lists.json"
with open(fn_lists, "r") as f:
    word_lists = json.load(f)
    BLACKLIST_UNIGRAM_LAST = set(word_lists["last_word"])
    # print(len(BLACKLIST_UNIGRAM_LAST))
    BLACKLIST_UNIGRAM_IN_TEXT = set(word_lists["in_text_word"])
    # print(len(BLACKLIST_UNIGRAM_IN_TEXT))
    BLACKLIST_UNIGRAM_FIRST = set(word_lists["first_word"])
    # print(len(BLACKLIST_UNIGRAM_FIRST))
    BLACKLIST_BIGRAM_IN_TEXT = set(word_lists["in_text_n_gram"])
    # print(len(BLACKLIST_BIGRAM_IN_TEXT))
    BLACKLIST_FIRST_LAST = set(word_lists["begin_end_words"])
    # print(len(BLACKLIST_FIRST_LAST))
    BLACKLIST_STOPWORD_PATTERN = set(word_lists["high_frequent_stopwords"])
    # print(len(BLACKLIST_STOPWORD_PATTERN))

    BLACKLIST_UNIGRAM = set(word_lists["identity"])  # 5.9
    # print(len(BLACKLIST_UNIGRAM))
    BLACKLIST_MOST_FREQ_TERM = set(word_lists["multiple_word_counts"])  # 4.37
    # print(len(BLACKLIST_MOST_FREQ_TERM))
    BLACKLIST_NGRAM = set(word_lists["long_texts"])  # 1.8
    # print(len(BLACKLIST_NGRAM))


def calc_max_ngrams(len_text, n=12):
    overhead = min(len_text, n - 1)
    overhead = ((overhead * overhead) + overhead) / 2
    overhead_plus = (max(n - len_text, 1) - 1) * len_text
    n_ngrams = len_text * n - overhead - overhead_plus
    return int(n_ngrams)


def generate_doc_ner_candidates(doc, n=12, do_filter=True):
    ner_candidates = []
    sent_begin = 0
    n_cands = 0
    false_negatives = 0
    true_positives = 0
    max_ngrams = 0
    n_token = 0
    for sent, sent_ner in zip(doc["sentences"], doc["ner"]):
        n_token += len(sent)
        max_ngrams += calc_max_ngrams(len(sent), n)
        sent_ner_gold = {(b, e): _ for b, e, _ in sent_ner}
        sent_cands = get_n_grams(sent, n, sent_begin, sent_ner, do_filter=do_filter)
        sent_ner_pred = {(b, e) for b, e, _ in sent_cands}
        sent_fn = len(set(sent_ner_gold) - sent_ner_pred)
        sent_tp = len(set(sent_ner_gold) & sent_ner_pred)
        false_negatives += sent_fn
        true_positives += sent_tp
        n_cands += len(sent_cands)
        ner_candidates.append(sent_cands)
        sent_begin += len(sent)
    doc["ner_candidates"] = ner_candidates
    false_positives = n_cands - true_positives
    # Alle gefundenen, die wirklich
    true_negatives = max_ngrams - true_positives - false_positives
    return (
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        max_ngrams,
        n_token,
    )


def get_n_grams(sentence, max_n=12, sentence_start=0, ner=[], do_filter=True):
    ner_candidates = []
    is_ent = set()
    for begin, end, label in ner:
        end = end
        is_ent.add((begin, end))
    len_sent = len(sentence)
    # skipped = 0
    for start in range(len_sent):
        for n in range(1, max_n + 1):
            if start + n > len_sent:
                break
            # Filtering
            n_gram = sentence[start : start + n]
            stop = False if not do_filter else need_stop(n, n_gram)
            if stop:
                break
            if not do_filter or use(n, n_gram):
                begin = sentence_start + start
                end = sentence_start + start + n - 1
                label = (begin, end) in is_ent
                ner_candidates.append([begin, end, label])
    return ner_candidates


def need_stop(n, n_gram):
    stop = True
    if n_gram[-1] in BLACKLIST_UNIGRAM_IN_TEXT:  # 27.32, 5fp (/6417)
        return stop
    if n > 1:
        first_unigram = n_gram[0]
        if first_unigram in BLACKLIST_UNIGRAM_FIRST:  # 27.7, 6fn
            return stop
        bi_gram_at_end = " ".join(n_gram[-2:])
        if bi_gram_at_end in BLACKLIST_BIGRAM_IN_TEXT:  # 25.72, 5fn
            return stop

        # multiple word count
        # Intuition: if "the" is two times in an n_gram,
        # it is not a valid entity mention
        # only the word with the highest frequence is returned
        # (incl. the frequence with a max threshold of 3)
        threshold_max = 3
        threshold_min = 2
        unigram_count = Counter(n_gram).most_common()
        unigram_count.sort(key=lambda x: -x[1])
        pattern = [
            f"{t} ({min(threshold_max, freq)})"
            for t, freq in unigram_count
            if freq >= threshold_min
        ]
        if pattern and pattern[0] in BLACKLIST_MOST_FREQ_TERM:  # 4.37, 3fn
            return stop
    return not stop


def use(n, n_gram):
    use = True
    if n == 1 and n_gram[0] in BLACKLIST_UNIGRAM:  # 5.92, 6fn
        return not use
    if n > 1:
        if n_gram[-1] in BLACKLIST_UNIGRAM_LAST:  # 28.76, 5fn
            return not use
        if n > 2:
            first_last = n_gram[0] + " " + n_gram[-1]
            if first_last in BLACKLIST_FIRST_LAST:  # 20.25%, 6fp
                return not use
        text = " ".join(n_gram)
        if n < 8 and text in BLACKLIST_NGRAM:  # 1.85, 4fn
            return not use
        # unk_stopword_pattern
        unk_stopword_pattern = ["unk" if w not in stopwords_set else w for w in n_gram]
        if "unk" in unk_stopword_pattern:
            unk_stopword_pattern = " ".join(unk_stopword_pattern)
            if unk_stopword_pattern in BLACKLIST_STOPWORD_PATTERN:  # 14.12, 7fn
                return not use
    return use


## Not used anymore!


def create_bool_fun(words, fun):
    def fun_bool(text):
        trans = fun(text)
        return trans not in words

    return fun_bool


def create_is_good_candidate_fun(funs):
    funs_parametrized = [create_bool_fun(word_list, f) for f, word_list in funs]

    # print(len(funs_parametrized))
    def is_good_candidate(word):
        for is_clean_fun in funs_parametrized:
            if not is_clean_fun(word):
                return False
        return True

    return is_good_candidate


### checker


def last_word(text):
    """28.76%"""
    text = text.split()
    if len(text) > 1:
        return text[-1]
    return True


def first_word(text):
    """27.7% filtered out."""
    text = text.split()
    if len(text) > 1:
        return text[0]
    return "<one_word>"


def create_in_text_word(top_words):
    """27.32%"""

    def in_text_word(text):
        words = text.split()
        if len(words) < 1:
            return "<to_short>"
        two_grams = words
        two_grams.sort(key=lambda x: -top_words.get(x, 0))
        high_freq_two_gram = two_grams[0]
        return high_freq_two_gram

    return in_text_word


def create_in_text_n_gram(top_n_grams):
    """25.72%"""

    def in_word_n_gram(text):
        words = text.split()
        if len(words) < 2:
            return "<to_short>"
        two_grams = [f"{t1} {t2}" for t1, t2 in zip(words[:-1], words[1:])]
        two_grams.sort(key=lambda x: -top_n_grams.get(x, 0))
        high_freq_two_gram = two_grams[0]
        return high_freq_two_gram

    return in_word_n_gram


def begin_end_words(text):
    """20.25%"""
    text = text.split()
    if len(text) > 2:
        return text[0] + " " + text[-1]
    return "<to_short>"


def create_high_frequent_stopwords(stopwords):
    """14.06%"""

    def high_frequent_stopwords(text):
        text = text.split()
        text = [w if w in stopwords_set else "unk" for w in text]
        if "unk" in set(text):
            return " ".join(text)
        return "<no_unk>"

    return high_frequent_stopwords


def identity(text):
    """5.92%"""
    if len(text.split()) == 1:
        return text
    return "<non_one_word>"


def multiple_word_counts(text):
    """4.29% filtered out.
    Intuition: if "the" is two times in an n_gram, it is not a valid entity mention
    only the word with the highest frequence is returned (incl. the frequence with a max threshold of 3)
    """
    threshold_max = 3
    threshold_min = 1
    text = text.split()
    text = sorted(Counter(text).most_common(), key=lambda x: -x[1])
    text = [
        f"{t} ({min(threshold_max, freq)})" for t, freq in text if freq > threshold_min
    ]
    if text:
        return text[0]
    return "<no_double>"


def long_texts(text):
    """1.79%"""
    min_val = 2
    length = len([w for w in text.split()])
    if length >= min_val:
        return text
    return f"<lt_{min_val-1}_words>"


stopwords_set = {
    ",",
    "the",
    ".",
    "of",
    "and",
    "-",
    "to",
    ")",
    "(",
    "a",
    "in",
    "is",
    "for",
    "on",
    "that",
    "with",
    "we",
    "as",
    "model",
    "al",
    "et",
    "are",
    "from",
    "by",
    "]",
    "models",
    "[",
    #'training',
    "data",
    "be",
    "1",
    "which",
    "can",
    '"',
    ":",
    "an",
    "this",
    "our",
    ";",
    "not",
    "each",
    #'learning',
    "it",
    "/",
    "2",
    "=",
    "or",
}
