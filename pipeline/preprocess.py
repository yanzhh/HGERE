# Get a JSONL file and generate verctorized input for model

# * need dataloader
# * need vectorizer

example_doc = dict(
   sentences = [
		["Hallo", "I", "am", "a", "sentence", "."],
		["This", "is", "the", "second", "sentence", "!"],
   ]
)

example_annnoted_doc = dict(
    text = "An example doc. It contains only two sentences."
    words = ["An", "example", "doc", ".", "It", "contains", "only", "two", "sentences", "."]
    # sentences are word_idx [0] (begining of first sentence) + [end_of_sentence_1, ...]
    sents = [0,4,9]
    # words char span: dim: n_words, 2: char_begin, char_end
    #words_char_span = [[0, 1], [3, 9], ...]
    # menitons: dim(n_mentions, 3) begin_token, end_token, label_idx
    mentions = [[1,2,0], [4,4,1]] # [Data:"example doc"], [DataGeneric: "it"]
)
example_annnoted_doc_with_token = dict(
    text = "An example doc. It contains only two sentences."
    words =         ["An"  , "example", "doc", "."  , "It" , "contains", "only", "two", "sentences", "."    ]
    word_to_token = [[0, 0], [1,2]    , [3,3], [4,4], [5,5], [6,6]     , [7,7] , [8,8], [9,9]      , [10,10]]
    
    token =     ["An", "e", "#xample", "doc", "." , "It", "contain", "#s" , "only", "two", "sentences", "."  ]
    token_ids = [ 34 , 324, 32       , 439  , 9348, 239 , 234      , 9298 , 948   ,  498 , 423        , 9348 ]
    # sentences are word_idx [0] (begining of first sentence) + [end_of_sentence_1, ...]
    sents = [0,4,9]
    # words char span: dim: n_words, 2: char_begin, char_end
    #words_char_span = [[0, 1], [3, 9], ...]
    # menitons: dim(n_mentions, 3) begin_token, end_token, label_idx
    mentions = [[1,2,0], [4,4,1]] # [Data:"example doc"], [DataGeneric: "it"]
)

def gen_sentence_samples(doc):
    sent_word_idxs = [begin, end for begin, end in zip(doc.sents[:-1], doc.sents[1:])] 
    sent_token_idxs = [word_to_token[0, begin

"""
# Sample: (with 2 spans)
| context | sentence | context | <sep> | span    | span    | span  | span  |
| before  |          | after   |       | start_1 | start_2 | end_1 | end_2 |
 * before sep length: max_sequence_length ( default: 384 token; setting: 256 )
 * max size of n grams: max_mention_ori_length ( default: 8; setting: 12)
 * maximum number of pairs: max_pair_length (doubled for start, end) (default: 256; setting: 64)

 value			|	default	|	our setting	|	description
 context_length	|	384		|	265			|	Half of the token for context
 max_spans_per_sample |	256	|	64			|	
 length total 	|	
 n_grams_max	|	12 		|	8			|	(e.g., urls, maybe optimization potential via heruistic?)

Example of n_grams for a 10 word sentence:
 10 * 12 = 120 n_grams

"""

#    train_dataset = ACEDatasetNER(
#        logger=logger, tokenizer=tokenizer, file_path=train_file, args=args
#    )


# Data Preprocessor
# * input
#   * label list
#   * tokenizer

# Plan
#  I need a preprocessor taking the json document and returning the vectorized version
# use ACEDatasetNER

# tokenize
#  all sentence tokens in one array. (store sentence beginnings)
#    * tokenize each word
#    * keep indices for sentence boundaries
#    * keep indices for sentences
#    * create subword_tokens
#    * subword_to_token
#    * token_to_subword

#    * Create ofset n grams
#    * 


# for BERT:
# input_ids + [1] (for each n_gram) + [0] (for each pad till max_pair_length)
# input_ids + [2] (for each n_gram) + [0] (for each pad till max_pair_length)

# Target
'''
sentences
  sent to subword token start and end

 subwords for doc
   SubwordToken 
      berdId: int
      begin: int
      end: int

Subword Token

''' 



import numpy as np

def generate_start_end(subword_to_word: np.ndarray) -> (np.ndarray, np.ndarray):
    #subword_to_word = np.array(subword_to_word)
    # Find where the word index changes — these are word boundaries
    subword_is_word_border = np.r_[True, np.diff(subword_to_word) != 0, True]
    # idxs of word_borders (idxs of True values)
    change_points = np.flatnonzero(subword_is_word_border)

    # Start indices are all but the last change point
    word_to_subword_start = change_points[:-1]
    # End indices are one before each next change point
    word_to_subword_end = change_points[1:] - 1
    return word_to_subword_start, word_to_subword_end


def test_generate_start_end():
    subword_to_word = np.array([0, 1, 1, 1, 2, 2, 3, 4])
    starts, ends = generate_start_end(subword_to_word)
    print(list(starts))
    print(list(ends))
    assert list(starts) == [0, 1, 4, 6, 7]
    print(list(ends))
    assert list(ends) == [0, 3, 5, 6, 7]

test_generate_start_end()
def sentences_to_words(sentences):
    word_sent_info = [(word, sent_idx) for sent_idx, words in enumerate(sentences) for word in words]
    words, word_to_sent = zip(*word_sent_info)
    return list(words), np.array(word_to_sent)

def test_sentences_to_words():
    sents = example_doc["sentences"]
    words, word_to_sent = sentences_to_words(sents)
    assert words == ['Hallo', 'I', 'am', 'a', 'sentence', '.', 'This', 'is', 'the', 'second', 'sentence', '!']
test_sentences_to_words()


    

class Document:
    def __init__(self, doc, tokenizer):
        self.words, self.word_to_sent = sentences_to_words(doc["sentences"])
        self.sent_to_word_start, self.sent_to_word_end = generate_start_end(self.word_to_sent)
        # can not use BertTokenizerFast because of old transformers version!
        encoding = [tokenizer.tokenize(word, returns_offsets_mapping=True) for word in self.words]
        self.subwords, self.subword_to_word = sentences_to_words(encoding)
        self.subword_token_ids = tokenizer.convert_tokens_to_ids(self.subwords)
        self.word_to_subword_start, self.word_to_subword_end = generate_start_end(self.subword_to_word)
        self.sent_to_subword_start = self.word_to_subword_start[self.sent_to_word_start]
        self.sent_to_subword_end = self.word_to_subword_end[self.sent_to_word_end]

    def sentence_with_context(self, sentence_idx):
        subword_start = self.sent_to_subword_start[sentence_idx]
        subword_end = self.sent_to_subword_end[sentence_idx]
        print(self.sent_to_subword_start)
        print(self.sent_to_subword_end)
        sent = self.words[subword_start:subword_end]
        print(sent)

def test_doc_to_samples():
    print("load transformers")
    from transformers import BertTokenizer
    model_path = "../pretrained_models/scibert_scivocab_uncased"
    print("load tokenizer")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    doc = Document(example_doc, tokenizer)
    print(doc)
    print(doc.sentence_with_context(0))
    print(doc.sentence_with_context(1))
test_doc_to_samples()


def sentences_to_token(tokenizer, sentences):
	sentence_boundaries = [0] + list(accumulate([len(s) for s in sentences]))
	words = chain(*sentences)
	return token, sentence_starts

def token_to_subwords(token):
	return subword_token, token_to_subwords, subwords_to_token

# args.max_seq_length

def maximize_context(subword_token, sentence_start, sentence_end):
	""" 293-342
	args.max_seq_length - 2 (2 are: start_token and sep token at end)
	"""
	return subword_token, sentence_start_maximized, sentence_end_maximized

# args.max_mention_ori_length is the maximum length of entities in subtokens

def get_n_grams(sentence_start, sentence_end, max_length, subword_to_word, word_to_subword, span_label_lookup):
	""" 347 ff.
	* get indices of words in sentence (not subtokens)
	* for each word collect all n_grams until max_length or word is not in sentence anymore
	"""
	n_grams = []
	sent_word_starts = subword_to_word[sentence_start]
	sent_word_ends = subword_to_word[sentence_end] + 1
	words_start = range(sent_word_starts, sent_wod_ends)
	words_subword_idxs = [word_to_subword[w_idx] for w_idx in words_start]
	for start_idx in range(0, max_length):
		for n in range(1, max_length):
			end_idx = start_idx + n
			start_subword, start_word = words_subword_idxs[start_idx]
			end_subword, end_word = words_subword_idxs[end_idx]
			# max subword token length check
			if end_subword - start_subword > max_length:
				break
			span_word = start_word, end_word
			span_subword = start_subword, end_subword
			label = span_label_lookup[span_word]
			n_grams.append((span_subword, label, span_word))
	return n_grams

# args.max_pair_length aka self.max_entity_length / 2
# Generate sentence, candidate tuples (one sentence can be used as input more than ones, because the number of entities are restricted (by args.max_pair_length)
def sentences_to_samples():
	pass

def add_standoff_tokens(padding=True):
	pass


def generate_attention_mask():
	#	only the input ids not the standoff annotations should be attended to
    pass

def generate_full_attention_mask():
    pass
 
"""
 attention from standoff start to standoff end and sequence
attention inside of sequence
no intention from sequence to standoff (???information flow need to pass sequence dependencies???)

"""
""" final output:
 600 ff.
item = [
	input_ids,
	attention_mask, # (only inner sequence and between standoff and full string),
	position_id, # position for sequence and for standoff: position of original start and end (used for positional embeddings I guess)
    labels, # list of labels for standoffs (one for each pair (start, end))
    mention_pos, # list of tuple: start_idx, end_idx for each span
    full_attention_mask, # An array with input length. 1 if not in padding for input and for standoff (I do not know why)
]
"""
