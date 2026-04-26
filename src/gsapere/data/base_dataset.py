"""
base_dataset.py — DocumentDataset base class.

Handles:
- JSONL file indexing (byte offsets for lazy loading)
- PTB unescape, tokenize_word, is_punctuation (static helpers)
- Subword alignment (_tokenize_document)
- Context window computation (_get_context_window)

Both PrunerDataset and RelationDataset inherit from this class.
"""

from __future__ import annotations

import json
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from gsapere.data.data_types import (
    ContextWindow,
    Document,
    NerSpan,
    Relation,
    Sentence,
    SubwordIndex,
)


class DocumentDataset(Dataset, ABC):
    """Abstract base class for all HGERE datasets.

    Provides shared JSONL loading (eager + lazy), PTB unescape, subword
    alignment, and context-window computation.  Subclasses implement
    _build_index(), __len__(), and __getitem__().
    """

    def __init__(
        self,
        file_path: str | Path | None,
        tokenizer: Any,
        max_seq_length: int,
        lazy: bool = False,
        pre_filter_params: dict[Any, Any] | None = None,
        doc_limit: int | None = None,
        in_memory_docs: list[dict] | None = None,
    ) -> None:
        self._file_path = Path(file_path) if file_path is not None else None
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._lazy = lazy
        self._doc_limit = doc_limit

        self.do_pre_filter = pre_filter_params is not None
        self.pre_filter_params = pre_filter_params
        self._in_memory_docs = in_memory_docs

        if in_memory_docs is not None:
            n = (
                min(len(in_memory_docs), doc_limit)
                if doc_limit is not None
                else len(in_memory_docs)
            )
            self._offsets = list(range(n))
        else:
            # Scan file for byte offsets — fast, no JSON parsing
            self._offsets = self._index_file()

    @property
    def file_path(self) -> Path:
        """Public accessor for the dataset file path (backward compat with ACEDatasetNER)."""
        if self._file_path is None:
            raise AttributeError(
                "Dataset was constructed from in-memory docs; no file path is available."
            )
        return self._file_path

    # ------------------------------------------------------------------
    # File indexing
    # ------------------------------------------------------------------

    def _index_file(self) -> list[int]:
        """Scan file in binary mode and record byte offset of each line.

        Returns a list of length <= number of lines (capped by doc_limit).
        """
        offsets: list[int] = []
        with open(self._file_path, "rb") as f:
            offset = 0
            for line in f:
                if self._doc_limit is not None and len(offsets) >= self._doc_limit:
                    break
                offsets.append(offset)
                offset += len(line)
        return offsets

    # ------------------------------------------------------------------
    # Raw document loading
    # ------------------------------------------------------------------

    def _load_raw_document(self, line_idx: int) -> Document:
        """Read line at _offsets[line_idx], parse JSON, apply PTB unescape,
        and return a frozen Document domain object."""
        if self._in_memory_docs is not None:
            raw = self._in_memory_docs[line_idx]
        else:
            with open(self._file_path, "rb") as f:
                f.seek(self._offsets[line_idx])
                raw = json.loads(f.readline())

        sentences_raw: list[list[str]] = raw["sentences"]
        ner_raw: list[list[list[Any]]] = raw.get("ner", [[] for _ in sentences_raw])
        relations_raw: list[list[list[Any]]] = raw.get(
            "relations", [[] for _ in sentences_raw]
        )

        sentences: list[Sentence] = []
        for sent_tokens, sent_ner, sent_rel in zip(
            sentences_raw, ner_raw, relations_raw
        ):
            # PTB unescape
            tokens = tuple(self.get_original_token(t) for t in sent_tokens)
            ner = tuple(NerSpan(start=s[0], end=s[1], label=s[2]) for s in sent_ner)
            relations = tuple(
                Relation(
                    subj_start=r[0],
                    subj_end=r[1],
                    obj_start=r[2],
                    obj_end=r[3],
                    label=r[4],
                )
                for r in sent_rel
            )
            sentences.append(Sentence(tokens=tokens, ner=ner, relations=relations))

        has_doc_key = "doc_key" in raw
        has_doc_id = "doc_id" in raw
        if has_doc_key and has_doc_id:
            raise KeyError(
                f"Document at line {line_idx} has both 'doc_key' and 'doc_id' fields; "
                "use exactly one."
            )
        if has_doc_key:
            doc_key = raw["doc_key"]
        elif has_doc_id:
            doc_key = raw["doc_id"]
        else:
            raise KeyError(
                f"Document at line {line_idx} has neither 'doc_key' nor 'doc_id'."
            )
        return Document(doc_key=doc_key, sentences=tuple(sentences))

    # ------------------------------------------------------------------
    # Subword alignment
    # ------------------------------------------------------------------

    def _tokenize_document(self, doc: Document) -> SubwordIndex:
        """Build token-to-subword and subword-to-token mappings for a document.

        All sentences are concatenated into a single flat pre-tokenized word list
        and passed to the tokenizer in one call via is_split_into_words=True.
        This is faster than calling tokenize() per word and lets the tokenizer
        handle all prefix-space / BPE context internally.
        """
        all_tokens: list[str] = []
        sentence_boundaries: list[int] = [0]
        for sent in doc.sentences:
            all_tokens.extend(sent.tokens)
            sentence_boundaries.append(len(all_tokens))

        # Whitespace-only tokens and tokens consisting entirely of characters that
        # tokenizers cannot map to subwords (combining marks, control chars, format
        # chars — e.g. U+0302 combining circumflex) produce no subwords and break
        # the word_id alignment.  Replace them with "." so every position gets at
        # least one subword; the original all_tokens list (and all span indices) are
        # unchanged.
        _INVISIBLE_CATEGORIES = frozenset({"Mn", "Mc", "Me", "Cc", "Cf", "Cs", "Co"})

        def _needs_substitution(tok: str) -> bool:
            if not tok.strip():
                return True
            return all(unicodedata.category(ch) in _INVISIBLE_CATEGORIES for ch in tok)

        tokens_for_tokenizer = [
            tok if not _needs_substitution(tok) else "." for tok in all_tokens
        ]

        # Single tokenizer call for the whole document
        enc = self.tokenizer(
            tokens_for_tokenizer, is_split_into_words=True, add_special_tokens=False
        )
        word_ids_list: list[int] = enc.word_ids()
        subword_tokens: list[str] = enc.tokens()

        # token2subword[i] = first subword index for token i (+ sentinel = total subwords)
        token2subword: list[int] = [0]
        for j, (prev, curr) in enumerate(zip(word_ids_list, word_ids_list[1:]), 1):
            if prev != curr:
                token2subword.append(j)
        token2subword.append(len(word_ids_list))

        if len(token2subword) != len(all_tokens) + 1:
            seen_ids = set(word_ids_list)
            missing = [
                (i, repr(tok)) for i, tok in enumerate(all_tokens) if i not in seen_ids
            ]
            raise ValueError(
                f"Tokenizer produced no subwords for {len(missing)} token(s) "
                f"in document '{doc.doc_key}': {missing[:10]}. "
                "Check for empty strings or unsupported characters in the input."
            )

        subword2token: list[int] = word_ids_list
        subword_start_positions: frozenset[int] = frozenset(token2subword[:-1])
        subword_sentence_boundaries: list[int] = [
            token2subword[boundary] for boundary in sentence_boundaries
        ]

        assert len(subword_sentence_boundaries) == len(doc.sentences) + 1

        return SubwordIndex(
            token2subword=token2subword,
            subword2token=subword2token,
            subword_sentence_boundaries=subword_sentence_boundaries,
            subword_start_positions=subword_start_positions,
            subword_tokens=subword_tokens,
        )

    # ------------------------------------------------------------------
    # Context window
    # ------------------------------------------------------------------

    def _get_context_window(
        self,
        subword_tokens: list[Any],
        doc_sent_start: int,
        doc_sent_end: int,
        max_num_subwords: int,
        nocross: bool = False,
    ) -> ContextWindow:
        """Compute left/right context around a sentence within its document.

        Returns a ContextWindow with the windowed token list (without CLS/SEP)
        and metadata needed for entity position remapping.
        """
        sentence_length = doc_sent_end - doc_sent_start

        # Clip over-long sentences
        if sentence_length > max_num_subwords:
            doc_sent_end -= sentence_length - max_num_subwords
            sentence_length = doc_sent_end - doc_sent_start

        left_length = doc_sent_start
        right_length = len(subword_tokens) - doc_sent_end

        if nocross or sentence_length >= max_num_subwords:
            left_context_length = 0
            right_context_length = 0
        else:
            half_context_length = (max_num_subwords - sentence_length) // 2
            if left_length < right_length:
                left_context_length = min(left_length, half_context_length)
                right_context_length = min(
                    right_length,
                    max_num_subwords - left_context_length - sentence_length,
                )
            else:
                right_context_length = min(right_length, half_context_length)
                left_context_length = min(
                    left_length,
                    max_num_subwords - right_context_length - sentence_length,
                )

        doc_offset = doc_sent_start - left_context_length
        target_tokens = list(
            subword_tokens[doc_offset : doc_sent_end + right_context_length]
        )

        return ContextWindow(
            target_tokens=target_tokens,
            doc_offset=doc_offset,
            left_context_length=left_context_length,
            sentence_length=sentence_length,
        )

    # ------------------------------------------------------------------
    # Static helpers (tokenization, PTB)
    # ------------------------------------------------------------------

    @staticmethod
    def is_punctuation(char: str) -> bool:
        """Return True if char is a punctuation character (Unicode-aware)."""
        cp = ord(char)
        if (
            (33 <= cp <= 47)
            or (58 <= cp <= 64)
            or (91 <= cp <= 96)
            or (123 <= cp <= 126)
        ):
            return True
        return unicodedata.category(char).startswith("P")

    @staticmethod
    def get_original_token(token: str) -> str:
        """Convert PTB escape sequences back to their original characters."""
        escape_to_original = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        return escape_to_original.get(token, token)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any: ...
