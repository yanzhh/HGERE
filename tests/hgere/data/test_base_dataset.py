"""Tests for base_dataset.py — Steps 2-5 and 14 of TDD plan."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader

from hgere.data.base_dataset import DocumentDataset
from hgere.data.data_types import Document


class _ConcreteDataset(DocumentDataset):
    """Minimal concrete subclass for testing DocumentDataset base methods."""

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, idx: int) -> Document:
        return self._load_raw_document(idx)


# ---------------------------------------------------------------------------
# Step 2 — _index_file: line counting and offset correctness
# ---------------------------------------------------------------------------


class TestIndexFile:
    def test_counts_lines(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        assert len(ds._offsets) == 3

    def test_offsets_seek_to_correct_doc(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        with open(tiny_jsonl_path, "rb") as f:
            for i, expected_key in enumerate(["doc_0", "doc_1", "doc_2"]):
                f.seek(ds._offsets[i])
                line = f.readline()
                doc = json.loads(line)
                assert doc["doc_key"] == expected_key

    def test_doc_limit(self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock) -> None:
        ds = _ConcreteDataset(
            tiny_jsonl_path, mock_tokenizer, max_seq_length=128, doc_limit=2
        )
        assert len(ds._offsets) == 2


# ---------------------------------------------------------------------------
# Step 3 — PTB unescape, tokenize_word, is_punctuation
# ---------------------------------------------------------------------------


class TestGetOriginalToken:
    def test_lrb(self, mock_tokenizer: MagicMock) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.get_original_token("-LRB-") == "("

    def test_rrb(self, mock_tokenizer: MagicMock) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.get_original_token("-RRB-") == ")"

    def test_lsb(self, mock_tokenizer: MagicMock) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.get_original_token("-LSB-") == "["

    def test_rsb(self, mock_tokenizer: MagicMock) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.get_original_token("-RSB-") == "]"

    def test_no_escape_passthrough(self, mock_tokenizer: MagicMock) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.get_original_token("hello") == "hello"


class TestIsPunctuation:
    def test_period_is_punctuation(self) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.is_punctuation(".")

    def test_comma_is_punctuation(self) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.is_punctuation(",")

    def test_letter_not_punctuation(self) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert not ds.is_punctuation("a")

    def test_digit_not_punctuation(self) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert not ds.is_punctuation("5")

    def test_exclamation_is_punctuation(self) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        assert ds.is_punctuation("!")


# ---------------------------------------------------------------------------
# Step 4 — _tokenize_document: subword alignment
# ---------------------------------------------------------------------------


class TestTokenizeDocument:
    """Uses the real mock_tokenizer fixture (words >4 chars → 2 subwords)."""

    def test_token2subword_length(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)  # doc_0: ["Hello", "world", "."]
        idx = ds._tokenize_document(doc)
        # n_tokens for doc_0: 3 tokens → token2subword has n_tokens+1 entries
        n_tokens = sum(len(s.tokens) for s in doc.sentences)
        assert len(idx.token2subword) == n_tokens + 1

    def test_token2subword_starts_at_zero(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        assert idx.token2subword[0] == 0

    def test_subword2token_roundtrip(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        n_tokens = sum(len(s.tokens) for s in doc.sentences)
        for token_i in range(n_tokens):
            first_subword = idx.token2subword[token_i]
            assert idx.subword2token[first_subword] == token_i

    def test_subword_sentence_boundaries_length(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(1)  # doc_1: 2 sentences
        idx = ds._tokenize_document(doc)
        n_sentences = len(doc.sentences)
        # boundaries has one entry per sentence start + final end
        assert len(idx.subword_sentence_boundaries) == n_sentences + 1

    def test_first_sentence_boundary_is_zero(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        assert idx.subword_sentence_boundaries[0] == 0

    def test_subword_start_positions_type(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        assert isinstance(idx.subword_start_positions, frozenset)

    def test_subword_tokens_populated(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """subword_tokens must be populated and match the number of subwords."""
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        assert len(idx.subword_tokens) > 0
        assert len(idx.subword_tokens) == len(idx.subword2token)

    def test_subword_tokens_match_per_word_tokenization(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """Tokens from is_split_into_words must equal per-word tokenize() results."""
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        # Re-derive expected tokens using the legacy per-word approach
        expected: list[str] = []
        for sent in doc.sentences:
            for token in sent.tokens:
                expected.extend(mock_tokenizer.tokenize(token))
        assert idx.subword_tokens == expected

    def test_subword_start_positions_includes_zero(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        idx = ds._tokenize_document(doc)
        assert 0 in idx.subword_start_positions


# ---------------------------------------------------------------------------
# Step 5 — _get_context_window
# ---------------------------------------------------------------------------


class TestGetContextWindow:
    def _make_ds(self, mock_tokenizer: MagicMock) -> DocumentDataset:
        return _ConcreteDataset.__new__(_ConcreteDataset)

    def test_short_sentence_no_context(self, mock_tokenizer: MagicMock) -> None:
        """When the sentence fills all available subwords, context = 0."""
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        # 10 subwords available (max_seq_length=12, minus CLS+SEP = 10)
        # sentence exactly fills 10 subwords: no room for context
        subwords = list(range(10))
        sentence_start = 0
        sentence_end = 10
        cw = ds._get_context_window(
            subword_tokens=subwords,
            doc_sent_start=sentence_start,
            doc_sent_end=sentence_end,
            max_num_subwords=10,
            nocross=False,
        )
        assert cw.left_context_length == 0
        assert cw.sentence_length == 10

    def test_symmetric_context_fill(self, mock_tokenizer: MagicMock) -> None:
        """When left == right, context fills symmetrically."""
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        # Document: [5 left] [4 sentence] [5 right], max_num_subwords=14
        # half_context = (14-4)/2 = 5 — should take 5 from each side
        subwords = list(range(14))
        cw = ds._get_context_window(
            subword_tokens=subwords,
            doc_sent_start=5,
            doc_sent_end=9,
            max_num_subwords=14,
            nocross=False,
        )
        assert cw.left_context_length == 5
        assert cw.sentence_length == 4
        assert len(cw.target_tokens) <= 14

    def test_nocross_disables_context(self, mock_tokenizer: MagicMock) -> None:
        """nocross=True forces left and right context to 0."""
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        subwords = list(range(20))
        cw = ds._get_context_window(
            subword_tokens=subwords,
            doc_sent_start=5,
            doc_sent_end=10,
            max_num_subwords=18,
            nocross=True,
        )
        assert cw.left_context_length == 0
        assert cw.doc_offset == 5

    def test_asymmetric_more_right(self, mock_tokenizer: MagicMock) -> None:
        """When right is longer, bias context toward right."""
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        # [2 left] [4 sentence] [10 right], max_num_subwords=12
        # Since left < right: left_context = min(2, half=4) = 2
        subwords = list(range(16))
        cw = ds._get_context_window(
            subword_tokens=subwords,
            doc_sent_start=2,
            doc_sent_end=6,
            max_num_subwords=12,
            nocross=False,
        )
        assert cw.left_context_length == 2
        assert cw.sentence_length == 4

    def test_doc_offset_equals_start_minus_left(
        self, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset.__new__(_ConcreteDataset)
        subwords = list(range(20))
        cw = ds._get_context_window(
            subword_tokens=subwords,
            doc_sent_start=5,
            doc_sent_end=10,
            max_num_subwords=18,
            nocross=False,
        )
        assert cw.doc_offset == cw.doc_offset  # basic sanity
        # doc_offset = doc_sent_start - left_context_length
        assert cw.doc_offset == 5 - cw.left_context_length


# ---------------------------------------------------------------------------
# _load_raw_document: PTB unescape applied during load
# ---------------------------------------------------------------------------


class TestLoadRawDocument:
    def test_ptb_unescaped(self, tmp_path: Path, mock_tokenizer: MagicMock) -> None:
        doc = {
            "doc_key": "doc_ptb",
            "sentences": [["-LRB-", "hello", "-RRB-"]],
            "ner": [[]],
            "relations": [[]],
        }
        path = tmp_path / "ptb.jsonl"
        path.write_text(json.dumps(doc) + "\n")
        ds = _ConcreteDataset(path, mock_tokenizer, max_seq_length=128)
        raw_doc = ds._load_raw_document(0)
        assert raw_doc.sentences[0].tokens == ("(", "hello", ")")

    def test_returns_document_type(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        doc = ds._load_raw_document(0)
        assert isinstance(doc, Document)
        assert doc.doc_key == "doc_0"


# ---------------------------------------------------------------------------
# Step 14 — Lazy loading: pickle-safety, consistency, multi-worker DataLoader
# ---------------------------------------------------------------------------


class TestLazyLoading:
    def test_no_file_handle_on_self(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """DocumentDataset must not store any open file handles (pickle-safety for workers)."""
        import io

        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        for attr_value in vars(ds).values():
            assert not isinstance(attr_value, io.IOBase), (
                f"Found open file handle on dataset instance: {attr_value!r}"
            )

    def test_lazy_true_len_equals_lazy_false(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """lazy=True and lazy=False datasets must have identical __len__."""
        ds_eager = _ConcreteDataset(
            tiny_jsonl_path, mock_tokenizer, max_seq_length=128, lazy=False
        )
        ds_lazy = _ConcreteDataset(
            tiny_jsonl_path, mock_tokenizer, max_seq_length=128, lazy=True
        )
        assert len(ds_lazy) == len(ds_eager)

    def test_lazy_true_getitem_equals_eager(
        self, tiny_jsonl_path: Path, mock_tokenizer: MagicMock
    ) -> None:
        """Items returned by lazy and eager modes must be identical Documents."""
        ds_eager = _ConcreteDataset(
            tiny_jsonl_path, mock_tokenizer, max_seq_length=128, lazy=False
        )
        ds_lazy = _ConcreteDataset(
            tiny_jsonl_path, mock_tokenizer, max_seq_length=128, lazy=True
        )
        for i in range(len(ds_eager)):
            assert ds_lazy[i].doc_key == ds_eager[i].doc_key
            assert len(ds_lazy[i].sentences) == len(ds_eager[i].sentences)

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_dataloader_multiworker_safe(
        self,
        tiny_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        num_workers: int,
    ) -> None:
        """DataLoader with num_workers>0 must complete without PicklingError."""
        ds = _ConcreteDataset(tiny_jsonl_path, mock_tokenizer, max_seq_length=128)
        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=list,
        )
        batches = list(loader)
        # 3 docs in tiny_jsonl_path → 3 batches
        assert len(batches) == len(ds)
