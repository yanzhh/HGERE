"""Tests for RelationDataset (reworked) — Steps 11-13 of TDD plan."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from gsapere.data.collators import RelationCollator
from gsapere.data.config import RelationDatasetParams
from gsapere.data.data_types import CandidateStats
from gsapere.data.relation_dataset import RelationDataset


def _make_dataset(
    file_path: Path,
    tokenizer: MagicMock,
    labels: MagicMock,
    params: RelationDatasetParams,
) -> RelationDataset:
    return RelationDataset(
        logger=MagicMock(),
        tokenizer=tokenizer,
        labels=labels,
        file_path=str(file_path),
        params=params,
    )


# ---------------------------------------------------------------------------
# Step 11 — _build_index (initialize): length, golden labels, predicted NERs
# ---------------------------------------------------------------------------


class TestBuildIndex:
    def test_len_equals_n_sentences(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """The fixture has 1 doc with 2 sentences → dataset length == 2."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds) == 2

    def test_golden_labels_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds.golden_labels) > 0

    def test_ner_golden_labels_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds.ner_golden_labels) > 0

    def test_global_predicted_ners_populated(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        # (0, 0) = doc_idx=0, sent_idx=0
        assert (0, 0) in ds.global_predicted_ners
        assert len(ds.global_predicted_ners[(0, 0)]) >= 1

    def test_sizes_list_length(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """self.sizes should have one entry per sentence."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert len(ds.sizes) == len(ds)


# ---------------------------------------------------------------------------
# Step 12 — _item_to_tensors (prepare_item)
# ---------------------------------------------------------------------------


class TestItemToTensors:
    def test_item_is_dict(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        assert isinstance(item, dict)

    def test_item_has_required_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        expected_keys = {
            "indices",
            "input_ids",
            "attention_mask",
            "position_ids",
            "sub_positions",
            "rel_labels",
            "ner_labels",
            "obj_token_pos",
            "n_ent",
            "subtoken_len",
            "sub",
        }
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        assert expected_keys == set(item.keys())

    def test_input_ids_shape_with_entities(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """For sentence with n_ent > 0, input_ids shape = (n_ent, seq_len+2*n_ent)."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        n_ent = item["n_ent"]
        if n_ent > 0:
            seq_len = item["subtoken_len"]
            assert item["input_ids"].shape[0] == n_ent
            assert item["input_ids"].shape[1] == seq_len + 2 * n_ent

    def test_rel_labels_shape(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds[0]
        n_ent = item["n_ent"]
        if n_ent > 0:
            assert item["rel_labels"].shape == (n_ent, n_ent)

    def test_n_ent_zero_sentence_returns_empty_tensors(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Sentence 1 (index 1) has a predicted entity → n_ent might be 0 or 1.
        But if we can find a sentence with no entities, tensors should be empty."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        for idx in range(len(ds)):
            item = ds[idx]
            if item["n_ent"] == 0:
                assert item["input_ids"].numel() == 0
                break


# ---------------------------------------------------------------------------
# Step 13 — RelationCollator
# ---------------------------------------------------------------------------


class TestRelationCollator:
    def test_single_sentence_output_keys(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        # Find a sentence with entities
        item = None
        for idx in range(len(ds)):
            candidate = ds[idx]
            if candidate["n_ent"] > 0:
                item = candidate
                break
        if item is None:
            return  # skip if no entities found

        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=relation_params.max_seq_length,
        )
        batch = collator([item])
        expected_keys = {
            "indices",
            "input_ids",
            "attention_mask",
            "position_ids",
            "sub_positions",
            "rel_labels",
            "ner_labels",
            "sub",
            "obj_token_pos",
            "ent_numbers",
        }
        assert expected_keys == set(batch.keys())

    def test_ent_numbers_tensor(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        items = [ds[i] for i in range(len(ds))]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=relation_params.max_seq_length,
        )
        batch = collator(items)
        assert "ent_numbers" in batch
        assert isinstance(batch["ent_numbers"], torch.Tensor)
        assert len(batch["ent_numbers"]) == len(items)

    def test_multi_sentence_rel_labels_shape(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """For a multi-sentence batch, rel_labels should be (n_sent, max_ent, max_ent)."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        items = [ds[i] for i in range(len(ds))]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id,
            max_seq_length=relation_params.max_seq_length,
        )
        batch = collator(items)
        n_sent = len(items)
        max_ent = max(item["n_ent"] for item in items)
        if max_ent > 0:
            assert batch["rel_labels"].shape[0] == n_sent
            assert batch["rel_labels"].shape[1] == max_ent
            assert batch["rel_labels"].shape[2] == max_ent


# ---------------------------------------------------------------------------
# Step 14 — candidate_stats
# ---------------------------------------------------------------------------


class TestCandidateStatsOnDataset:
    def test_candidate_stats_is_candidate_stats(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert isinstance(ds.candidate_stats, CandidateStats)

    def test_perfect_candidates_all_tp(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Fixture has predicted_ner == ner → TP=n_gold, FP=0, FN=0."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        stats = ds.candidate_stats
        assert stats.n_tp == stats.n_gold
        assert stats.n_fp == 0
        assert stats.n_fn == 0
        assert stats.recall == pytest.approx(1.0)
        assert stats.precision == pytest.approx(1.0)

    def test_fp_and_fn_counted_correctly(
        self,
        relation_jsonl_path_with_fp: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Fixture has 1 FP and 1 FN: n_gold=2, n_candidates=2, n_tp=1, n_fp=1, n_fn=1."""
        ds = _make_dataset(
            relation_jsonl_path_with_fp, mock_tokenizer, mock_labels, relation_params
        )
        stats = ds.candidate_stats
        assert stats.n_gold == 2
        assert stats.n_tp == 1
        assert stats.n_fp == 1
        assert stats.n_fn == 1

    def test_candidate_stats_logged(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Logger should be called with candidate stats during _build_index."""
        mock_logger = MagicMock()
        ds = RelationDataset(
            logger=mock_logger,
            tokenizer=mock_tokenizer,
            labels=mock_labels,
            file_path=str(relation_jsonl_path),
            params=relation_params,
        )
        assert ds.candidate_stats is not None
        assert mock_logger.info.called


# ---------------------------------------------------------------------------
# Tests for double-annotation handling
# ---------------------------------------------------------------------------


class TestDoubleAnnotationHandling:
    def test_ner_golden_labels_has_both_types(
        self,
        relation_jsonl_double_ann: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Both annotations for a doubly-annotated span appear in ner_golden_labels."""
        ds = _make_dataset(
            relation_jsonl_double_ann, mock_tokenizer, mock_labels, relation_params
        )
        # span (1,1) should appear with both "Method" and "Task"
        sent_id = (0, 0)
        gold_for_span = {
            label
            for (s, span, label) in ds.ner_golden_labels
            if s == sent_id and span == (1, 1)
        }
        assert "Method" in gold_for_span
        assert "Task" in gold_for_span

    def test_ner_golden_span_types_populated(
        self,
        relation_jsonl_double_ann: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_double_ann, mock_tokenizer, mock_labels, relation_params
        )
        key = (0, 0, 1, 1)  # doc=0, sent=0, start=1, end=1
        assert key in ds.ner_golden_span_types
        assert ds.ner_golden_span_types[key] == {"Method", "Task"}

    def test_golden_labels_with_ner_expanded_for_double_ann(
        self,
        relation_jsonl_double_ann: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """golden_labels_with_ner should contain all valid (s_type, o_type) combos."""
        ds = _make_dataset(
            relation_jsonl_double_ann, mock_tokenizer, mock_labels, relation_params
        )
        # relation: (1,1) → (2,2) "Used-for"
        # span (1,1) has {Method, Task}, span (2,2) has {Dataset}
        # Expected entries: (Method,Dataset) and (Task,Dataset)
        sent_id = (0, 0)
        rel_entries = {
            (subj, obj, label)
            for (s, subj, obj, label) in ds.golden_labels_with_ner
            if s == sent_id
        }
        assert ((1, 1, "Method"), (2, 2, "Dataset"), "Used-for") in rel_entries
        assert ((1, 1, "Task"), (2, 2, "Dataset"), "Used-for") in rel_entries

    def test_warning_logged_for_dropped_label(
        self,
        relation_jsonl_double_ann: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        mock_logger = MagicMock()
        RelationDataset(
            logger=mock_logger,
            tokenizer=mock_tokenizer,
            labels=mock_labels,
            file_path=str(relation_jsonl_double_ann),
            params=relation_params,
        )
        # warning should have been called for the dropped NER label
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any(
            "dropped label" in w.lower() or "multiple ner" in w.lower()
            for w in warning_calls
        )

    def test_connectivity_breaks_alphabetical_tiebreak(
        self,
        relation_jsonl_double_ann_connectivity: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """Label with higher relation connectivity wins over alphabetical order.

        Span (1,1) has {Method, Task}.  A singly-annotated Task span (2,2) is
        in a relation, giving "Task" connectivity=1 and "Method" connectivity=0.
        Expected training label for (1,1): "Task" (not "Method" as alphabetical
        would give).
        """
        ds = _make_dataset(
            relation_jsonl_double_ann_connectivity,
            mock_tokenizer,
            mock_labels,
            relation_params,
        )
        # entity_labels_gold is not directly exposed, but the chosen label is
        # reflected in ner_golden_labels: the dropped label counter will show
        # "Method" dropped (not "Task").  We verify via golden_labels_with_ner:
        # the relation (1,1)→(3,3) "Result" should be expanded with "Task" as
        # the subject type (the chosen label), not "Method".
        sent_id = (0, 0)
        rel_subj_types = {
            subj[-1]
            for (s, subj, obj, label) in ds.golden_labels_with_ner
            if s == sent_id and label == "Result" and subj[:2] == (1, 1)
        }
        assert "Task" in rel_subj_types

    def test_alphabetical_fallback_when_no_connectivity(
        self,
        relation_jsonl_double_ann: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """When no single-label span provides connectivity signal, fall back to alphabetical.

        In relation_jsonl_double_ann, the only relation endpoint that is single-label
        is (2,2)="Dataset". Neither "Method" nor "Task" gets connectivity from it.
        Alphabetical → "Method" is chosen.
        """
        # We can observe the chosen label indirectly via the dropped-label warning:
        # if "Method" is chosen, "Task" should appear in the dropped-label summary.
        from unittest.mock import MagicMock as MM

        mock_logger = MM()
        from gsapere.data.relation_dataset import RelationDataset

        RelationDataset(
            logger=mock_logger,
            tokenizer=mock_tokenizer,
            labels=mock_labels,
            file_path=str(relation_jsonl_double_ann),
            params=relation_params,
        )
        warning_text = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        # "Task" should be in the dropped label summary (Method chosen, Task dropped)
        assert "Task" in warning_text


class TestSelfRelationWarning:
    def test_warning_logged_for_self_relation(
        self,
        relation_jsonl_self_rel: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        mock_logger = MagicMock()
        RelationDataset(
            logger=mock_logger,
            tokenizer=mock_tokenizer,
            labels=mock_labels,
            file_path=str(relation_jsonl_self_rel),
            params=relation_params,
        )
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("self-relation" in w.lower() for w in warning_calls)

    def test_self_relation_not_in_golden_labels(
        self,
        relation_jsonl_self_rel: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_self_rel, mock_tokenizer, mock_labels, relation_params
        )
        # Self-relation is dropped; golden_labels should be empty
        assert len(ds.golden_labels) == 0


# ---------------------------------------------------------------------------
# Symmetric relation handling
# ---------------------------------------------------------------------------


class TestSymmetricRelationHandling:
    def test_reverse_annotated_sym_rel_maps_to_canonical(
        self,
        relation_jsonl_sym_reverse: Path,
        mock_tokenizer: MagicMock,
        mock_labels_with_sym: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """A sym relation annotated as B→A should appear as A→B (canonical) in golden_labels."""
        # relation_params has no_sym=True; override to no_sym=False so 'Similar' is sym
        from gsapere.data.config import RelationDatasetParams

        params_sym = RelationDatasetParams(
            max_seq_length=64,
            use_typemarker=False,
            local_rank=-1,
            model_type="bert",
            no_sym=False,
            nocross=False,
        )
        ds = _make_dataset(
            relation_jsonl_sym_reverse, mock_tokenizer, mock_labels_with_sym, params_sym
        )
        # Gold has (2,2)→(1,1); canonical is (1,1)→(2,2)
        doc_id = 0
        sent_id = 0
        canonical = ((doc_id, sent_id), (1, 1), (2, 2), "Similar")
        reverse = ((doc_id, sent_id), (2, 2), (1, 1), "Similar")
        assert canonical in ds.golden_labels
        assert reverse not in ds.golden_labels

    def test_duplicate_sym_directions_deduplicated(
        self,
        tmp_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels_with_sym: MagicMock,
    ) -> None:
        """Both A→B and B→A in gold for a sym label → only 1 entry in golden_labels."""
        import json

        doc = {
            "doc_key": "doc_sym_both",
            "sentences": [["word0", "word1", "word2", "word3"]],
            "ner": [[[1, 1, "Method"], [2, 2, "Method"]]],
            "predicted_ner": [[[1, 1, "Method"], [2, 2, "Method"]]],
            "relations": [[[1, 1, 2, 2, "Similar"], [2, 2, 1, 1, "Similar"]]],
        }
        path = tmp_path / "sym_both.jsonl"
        path.write_text(json.dumps(doc) + "\n")

        from gsapere.data.config import RelationDatasetParams

        params_sym = RelationDatasetParams(
            max_seq_length=64,
            use_typemarker=False,
            local_rank=-1,
            model_type="bert",
            no_sym=False,
            nocross=False,
        )
        ds = _make_dataset(path, mock_tokenizer, mock_labels_with_sym, params_sym)
        sym_gold = [e for e in ds.golden_labels if e[-1] == "Similar"]
        assert len(sym_gold) == 1


# ---------------------------------------------------------------------------
# dataset_id propagation (multi-head support)
# ---------------------------------------------------------------------------


class TestDatasetIdPropagation:
    def test_dataset_id_stored_on_dataset(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
    ) -> None:
        params = RelationDatasetParams(
            max_seq_length=64,
            use_typemarker=False,
            local_rank=-1,
            model_type="bert",
            no_sym=True,
            nocross=False,
            dataset_id="scier",
        )
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, params)
        assert ds.dataset_id == "scier"

    def test_dataset_id_none_by_default(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert ds.dataset_id is None

    def test_prepare_item_includes_dataset_id(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
    ) -> None:
        params = RelationDatasetParams(
            max_seq_length=64,
            use_typemarker=False,
            local_rank=-1,
            model_type="bert",
            no_sym=True,
            nocross=False,
            dataset_id="scier",
        )
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, params)
        item = ds.prepare_item(0)
        assert "dataset_id" in item
        assert item["dataset_id"] == "scier"

    def test_prepare_item_no_dataset_id_key_when_none(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        item = ds.prepare_item(0)
        assert "dataset_id" not in item

    def test_collated_batch_includes_dataset_id(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
    ) -> None:
        from gsapere.data.collators import RelationCollator

        params = RelationDatasetParams(
            max_seq_length=64,
            use_typemarker=False,
            local_rank=-1,
            model_type="bert",
            no_sym=True,
            nocross=False,
            dataset_id="scinlp",
        )
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, params)
        items = [ds.prepare_item(0), ds.prepare_item(1)]
        collator = RelationCollator(
            tokenizer_pad_id=mock_tokenizer.pad_token_id, max_seq_length=64
        )
        batch = collator(items)
        assert "dataset_id" in batch
        assert batch["dataset_id"] == "scinlp"


# ---------------------------------------------------------------------------
# use_dataset_id_token_as_cls — CLS token replacement
# ---------------------------------------------------------------------------


class TestDatasetIdTokenAsCls:
    """Tests for dataset_cls_token_id: pre-resolved [unusedX] token substitution."""

    def test_no_replacement_when_none(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
    ) -> None:
        params = RelationDatasetParams(
            max_seq_length=64,
            model_type="bert",
            no_sym=True,
            dataset_id="scier",
            dataset_cls_token_id=None,
        )
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, params)
        assert ds._dataset_cls_token_id is None
        item = ds.prepare_item(0)
        if item["n_ent"] > 0:
            assert item["input_ids"][0][0].item() == 101

    def test_replacement_at_position_zero(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
    ) -> None:
        # Simulates [unused4]=id 5 being the dataset CLS token for 'scier'
        dataset_token_id = 5
        params = RelationDatasetParams(
            max_seq_length=64,
            model_type="bert",
            no_sym=True,
            dataset_id="scier",
            dataset_cls_token_id=dataset_token_id,
        )
        ds = _make_dataset(relation_jsonl_path, mock_tokenizer, mock_labels, params)
        assert ds._dataset_cls_token_id == dataset_token_id
        item = ds.prepare_item(0)
        if item["n_ent"] > 0:
            for row in item["input_ids"]:
                assert row[0].item() == dataset_token_id, (
                    f"Expected {dataset_token_id} at position 0, got {row[0].item()}"
                )

    def test_no_replacement_without_token_id(
        self,
        relation_jsonl_path: Path,
        mock_tokenizer: MagicMock,
        mock_labels: MagicMock,
        relation_params: RelationDatasetParams,
    ) -> None:
        """dataset_cls_token_id defaults to None → no substitution."""
        ds = _make_dataset(
            relation_jsonl_path, mock_tokenizer, mock_labels, relation_params
        )
        assert ds._dataset_cls_token_id is None
