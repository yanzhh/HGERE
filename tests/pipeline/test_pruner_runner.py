"""Tests for pipeline/pruner_runner.py."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch


from hgere.pipeline.config import FinalPruningConfig, PrunerConfig
from hgere.pipeline.pruner_runner import (
    PrunerRunner,
    apply_final_pruning,
    apply_threshold,
    apply_topk,
)


# ---------------------------------------------------------------------------
# apply_threshold (pure function — no mocking needed)
# ---------------------------------------------------------------------------


class TestApplyThreshold:
    def test_keeps_spans_above_threshold(
        self, sentences_predictions_fixture: dict
    ) -> None:
        result = apply_threshold(sentences_predictions_fixture, threshold=0.5)
        # Only spans with prob >= 0.5 should remain
        sent = result[(0, 0)]
        assert all(prob >= 0.5 for _, _, prob, _ in sent)

    def test_removes_spans_below_threshold(
        self, sentences_predictions_fixture: dict
    ) -> None:
        result = apply_threshold(sentences_predictions_fixture, threshold=0.5)
        sent = result[(0, 0)]
        # (0, 0, 0.001, "NIL") and (0, 2, 0.0001, "NIL") should be gone
        starts = [s for s, _, _, _ in sent]
        assert starts.count(0) == 0

    def test_zero_threshold_keeps_all(
        self, sentences_predictions_fixture: dict
    ) -> None:
        result = apply_threshold(sentences_predictions_fixture, threshold=0.0)
        assert len(result[(0, 0)]) == 4

    def test_threshold_one_keeps_none(
        self, sentences_predictions_fixture: dict
    ) -> None:
        result = apply_threshold(sentences_predictions_fixture, threshold=1.0)
        assert result[(0, 0)] == []

    def test_empty_predictions(self) -> None:
        result = apply_threshold({}, threshold=0.5)
        assert result == {}

    def test_output_preserves_tuple_structure(
        self, sentences_predictions_fixture: dict
    ) -> None:
        result = apply_threshold(sentences_predictions_fixture, threshold=0.5)
        for span in result[(0, 0)]:
            assert len(span) == 4  # (start, end, prob, label)


# ---------------------------------------------------------------------------
# apply_topk
# ---------------------------------------------------------------------------


class TestApplyTopk:
    def _pool(self) -> dict[tuple[int, int], list[tuple[int, int, float, str]]]:
        return {
            (0, 0): [
                (0, 1, 0.9, "Method"),
                (1, 2, 0.6, "Task"),
                (0, 2, 0.3, "NIL"),
                (3, 4, 0.1, "NIL"),
            ]
        }

    def test_keeps_top_k_by_probability(self) -> None:
        result = apply_topk(
            self._pool(),
            topk_ratio=1.0,
            min_mentions_by_sentence=1,
            max_mentions_by_sentence=2,
        )
        # sent_len = max(4) + 1 = 5; K = clamp(1.0*5=5, 1, 2) = 2
        assert len(result[(0, 0)]) == 2
        probs = [p for _, _, p, _ in result[(0, 0)]]
        assert probs == sorted(probs, reverse=True)

    def test_min_mentions_floor(self) -> None:
        result = apply_topk(
            self._pool(),
            topk_ratio=0.01,
            min_mentions_by_sentence=3,
            max_mentions_by_sentence=18,
        )
        # topk_ratio*sent_len would be <1, but floor is 3
        assert len(result[(0, 0)]) == 3

    def test_max_mentions_cap(self) -> None:
        result = apply_topk(
            self._pool(),
            topk_ratio=10.0,
            min_mentions_by_sentence=1,
            max_mentions_by_sentence=2,
        )
        assert len(result[(0, 0)]) == 2

    def test_empty_sentence_returns_empty(self) -> None:
        pool: dict[tuple[int, int], list[tuple[int, int, float, str]]] = {(0, 0): []}
        result = apply_topk(
            pool,
            topk_ratio=0.5,
            min_mentions_by_sentence=1,
            max_mentions_by_sentence=10,
        )
        assert result[(0, 0)] == []

    def test_empty_pool_returns_empty(self) -> None:
        result = apply_topk(
            {}, topk_ratio=0.5, min_mentions_by_sentence=1, max_mentions_by_sentence=10
        )
        assert result == {}


# ---------------------------------------------------------------------------
# apply_final_pruning
# ---------------------------------------------------------------------------


class TestApplyFinalPruning:
    def _pool(self) -> dict[tuple[int, int], list[tuple[int, int, float, str]]]:
        return {
            (0, 0): [
                (0, 1, 0.9, "Method"),
                (1, 2, 0.4, "NIL"),
            ]
        }

    def test_threshold_method_filters_correctly(self) -> None:
        cfg = FinalPruningConfig(method="threshold", threshold=0.5)
        result = apply_final_pruning(self._pool(), cfg)
        assert len(result[(0, 0)]) == 1
        assert result[(0, 0)][0][2] == 0.9

    def test_topk_method_keeps_top_k(self) -> None:
        cfg = FinalPruningConfig(
            method="topk",
            topk_ratio=10.0,
            min_mentions_by_sentence=1,
            max_mentions_by_sentence=1,
        )
        result = apply_final_pruning(self._pool(), cfg)
        assert len(result[(0, 0)]) == 1
        assert result[(0, 0)][0][2] == 0.9  # highest prob


# ---------------------------------------------------------------------------
# PrunerRunner.build_predicted_ner_from_pool
# ---------------------------------------------------------------------------


class TestBuildPredictedNerFromPool:
    def test_strips_probability(self, sentences_predictions_fixture: dict) -> None:
        from hgere.pipeline.pruner_runner import build_predicted_ner_from_pool

        pool = apply_threshold(sentences_predictions_fixture, threshold=0.5)
        result = build_predicted_ner_from_pool(pool, n_docs=1, n_sents_per_doc=[1])
        # Each span should be (start, end, label) — 3-tuples
        for span in result[0][0]:
            assert len(span) == 3

    def test_correct_doc_and_sent_assignment(self) -> None:
        from hgere.pipeline.pruner_runner import build_predicted_ner_from_pool

        pool = {
            (0, 0): [(1, 2, 0.9, "Method")],
            (0, 1): [(3, 3, 0.8, "Task")],
            (1, 0): [(0, 1, 0.7, "Dataset")],
        }
        result = build_predicted_ner_from_pool(pool, n_docs=2, n_sents_per_doc=[2, 1])
        assert len(result) == 2
        assert len(result[0]) == 2  # doc 0 has 2 sentences
        assert len(result[1]) == 1  # doc 1 has 1 sentence
        assert result[0][0][0][:2] == [1, 2]
        assert result[1][0][0][:2] == [0, 1]


# ---------------------------------------------------------------------------
# PrunerRunner._load_prefilter_params
# ---------------------------------------------------------------------------


class TestLoadPrefilterParams:
    def test_best_config_topk_format(self, tmp_path: Any) -> None:
        from hgere.pipeline.pruner_runner import _load_prefilter_params

        cfg_file = tmp_path / "best_config.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "best_method": "topk",
                    "parameters": {
                        "topk_ratio": 0.3,
                        "min_mentions_num": 2,
                        "max_mentions_num": 15,
                    },
                }
            )
        )
        params = _load_prefilter_params(str(cfg_file))
        assert params["topk_ratio"] == 0.3
        assert params["min_mentions_num"] == 2
        assert params["max_mentions_num"] == 15

    def test_best_config_threshold_format_uses_defaults(self, tmp_path: Any) -> None:
        from hgere.pipeline.pruner_runner import (
            _DEFAULT_PREFILTER,
            _load_prefilter_params,
        )

        cfg_file = tmp_path / "best_config.json"
        cfg_file.write_text(
            json.dumps({"best_method": "threshold", "parameters": 0.01})
        )
        params = _load_prefilter_params(str(cfg_file))
        assert params == _DEFAULT_PREFILTER

    def test_plain_format(self, tmp_path: Any) -> None:
        from hgere.pipeline.pruner_runner import _load_prefilter_params

        cfg_file = tmp_path / "prefilter.json"
        cfg_file.write_text(
            json.dumps(
                {"topk_ratio": 0.4, "min_mentions_num": 1, "max_mentions_num": 12}
            )
        )
        params = _load_prefilter_params(str(cfg_file))
        assert params["topk_ratio"] == 0.4
        assert params["min_mentions_num"] == 1
        assert params["max_mentions_num"] == 12


# ---------------------------------------------------------------------------
# PrunerRunner (mocked model)
# ---------------------------------------------------------------------------


class TestPrunerRunner:
    def _make_config(self, tmp_path: Any, threshold: float = 0.5) -> PrunerConfig:
        return PrunerConfig(
            model_dir=str(tmp_path / "pruner"),
            base_model_name_or_path=str(tmp_path / "scibert"),
            final_pruning=FinalPruningConfig(method="threshold", threshold=threshold),
        )

    def test_run_empty_list_returns_empty(self, tmp_path: Any, tiny_doc: dict) -> None:
        cfg = self._make_config(tmp_path)
        with patch("hgere.pipeline.pruner_runner.PrunerRunner._load_model"):
            runner = PrunerRunner.__new__(PrunerRunner)
            runner._config = cfg
            runner._label_set = "gsap"
            runner._prefilter_params = {
                "topk_ratio": 0.5,
                "min_mentions_num": 3,
                "max_mentions_num": 18,
            }
            runner._model = MagicMock()
            runner._tokenizer = MagicMock()
            runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = {}
            result = runner.run([])
        assert result == []

    def test_run_adds_ner_candidates_proba_key(
        self, tmp_path: Any, tiny_doc: dict
    ) -> None:
        cfg = self._make_config(tmp_path)
        fake_pool = {(0, 0): [(1, 2, 0.95, "Method"), (0, 0, 0.001, "NIL")]}

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = fake_pool
            result = runner.run([tiny_doc])

        assert "ner_candidates_proba" in result[0]

    def test_run_adds_predicted_ner_key(self, tmp_path: Any, tiny_doc: dict) -> None:
        cfg = self._make_config(tmp_path)
        fake_pool = {(0, 0): [(1, 2, 0.95, "Method"), (0, 0, 0.001, "NIL")]}

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = fake_pool
            result = runner.run([tiny_doc])

        assert "predicted_ner" in result[0]

    def test_threshold_filters_predicted_ner(
        self, tmp_path: Any, tiny_doc: dict
    ) -> None:
        cfg = self._make_config(tmp_path, threshold=0.5)
        # pool has one span above threshold and one below
        fake_pool = {(0, 0): [(1, 2, 0.95, "Method"), (0, 0, 0.001, "NIL")]}

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = fake_pool
            result = runner.run([tiny_doc])

        predicted_ner_sent = result[0]["predicted_ner"][0]
        # Only the span with prob=0.95 should pass threshold=0.5
        assert len(predicted_ner_sent) == 1
        assert predicted_ner_sent[0][:2] == [1, 2]

    def test_topk_final_pruning(self, tmp_path: Any, tiny_doc: dict) -> None:
        cfg = PrunerConfig(
            model_dir=str(tmp_path / "pruner"),
            base_model_name_or_path=str(tmp_path / "scibert"),
            final_pruning=FinalPruningConfig(
                method="topk",
                topk_ratio=10.0,
                min_mentions_by_sentence=1,
                max_mentions_by_sentence=1,
            ),
        )
        fake_pool = {(0, 0): [(1, 2, 0.95, "Method"), (0, 0, 0.001, "NIL")]}

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = fake_pool
            result = runner.run([tiny_doc])

        # max_mentions_by_sentence=1 → only top-1 span kept
        assert len(result[0]["predicted_ner"][0]) == 1
        assert result[0]["predicted_ner"][0][0][:2] == [1, 2]

    def test_ner_candidates_proba_contains_all_pool_spans(
        self, tmp_path: Any, tiny_doc: dict
    ) -> None:
        cfg = self._make_config(tmp_path)
        fake_pool = {(0, 0): [(1, 2, 0.95, "Method"), (0, 0, 0.001, "NIL")]}

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = fake_pool
            result = runner.run([tiny_doc])

        # Full pool (both spans) should be in ner_candidates_proba
        candidates = result[0]["ner_candidates_proba"][0]
        assert len(candidates) == 2

    def test_doc_key_preserved(self, tmp_path: Any, tiny_doc: dict) -> None:
        cfg = self._make_config(tmp_path)

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = {(0, 0): [(1, 2, 0.9, "Method")]}
            result = runner.run([tiny_doc])

        assert result[0]["doc_key"] == tiny_doc["doc_key"]

    def test_batch_returns_same_count(self, tmp_path: Any, tiny_docs: list) -> None:
        cfg = self._make_config(tmp_path)
        fake_pool = {
            (0, 0): [(1, 2, 0.9, "Method")],
            (1, 0): [(2, 2, 0.8, "Task")],
        }

        runner = PrunerRunner.__new__(PrunerRunner)
        runner._config = cfg
        runner._label_set = "gsap"
        runner._prefilter_params = {
            "topk_ratio": 0.5,
            "min_mentions_num": 3,
            "max_mentions_num": 18,
        }
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"

        with patch("hgere.pipeline.pruner_runner.run_pruner_inference") as mock_inf:
            mock_inf.return_value = fake_pool
            result = runner.run(tiny_docs)

        assert len(result) == len(tiny_docs)
