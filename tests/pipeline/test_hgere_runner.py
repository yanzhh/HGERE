"""Tests for pipeline/hgere_runner.py."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch


from gsapere.pipeline.config import HGEREConfig
from gsapere.pipeline.hgere_runner import HGERERunner


def _make_hgere_output_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Simulate HGERE output format for a document."""
    out = dict(doc)
    out["predicted_ner"] = [[[1, 2, "Method"]]]
    out["predicted_ner_proba"] = [[[1, 2, "Method", 0.9, 0.95]]]
    out["predicted_rel"] = [[]]
    out["predicted_rel_proba"] = [[]]
    return out


class TestHGERERunner:
    def _make_config(self, tmp_path: Any) -> HGEREConfig:
        return HGEREConfig(
            model_dir=str(tmp_path / "hgere"),
            base_model_name_or_path=str(tmp_path / "scibert"),
        )

    def test_run_empty_list_returns_empty(self, tmp_path: Any) -> None:
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        result = runner.run([])
        assert result == []

    def test_run_adds_predicted_ner_key(
        self, tmp_path: Any, tiny_doc_with_candidates: dict
    ) -> None:
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        with patch(
            "gsapere.pipeline.hgere_runner.HGERERunner._run_inference"
        ) as mock_inf:
            mock_inf.return_value = [_make_hgere_output_doc(tiny_doc_with_candidates)]
            result = runner.run([tiny_doc_with_candidates])

        assert "predicted_ner" in result[0]

    def test_run_adds_predicted_rel_key(
        self, tmp_path: Any, tiny_doc_with_candidates: dict
    ) -> None:
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        with patch(
            "gsapere.pipeline.hgere_runner.HGERERunner._run_inference"
        ) as mock_inf:
            mock_inf.return_value = [_make_hgere_output_doc(tiny_doc_with_candidates)]
            result = runner.run([tiny_doc_with_candidates])

        assert "predicted_rel" in result[0]

    def test_run_adds_predicted_ner_proba_key(
        self, tmp_path: Any, tiny_doc_with_candidates: dict
    ) -> None:
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        with patch(
            "gsapere.pipeline.hgere_runner.HGERERunner._run_inference"
        ) as mock_inf:
            mock_inf.return_value = [_make_hgere_output_doc(tiny_doc_with_candidates)]
            result = runner.run([tiny_doc_with_candidates])

        assert "predicted_ner_proba" in result[0]

    def test_doc_key_preserved(
        self, tmp_path: Any, tiny_doc_with_candidates: dict
    ) -> None:
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        with patch(
            "gsapere.pipeline.hgere_runner.HGERERunner._run_inference"
        ) as mock_inf:
            mock_inf.return_value = [_make_hgere_output_doc(tiny_doc_with_candidates)]
            result = runner.run([tiny_doc_with_candidates])

        assert result[0]["doc_key"] == tiny_doc_with_candidates["doc_key"]

    def test_batch_returns_same_count(self, tmp_path: Any, tiny_docs: list) -> None:
        # Give each doc a predicted_ner
        docs_with_candidates = [
            dict(d, predicted_ner=[[[0, 0, "Method"]]]) for d in tiny_docs
        ]
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        with patch(
            "gsapere.pipeline.hgere_runner.HGERERunner._run_inference"
        ) as mock_inf:
            mock_inf.return_value = [
                _make_hgere_output_doc(d) for d in docs_with_candidates
            ]
            result = runner.run(docs_with_candidates)

        assert len(result) == len(tiny_docs)

    def test_run_inference_uses_tempfiles_then_cleans_up(
        self, tmp_path: Any, tiny_doc_with_candidates: dict
    ) -> None:
        """No temp files should remain after _run_inference completes."""
        cfg = self._make_config(tmp_path)
        runner = HGERERunner.__new__(HGERERunner)
        runner._config = cfg
        runner._model = MagicMock()
        runner._tokenizer = MagicMock()
        runner._device = "cpu"
        runner._label_set = "gsap"

        created_paths: list[str] = []

        def _mock_infer_hgere(
            model, eval_dataset, args, logger, source_file_path, output_path, gold_only, **kwargs
        ) -> None:
            created_paths.append(str(output_path))
            # Write minimal output
            with open(output_path, "w") as f:
                out = _make_hgere_output_doc(tiny_doc_with_candidates)
                f.write(json.dumps(out) + "\n")

        with (
            patch("gsapere.pipeline.hgere_runner.RelationDataset") as MockDS,
            patch(
                "gsapere.pipeline.hgere_runner.infer_hgere",
                side_effect=_mock_infer_hgere,
            ),
            patch("gsapere.pipeline.hgere_runner.LABELS") as MockLabels,
        ):
            MockDS.return_value = MagicMock()
            MockDS.return_value.build = MagicMock()
            MockLabels.__getitem__ = MagicMock(return_value=MagicMock())
            runner._run_inference([tiny_doc_with_candidates])

        # All temp files should be cleaned up
        for p in created_paths:
            assert not os.path.exists(p), f"Temp file not cleaned up: {p}"
