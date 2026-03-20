"""Tests for pipeline/pipeline.py — Pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from gsapere.pipeline.pipeline import Pipeline


class TestPipelineFromYaml:
    def test_from_yaml_returns_pipeline(self, minimal_config_yaml: Path) -> None:
        with (
            patch("gsapere.pipeline.pruner_runner.PrunerRunner._load_model"),
            patch("gsapere.pipeline.hgere_runner.HGERERunner._load_model"),
        ):
            pipeline = Pipeline.from_yaml(minimal_config_yaml)
        assert isinstance(pipeline, Pipeline)

    def test_from_yaml_string_path(self, minimal_config_yaml: Path) -> None:
        with (
            patch("gsapere.pipeline.pruner_runner.PrunerRunner._load_model"),
            patch("gsapere.pipeline.hgere_runner.HGERERunner._load_model"),
        ):
            pipeline = Pipeline.from_yaml(str(minimal_config_yaml))
        assert isinstance(pipeline, Pipeline)


class TestPipelineProcessDocument:
    def _make_pipeline(self, minimal_config_yaml: Path) -> Pipeline:
        with (
            patch("gsapere.pipeline.pruner_runner.PrunerRunner._load_model"),
            patch("gsapere.pipeline.hgere_runner.HGERERunner._load_model"),
        ):
            return Pipeline.from_yaml(minimal_config_yaml)

    def test_process_document_calls_pruner_then_hgere(
        self, minimal_config_yaml: Path, tiny_doc: dict
    ) -> None:
        pipeline = self._make_pipeline(minimal_config_yaml)
        call_order: list[str] = []

        def mock_pruner_run(docs: list) -> list:
            call_order.append("pruner")
            return [
                dict(d, predicted_ner=[[]], ner_candidates_proba=[[]]) for d in docs
            ]

        def mock_hgere_run(docs: list) -> list:
            call_order.append("hgere")
            return [dict(d, predicted_ner=[[]], predicted_rel=[[]]) for d in docs]

        pipeline._pruner.run = mock_pruner_run  # type: ignore[assignment]
        pipeline._hgere.run = mock_hgere_run  # type: ignore[assignment]

        pipeline.process_document(tiny_doc)
        assert call_order == ["pruner", "hgere"]

    def test_process_document_returns_dict(
        self, minimal_config_yaml: Path, tiny_doc: dict
    ) -> None:
        pipeline = self._make_pipeline(minimal_config_yaml)
        enriched_doc = dict(tiny_doc, predicted_ner=[[]], predicted_rel=[[]])

        pipeline._pruner.run = lambda docs: [
            dict(d, predicted_ner=[[]], ner_candidates_proba=[[]]) for d in docs
        ]  # type: ignore[assignment]
        pipeline._hgere.run = lambda docs: [enriched_doc]  # type: ignore[assignment]

        result = pipeline.process_document(tiny_doc)
        assert isinstance(result, dict)

    def test_process_document_output_has_expected_keys(
        self, minimal_config_yaml: Path, tiny_doc: dict
    ) -> None:
        pipeline = self._make_pipeline(minimal_config_yaml)

        def mock_pruner_run(docs: list) -> list:
            return [
                dict(d, predicted_ner=[[]], ner_candidates_proba=[[]]) for d in docs
            ]

        def mock_hgere_run(docs: list) -> list:
            return [
                dict(
                    d,
                    predicted_ner=[[]],
                    predicted_ner_proba=[[]],
                    predicted_rel=[[]],
                    predicted_rel_proba=[[]],
                )
                for d in docs
            ]

        pipeline._pruner.run = mock_pruner_run  # type: ignore[assignment]
        pipeline._hgere.run = mock_hgere_run  # type: ignore[assignment]

        result = pipeline.process_document(tiny_doc)
        assert "ner_candidates_proba" in result
        assert "predicted_ner" in result
        assert "predicted_rel" in result


class TestPipelineProcessDocuments:
    def _make_pipeline(self, minimal_config_yaml: Path) -> Pipeline:
        with (
            patch("gsapere.pipeline.pruner_runner.PrunerRunner._load_model"),
            patch("gsapere.pipeline.hgere_runner.HGERERunner._load_model"),
        ):
            return Pipeline.from_yaml(minimal_config_yaml)

    def test_process_documents_returns_same_count(
        self, minimal_config_yaml: Path, tiny_docs: list
    ) -> None:
        pipeline = self._make_pipeline(minimal_config_yaml)

        pipeline._pruner.run = lambda docs: [  # type: ignore[assignment]
            dict(d, predicted_ner=[[]], ner_candidates_proba=[[]]) for d in docs
        ]
        pipeline._hgere.run = lambda docs: [  # type: ignore[assignment]
            dict(d, predicted_ner=[[]], predicted_rel=[[]]) for d in docs
        ]

        results = pipeline.process_documents(tiny_docs)
        assert len(results) == len(tiny_docs)

    def test_process_documents_empty_list(self, minimal_config_yaml: Path) -> None:
        pipeline = self._make_pipeline(minimal_config_yaml)
        pipeline._pruner.run = lambda docs: docs  # type: ignore[assignment]
        pipeline._hgere.run = lambda docs: docs  # type: ignore[assignment]

        results = pipeline.process_documents([])
        assert results == []

    def test_process_document_delegates_to_process_documents(
        self, minimal_config_yaml: Path, tiny_doc: dict
    ) -> None:
        pipeline = self._make_pipeline(minimal_config_yaml)
        enriched = dict(tiny_doc, predicted_ner=[[]])

        call_args: list[list] = []

        def mock_process_docs(docs: list) -> list:
            call_args.append(docs)
            return [enriched]

        pipeline.process_documents = mock_process_docs  # type: ignore[method-assign]
        result = pipeline.process_document(tiny_doc)

        assert call_args[0] == [tiny_doc]
        assert result == enriched
