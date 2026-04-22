"""Pipeline — combines PrunerRunner and HGERERunner for on-demand inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsapere.pipeline.config import PipelineConfig
from gsapere.pipeline.hgere_runner import HGERERunner
from gsapere.pipeline.pruner_runner import PrunerRunner


class Pipeline:
    """Two-stage entity and relation extraction pipeline.

    Load once, call process_document() or process_documents() on demand.
    Both models stay resident in memory between calls — suitable for use as
    a service backend.

    Example::

        pipeline = Pipeline.from_yaml("config.yaml")

        # On each incoming request:
        result = pipeline.process_document(doc)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._pruner = PrunerRunner(config.pruner, config.label_sets[0])
        self._hgere = HGERERunner(config.hgere, config.label_sets)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        """Initialise the pipeline from a YAML config file.

        Args:
            path: Path to the pipeline YAML configuration file.

        Returns:
            Fully initialised Pipeline with both models loaded.
        """
        config = PipelineConfig.from_yaml(path)
        return cls(config)

    def process_document(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Process a single document.

        Args:
            doc: Document dict with at least ``doc_key`` and ``sentences``.

        Returns:
            Enriched document with ``ner_candidates_proba``, ``predicted_ner``,
            ``predicted_ner_proba``, ``predicted_rel``, ``predicted_rel_proba``.
        """
        results = self.process_documents([doc])
        return results[0]

    def process_documents(
        self,
        docs: list[dict[str, Any]],
        show_progress: bool = False,
        debug_break_on_first_rel: bool = False,
        debug_log_rel_probs: bool = False,
    ) -> list[dict[str, Any]]:
        """Process a batch of documents using the first configured label set.

        For single-label-set configs this is the only label set. For
        multi-label-set configs, use :meth:`process_documents_multi` to get
        predictions for all label sets.

        Args:
            docs: List of document dicts.
            show_progress: Show tqdm progress bars.
            debug_break_on_first_rel: Raise after the first predicted relation.

        Returns:
            List of enriched documents in the same order.
        """
        if not docs:
            return []
        docs_with_candidates = self._pruner.run(docs, show_progress=show_progress)
        return self._hgere.run(
            docs_with_candidates,
            show_progress=show_progress,
            debug_break_on_first_rel=debug_break_on_first_rel,
            debug_log_rel_probs=debug_log_rel_probs,
        )

    def process_documents_multi(
        self,
        docs: list[dict[str, Any]],
        show_progress: bool = False,
        debug_break_on_first_rel: bool = False,
        debug_log_rel_probs: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """Process documents through all configured label sets.

        The pruner runs once; HGERE runs once per label set, each time
        activating the corresponding head. Returns a dict so callers can write
        separate output files per label set.

        Args:
            docs: List of document dicts.
            show_progress: Show tqdm progress bars.
            debug_break_on_first_rel: Raise after the first predicted relation.

        Returns:
            ``{label_set: enriched_docs}`` — one list per label set, same order
            as input.
        """
        if not docs:
            return {ls: [] for ls in self._config.label_sets}
        docs_with_candidates = self._pruner.run(docs, show_progress=show_progress)
        return self._hgere.run_multi(
            docs_with_candidates,
            show_progress=show_progress,
            debug_break_on_first_rel=debug_break_on_first_rel,
            debug_log_rel_probs=debug_log_rel_probs,
        )
