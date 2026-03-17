"""Label scheme definitions for NER and relation extraction.

Built-in label sets live as YAML files inside the package at
``hgere/label_configs/<name>.yaml``.  Users can add new datasets by dropping
a YAML file there *or* by pointing ``--label_set`` at any path on disk.

YAML format::

    ner:
      - TypeA
      - TypeB
    relations:
      symmetric:
        - SYM_REL
      non_symmetric:
        - ASYM_REL

The ``relations`` key is optional (useful for pruner-only configs).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import List

import yaml


@dataclass
class RelationLabelScheme:
    _sym: List[str]
    _non_sym: List[str]

    @property
    def all(self) -> List[str]:
        return ["NIL"] + self._sym + self._non_sym

    def symmetric(self, only_nil: bool = False) -> List[str]:
        return ["NIL"] if only_nil else ["NIL"] + self._sym


class LabelScheme:
    def __init__(
        self, ner: List[str], rel_sym: List[str], rel_non_sym: List[str]
    ) -> None:
        self.ner = ["NIL"] + ner
        self.rel = RelationLabelScheme(rel_sym, rel_non_sym)

    @property
    def num_ner_labels(self) -> int:
        return len(self.ner)

    def num_rel_labels(self, no_sym: bool) -> int:
        n_all_rels = len(self.rel.all)
        n_symmetric_rels = len(self.rel.symmetric(only_nil=no_sym))
        return (2 * n_all_rels) - n_symmetric_rels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_from_dict(data: dict) -> LabelScheme:
    ner: List[str] = data["ner"]
    relations: dict = data.get("relations", {})
    rel_sym: List[str] = relations.get("symmetric") or []
    rel_non_sym: List[str] = relations.get("non_symmetric") or []
    return LabelScheme(ner=ner, rel_sym=rel_sym, rel_non_sym=rel_non_sym)


def _list_builtin_names() -> list[str]:
    """Return sorted list of built-in label set names (without .yaml suffix)."""
    config_dir = files("hgere") / "label_configs"
    return sorted(
        p.name.removesuffix(".yaml")
        for p in config_dir.iterdir()
        if p.name.endswith(".yaml")
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_label_scheme(name_or_path: str) -> LabelScheme:
    """Load a :class:`LabelScheme` by built-in name or path to a YAML file.

    Parameters
    ----------
    name_or_path:
        Either a registered name (e.g. ``"scierc"``, ``"gsap"``) that maps to
        a bundled YAML file, or an absolute/relative path to a custom YAML file
        on disk.

    Raises
    ------
    ValueError
        If ``name_or_path`` is neither a known built-in name nor an existing
        file path.
    """
    path = Path(name_or_path)
    if path.is_file():
        with path.open() as fh:
            data = yaml.safe_load(fh)
        return _load_from_dict(data)

    config_file = files("hgere") / "label_configs" / f"{name_or_path}.yaml"
    if not config_file.is_file():
        available = _list_builtin_names()
        raise ValueError(
            f"Unknown label set '{name_or_path}'. "
            f"Built-in sets: {available}. "
            "Provide a registered name or a path to a YAML file."
        )
    with config_file.open() as fh:
        data = yaml.safe_load(fh)
    return _load_from_dict(data)


# ---------------------------------------------------------------------------
# Backward-compatible registry
# ---------------------------------------------------------------------------

#: Pre-built mapping of built-in label set names → LabelScheme objects.
#: Equivalent to calling ``load_label_scheme(name)`` for each built-in name.
LABELS: dict[str, LabelScheme] = {
    name: load_label_scheme(name) for name in _list_builtin_names()
}
