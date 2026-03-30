"""Shared config utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml


class _NoDuplicatesLoader(yaml.SafeLoader):
    pass


def _no_duplicates_constructor(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict:
    keys: set[str] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node)
        if key in keys:
            raise ValueError(f"Duplicate key in YAML: '{key}'")
        keys.add(key)
    return loader.construct_mapping(node, deep=True)


_NoDuplicatesLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _no_duplicates_constructor,
)


def load_yaml_strict(path: Union[str, Path]) -> dict:
    """Load a YAML file, raising ValueError on duplicate keys."""
    with open(path) as f:
        return yaml.load(f, Loader=_NoDuplicatesLoader)
