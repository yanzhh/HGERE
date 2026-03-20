"""Generic machinery for building argparse parsers from Pydantic models.

Design
------
- **Single source of truth**: Pydantic models define fields, types, defaults,
  and descriptions.  No argparse boilerplate needs to be written by hand.
- **Flat CLI namespace**: Nested model fields are flattened using ``__`` as
  the separator (e.g. ``train_params.learning_rate`` →
  ``--train_params__learning_rate``).  The double underscore is chosen so that
  single-underscore field names (``do_lower_case``) are unambiguous.
- **Disambiguation guarantee**: :func:`check_no_ambiguity` verifies at
  import/test time that no two paths in the model flatten to the same CLI
  flag name, and that no field name component itself contains ``__``.
- **Pydantic-supplied defaults**: Every flag has ``default=None`` in argparse
  so that only *explicitly supplied* CLI values are passed to Pydantic.
  Pydantic supplies all real defaults.
- **Bool handling**: Both ``--flag`` (*store_true*) and ``--no_flag``
  (*store_false*) are registered for every boolean field.  The argparse
  default is ``None`` so absent flags are distinguishable from explicit ones.

Public API
----------
- :func:`check_no_ambiguity`
- :func:`collect_flat_fields`
- :func:`build_argparser`
- :func:`namespace_to_nested_dict`
- :func:`apply_config_and_cli`
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore[assignment]

# Separator used when flattening nested field paths into a single CLI flag name.
SEP: str = "__"

# Fields always excluded from the CLI (internal / structural).
_SKIP_TOP_LEVEL: frozenset[str] = frozenset({"schema_version"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_pydantic_model(t: Any) -> bool:
    try:
        return isinstance(t, type) and issubclass(t, BaseModel)
    except TypeError:
        return False


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner_type, is_optional)``.

    Handles ``Optional[X]`` (= ``Union[X, None]``) from ``typing``.
    Returns the annotation unchanged if it is not optional.
    """
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


def _get_literal_choices(annotation: Any) -> list[Any] | None:
    """Return the choices list if *annotation* is a ``Literal[...]``, else ``None``."""
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    return None


# ---------------------------------------------------------------------------
# FlatField — the intermediate representation
# ---------------------------------------------------------------------------


@dataclass
class FlatField:
    """A single leaf field from a (possibly nested) Pydantic model.

    Attributes
    ----------
    flat_name:
        The CLI flag name after flattening (e.g. ``train_params__learning_rate``).
    path:
        The original field path as a tuple (e.g. ``("train_params", "learning_rate")``).
    annotation:
        The leaf type annotation (may still be ``Optional[X]``).
    default:
        The field's default value, or ``None`` if the field is optional with
        no explicit default.  ``PydanticUndefined`` means the field is required.
    is_required:
        ``True`` if the field has no default and is not optional.
    description:
        Human-readable description from the ``Field(description=...)`` call.
    """

    flat_name: str
    path: tuple[str, ...]
    annotation: Any
    default: Any
    is_required: bool
    description: str | None


# ---------------------------------------------------------------------------
# Core collection
# ---------------------------------------------------------------------------


def collect_flat_fields(
    model_cls: type[BaseModel],
    _prefix: tuple[str, ...] = (),
    _skip_top: frozenset[str] = _SKIP_TOP_LEVEL,
) -> list[FlatField]:
    """Recursively collect all *leaf* fields from *model_cls*.

    Nested Pydantic sub-models are traversed automatically; the nesting
    path is reflected in ``FlatField.flat_name`` via the ``__`` separator.

    Parameters
    ----------
    model_cls:
        The Pydantic model class to inspect.
    _prefix:
        Internal — the accumulated path prefix during recursion.
    _skip_top:
        Field names to skip at the **top level only** (e.g. ``schema_version``).
    """
    result: list[FlatField] = []

    for name, field_info in model_cls.model_fields.items():
        # Skip internal fields at the outermost level only.
        if not _prefix and name in _skip_top:
            continue

        path = _prefix + (name,)
        annotation: Any = field_info.annotation

        # Unwrap Optional[X] → X
        inner, is_optional = _unwrap_optional(annotation)

        # Recurse into nested Pydantic models (e.g. train_params: PrunerTrainParams).
        if _is_pydantic_model(inner):
            result.extend(
                collect_flat_fields(inner, _prefix=path, _skip_top=frozenset())
            )
            continue

        # Leaf field — determine default and required status.
        raw_default = field_info.default
        if raw_default is PydanticUndefined:
            factory = field_info.default_factory
            effective_default = factory() if factory is not None else PydanticUndefined
        else:
            effective_default = raw_default

        is_required = (effective_default is PydanticUndefined) and not is_optional
        if is_optional and effective_default is PydanticUndefined:
            effective_default = None

        result.append(
            FlatField(
                flat_name=SEP.join(path),
                path=path,
                annotation=annotation,
                default=effective_default,
                is_required=is_required,
                description=field_info.description,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Disambiguation check
# ---------------------------------------------------------------------------


def check_no_ambiguity(model_cls: type[BaseModel]) -> None:
    """Raise ``ValueError`` if flattening *model_cls* would produce ambiguous flag names.

    Two sources of ambiguity are detected:

    1. A field name component contains ``__`` (the separator), which would
       make it impossible to round-trip the flat name back to the original
       path unambiguously.
    2. Two different field paths produce the same flat name after joining
       with ``__`` (e.g. a top-level field ``a__b`` clashes with a nested
       field ``a.b``).
    """
    fields = collect_flat_fields(model_cls)

    # Rule 1 — no field name component may contain the separator.
    for field in fields:
        for component in field.path:
            if SEP in component:
                raise ValueError(
                    f"Field name component '{component}' at path "
                    f"'{'.'.join(field.path)}' contains the separator '{SEP}'. "
                    "This makes round-tripping the flattened CLI name ambiguous. "
                    "Rename the field to remove double underscores."
                )

    # Rule 2 — all flat names must be unique.
    seen: dict[str, tuple[str, ...]] = {}
    for field in fields:
        if field.flat_name in seen:
            raise ValueError(
                f"Ambiguous flattened parameter name '{field.flat_name}': "
                f"path {seen[field.flat_name]} and path {field.path} both "
                "produce the same CLI flag name after flattening."
            )
        seen[field.flat_name] = field.path


# ---------------------------------------------------------------------------
# Argparse builder
# ---------------------------------------------------------------------------


def _add_bool_field(
    parser: argparse.ArgumentParser,
    group: argparse._ArgumentGroup,
    field: FlatField,
) -> None:
    """Register ``--{name}`` and ``--no_{name}`` for a boolean field."""
    dest = field.flat_name
    help_text = field.description or ""

    group.add_argument(f"--{dest}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no_{dest}", dest=dest, action="store_false", help=f"Disable: {help_text}"
    )
    # Override the store_true/store_false defaults so absent flags are None.
    parser.set_defaults(**{dest: None})


def _add_scalar_field(
    group: argparse._ArgumentGroup,
    field: FlatField,
) -> None:
    """Register a non-boolean leaf field."""
    annotation = field.annotation
    inner, _ = _unwrap_optional(annotation)

    choices = _get_literal_choices(inner)

    kwargs: dict[str, Any] = {
        "dest": field.flat_name,
        "default": None,  # sentinel — Pydantic provides the real default
        "required": False,  # Pydantic validates required-ness after merging
        "help": (field.description or ""),
        "metavar": field.flat_name.upper(),
    }

    if choices:
        kwargs["type"] = str
        kwargs["choices"] = choices
    elif inner is str or inner is type(None):
        kwargs["type"] = str
    elif inner is int:
        kwargs["type"] = int
    elif inner is float:
        kwargs["type"] = float
    else:
        kwargs["type"] = str  # fallback for unrecognised types

    group.add_argument(f"--{field.flat_name}", **kwargs)


def build_argparser(
    model_cls: type[BaseModel],
    description: str = "",
) -> argparse.ArgumentParser:
    """Build an :class:`argparse.ArgumentParser` from *model_cls*.

    All model fields are exposed as ``--flag`` arguments.  Top-level fields
    and nested fields are separated into labelled argument groups in the
    ``--help`` output.  Every flag defaults to ``None`` so that only
    *explicitly provided* CLI values are forwarded to Pydantic (which then
    applies its own defaults for anything not supplied).

    Parameters
    ----------
    model_cls:
        The Pydantic model to introspect.
    description:
        Parser description shown in ``--help``.
    """
    check_no_ambiguity(model_cls)

    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)

    # Group fields by their top-level section (first path component).
    groups: dict[str, argparse._ArgumentGroup] = {}
    top_group = parser.add_argument_group("parameters")

    for field in collect_flat_fields(model_cls):
        section = field.path[0] if len(field.path) > 1 else None
        if section is None:
            group = top_group
        else:
            if section not in groups:
                groups[section] = parser.add_argument_group(f"{section} parameters")
            group = groups[section]

        inner, _ = _unwrap_optional(field.annotation)
        if inner is bool:
            _add_bool_field(parser, group, field)
        else:
            _add_scalar_field(group, field)

    return parser


# ---------------------------------------------------------------------------
# Namespace → nested dict  (round-trip after parsing)
# ---------------------------------------------------------------------------


def namespace_to_nested_dict(namespace: argparse.Namespace) -> dict[str, Any]:
    """Convert a flat argparse *namespace* back to a nested dict.

    Flat keys containing ``__`` are split on the *first* occurrence to
    reconstruct the nesting.  For example::

        {"train_params__learning_rate": 1e-6}
        →  {"train_params": {"learning_rate": 1e-6}}

    Only non-``None`` values are included (``None`` means the flag was not
    explicitly supplied on the command line).
    """
    nested: dict[str, Any] = {}

    for flat_key, value in vars(namespace).items():
        if value is None:
            continue
        parts = flat_key.split(SEP, maxsplit=1)
        if len(parts) == 1:
            nested[flat_key] = value
        else:
            section, remainder = parts
            nested.setdefault(section, {})[remainder] = value

    return nested


# ---------------------------------------------------------------------------
# Config file + CLI merge → Pydantic model
# ---------------------------------------------------------------------------


def apply_config_and_cli(
    namespace: argparse.Namespace,
    model_cls: type[BaseModel],
) -> BaseModel:
    """Build a validated *model_cls* instance from CLI flags in *namespace*.

    Only flags explicitly provided on the command line (non-``None`` values)
    are forwarded; Pydantic supplies defaults for everything else.

    Parameters
    ----------
    namespace:
        The parsed :class:`argparse.Namespace` from :func:`build_argparser`.
    model_cls:
        The Pydantic model to validate against.
    """
    cli_overrides = namespace_to_nested_dict(namespace)
    return model_cls.model_validate(cli_overrides)
