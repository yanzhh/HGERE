"""Tests for hgere.config.cli_gen.

Covers:
- collect_flat_fields: field collection, skipping, recursion
- check_no_ambiguity: passes for clean models, fails for __ in names and collisions
- build_argparser: correct flags, types, bool handling
- namespace_to_nested_dict: round-trip flattening
- apply_config_and_cli: merge priority (CLI > file > Pydantic defaults)
- PrunerTrainConfig specifically: no ambiguity guarantee
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import pytest
from pydantic import BaseModel, Field

from gsapere.config.cli_gen import (
    SEP,
    apply_config_and_cli,
    build_argparser,
    check_no_ambiguity,
    collect_flat_fields,
    namespace_to_nested_dict,
)
from gsapere.span_classifier.config import PrunerTrainConfig


# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------


class Inner(BaseModel):
    x: int = Field(default=1, description="An integer.")
    flag: bool = Field(default=False, description="A flag.")


class Outer(BaseModel):
    name: str = Field(description="Required string.")
    count: int = Field(default=5, description="A count.")
    nested: Inner = Field(default_factory=Inner)


class WithOptional(BaseModel):
    value: Optional[str] = Field(default=None, description="Optional string.")


class WithLiteral(BaseModel):
    mode: str = Field(default="a")  # not Literal, just str
    choice: str = Field(default="x", description="Choices.")


# ---------------------------------------------------------------------------
# collect_flat_fields
# ---------------------------------------------------------------------------


class TestCollectFlatFields:
    def test_top_level_fields_returned(self) -> None:
        fields = collect_flat_fields(Outer)
        names = [f.flat_name for f in fields]
        assert "name" in names
        assert "count" in names

    def test_nested_fields_flattened(self) -> None:
        fields = collect_flat_fields(Outer)
        names = [f.flat_name for f in fields]
        assert f"nested{SEP}x" in names
        assert f"nested{SEP}flag" in names

    def test_nested_model_itself_not_in_results(self) -> None:
        fields = collect_flat_fields(Outer)
        names = [f.flat_name for f in fields]
        assert "nested" not in names

    def test_required_field_marked_correctly(self) -> None:
        fields = {f.flat_name: f for f in collect_flat_fields(Outer)}
        assert fields["name"].is_required is True
        assert fields["count"].is_required is False

    def test_default_values_preserved(self) -> None:
        fields = {f.flat_name: f for f in collect_flat_fields(Outer)}
        assert fields["count"].default == 5
        assert fields[f"nested{SEP}x"].default == 1

    def test_optional_field_not_required(self) -> None:
        fields = {f.flat_name: f for f in collect_flat_fields(WithOptional)}
        assert fields["value"].is_required is False
        assert fields["value"].default is None

    def test_schema_version_skipped_at_top_level(self) -> None:
        fields = collect_flat_fields(PrunerTrainConfig)
        names = [f.flat_name for f in fields]
        assert "schema_version" not in names

    def test_path_tuple_correct(self) -> None:
        fields = {f.flat_name: f for f in collect_flat_fields(Outer)}
        assert fields[f"nested{SEP}x"].path == ("nested", "x")
        assert fields["name"].path == ("name",)

    def test_description_propagated(self) -> None:
        fields = {f.flat_name: f for f in collect_flat_fields(Outer)}
        assert fields["name"].description == "Required string."


# ---------------------------------------------------------------------------
# check_no_ambiguity
# ---------------------------------------------------------------------------


class TestCheckNoAmbiguity:
    def test_clean_model_passes(self) -> None:
        check_no_ambiguity(Outer)  # must not raise

    def test_pruner_train_config_passes(self) -> None:
        check_no_ambiguity(PrunerTrainConfig)  # guaranteed by design

    def test_field_with_double_underscore_fails(self) -> None:
        class Bad(BaseModel):
            train__lr: float = 1e-5  # contains SEP

        with pytest.raises(ValueError, match="separator"):
            check_no_ambiguity(Bad)

    def test_collision_between_top_and_nested_fails(self) -> None:
        class InnerC(BaseModel):
            b: str = "x"

        class Colliding(BaseModel):
            a: InnerC  # flattens to a__b
            a__b: str = "y"  # also a__b → collision

        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            check_no_ambiguity(Colliding)

    def test_two_nested_sections_same_leaf_name_no_collision(self) -> None:
        """Different prefixes → different flat names → no collision."""

        class SectionA(BaseModel):
            value: int = 1

        class SectionB(BaseModel):
            value: int = 2

        class TwoSections(BaseModel):
            a: SectionA = Field(default_factory=SectionA)
            b: SectionB = Field(default_factory=SectionB)

        # a__value and b__value are distinct → no error
        check_no_ambiguity(TwoSections)


# ---------------------------------------------------------------------------
# build_argparser
# ---------------------------------------------------------------------------


class TestBuildArgparser:
    def test_required_str_field_has_flag(self) -> None:
        parser = build_argparser(Outer)
        action_dests = {a.dest for a in parser._actions}
        assert "name" in action_dests

    def test_nested_field_has_prefixed_flag(self) -> None:
        parser = build_argparser(Outer)
        action_dests = {a.dest for a in parser._actions}
        assert f"nested{SEP}x" in action_dests

    def test_bool_field_has_both_on_and_off_flags(self) -> None:
        parser = build_argparser(Outer)
        option_strings = {s for a in parser._actions for s in a.option_strings}
        assert f"--nested{SEP}flag" in option_strings
        assert f"--no_nested{SEP}flag" in option_strings

    def test_bool_default_is_none(self) -> None:
        parser = build_argparser(Outer)
        ns = parser.parse_args([])
        assert getattr(ns, f"nested{SEP}flag") is None

    def test_bool_flag_sets_true(self) -> None:
        parser = build_argparser(Outer)
        ns = parser.parse_args([f"--nested{SEP}flag"])
        assert getattr(ns, f"nested{SEP}flag") is True

    def test_no_bool_flag_sets_false(self) -> None:
        parser = build_argparser(Outer)
        ns = parser.parse_args([f"--no_nested{SEP}flag"])
        assert getattr(ns, f"nested{SEP}flag") is False

    def test_int_field_parsed_correctly(self) -> None:
        parser = build_argparser(Outer)
        ns = parser.parse_args(["--name", "test", "--count", "42"])
        assert ns.count == 42

    def test_absent_flag_defaults_to_none(self) -> None:
        parser = build_argparser(Outer)
        ns = parser.parse_args(["--name", "x"])
        assert ns.count is None


# ---------------------------------------------------------------------------
# namespace_to_nested_dict
# ---------------------------------------------------------------------------


class TestNamespaceToNestedDict:
    def test_flat_keys_stay_flat(self) -> None:
        ns = argparse.Namespace(name="foo", count=3)
        result = namespace_to_nested_dict(ns)
        assert result == {"name": "foo", "count": 3}

    def test_prefixed_key_becomes_nested(self) -> None:
        ns = argparse.Namespace(train_params__learning_rate=1e-5)
        result = namespace_to_nested_dict(ns)
        assert result == {"train_params": {"learning_rate": 1e-5}}

    def test_none_values_excluded(self) -> None:
        ns = argparse.Namespace(name="foo", count=None)
        result = namespace_to_nested_dict(ns)
        assert "count" not in result

    def test_multiple_nested_fields_same_section(self) -> None:
        ns = argparse.Namespace(
            train_params__learning_rate=1e-5,
            train_params__seed=42,
        )
        result = namespace_to_nested_dict(ns)
        assert result == {"train_params": {"learning_rate": 1e-5, "seed": 42}}


# ---------------------------------------------------------------------------
# apply_config_and_cli
# ---------------------------------------------------------------------------


class TestApplyConfigAndCli:
    def _minimal_ns(self, **overrides: Any) -> argparse.Namespace:
        base = {
            "label_set": "gsap",
            "model_dir": "saves/pruner",
            "base_model_name_or_path": "pretrained_models/scibert",
            "train_params__data_dir": "datasets/gsap",
            "train_params__learning_rate": 1e-6,
            "train_params__num_train_epochs": 8,
            "train_params__per_gpu_train_batch_size": 8,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_cli_values_produce_valid_config(self) -> None:
        ns = self._minimal_ns()
        result = apply_config_and_cli(ns, PrunerTrainConfig)
        assert result.label_set == "gsap"
        assert result.train_params.learning_rate == 1e-6

    def test_cli_override_applied(self) -> None:
        ns = self._minimal_ns(train_params__learning_rate=2e-6)
        result = apply_config_and_cli(ns, PrunerTrainConfig)
        assert result.train_params.learning_rate == 2e-6

    def test_pydantic_defaults_fill_missing_fields(self) -> None:
        ns = self._minimal_ns()
        result = apply_config_and_cli(ns, PrunerTrainConfig)
        assert result.train_params.pruner_loss == "bce"
        assert result.train_params.seed == 42

    def test_invalid_config_raises(self) -> None:
        ns = argparse.Namespace(label_set="gsap")  # missing required fields
        with pytest.raises(Exception):
            apply_config_and_cli(ns, PrunerTrainConfig)
