"""Tests for hgere.labels — YAML-based label scheme loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hgere.labels import LabelScheme, _list_builtin_names, load_label_scheme


class TestListBuiltinNames:
    def test_returns_list_of_strings(self) -> None:
        names = _list_builtin_names()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_known_datasets_present(self) -> None:
        names = _list_builtin_names()
        for expected in ("gsap", "scierc", "scier", "scinlp", "somd", "ace04", "ace05"):
            assert expected in names, f"'{expected}' missing from built-in names"


class TestLoadLabelSchemeByName:
    def test_returns_label_scheme(self) -> None:
        scheme = load_label_scheme("scierc")
        assert isinstance(scheme, LabelScheme)

    def test_ner_starts_with_nil(self) -> None:
        for name in _list_builtin_names():
            scheme = load_label_scheme(name)
            assert scheme.ner[0] == "NIL", (
                f"{name}: ner[0] is '{scheme.ner[0]}', expected 'NIL'"
            )

    def test_scierc_ner_types(self) -> None:
        scheme = load_label_scheme("scierc")
        assert "Method" in scheme.ner
        assert "Task" in scheme.ner
        assert "Material" in scheme.ner

    def test_scierc_relations(self) -> None:
        scheme = load_label_scheme("scierc")
        assert "COMPARE" in scheme.rel.symmetric()
        assert "PART-OF" in scheme.rel.all
        assert "USED-FOR" in scheme.rel.all

    def test_gsap_ner_types(self) -> None:
        scheme = load_label_scheme("gsap")
        assert "Dataset" in scheme.ner
        assert "Method" in scheme.ner
        assert "MLModel" in scheme.ner

    def test_scier_no_symmetric_relations(self) -> None:
        scheme = load_label_scheme("scier")
        # Only NIL in symmetric when there are no symmetric relations
        assert scheme.rel.symmetric() == ["NIL"]

    def test_unknown_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown label set"):
            load_label_scheme("does_not_exist_dataset")

    def test_all_builtins_loadable(self) -> None:
        for name in _list_builtin_names():
            scheme = load_label_scheme(name)
            assert len(scheme.ner) >= 2  # at least NIL + one type


class TestLoadLabelSchemeFromPath:
    def test_load_custom_yaml(self, tmp_path: Path) -> None:
        data = {
            "ner": ["TypeA", "TypeB"],
            "relations": {
                "symmetric": ["SYM"],
                "non_symmetric": ["NON-SYM"],
            },
        }
        yaml_file = tmp_path / "custom.yaml"
        yaml_file.write_text(yaml.dump(data))

        scheme = load_label_scheme(str(yaml_file))
        assert scheme.ner == ["NIL", "TypeA", "TypeB"]
        assert "SYM" in scheme.rel.symmetric()
        assert "NON-SYM" in scheme.rel.all

    def test_load_ner_only_yaml(self, tmp_path: Path) -> None:
        data = {"ner": ["TypeA"]}
        yaml_file = tmp_path / "ner_only.yaml"
        yaml_file.write_text(yaml.dump(data))

        scheme = load_label_scheme(str(yaml_file))
        assert scheme.ner == ["NIL", "TypeA"]
        assert scheme.rel.all == ["NIL"]

    def test_nonexistent_path_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown label set"):
            load_label_scheme("/nonexistent/path/to/labels.yaml")


class TestLabelScheme:
    def test_num_ner_labels(self) -> None:
        scheme = load_label_scheme("scier")
        # scier: Method, Dataset, Task → NIL + 3 = 4
        assert scheme.num_ner_labels == 4

    def test_num_rel_labels_no_sym(self) -> None:
        scheme = load_label_scheme("scierc")
        # With no_sym=True: only NIL as symmetric → (2 * all) - 1
        n = scheme.num_rel_labels(no_sym=True)
        assert isinstance(n, int)
        assert n > 0

    def test_num_rel_labels_with_sym(self) -> None:
        scheme = load_label_scheme("scierc")
        n_with = scheme.num_rel_labels(no_sym=False)
        n_without = scheme.num_rel_labels(no_sym=True)
        assert n_with < n_without  # allowing sym reduces the count


class TestBackwardCompatLABELS:
    def test_labels_registry_accessible(self) -> None:
        from hgere.labels import LABELS

        assert "scierc" in LABELS
        assert "gsap" in LABELS
        assert isinstance(LABELS["scierc"], LabelScheme)
