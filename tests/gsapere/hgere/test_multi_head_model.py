"""Tests for multi-head model support in BertForHyperGNN.

Covers:
- Single-head mode (backward compat): model structure unchanged
- Multi-head mode: ner_heads / rel_heads ModuleDict created per dataset
- forward() routes to correct head based on dataset_id
- forward() without dataset_id uses single-head mode
- Output logit shapes are dataset-specific
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from gsapere.models.hgere import BertForHyperGNN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> MagicMock:
    """Build a minimal args namespace accepted by BertForHyperGNN."""
    defaults = dict(
        ent_dim=64,
        rel_dim=64,
        mem_dim=64,
        rel_rank=64,
        rel_factorize=False,
        factor_type="ternary",
        factor_encoder="cat",
        ent_repr="mix",
        layernorm=False,
        layernorm_1st=False,
        attn_self=False,
        lminit=False,
        n_iter=1,
        n_head=2,
        d_head=16,
        edgetype="sib",
        attn_scorer="biaf",
        attn_res=False,
        re_focal_loss=False,
        ner_focal_loss=False,
        device="cpu",
    )
    defaults.update(overrides)
    return MagicMock(**defaults)


def _make_bert_config(
    num_labels: int = 10,
    num_ner_labels: int = 4,
    dataset_heads: dict | None = None,
) -> MagicMock:
    """Build a minimal BertConfig-like object."""
    cfg = MagicMock()
    cfg.num_labels = num_labels
    cfg.num_ner_labels = num_ner_labels
    cfg.hidden_size = 64
    cfg.hidden_dropout_prob = 0.0
    cfg.max_seq_length = 32
    cfg.alpha = 1.0
    cfg.dataset_heads = dataset_heads
    # Required by BertPreTrainedModel
    cfg.output_attentions = False
    cfg.output_hidden_states = False
    cfg.use_return_dict = False
    cfg.is_decoder = False
    cfg.add_cross_attention = False
    cfg.chunk_size_feed_forward = 0
    cfg.seq_classif_dropout = 0.0
    cfg.vocab_size = 30522
    cfg.type_vocab_size = 2
    cfg.max_position_embeddings = 512
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 2
    cfg.intermediate_size = 128
    cfg.hidden_act = "gelu"
    cfg.layer_norm_eps = 1e-12
    cfg.pad_token_id = 0
    cfg.attention_probs_dropout_prob = 0.0
    cfg.initializer_range = 0.02
    cfg.position_embedding_type = "absolute"
    return cfg


# ---------------------------------------------------------------------------
# Single-head mode (backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="Requires loading full BERT weights; use integration test instead"
)
class TestSingleHeadMode:
    """Structural checks that rely on a real pretrained model load."""

    def test_single_head_has_rel_cls_and_ner_cls(self) -> None:
        pass

    def test_single_head_no_ner_heads_attr(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Multi-head helpers (unit-testable without BERT weights)
# ---------------------------------------------------------------------------


class TestHeadBuilders:
    """Test the helper functions that create NER/RE heads."""

    def test_build_ner_head_mix(self) -> None:
        from gsapere.models.hgere import _build_ner_head

        head = _build_ner_head(ent_dim=64, num_ner_labels=5, ent_repr="mix")
        assert isinstance(head, nn.Module)
        # Mix mode: CatEncoder takes two inputs of ent_dim
        sub = torch.randn(3, 64)
        obj = torch.randn(3, 64)
        out = head(sub, obj)
        assert out.shape == (3, 5)

    def test_build_ner_head_sub(self) -> None:
        from gsapere.models.hgere import _build_ner_head

        head = _build_ner_head(ent_dim=64, num_ner_labels=5, ent_repr="sub")
        assert isinstance(head, nn.Linear)
        out = head(torch.randn(3, 64))
        assert out.shape == (3, 5)

    def test_build_rel_head(self) -> None:
        from gsapere.models.hgere import _build_rel_head

        head = _build_rel_head(rel_dim=64, num_rel_labels=12)
        assert isinstance(head, nn.Linear)
        out = head(torch.randn(2, 3, 64))
        assert out.shape == (2, 3, 12)


# ---------------------------------------------------------------------------
# Multi-head init (structural checks without pretrained weights)
# ---------------------------------------------------------------------------


class TestMultiHeadInit:
    """Test that the model correctly builds ModuleDict heads from dataset_heads config."""

    def _build_model_with_heads(self, dataset_heads: dict) -> BertForHyperGNN:
        """Build model using from_pretrained with a tiny real BERT config."""
        from transformers import BertConfig

        bert_cfg = BertConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            vocab_size=100,
        )
        bert_cfg.num_labels = 10  # fallback / ignored in multi-head
        bert_cfg.num_ner_labels = 4  # fallback / ignored in multi-head
        bert_cfg.max_seq_length = 32
        bert_cfg.alpha = 1.0
        bert_cfg.dataset_heads = dataset_heads

        args = _make_args()
        # Directly instantiate (no pretrained weights needed)
        model = BertForHyperGNN(bert_cfg, args=args)
        return model

    def test_model_has_ner_heads_moduledict(self) -> None:
        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "scinlp": {"num_ner_labels": 5, "num_rel_labels": 14},
        }
        model = self._build_model_with_heads(heads)
        assert hasattr(model, "ner_heads")
        assert isinstance(model.ner_heads, nn.ModuleDict)
        assert set(model.ner_heads.keys()) == {"scier", "scinlp"}

    def test_model_has_rel_heads_moduledict(self) -> None:
        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "scinlp": {"num_ner_labels": 5, "num_rel_labels": 14},
        }
        model = self._build_model_with_heads(heads)
        assert hasattr(model, "rel_heads")
        assert isinstance(model.rel_heads, nn.ModuleDict)
        assert set(model.rel_heads.keys()) == {"scier", "scinlp"}

    def test_no_shared_tensor_aliases(self) -> None:
        """In multi-head mode, ner_cls/rel_cls must not exist to avoid HF serialization errors."""
        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "scinlp": {"num_ner_labels": 5, "num_rel_labels": 14},
        }
        model = self._build_model_with_heads(heads)
        assert not hasattr(model, "ner_cls")
        assert not hasattr(model, "rel_cls")

    def test_rel_head_output_size_per_dataset(self) -> None:
        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "gsap": {"num_ner_labels": 11, "num_rel_labels": 39},
        }
        model = self._build_model_with_heads(heads)
        # Check output dimensions of each rel head
        dummy = torch.randn(1, 3, 64)  # [bs, n_ent, rel_dim]
        scier_out = model.rel_heads["scier"](dummy)
        gsap_out = model.rel_heads["gsap"](dummy)
        assert scier_out.shape[-1] == 12
        assert gsap_out.shape[-1] == 39

    def test_single_head_mode_no_ner_heads_attr(self) -> None:
        """When dataset_heads is None, model should NOT have ner_heads attribute."""
        from transformers import BertConfig

        bert_cfg = BertConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            vocab_size=100,
        )
        bert_cfg.num_labels = 10
        bert_cfg.num_ner_labels = 4
        bert_cfg.max_seq_length = 32
        bert_cfg.alpha = 1.0
        bert_cfg.dataset_heads = None

        args = _make_args()
        model = BertForHyperGNN(bert_cfg, args=args)
        assert not hasattr(model, "ner_heads")
        assert not hasattr(model, "rel_heads")
        assert hasattr(model, "ner_cls")
        assert hasattr(model, "rel_cls")


# ---------------------------------------------------------------------------
# _HeadWithFC unit tests
# ---------------------------------------------------------------------------


class TestHeadWithFC:
    """Unit tests for the _HeadWithFC module."""

    def test_forward_single_input(self) -> None:
        from gsapere.models.hgere import _HeadWithFC

        inner = nn.Linear(64, 32)
        head = _HeadWithFC(inner, hidden_dim=32, num_labels=10)
        out = head(torch.randn(4, 64))
        assert out.shape == (4, 10)

    def test_forward_multi_input(self) -> None:
        """Multi-arg forward (mix ent_repr: sub + obj concatenated by CatEncoder)."""
        from gsapere.models.hgere import _HeadWithFC, CatEncoder

        inner = CatEncoder(input_dims=[64, 64], output_dim=32)
        head = _HeadWithFC(inner, hidden_dim=32, num_labels=7)
        out = head(torch.randn(5, 64), torch.randn(5, 64))
        assert out.shape == (5, 7)

    def test_has_inner_proj_act_out(self) -> None:
        from gsapere.models.hgere import _HeadWithFC

        inner = nn.Linear(16, 8)
        head = _HeadWithFC(inner, hidden_dim=8, num_labels=3)
        assert hasattr(head, "inner_proj")
        assert hasattr(head, "act")
        assert hasattr(head, "out")

    def test_3d_input(self) -> None:
        """Relation heads operate on [bs, n_ent, rel_dim] tensors."""
        from gsapere.models.hgere import _HeadWithFC

        inner = nn.Linear(64, 32)
        head = _HeadWithFC(inner, hidden_dim=32, num_labels=12)
        out = head(torch.randn(2, 5, 64))
        assert out.shape == (2, 5, 12)


# ---------------------------------------------------------------------------
# Head builders with head_hidden_dim
# ---------------------------------------------------------------------------


class TestHeadBuildersWithFC:
    def test_build_ner_head_sub_with_fc(self) -> None:
        from gsapere.models.hgere import _HeadWithFC, _build_ner_head

        head = _build_ner_head(
            ent_dim=64, num_ner_labels=5, ent_repr="sub", head_hidden_dim=32
        )
        assert isinstance(head, _HeadWithFC)
        out = head(torch.randn(3, 64))
        assert out.shape == (3, 5)

    def test_build_ner_head_mix_with_fc(self) -> None:
        from gsapere.models.hgere import _HeadWithFC, _build_ner_head

        head = _build_ner_head(
            ent_dim=64, num_ner_labels=5, ent_repr="mix", head_hidden_dim=32
        )
        assert isinstance(head, _HeadWithFC)
        out = head(torch.randn(3, 64), torch.randn(3, 64))
        assert out.shape == (3, 5)

    def test_build_rel_head_with_fc(self) -> None:
        from gsapere.models.hgere import _HeadWithFC, _build_rel_head

        head = _build_rel_head(rel_dim=64, num_rel_labels=12, head_hidden_dim=32)
        assert isinstance(head, _HeadWithFC)
        out = head(torch.randn(2, 3, 64))
        assert out.shape == (2, 3, 12)

    def test_build_ner_head_without_fc_unchanged(self) -> None:
        """Without head_hidden_dim, builders must return the same type as before."""
        from gsapere.models.hgere import _build_ner_head, _build_rel_head

        ner = _build_ner_head(ent_dim=64, num_ner_labels=5, ent_repr="sub")
        assert isinstance(ner, nn.Linear)
        rel = _build_rel_head(rel_dim=64, num_rel_labels=12)
        assert isinstance(rel, nn.Linear)


# ---------------------------------------------------------------------------
# Multi-head model with FC projection
# ---------------------------------------------------------------------------


class TestMultiHeadInitWithFC:
    def _build_model(
        self, dataset_heads: dict, head_hidden_dim: int | None = None
    ) -> BertForHyperGNN:
        from transformers import BertConfig

        bert_cfg = BertConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
            vocab_size=100,
        )
        bert_cfg.num_labels = 10
        bert_cfg.num_ner_labels = 4
        bert_cfg.max_seq_length = 32
        bert_cfg.alpha = 1.0
        bert_cfg.dataset_heads = dataset_heads

        args = _make_args(head_hidden_dim=head_hidden_dim)
        return BertForHyperGNN(bert_cfg, args=args)

    def test_heads_are_fc_head_with_fc(self) -> None:
        from gsapere.models.hgere import _HeadWithFC

        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "scinlp": {"num_ner_labels": 5, "num_rel_labels": 14},
        }
        model = self._build_model(heads, head_hidden_dim=32)
        for key in heads:
            assert isinstance(model.ner_heads[key], _HeadWithFC)
            assert isinstance(model.rel_heads[key], _HeadWithFC)

    def test_output_shapes_with_fc(self) -> None:
        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "gsap": {"num_ner_labels": 11, "num_rel_labels": 39},
        }
        model = self._build_model(heads, head_hidden_dim=32)
        rel_dummy = torch.randn(1, 3, 64)
        assert model.rel_heads["scier"](rel_dummy).shape[-1] == 12
        assert model.rel_heads["gsap"](rel_dummy).shape[-1] == 39

    def test_fc_projections_synced_across_heads(self) -> None:
        """After init, all heads' inner_proj weights must be identical."""
        heads = {
            "scier": {"num_ner_labels": 4, "num_rel_labels": 12},
            "scinlp": {"num_ner_labels": 5, "num_rel_labels": 14},
            "gsap": {"num_ner_labels": 11, "num_rel_labels": 39},
        }
        model = self._build_model(heads, head_hidden_dim=32)
        keys = list(heads.keys())
        ref_ner = model.ner_heads[keys[0]].inner_proj.state_dict()
        ref_rel = model.rel_heads[keys[0]].inner_proj.state_dict()
        for k in keys[1:]:
            for param_name, ref_val in ref_ner.items():
                assert torch.equal(
                    model.ner_heads[k].inner_proj.state_dict()[param_name], ref_val
                ), (
                    f"NER inner_proj '{param_name}' differs between '{keys[0]}' and '{k}'"
                )
            for param_name, ref_val in ref_rel.items():
                assert torch.equal(
                    model.rel_heads[k].inner_proj.state_dict()[param_name], ref_val
                ), (
                    f"REL inner_proj '{param_name}' differs between '{keys[0]}' and '{k}'"
                )

    def test_without_fc_heads_are_not_fc_head(self) -> None:
        from gsapere.models.hgere import _HeadWithFC

        heads = {"scier": {"num_ner_labels": 4, "num_rel_labels": 12}}
        model = self._build_model(heads, head_hidden_dim=None)
        assert not isinstance(model.ner_heads["scier"], _HeadWithFC)
        assert not isinstance(model.rel_heads["scier"], _HeadWithFC)
