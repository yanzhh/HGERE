# coding=utf-8
"""
Legacy Albert-based ACE models.

These were used in ACE-dataset experiments and are no longer part of the
active GSAP-ERE pipeline.  They are preserved here so the transformers/
sub-tree can be removed while keeping the code accessible.

Models:
  AlbertForACEBothOneDropoutSub  — dual-path NER + RE with subject marker
                                   NOTE: contains a pdb.set_trace() debug breakpoint
  AlBertForBaselines             — switchable baseline: firstorder / mfvi / gnn
                                   NOTE: known bug: `bsz` is undefined in the
                                   non-GNN else-branch of forward()
  AlbertForHyperGNN              — Albert counterpart of BertForHyperGNN

WARNING: AlbertModel / AlbertPreTrainedModel in the pip version of transformers
may differ from the locally-patched version these classes were originally tested
against.  They are kept here as reference code only.
"""

import pdb

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertModel, AlbertPreTrainedModel

from utils.model_ere import (
    CatEncoder,
    HyperGNNBinaryGraph,
    HyperGNNHybridGraph,
    HyperGNNTernaryGraph,
    get_ent_mask1d,
    get_ent_mask2d,
)
from utils.modules_ace import GNN, MFVI


class AlbertForACEBothOneDropoutSub(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
        self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
        self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor(
            [config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32
        )
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sub_positions=None,
        labels=None,
        ner_labels=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = self.dropout(outputs[0])
        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        pdb.set_trace()  # NOTE: debug breakpoint present in original code
        e1_hidden_states = hidden_states[:, seq_len : seq_len + ent_len]
        e2_hidden_states = hidden_states[:, seq_len + ent_len :]
        feature_vector = torch.cat([e1_hidden_states, e2_hidden_states], dim=2)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        m1_start_states = hidden_states[torch.arange(bsz), sub_positions[:, 0]]
        m1_end_states = hidden_states[torch.arange(bsz), sub_positions[:, 1]]
        m1_states = torch.cat([m1_start_states, m1_end_states], dim=-1)
        m1_scores = self.re_classifier_m1(m1_states)
        m2_scores = self.re_classifier_m2(feature_vector)
        re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[2:]

        if labels is not None:
            loss_fct_re = CrossEntropyLoss(
                ignore_index=-1, weight=self.alpha.to(re_prediction_scores)
            )
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(
                re_prediction_scores.view(-1, self.num_labels), labels.view(-1)
            )
            ner_loss = loss_fct_ner(
                ner_prediction_scores.view(-1, self.num_ner_labels), ner_labels.view(-1)
            )
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs


class AlBertForBaselines(AlbertPreTrainedModel):
    """Switchable baselines: firstorder / mfvi / gnn.

    NOTE: known bug in original — `bsz` is undefined in the non-GNN else-branch.
    """

    def __init__(self, config, args=None):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args

        if self.args.baseline == "firstorder":
            self.sub_encoder = CatEncoder(
                input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim, proj=True
            )
            self.obj_encoder = CatEncoder(
                input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim, proj=True
            )
            sub_dim = self.sub_encoder.output_dim
            obj_dim = self.obj_encoder.output_dim
            self.rel_encoder = CatEncoder(
                input_dims=[sub_dim, obj_dim], output_dim=args.rel_dim, proj=True
            )
            rel_dim = self.rel_encoder.output_dim
            ent_dim = args.ent_dim
            self.ner_cls = CatEncoder(
                input_dims=[sub_dim, obj_dim], output_dim=self.num_ner_labels
            )
            self.rel_cls = nn.Linear(rel_dim, self.num_labels)

        elif args.baseline == "mfvi":
            self.sub_encoder = CatEncoder(
                input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim, proj=True
            )
            self.obj_encoder = CatEncoder(
                input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim, proj=True
            )
            sub_dim = self.sub_encoder.output_dim
            obj_dim = self.obj_encoder.output_dim
            self.rel_encoder = CatEncoder(
                input_dims=[sub_dim, obj_dim], output_dim=args.rel_dim, proj=True
            )
            rel_dim = self.rel_encoder.output_dim
            ent_dim = args.ent_dim
            self.mfvigraph = MFVI(
                ent_dim=sub_dim,
                rel_dim=rel_dim,
                mem_dim=args.mem_dim,
                n_ent_labels=self.num_ner_labels,
                n_rel_labels=self.num_labels,
                args=args,
            )
            self.sub_scorer = nn.Linear(sub_dim, self.num_ner_labels)
            self.obj_scorer = nn.Linear(obj_dim, self.num_ner_labels)
            self.rel_cls = nn.Linear(rel_dim, self.num_labels)

        elif args.baseline == "gnn":
            self.sub_encoder = CatEncoder(
                input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim
            )
            self.obj_encoder = CatEncoder(
                input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim
            )
            sub_dim = args.ent_dim
            obj_dim = args.ent_dim
            self.rel_encoder = CatEncoder(
                input_dims=[sub_dim, obj_dim], output_dim=args.rel_dim
            )
            rel_dim = args.rel_dim
            ent_dim = args.ent_dim
            self.gnn = GNN(
                ent_dim=ent_dim,
                rel_dim=rel_dim,
                dropout=config.hidden_dropout_prob,
                args=args,
            )
            self.ner_cls = CatEncoder(
                input_dims=[ent_dim] * 2, output_dim=self.num_ner_labels
            )
            self.rel_cls = nn.Linear(rel_dim, self.num_labels)

        else:
            self.ner_classifier = nn.Linear(config.hidden_size * 2, self.num_ner_labels)
            self.re_classifier_m1 = nn.Linear(config.hidden_size * 2, self.num_labels)
            self.re_classifier_m2 = nn.Linear(config.hidden_size * 2, self.num_labels)

        self.alpha = torch.tensor(
            [config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32
        )

        if self.args.layernorm_1st and self.args.baseline in {
            "firstorder",
            "mfvi",
            "gnn",
        }:
            self.sub_layernorm = (
                nn.LayerNorm(ent_dim, eps=1e-6) if args.layernorm else nn.Identity()
            )
            self.obj_layernorm = (
                nn.LayerNorm(ent_dim, eps=1e-6) if args.layernorm else nn.Identity()
            )
            self.rel_layernorm = (
                nn.LayerNorm(rel_dim, eps=1e-6) if args.layernorm else nn.Identity()
            )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sub_positions=None,
        rel_labels=None,
        ner_labels=None,
        ent_numbers=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = self.dropout(outputs[0])
        seq_len = self.max_seq_length
        max_ent_num = max(ent_numbers)
        tot_seq_len = input_ids.shape[-1]
        ent_len = (tot_seq_len - seq_len) // 2

        obj_start_states = hidden_states[:, seq_len : seq_len + ent_len][
            :, :max_ent_num, :
        ]
        obj_end_states = hidden_states[:, seq_len + ent_len :][:, :max_ent_num, :]

        sub_start_states = hidden_states[
            torch.arange(sum(ent_numbers)), sub_positions[:, 0]
        ]
        sub_end_states = hidden_states[
            torch.arange(sum(ent_numbers)), sub_positions[:, 1]
        ]

        sub_reprs = self.sub_encoder(sub_start_states, sub_end_states)
        obj_reprs = self.obj_encoder(obj_start_states, obj_end_states)
        rel_reprs = self.rel_encoder(
            sub_reprs.unsqueeze(-2).expand(obj_reprs.shape), obj_reprs
        )
        rel_reprs_split = torch.split(rel_reprs, ent_numbers.tolist())
        rel_reprs = pad_sequence(rel_reprs_split, batch_first=True, padding_value=0)
        obj_reprs_split = torch.split(obj_reprs, ent_numbers.tolist())
        obj_reprs = pad_sequence(obj_reprs_split, batch_first=True, padding_value=-1e4)
        uni_obj_reprs = torch.max(obj_reprs, dim=1)[0]
        sub_reprs_split = torch.split(sub_reprs, ent_numbers.tolist())
        sub_reprs = pad_sequence(sub_reprs_split, batch_first=True, padding_value=0)

        mask1d = get_ent_mask1d(ent_numbers)
        mask2d = get_ent_mask2d(ent_numbers)
        uni_obj_reprs *= mask1d.unsqueeze(-1)
        rel_reprs *= mask2d.unsqueeze(-1)

        if self.args.layernorm_1st and self.args.baseline in {
            "firstorder",
            "mfvi",
            "gnn",
        }:
            sub_reprs = self.sub_layernorm(sub_reprs)
            uni_obj_reprs = self.obj_layernorm(uni_obj_reprs)
            rel_reprs = self.rel_layernorm(rel_reprs)

        if self.args.baseline == "firstorder":
            ner_prediction_scores = self.ner_cls(sub_reprs, uni_obj_reprs)
            re_prediction_scores = self.rel_cls(rel_reprs)

        elif self.args.baseline == "mfvi":
            subscores = self.sub_scorer(sub_reprs)
            objscores = self.obj_scorer(uni_obj_reprs)
            relscores = self.rel_cls(rel_reprs)
            subscores, objscores, re_prediction_scores = self.mfvigraph(
                sub_reprs,
                uni_obj_reprs,
                rel_reprs,
                subscores,
                objscores,
                relscores,
                ent_numbers,
            )
            ner_prediction_scores = subscores + objscores

        elif self.args.baseline == "gnn":
            sub_reprs, uni_obj_reprs, rel_reprs = self.gnn(
                sub_reprs, uni_obj_reprs, rel_reprs, ent_numbers
            )
            ner_prediction_scores = self.ner_cls(sub_reprs, uni_obj_reprs)
            re_prediction_scores = self.rel_cls(rel_reprs)

        else:
            # NOTE: `bsz` is undefined here — known bug in original code
            ner_reprs = torch.cat([obj_start_states, obj_end_states], dim=-1)
            ner_prediction_scores = self.ner_classifier(ner_reprs)
            m1_states = torch.cat([sub_start_states, sub_end_states], dim=-1)
            m1_scores = self.re_classifier_m1(m1_states)
            m2_scores = self.re_classifier_m2(ner_reprs)
            re_prediction_scores = m1_scores.unsqueeze(1) + m2_scores

        outputs = (re_prediction_scores, ner_prediction_scores) + outputs[2:]

        if rel_labels is not None:
            loss_fct_re = CrossEntropyLoss(
                ignore_index=-1, weight=self.alpha.to(re_prediction_scores)
            )
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_loss = loss_fct_re(
                re_prediction_scores.view(-1, self.num_labels), rel_labels.view(-1)
            )
            if self.args.baseline in {"firstorder", "mfvi", "gnn"}:
                ner_loss = loss_fct_ner(
                    ner_prediction_scores.view(-1, self.num_ner_labels),
                    ner_labels.view(-1),
                )
            else:
                # NOTE: `bsz` is undefined — known bug in original code
                ner_labels_exp = ner_labels.unsqueeze(0).repeat(bsz, 1)  # noqa: F821
                ner_loss = loss_fct_ner(
                    ner_prediction_scores.view(-1, self.num_ner_labels),
                    ner_labels_exp.view(-1),
                )
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs


class AlbertForHyperGNN(AlbertPreTrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args

        self.sub_encoder = CatEncoder(
            input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim
        )
        self.obj_encoder = CatEncoder(
            input_dims=[config.hidden_size] * 2, output_dim=args.ent_dim
        )
        sub_dim = args.ent_dim
        obj_dim = args.ent_dim
        self.rel_encoder = CatEncoder(
            input_dims=[sub_dim, obj_dim], output_dim=args.rel_dim
        )
        rel_dim = args.rel_dim
        ent_dim = args.ent_dim

        if args.factor_type == "ternary":
            self.htnnlayer = HyperGNNTernaryGraph(
                ent_dim=ent_dim,
                rel_dim=rel_dim,
                dropout=config.hidden_dropout_prob,
                args=args,
            )
        elif self.args.factor_type in {
            "sib",
            "cop",
            "gp",
            "sibcop",
            "sibgp",
            "copgp",
            "sibcopgp",
        }:
            self.htnnlayer = HyperGNNBinaryGraph(
                rel_dim=rel_dim, dropout=config.hidden_dropout_prob, args=args
            )
        elif self.args.factor_type in {
            "tersib",
            "tercop",
            "tergp",
            "tersibcop",
            "tersibgp",
            "tercopgp",
            "tersibcopgp",
        }:
            self.htnnlayer = HyperGNNHybridGraph(
                ent_dim=ent_dim,
                rel_dim=rel_dim,
                dropout=config.hidden_dropout_prob,
                args=args,
            )
        else:
            pdb.set_trace()

        self.ner_cls = CatEncoder(
            input_dims=[ent_dim] * 2, output_dim=self.num_ner_labels
        )
        self.rel_cls = nn.Linear(rel_dim, self.num_labels)

        self.alpha = torch.tensor(
            [config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32
        )

        if self.args.layernorm_1st:
            self.sub_layernorm = (
                nn.LayerNorm(ent_dim, eps=1e-6) if args.layernorm else nn.Identity()
            )
            self.obj_layernorm = (
                nn.LayerNorm(ent_dim, eps=1e-6) if args.layernorm else nn.Identity()
            )
            self.rel_layernorm = (
                nn.LayerNorm(rel_dim, eps=1e-6) if args.layernorm else nn.Identity()
            )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sub_positions=None,
        rel_labels=None,
        ner_labels=None,
        ent_numbers=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = self.dropout(outputs[0])
        seq_len = self.max_seq_length
        max_ent_num = max(ent_numbers)
        tot_seq_len = input_ids.shape[-1]
        ent_len = (tot_seq_len - seq_len) // 2

        obj_start_states = hidden_states[:, seq_len : seq_len + ent_len][
            :, :max_ent_num, :
        ]
        obj_end_states = hidden_states[:, seq_len + ent_len :][:, :max_ent_num, :]

        sub_start_states = hidden_states[
            torch.arange(sum(ent_numbers)), sub_positions[:, 0]
        ]
        sub_end_states = hidden_states[
            torch.arange(sum(ent_numbers)), sub_positions[:, 1]
        ]

        sub_reprs = self.sub_encoder(sub_start_states, sub_end_states)
        obj_reprs = self.obj_encoder(obj_start_states, obj_end_states)
        rel_reprs = self.rel_encoder(
            sub_reprs.unsqueeze(-2).expand(obj_reprs.shape), obj_reprs
        )
        rel_reprs_split = torch.split(rel_reprs, ent_numbers.tolist())
        rel_reprs = pad_sequence(rel_reprs_split, batch_first=True, padding_value=0)
        obj_reprs_split = torch.split(obj_reprs, ent_numbers.tolist())
        obj_reprs = pad_sequence(obj_reprs_split, batch_first=True, padding_value=-1e4)
        uni_obj_reprs = torch.max(obj_reprs, dim=1)[0]
        sub_reprs_split = torch.split(sub_reprs, ent_numbers.tolist())
        sub_reprs = pad_sequence(sub_reprs_split, batch_first=True, padding_value=0)

        mask1d = get_ent_mask1d(ent_numbers)
        mask2d = get_ent_mask2d(ent_numbers)
        uni_obj_reprs *= mask1d.unsqueeze(-1)
        rel_reprs *= mask2d.unsqueeze(-1)

        if self.args.layernorm_1st:
            sub_reprs = self.sub_layernorm(sub_reprs)
            uni_obj_reprs = self.obj_layernorm(uni_obj_reprs)
            rel_reprs = self.rel_layernorm(rel_reprs)

        if self.args.factor_type in {
            "ternary",
            "tersib",
            "tercop",
            "tergp",
            "tersibcop",
            "tersibgp",
            "tercopgp",
            "tersibcopgp",
        }:
            sub_reprs, uni_obj_reprs, rel_reprs = self.htnnlayer(
                sub_reprs, uni_obj_reprs, rel_reprs, ent_numbers
            )
        elif self.args.factor_type in {"sib", "cop", "gp", "sibcop", "sibgp", "copgp"}:
            rel_reprs = self.htnnlayer(rel_reprs, ent_numbers)

        ner_prediction_scores = self.ner_cls(sub_reprs, uni_obj_reprs)
        re_prediction_scores = self.rel_cls(rel_reprs)

        outputs = (re_prediction_scores, ner_prediction_scores)

        if rel_labels is not None:
            loss_fct_re = CrossEntropyLoss(ignore_index=-1)
            loss_fct_ner = CrossEntropyLoss(ignore_index=-1)
            re_prediction_scores = re_prediction_scores.float()
            ner_prediction_scores = ner_prediction_scores.float()
            re_loss = loss_fct_re(
                re_prediction_scores.reshape(-1, self.num_labels),
                rel_labels.reshape(-1),
            )
            ner_loss = loss_fct_ner(
                ner_prediction_scores.reshape(-1, self.num_ner_labels),
                ner_labels.reshape(-1),
            )
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs
