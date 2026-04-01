"""
Pruner model definitions.

BertForSpanMarkerNerPruner
    Binary span classifier built on top of BERT (or SciBERT).  It scores every
    candidate n-gram with a single logit P(entity), using span-marker position
    embeddings.  The span representation is either a simple concatenation of four
    hidden states (start marker, end marker, start subtoken, end subtoken) or a
    biaffine projection thereof (BiSpanRepr).  An optional attention-pooled
    representation (AttnSpanRepr) can be appended for richer context.  Supports
    BCE and focal loss via --pruner_loss.

Supporting modules
    BiSpanRepr         – biaffine span encoder (marker × subtoken bilinear)
    AttnSpanRepr       – attention-pooled span representation over subtokens
    CatEncoder         – concatenation + optional linear projection
"""

from transformers import (
    BertPreTrainedModel,
    BertModel,
    ModernBertPreTrainedModel,
    ModernBertModel,
)
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
# import pdb


class BertForSpanMarkerNerPruner(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.args = args
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.biaf_span = args.biaf_span

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.biaf_span:
            # self.span_encoder = BiaffineSpanRepr(
            #           input_size=config.hidden_size, hidden_dim=args.span_hidden_size,
            #           span_dim=args.span_size, rank=args.rank,
            #           factorize=args.biaf_factorize, mode=args.biaf_mode)
            self.span_encoder = BiSpanRepr(
                input_size=config.hidden_size,
                span_dim=args.span_size,
                hidden_size=args.span_hidden_size,
            )
            if self.args.extra_repr == "attn":
                self.extra_encoder = AttnSpanRepr(
                    input_dim=config.hidden_size,
                    proj_dim=args.span_hidden_size,
                    output_dim=args.span_size,
                    dropout=config.hidden_dropout_prob,
                )
                self.ner_classifier = nn.Linear(
                    self.span_encoder.output_dim + self.extra_encoder.output_dim, 1
                )
            elif self.args.extra_repr == "cat":
                self.extra_encoder = CatEncoder(
                    input_dims=[config.hidden_size] * 4,
                    output_dim=args.span_size,
                    proj=True,
                )
                self.ner_classifier = nn.Linear(
                    self.span_encoder.output_dim + self.extra_encoder.output_dim, 1
                )
            else:
                self.ner_classifier = nn.Linear(self.span_encoder.output_dim, 1)
        else:
            self.ner_classifier = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.alpha = torch.tensor(
            [config.alpha], dtype=torch.float32
        )  # config.alpha: loss weight for Nil type
        self.onedropout = config.onedropout

        self.post_init()

        # post_init() does not handle raw nn.Parameter objects inside custom
        # modules like BiSpanRepr (only nn.Linear / Embedding / LayerNorm).
        # Re-run reset_parameters() to ensure weight and bias are properly set.
        if self.biaf_span:
            self.span_encoder.reset_parameters()

    def get_extended_attention_mask(
        self, attention_mask, input_shape, device=None, dtype=None
    ):
        extended = super().get_extended_attention_mask(
            attention_mask, input_shape, device, dtype
        )
        return extended.clamp(min=-1e4)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mention_pos=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]  # bs * tot_seq_len * hidden_dim
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = (
            input_ids.shape
        )  # tot_seq_len = max_seq_length + 2 * max_pair_length
        ent_len = (
            tot_seq_len - seq_len
        ) // 2  # a pair of markers for an entity.     ent_len = max_pair_length

        e1_hidden_states = hidden_states[
            :, seq_len : seq_len + ent_len
        ]  # hidden states for entity_starts,  shape: bs * ent_len(max_pair_length) * hid_dim
        e2_hidden_states = hidden_states[
            :, seq_len + ent_len :
        ]  # hidden states for entity_ends

        m1_start_states = hidden_states[
            torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]
        ]
        m1_end_states = hidden_states[
            torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]
        ]

        if self.biaf_span:
            feature_vector = self.span_encoder(
                e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states
            )
            if self.args.extra_repr == "attn":
                extra_feature = self.extra_encoder(hidden_states, mention_pos)
                feature_vector = torch.cat([feature_vector, extra_feature], dim=-1)
            elif self.args.extra_repr == "cat":
                extra_feature = self.extra_encoder(
                    e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states
                )
                feature_vector = torch.cat([feature_vector, extra_feature], dim=-1)
        else:
            feature_vector = torch.cat(
                [e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states],
                dim=2,
            )

        if not self.onedropout:
            feature_vector = self.dropout(feature_vector)

        ner_prediction_scores = self.ner_classifier(
            feature_vector
        )  # bs * max_pair_length * num_labels

        outputs = (ner_prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here

        if labels is not None:
            ner_mask = labels > -1
            gold_entities = (labels > 0).float()
            masked_scores = (
                ner_prediction_scores.squeeze(-1).masked_select(ner_mask).float()
            )
            masked_gold_entities = gold_entities.masked_select(ner_mask)

            pruner_loss = getattr(self.args, "pruner_loss", "bce")
            if pruner_loss == "focal":
                gamma = getattr(self.args, "focal_gamma", 2.0)
                alpha = getattr(self.args, "focal_alpha", 0.25)
                # Focal loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
                bce = BCEWithLogitsLoss(reduction="none")(
                    masked_scores, masked_gold_entities
                )
                p_t = torch.exp(-bce)
                alpha_t = alpha * masked_gold_entities + (1 - alpha) * (
                    1 - masked_gold_entities
                )
                ner_loss = (alpha_t * (1 - p_t) ** gamma * bce).sum() / ner_mask.sum()
            else:
                ner_loss = (
                    BCEWithLogitsLoss(reduction="none")(
                        masked_scores, masked_gold_entities
                    ).sum()
                    / ner_mask.sum()
                )

            outputs = (ner_loss,) + outputs

        return outputs


class ModernBertForSpanMarkerNerPruner(ModernBertPreTrainedModel):
    """ModernBERT-backed span marker pruner.

    Identical task head to BertForSpanMarkerNerPruner; only the encoder and
    base class differ.  ModernBERT does not use token_type_ids.

    ModernBERT init note
    --------------------
    Any custom head or extra Linear on top of ModernBERT must be treated as
    manual-init territory:

    1. Set ``_supports_assign_param_buffer = False`` — disables the fast-init
       path in ``from_pretrained`` that leaves missing-checkpoint parameters as
       raw uninitialized memory.
    2. Call ``post_init()`` at the end of ``__init__`` — triggers the
       module-wide ``_init_weights`` pass.
    3. Override ``_init_weights`` to initialize every custom layer explicitly,
       especially nested Linears not covered by ModernBERT's built-in module
       checks (ModernBertMLP, ModernBertAttention, …).  ModernBERT's default
       ``_init_weights`` zeroes biases for any ``nn.Linear`` but only gives
       weights a real distribution for its own recognized internal types; a
       plain custom ``nn.Linear`` in a task head gets garbage weights otherwise.
    """

    # Disable Transformers' "superfast init" path, which leaves parameters
    # that are absent from the pretrained checkpoint as raw uninitialized
    # memory.  With this flag False, from_pretrained falls back to the safe
    # init path and _init_weights is applied to every submodule including our
    # custom head layers.
    _supports_assign_param_buffer = False

    def _init_weights(self, module):
        super()._init_weights(module)
        # ModernBERT's _init_weights zeroes nn.Linear biases for all layers
        # but only initialises weights for its own recognised internal types
        # (ModernBertMLP, ModernBertAttention, …).  Custom task-head linears
        # are not among those types so their weights keep whatever garbage was
        # in the freshly-allocated tensor.  Catch every nn.Linear that is not
        # part of self.bert and give it a clean init with the same scale
        # ModernBERT uses for its built-in classification heads.
        if isinstance(module, nn.Linear) and not any(
            module is m for m in self.bert.modules()
        ):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.hidden_size**-0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, config, args=None):
        super().__init__(config)
        self.args = args
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels

        self.biaf_span = args.biaf_span

        self.bert = ModernBertModel(config)
        _dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(_dropout)

        if self.biaf_span:
            self.span_encoder = BiSpanRepr(
                input_size=config.hidden_size,
                span_dim=args.span_size,
                hidden_size=args.span_hidden_size,
            )
            if self.args.extra_repr == "attn":
                self.extra_encoder = AttnSpanRepr(
                    input_dim=config.hidden_size,
                    proj_dim=args.span_hidden_size,
                    output_dim=args.span_size,
                    dropout=_dropout,
                )
                self.ner_classifier = nn.Linear(
                    self.span_encoder.output_dim + self.extra_encoder.output_dim, 1
                )
            elif self.args.extra_repr == "cat":
                self.extra_encoder = CatEncoder(
                    input_dims=[config.hidden_size] * 4,
                    output_dim=args.span_size,
                    proj=True,
                )
                self.ner_classifier = nn.Linear(
                    self.span_encoder.output_dim + self.extra_encoder.output_dim, 1
                )
            else:
                self.ner_classifier = nn.Linear(self.span_encoder.output_dim, 1)
        else:
            self.ner_classifier = nn.Linear(config.hidden_size * 4, self.num_labels)

        self.alpha = torch.tensor([config.alpha], dtype=torch.float32)
        self.onedropout = config.onedropout

        self.post_init()

        if self.biaf_span:
            self.span_encoder.reset_parameters()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,  # accepted but not forwarded (ModernBERT has no segment embeddings)
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mention_pos=None,
        sent_subword_length=None,
    ):
        # ModernBERT requires a 2D (batch, seq_len) padding mask, not the 2D
        # per-sample (seq, seq) attention matrix that BERT supports.  The data
        # loader already collapses the mask for modernbert model types, but as
        # a safety net we handle a stray 3D tensor here.
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, :, 0]

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        if self.onedropout:
            hidden_states = self.dropout(hidden_states)

        seq_len = self.max_seq_length
        bsz, tot_seq_len = input_ids.shape
        ent_len = (tot_seq_len - seq_len) // 2

        if sent_subword_length is not None:
            # Compact layout: markers sit at [L .. L+ent_len) and [L+ent_len .. L+2*ent_len)
            # where L is the per-sample actual subword sentence length.
            # Use gather so each sample uses its own offset.
            hidden_dim = hidden_states.shape[-1]
            arange = torch.arange(
                ent_len, device=hidden_states.device, dtype=torch.long
            )
            e1_offsets = sent_subword_length.unsqueeze(1) + arange.unsqueeze(
                0
            )  # (bsz, ent_len)
            e2_offsets = e1_offsets + ent_len

            def expand(o):
                return o.unsqueeze(-1).expand(-1, -1, hidden_dim)

            e1_hidden_states = torch.gather(hidden_states, 1, expand(e1_offsets))
            e2_hidden_states = torch.gather(hidden_states, 1, expand(e2_offsets))
        else:
            e1_hidden_states = hidden_states[:, seq_len : seq_len + ent_len]
            e2_hidden_states = hidden_states[:, seq_len + ent_len :]

        m1_start_states = hidden_states[
            torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 0]
        ]
        m1_end_states = hidden_states[
            torch.arange(bsz).unsqueeze(-1), mention_pos[:, :, 1]
        ]

        if self.biaf_span:
            feature_vector = self.span_encoder(
                e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states
            )
            if self.args.extra_repr == "attn":
                extra_feature = self.extra_encoder(hidden_states, mention_pos)
                feature_vector = torch.cat([feature_vector, extra_feature], dim=-1)
            elif self.args.extra_repr == "cat":
                extra_feature = self.extra_encoder(
                    e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states
                )
                feature_vector = torch.cat([feature_vector, extra_feature], dim=-1)
        else:
            feature_vector = torch.cat(
                [e1_hidden_states, e2_hidden_states, m1_start_states, m1_end_states],
                dim=2,
            )

        if not self.onedropout:
            feature_vector = self.dropout(feature_vector)

        ner_prediction_scores = self.ner_classifier(feature_vector)

        outputs = (ner_prediction_scores,) + outputs[2:]

        if labels is not None:
            ner_mask = labels > -1
            gold_entities = (labels > 0).float()
            masked_scores = (
                ner_prediction_scores.squeeze(-1).masked_select(ner_mask).float()
            )
            masked_gold_entities = gold_entities.masked_select(ner_mask)

            pruner_loss = getattr(self.args, "pruner_loss", "bce")
            if pruner_loss == "focal":
                gamma = getattr(self.args, "focal_gamma", 2.0)
                alpha = getattr(self.args, "focal_alpha", 0.25)
                bce = BCEWithLogitsLoss(reduction="none")(
                    masked_scores, masked_gold_entities
                )
                p_t = torch.exp(-bce)
                alpha_t = alpha * masked_gold_entities + (1 - alpha) * (
                    1 - masked_gold_entities
                )
                ner_loss = (alpha_t * (1 - p_t) ** gamma * bce).sum() / ner_mask.sum()
            else:
                ner_loss = (
                    BCEWithLogitsLoss(reduction="none")(
                        masked_scores, masked_gold_entities
                    ).sum()
                    / ner_mask.sum()
                )

            outputs = (ner_loss,) + outputs

        return outputs


class CatEncoder(nn.Module):
    def __init__(self, input_dims, output_dim=None, proj=True):
        super().__init__()
        inputdims = [dim for dim in input_dims]
        self.input_dims = inputdims
        self.proj = proj
        self.output_dim = output_dim if self.proj else sum(self.inputdims)
        if proj:
            self.projection = nn.Linear(sum(inputdims), output_dim)

    def forward(self, *reprs):
        repr = torch.cat(reprs, dim=-1)
        if self.proj:
            repr = self.projection(repr)
        return repr


class AttnSpanRepr(nn.Module):
    """Class implementing the attention-based span representation."""

    def __init__(self, input_dim, proj_dim, output_dim, dropout=0.1):
        """If use_endpoints is true then concatenate the end points to attention-pooled span repr.
        Otherwise just return the attention pooled term.
        """
        super(AttnSpanRepr, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim), nn.GELU(), nn.Dropout(p=dropout)
        )
        self.output_dim = output_dim
        self.attention_params = nn.Linear(proj_dim, 1)
        self.output_proj = nn.Linear(proj_dim, output_dim)

        # Initialize weight to zero weight
        # self.attention_params.weight.data.fill_(0)
        # self.attention_params.bias.data.fill_(0)

    def forward(self, encoded_input, mention_pos):
        """
        encoded_input: bs x seq_len x dh
        """
        start_ids = mention_pos[:, :, 0]  # bs x max_n_ent
        end_ids = mention_pos[:, :, 1]  # bs x max_n_ent
        encoded_input = self.proj(encoded_input)
        # b x ns x ne    (ns: subtokens, ne: entities)
        span_mask = _get_span_mask(start_ids, end_ids, encoded_input.shape[1])
        attn_mask = (1 - span_mask) * (-1e4)
        # b x ns x 1 + b x ns x ne --> b x ns x ne
        attn_logits = self.attention_params(encoded_input) + attn_mask
        # b x ns x ne --> b x ne x ns
        attention_wts = nn.functional.softmax(attn_logits, dim=1).permute(0, 2, 1)
        attention_term = torch.einsum("bes, bsd->bed", attention_wts, encoded_input)
        span_repr = self.output_proj(attention_term)

        return span_repr


class BiSpanRepr(nn.Module):
    def __init__(self, input_size, span_dim, hidden_size):
        """
        repr1 = cat(e1, m1),  repr2 = cat(e2, m2), repr = biaf(repr1, repr2)
        """
        super().__init__()
        self.span_dim = span_dim
        self.proj11 = nn.Linear(input_size, hidden_size)
        self.proj12 = nn.Linear(input_size, hidden_size)
        self.proj21 = nn.Linear(input_size, hidden_size)
        self.proj22 = nn.Linear(input_size, hidden_size)
        self.proj1 = nn.Linear(2 * hidden_size, hidden_size)
        self.proj2 = nn.Linear(2 * hidden_size, hidden_size)
        # self.encode_proj = nn.Linear(rank, span_dim)
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size, span_dim))
        self.bias = nn.Parameter(torch.Tensor(span_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for w in [self.weight]:
            torch.nn.init.xavier_normal_(w)
        self.bias.data.fill_(0)
        # Explicitly reset all projection linears so that the fast-init path
        # in from_pretrained (which may leave missing-key params as garbage
        # memory) cannot produce corrupt weights or biases at forward time.
        for proj in [
            self.proj11,
            self.proj12,
            self.proj21,
            self.proj22,
            self.proj1,
            self.proj2,
        ]:
            proj.reset_parameters()
        return

    def forward(self, e1, e2, m1, m2):
        """
        e1, e2, m1, m2: bs * n_ent * input_size (bert output dim)
        e1, e2: entity start/end (subtokens)
        m1, m2: entity start/end (markers with the same position encoding as subtokens)
        """
        e1 = self.proj11(e1)
        e2 = self.proj12(e2)
        m1 = self.proj21(m1)
        m2 = self.proj22(m2)

        repr1 = self.proj1(torch.cat((e1, m1), dim=-1))
        repr2 = self.proj2(torch.cat((e2, m2), dim=-1))
        # span_repr = self.encode_proj(repr1*repr2)
        layer = torch.einsum(
            "bnd,bne,deo->bno", repr1, repr2, self.weight.to(repr1.dtype)
        )
        span_repr = layer + self.bias.to(layer.dtype)
        return span_repr

    @property
    def output_dim(self):
        return self.span_dim


def _get_span_mask(start_ids, end_ids, max_len):
    b, n = start_ids.shape
    # b x ns x n    (max_len: ns), end_ids: not for slicing. subtoken of end_ids is in span
    tmp = (
        torch.arange(max_len, device=start_ids.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(b, max_len, n)
    )
    batch_start_ids = start_ids.unsqueeze(1).expand_as(tmp)
    batch_end_ids = end_ids.unsqueeze(1).expand_as(tmp)
    mask = (tmp >= batch_start_ids).float() * (tmp <= batch_end_ids).float()
    return mask
