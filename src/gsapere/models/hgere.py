"""
HGERE model definitions (Entity and Relation Extraction).

BertForHyperGNN
    Main end-to-end model.  Encodes sentences with BERT/SciBERT, then runs a
    stack of HyperGNN message-passing layers over a hypergraph whose nodes are
    entity candidates and whose hyperedges are candidate relation triples.
    Outputs NER logits (multi-class) and relation logits (multi-class) jointly.

HyperGNN layers
    HyperGNNBinaryGraph        – binary (entity–entity) hypergraph
    HyperGNNTernaryGraph       – ternary (subject–object–relation) hypergraph
    HyperGNNHybridGraph        – hybrid binary + ternary graph
    HyperGNNBinaryComposeLayer / HyperGNNBinaryAggregateLayer
        – compose and aggregate messages for binary edges
    HyperGNNTernaryComposeLayer / HyperGNNTernaryAggregateLayer
        – compose and aggregate messages for ternary edges
    HyperGNNHybridAggregateLayer
        – aggregation for the hybrid graph

Supporting modules
    BiafEncoder     – biaffine relation encoder (subject × object → relation)
    CPDTrilinear    – CP-decomposed trilinear tensor for ternary interactions
    CatEncoder      – concatenation + optional linear projection
    LinearMessegePasser – single linear message-passing step
"""

from transformers import (
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    ModernBertConfig,
    ModernBertModel,
    ModernBertPreTrainedModel,
)
import torch
from torch import nn
from torch.nn import (
    CrossEntropyLoss,
    Dropout,
    Linear,
    Sequential,
    LayerNorm,
    Identity,
    Module,
    Parameter,
    GELU,
)
from torch.nn.utils.rnn import pad_sequence


# ---------------------------------------------------------------------------
# Per-dataset head builders (shared by both model classes)
# ---------------------------------------------------------------------------


def _build_ner_head(ent_dim: int, num_ner_labels: int, ent_repr: str) -> Module:
    """Build a single NER classification head."""
    if ent_repr == "mix":
        return CatEncoder(input_dims=[ent_dim] * 2, output_dim=num_ner_labels)
    return Linear(ent_dim, num_ner_labels)


def _build_rel_head(rel_dim: int, num_rel_labels: int) -> Module:
    """Build a single relation classification head."""
    return Linear(rel_dim, num_rel_labels)


def _apply_ner_head(
    ner_cls: Module, sub_reprs: torch.Tensor, uni_obj_reprs: torch.Tensor, ent_repr: str
) -> torch.Tensor:
    """Call the NER head with the right inputs depending on ent_repr mode."""
    if ent_repr == "mix":
        return ner_cls(sub_reprs, uni_obj_reprs)
    if ent_repr == "sub":
        return ner_cls(sub_reprs)
    if ent_repr == "obj":
        return ner_cls(uni_obj_reprs)
    raise ValueError(f"Unknown ent_repr: {ent_repr!r}")


def _compute_re_loss(
    re_prediction_scores: torch.Tensor,
    rel_labels: torch.Tensor,
    num_labels: int,
    args: object,
) -> torch.Tensor:
    """Compute RE loss (focal or cross-entropy)."""
    re_logits = re_prediction_scores.reshape(-1, num_labels)
    re_targets = rel_labels.reshape(-1)
    if getattr(args, "re_focal_loss", False):
        gamma = getattr(args, "re_focal_gamma", 2.0)
        mask = re_targets != -1
        logits_m = re_logits[mask]
        targets_m = re_targets[mask]
        log_p = torch.nn.functional.log_softmax(logits_m, dim=-1)
        p_t = log_p.exp().gather(1, targets_m.unsqueeze(1)).squeeze(1)
        return (
            -((1 - p_t) ** gamma) * log_p.gather(1, targets_m.unsqueeze(1)).squeeze(1)
        ).mean()
    return CrossEntropyLoss(ignore_index=-1)(re_logits, re_targets)


def _compute_ner_loss(
    ner_prediction_scores: torch.Tensor,
    ner_labels: torch.Tensor,
    num_ner_labels: int,
    args: object,
) -> torch.Tensor:
    """Compute NER loss (focal or cross-entropy)."""
    ner_logits = ner_prediction_scores.reshape(-1, num_ner_labels)
    ner_targets = ner_labels.reshape(-1)
    if getattr(args, "ner_focal_loss", False):
        gamma = getattr(args, "ner_focal_gamma", 2.0)
        mask = ner_targets != -1
        logits_m = ner_logits[mask]
        targets_m = ner_targets[mask]
        log_p = torch.nn.functional.log_softmax(logits_m, dim=-1)
        p_t = log_p.exp().gather(1, targets_m.unsqueeze(1)).squeeze(1)
        return (
            -((1 - p_t) ** gamma) * log_p.gather(1, targets_m.unsqueeze(1)).squeeze(1)
        ).mean()
    return CrossEntropyLoss(ignore_index=-1)(ner_logits, ner_targets)


class BertForHyperGNN(BertPreTrainedModel):
    def __init__(self, config, args=None):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = BertModel(config)
        self.dropout = Dropout(config.hidden_dropout_prob)

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
            print(f"No valid factor_type specifiec: {self.args.factor_type}")
            print(
                "Valid factor_types: {'sib', 'cop', 'gp', 'sibcop', 'sibgp', 'copgp','sibcopgp'}"
            )
            print(
                "                    {'tersib', 'tercop', 'tergp', 'tersibcop','tersibgp', 'tercopgp', 'tersibcopgp'}"
            )
            raise Exception()
        # Classification heads: single-head or per-dataset multi-head
        dataset_heads_cfg = getattr(config, "dataset_heads", None)
        if dataset_heads_cfg is not None:
            # Multi-head mode: one NER head and one RE head per dataset
            self._head_info: dict = dataset_heads_cfg
            self.ner_heads = nn.ModuleDict(
                {
                    name: _build_ner_head(
                        ent_dim, info["num_ner_labels"], args.ent_repr
                    )
                    for name, info in dataset_heads_cfg.items()
                }
            )
            self.rel_heads = nn.ModuleDict(
                {
                    name: _build_rel_head(rel_dim, info["num_rel_labels"])
                    for name, info in dataset_heads_cfg.items()
                }
            )
        else:
            # Single-head mode: unchanged behaviour
            self._head_info = None
            self.rel_cls = Linear(rel_dim, self.num_labels)
            if args.ent_repr == "mix":
                self.ner_cls = CatEncoder(
                    input_dims=[ent_dim] * 2, output_dim=self.num_ner_labels
                )
            else:
                self.ner_cls = Linear(ent_dim, self.num_ner_labels)
            self.alpha = torch.tensor(
                [config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32
            )

        if self.args.layernorm_1st:
            self.sub_layernorm = (
                LayerNorm(ent_dim, eps=1e-6) if args.layernorm else Identity()
            )
            self.obj_layernorm = (
                LayerNorm(ent_dim, eps=1e-6) if args.layernorm else Identity()
            )
            self.rel_layernorm = (
                LayerNorm(rel_dim, eps=1e-6) if args.layernorm else Identity()
            )

        self.post_init()

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
        # obj_positions=None,
        rel_labels=None,  # bsz * max_ent_num * max_ent_num
        ner_labels=None,  # bsz * max_ent_num
        ent_numbers=None,
        dataset_id=None,  # str | None — selects per-dataset head in multi-head mode
        # sub_ner_labels=None,
    ):
        # token_type_ids is never provided by the data loader.
        # In transformers 4.x a pre-registered buffer of size [1, 512] is used as fallback,
        # which fails to expand when seq_length > 512 (entity-marker sequences can reach ~530).
        # Explicitly pass zeros of the correct shape to avoid the buffer expansion error.
        if token_type_ids is None:
            ids = input_ids if input_ids is not None else inputs_embeds
            token_type_ids = torch.zeros(
                ids.shape[:2], dtype=torch.long, device=ids.device
            )

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]  # n_ent * seq_len * dh
        hidden_states = self.dropout(
            hidden_states
        )  # bs=4, 20 * seq_len * 4096  (20=bs*max_ent_num)
        seq_len = self.max_seq_length
        ent_numbers_list = ent_numbers.tolist()
        max_ent_num = max(ent_numbers_list)
        # bsz = len(ent_numbers)
        tot_seq_len = input_ids.shape[-1]

        # bsz, tot_seq_len = input_ids.shape                                                  # bs = n_ent,   max_ent_num: max ent number in this batch
        ent_len = (tot_seq_len - seq_len) // 2
        # objects
        obj_start_states = hidden_states[:, seq_len : seq_len + ent_len][
            :, :max_ent_num, :
        ]  # n_ent x max_ent_num x dh
        obj_end_states = hidden_states[:, seq_len + ent_len :][:, :max_ent_num, :]

        n_total_ents = sum(ent_numbers_list)
        sub_start_states = hidden_states[
            torch.arange(n_total_ents), sub_positions[:, 0]
        ]  # n_ent x dh
        sub_end_states = hidden_states[torch.arange(n_total_ents), sub_positions[:, 1]]

        sub_reprs = self.sub_encoder(sub_start_states, sub_end_states)  # n_ent x de
        obj_reprs = self.obj_encoder(
            obj_start_states, obj_end_states
        )  # n_ent x max_ent_num x de

        rel_reprs = self.rel_encoder(
            sub_reprs.unsqueeze(-2).expand(obj_reprs.shape), obj_reprs
        )  # n_ent x max_ent_num x dr
        rel_reprs_split = torch.split(rel_reprs, ent_numbers_list)
        rel_reprs = pad_sequence(rel_reprs_split, batch_first=True, padding_value=0)  #

        obj_reprs_split = torch.split(
            obj_reprs, ent_numbers_list
        )  # split on dim0, (n_ent)
        obj_reprs = pad_sequence(
            obj_reprs_split, batch_first=True, padding_value=-1e4
        )  # n_ent x max_ent_num x max_ent_num x dh
        uni_obj_reprs = torch.max(obj_reprs, dim=1)[0]  # n_ent x max_ent_num x dh

        sub_reprs_split = torch.split(sub_reprs, ent_numbers_list)
        sub_reprs = pad_sequence(
            sub_reprs_split, batch_first=True, padding_value=0
        )  # n_ent x max_ent_num x de

        mask1d = get_ent_mask1d(ent_numbers, max_num=max_ent_num)
        mask2d = get_ent_mask2d(ent_numbers, max_num=max_ent_num)
        uni_obj_reprs *= mask1d.unsqueeze(-1)
        rel_reprs *= mask2d.unsqueeze(-1)

        if self.args.layernorm_1st:
            sub_reprs = self.sub_layernorm(sub_reprs)
            uni_obj_reprs = self.obj_layernorm(uni_obj_reprs)
            rel_reprs = self.rel_layernorm(rel_reprs)

        # relmask = torch.ones(rel_reprs.shape[:-1]).to(rel_reprs)
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

        # Head selection: multi-head routes by dataset_id, single-head uses cls attrs
        if self._head_info is not None and dataset_id is not None:
            ner_cls = self.ner_heads[dataset_id]
            rel_cls = self.rel_heads[dataset_id]
            num_labels = self._head_info[dataset_id]["num_rel_labels"]
            num_ner_labels = self._head_info[dataset_id]["num_ner_labels"]
        else:
            ner_cls = self.ner_cls
            rel_cls = self.rel_cls
            num_labels = self.num_labels
            num_ner_labels = self.num_ner_labels

        re_prediction_scores = rel_cls(rel_reprs)
        ner_prediction_scores = _apply_ner_head(
            ner_cls, sub_reprs, uni_obj_reprs, self.args.ent_repr
        )

        outputs = (
            re_prediction_scores,
            ner_prediction_scores,
        )  # Add hidden states and attention if they are here

        ner_prediction_scores = ner_prediction_scores.float()
        re_prediction_scores = re_prediction_scores.float()

        if rel_labels is not None:
            re_loss = _compute_re_loss(
                re_prediction_scores, rel_labels, num_labels, self.args
            )
            ner_loss = _compute_ner_loss(
                ner_prediction_scores, ner_labels, num_ner_labels, self.args
            )
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs

    def forward_all_heads(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        sub_positions=None,
        ent_numbers=None,
        **kwargs,
    ) -> dict[str, tuple]:
        """Run encoder+HyperGNN once and apply every head.

        Returns ``{dataset_id: (re_logits, ner_logits)}`` for all heads.
        Only valid for multi-head checkpoints (``dataset_heads`` in config).
        """
        if self._head_info is None:
            raise RuntimeError("forward_all_heads requires a multi-head model.")

        ids = input_ids if input_ids is not None else kwargs.get("inputs_embeds")
        token_type_ids = torch.zeros(ids.shape[:2], dtype=torch.long, device=ids.device)

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        hidden_states = self.dropout(outputs[0])

        seq_len = self.max_seq_length
        ent_numbers_list = ent_numbers.tolist()
        max_ent_num = max(ent_numbers_list)
        tot_seq_len = input_ids.shape[-1]
        ent_len = (tot_seq_len - seq_len) // 2

        obj_start_states = hidden_states[:, seq_len : seq_len + ent_len][
            :, :max_ent_num, :
        ]
        obj_end_states = hidden_states[:, seq_len + ent_len :][:, :max_ent_num, :]
        n_total_ents = sum(ent_numbers_list)
        sub_start_states = hidden_states[
            torch.arange(n_total_ents), sub_positions[:, 0]
        ]
        sub_end_states = hidden_states[torch.arange(n_total_ents), sub_positions[:, 1]]

        sub_reprs = self.sub_encoder(sub_start_states, sub_end_states)
        obj_reprs = self.obj_encoder(obj_start_states, obj_end_states)
        rel_reprs = self.rel_encoder(
            sub_reprs.unsqueeze(-2).expand(obj_reprs.shape), obj_reprs
        )

        ent_numbers_list = ent_numbers.tolist()
        rel_reprs = pad_sequence(
            torch.split(rel_reprs, ent_numbers_list),
            batch_first=True,
            padding_value=0,
        )
        obj_reprs = pad_sequence(
            torch.split(obj_reprs, ent_numbers_list),
            batch_first=True,
            padding_value=-1e4,
        )
        uni_obj_reprs = torch.max(obj_reprs, dim=1)[0]
        sub_reprs = pad_sequence(
            torch.split(sub_reprs, ent_numbers_list),
            batch_first=True,
            padding_value=0,
        )

        mask1d = get_ent_mask1d(ent_numbers, max_num=max_ent_num)
        mask2d = get_ent_mask2d(ent_numbers, max_num=max_ent_num)
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

        return {
            ds_id: (
                self.rel_heads[ds_id](rel_reprs).float(),
                _apply_ner_head(
                    self.ner_heads[ds_id], sub_reprs, uni_obj_reprs, self.args.ent_repr
                ).float(),
            )
            for ds_id in self._head_info
        }


class ModernBertForHyperGNN(ModernBertPreTrainedModel):
    """ModernBERT-backed HGERE model.

    Identical task head (HyperGNN layers, NER/RE classifiers) to
    BertForHyperGNN; only the encoder and base class differ.
    ModernBERT does not use token_type_ids.

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

    # Disable Transformers' "superfast init" path so that custom head
    # parameters absent from the pretrained checkpoint are properly
    # initialized instead of left as raw uninitialized memory.
    _supports_assign_param_buffer = False

    def _init_weights(self, module):
        super()._init_weights(module)
        # ModernBERT's _init_weights only initialises weights for its own
        # recognised internal types; custom task-head linears get their bias
        # zeroed but their weight is never touched, leaving garbage memory.
        # Catch every nn.Linear outside self.bert and give it the same scale
        # ModernBERT uses for built-in classification heads.
        if isinstance(module, nn.Linear) and not any(
            module is m for m in self.bert.modules()
        ):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.hidden_size**-0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def __init__(self, config, args=None):
        super().__init__(config)
        self.max_seq_length = config.max_seq_length
        self.num_labels = config.num_labels
        self.num_ner_labels = config.num_ner_labels

        self.bert = ModernBertModel(config)
        _dropout = getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = Dropout(_dropout)

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
                dropout=_dropout,
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
                rel_dim=rel_dim, dropout=_dropout, args=args
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
                dropout=_dropout,
                args=args,
            )
        else:
            print(f"No valid factor_type specified: {self.args.factor_type}")
            raise Exception()

        # Classification heads: single-head or per-dataset multi-head
        dataset_heads_cfg = getattr(config, "dataset_heads", None)
        if dataset_heads_cfg is not None:
            self._head_info: dict = dataset_heads_cfg
            self.ner_heads = nn.ModuleDict(
                {
                    name: _build_ner_head(
                        ent_dim, info["num_ner_labels"], args.ent_repr
                    )
                    for name, info in dataset_heads_cfg.items()
                }
            )
            self.rel_heads = nn.ModuleDict(
                {
                    name: _build_rel_head(rel_dim, info["num_rel_labels"])
                    for name, info in dataset_heads_cfg.items()
                }
            )
        else:
            self._head_info = None
            self.rel_cls = Linear(rel_dim, self.num_labels)
            if args.ent_repr == "mix":
                self.ner_cls = CatEncoder(
                    input_dims=[ent_dim] * 2, output_dim=self.num_ner_labels
                )
            else:
                self.ner_cls = Linear(ent_dim, self.num_ner_labels)
            self.alpha = torch.tensor(
                [config.alpha] + [1.0] * (self.num_labels - 1), dtype=torch.float32
            )

        if self.args.layernorm_1st:
            self.sub_layernorm = (
                LayerNorm(ent_dim, eps=1e-6) if args.layernorm else Identity()
            )
            self.obj_layernorm = (
                LayerNorm(ent_dim, eps=1e-6) if args.layernorm else Identity()
            )
            self.rel_layernorm = (
                LayerNorm(rel_dim, eps=1e-6) if args.layernorm else Identity()
            )

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mentions=None,
        token_type_ids=None,  # accepted but not forwarded (ModernBERT has no segment embeddings)
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sub_positions=None,
        rel_labels=None,
        ner_labels=None,
        ent_numbers=None,
        dataset_id=None,  # str | None — selects per-dataset head in multi-head mode
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        seq_len = self.max_seq_length
        ent_numbers_list = ent_numbers.tolist()
        max_ent_num = max(ent_numbers_list)
        tot_seq_len = input_ids.shape[-1]

        ent_len = (tot_seq_len - seq_len) // 2
        obj_start_states = hidden_states[:, seq_len : seq_len + ent_len][
            :, :max_ent_num, :
        ]
        obj_end_states = hidden_states[:, seq_len + ent_len :][:, :max_ent_num, :]

        n_total_ents = sum(ent_numbers_list)
        sub_start_states = hidden_states[
            torch.arange(n_total_ents), sub_positions[:, 0]
        ]
        sub_end_states = hidden_states[torch.arange(n_total_ents), sub_positions[:, 1]]

        sub_reprs = self.sub_encoder(sub_start_states, sub_end_states)
        obj_reprs = self.obj_encoder(obj_start_states, obj_end_states)

        rel_reprs = self.rel_encoder(
            sub_reprs.unsqueeze(-2).expand(obj_reprs.shape), obj_reprs
        )
        rel_reprs_split = torch.split(rel_reprs, ent_numbers_list)
        rel_reprs = pad_sequence(rel_reprs_split, batch_first=True, padding_value=0)

        obj_reprs_split = torch.split(obj_reprs, ent_numbers_list)
        obj_reprs = pad_sequence(obj_reprs_split, batch_first=True, padding_value=-1e4)
        uni_obj_reprs = torch.max(obj_reprs, dim=1)[0]

        sub_reprs_split = torch.split(sub_reprs, ent_numbers_list)
        sub_reprs = pad_sequence(sub_reprs_split, batch_first=True, padding_value=0)

        mask1d = get_ent_mask1d(ent_numbers, max_num=max_ent_num)
        mask2d = get_ent_mask2d(ent_numbers, max_num=max_ent_num)
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

        # Head selection: multi-head routes by dataset_id, single-head uses cls attrs
        if self._head_info is not None and dataset_id is not None:
            ner_cls = self.ner_heads[dataset_id]
            rel_cls = self.rel_heads[dataset_id]
            num_labels = self._head_info[dataset_id]["num_rel_labels"]
            num_ner_labels = self._head_info[dataset_id]["num_ner_labels"]
        else:
            ner_cls = self.ner_cls
            rel_cls = self.rel_cls
            num_labels = self.num_labels
            num_ner_labels = self.num_ner_labels

        re_prediction_scores = rel_cls(rel_reprs)
        ner_prediction_scores = _apply_ner_head(
            ner_cls, sub_reprs, uni_obj_reprs, self.args.ent_repr
        )

        outputs = (re_prediction_scores, ner_prediction_scores)

        ner_prediction_scores = ner_prediction_scores.float()
        re_prediction_scores = re_prediction_scores.float()

        if rel_labels is not None:
            re_loss = _compute_re_loss(
                re_prediction_scores, rel_labels, num_labels, self.args
            )
            ner_loss = _compute_ner_loss(
                ner_prediction_scores, ner_labels, num_ner_labels, self.args
            )
            loss = re_loss + ner_loss
            outputs = (loss, re_loss, ner_loss) + outputs

        return outputs

    def forward_all_heads(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        sub_positions=None,
        ent_numbers=None,
        **kwargs,
    ) -> dict[str, tuple]:
        """Run encoder+HyperGNN once and apply every head.

        Returns ``{dataset_id: (re_logits, ner_logits)}`` for all heads.
        Only valid for multi-head checkpoints (``dataset_heads`` in config).
        """
        if self._head_info is None:
            raise RuntimeError("forward_all_heads requires a multi-head model.")

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = self.dropout(outputs[0])

        seq_len = self.max_seq_length
        ent_numbers_list = ent_numbers.tolist()
        max_ent_num = max(ent_numbers_list)
        tot_seq_len = input_ids.shape[-1]
        ent_len = (tot_seq_len - seq_len) // 2

        obj_start_states = hidden_states[:, seq_len : seq_len + ent_len][
            :, :max_ent_num, :
        ]
        obj_end_states = hidden_states[:, seq_len + ent_len :][:, :max_ent_num, :]
        n_total_ents = sum(ent_numbers_list)
        sub_start_states = hidden_states[
            torch.arange(n_total_ents), sub_positions[:, 0]
        ]
        sub_end_states = hidden_states[torch.arange(n_total_ents), sub_positions[:, 1]]

        sub_reprs = self.sub_encoder(sub_start_states, sub_end_states)
        obj_reprs = self.obj_encoder(obj_start_states, obj_end_states)
        rel_reprs = self.rel_encoder(
            sub_reprs.unsqueeze(-2).expand(obj_reprs.shape), obj_reprs
        )

        ent_numbers_list = ent_numbers.tolist()
        rel_reprs = pad_sequence(
            torch.split(rel_reprs, ent_numbers_list),
            batch_first=True,
            padding_value=0,
        )
        obj_reprs = pad_sequence(
            torch.split(obj_reprs, ent_numbers_list),
            batch_first=True,
            padding_value=-1e4,
        )
        uni_obj_reprs = torch.max(obj_reprs, dim=1)[0]
        sub_reprs = pad_sequence(
            torch.split(sub_reprs, ent_numbers_list),
            batch_first=True,
            padding_value=0,
        )

        mask1d = get_ent_mask1d(ent_numbers, max_num=max_ent_num)
        mask2d = get_ent_mask2d(ent_numbers, max_num=max_ent_num)
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

        return {
            ds_id: (
                self.rel_heads[ds_id](rel_reprs).float(),
                _apply_ner_head(
                    self.ner_heads[ds_id], sub_reprs, uni_obj_reprs, self.args.ent_repr
                ).float(),
            )
            for ds_id in self._head_info
        }


def get_ent_mask1d(n_ents, max_num=None):
    """
    n_ents: shape (b,), gold ent number.
    """
    if max_num is None:
        max_num = n_ents.max().item()
    bs = len(n_ents)
    mask = torch.arange(max_num, device=n_ents.device).unsqueeze(0).expand(bs, -1)
    mask = mask < n_ents.reshape(bs, 1)
    return mask


def get_ent_mask2d(n_ents, max_num=None):
    """
    n_ents: shape (bs,), ent number.
    return b x max_n_ent x max_n_ent
    """
    if isinstance(n_ents, list):
        n_ents = torch.tensor(n_ents)
    if max_num is None:
        max_num = n_ents.max().item()
    valid = torch.arange(max_num, device=n_ents.device).unsqueeze(0) < n_ents.view(
        -1, 1
    )  # bs x max_num
    return valid.unsqueeze(2) & valid.unsqueeze(1)  # bs x max_num x max_num


def get_ent_mask3d(n_ents):
    """
    mask2d: bs x ne x ne
    return bs x ne x ne x ne
    """
    mask2d = get_ent_mask2d(n_ents)
    bs, ne, _ = mask2d.shape
    m1 = mask2d.unsqueeze(-1).repeat(1, 1, 1, ne)  # bs x n1 x n2 x n3
    m2 = mask2d.unsqueeze(-2).repeat(1, 1, ne, 1)  # bs x n1 x n3 x n2
    mask = m1 * m2
    return mask


class CatEncoder(Module):
    def __init__(self, input_dims, output_dim=None, proj=True):
        super().__init__()
        inputdims = [dim for dim in input_dims]
        self.input_dims = inputdims
        self.proj = proj
        self.output_dim = output_dim if self.proj else sum(self.inputdims)
        if proj:
            self.projection = Linear(sum(inputdims), output_dim)

    def forward(self, *reprs):
        repr = torch.cat(reprs, dim=-1)
        if self.proj:
            repr = self.projection(repr)
        return repr


class HyperGNNBinaryGraph(Module):
    def __init__(self, rel_dim, dropout, args):
        super(HyperGNNBinaryGraph, self).__init__()
        self.args = args
        self.iter = args.n_iter
        aggregator = HyperGNNBinaryAggregateLayer

        self.hyperedgelayer = HyperGNNBinaryComposeLayer(
            rel_dim, dropout=dropout, args=args
        )
        self.aggregate = aggregator(rel_dim, dropout, args)

    def forward(self, rel_reprs, ent_numbers):
        """
        xx_reprs: node reprs
        """
        mask2d = get_ent_mask2d(ent_numbers)
        for i in range(self.iter):
            factor = self.hyperedgelayer(rel_reprs)
            rel_reprs = self.aggregate(rel_reprs, factor, ent_numbers)
            rel_reprs *= mask2d.unsqueeze(-1)
        return rel_reprs


class HyperGNNTernaryGraph(Module):
    def __init__(self, ent_dim, rel_dim, dropout, args):
        super(HyperGNNTernaryGraph, self).__init__()
        self.args = args
        self.iter = args.n_iter

        self.hyperedgelayer = HyperGNNTernaryComposeLayer(
            ent_dim, rel_dim, dropout=dropout, args=args
        )
        aggregator = HyperGNNTernaryAggregateLayer
        self.aggregate = aggregator(ent_dim, rel_dim, dropout, args)

    def forward(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
        """
        sub_reprs: bs x ns x no x d
        """
        mask1d = get_ent_mask1d(ent_numbers)
        mask2d = get_ent_mask2d(ent_numbers)

        for i in range(self.iter):
            factor = self.hyperedgelayer(sub_reprs, obj_reprs, rel_reprs)
            sub_reprs, obj_reprs, rel_reprs = self.aggregate(
                sub_reprs, obj_reprs, rel_reprs, factor, ent_numbers
            )
            sub_reprs *= mask1d.unsqueeze(-1)
            obj_reprs *= mask1d.unsqueeze(-1)
            rel_reprs *= mask2d.unsqueeze(-1)
        return sub_reprs, obj_reprs, rel_reprs


class HyperGNNBinaryComposeLayer(Module):
    """
    update hyperedge features
    """

    def __init__(self, rel_dim, dropout, args):
        super(HyperGNNBinaryComposeLayer, self).__init__()
        self.args = args
        self.factor_type = args.factor_type
        mem_dim = args.mem_dim
        self.dropout = Dropout(dropout)
        layernorm = args.layernorm
        dims = (rel_dim, rel_dim)
        if args.factor_type in {"sib", "cop", "gp", "tersib", "tercop", "tergp"}:
            if args.factor_encoder == "biaf":
                self.factor_compose1 = BiafEncoder(
                    input_dim1=rel_dim,
                    input_dim2=rel_dim,
                    output_dim=mem_dim,
                    rank=mem_dim,
                    factorize=True,
                )
            elif args.factor_encoder == "cat":
                self.factor_compose1 = CatEncoder(
                    input_dims=dims, output_dim=mem_dim, proj=True
                )
            self.layernorm1 = LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()
        elif args.factor_type in {
            "sibcop",
            "sibgp",
            "copgp",
            "tersibcop",
            "tersibgp",
            "tercopgp",
        }:
            if args.factor_encoder == "biaf":
                self.factor_compose1 = BiafEncoder(
                    input_dim1=rel_dim,
                    input_dim2=rel_dim,
                    output_dim=mem_dim,
                    rank=mem_dim,
                    factorize=True,
                )
                self.factor_compose2 = BiafEncoder(
                    input_dim1=rel_dim,
                    input_dim2=rel_dim,
                    output_dim=mem_dim,
                    rank=mem_dim,
                    factorize=True,
                )
            elif args.factor_encoder == "cat":
                self.factor_compose1 = CatEncoder(
                    input_dims=dims, output_dim=mem_dim, proj=True
                )
                self.factor_compose2 = CatEncoder(
                    input_dims=dims, output_dim=mem_dim, proj=True
                )
            self.layernorm1 = LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()
            self.layernorm2 = LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()
        elif args.factor_type in {"sibcopgp", "tersibcopgp"}:
            if args.factor_encoder == "biaf":
                self.factor_compose1 = BiafEncoder(
                    input_dim1=rel_dim,
                    input_dim2=rel_dim,
                    output_dim=mem_dim,
                    rank=mem_dim,
                    factorize=True,
                )
                self.factor_compose2 = BiafEncoder(
                    input_dim1=rel_dim,
                    input_dim2=rel_dim,
                    output_dim=mem_dim,
                    rank=mem_dim,
                    factorize=True,
                )
                self.factor_compose3 = BiafEncoder(
                    input_dim1=rel_dim,
                    input_dim2=rel_dim,
                    output_dim=mem_dim,
                    rank=mem_dim,
                    factorize=True,
                )
            elif args.factor_encoder == "cat":
                self.factor_compose1 = CatEncoder(
                    input_dims=dims, output_dim=mem_dim, proj=True
                )
                self.factor_compose2 = CatEncoder(
                    input_dims=dims, output_dim=mem_dim, proj=True
                )
                self.factor_compose3 = CatEncoder(
                    input_dims=dims, output_dim=mem_dim, proj=True
                )
            self.layernorm1 = LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()
            self.layernorm2 = LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()
            self.layernorm3 = LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()

    def forward(self, rel_reprs):
        """
        input old vertex features, update hyperedge features, then update vertex features
        """
        # if self.args.unirel:
        # 	rel_ha0 = rel_hb0 = reprs
        # else:
        # 	rel_ha0, rel_hb0 = reprs
        bs, ns, no, _ = rel_reprs.shape
        # initial the node reprs in hyperedge
        if self.factor_type in {"sib", "tersib"}:  # score for rij and rik
            b1_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b1_hb = rel_reprs.unsqueeze(-3).expand(
                -1, -1, no, -1, -1
            )  # bs x ns x no1 x no x dm
        elif self.factor_type in {"cop", "tercop"}:  # score for rik and rjk
            b1_ha = rel_reprs.unsqueeze(-3).expand(
                -1, -1, ns, -1, -1
            )  # bs x ns x ns1 x no x dm
            b1_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
        elif self.factor_type in {"gp", "tergp"}:  # score for rij and rjk
            b1_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b1_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
        elif self.factor_type in {"sibcop", "tersibcop"}:
            b1_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b1_hb = rel_reprs.unsqueeze(-3).expand(
                -1, -1, no, -1, -1
            )  # bs x ns x no1 x no x dm
            b2_ha = rel_reprs.unsqueeze(-3).expand(
                -1, -1, ns, -1, -1
            )  # bs x ns x ns1 x no x dm
            b2_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
        elif self.factor_type in {"sibgp", "tersibgp"}:
            b1_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b1_hb = rel_reprs.unsqueeze(-3).expand(
                -1, -1, no, -1, -1
            )  # bs x ns x no1 x no x dm
            b2_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b2_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
        elif self.factor_type in {"copgp", "tercopgp"}:
            b1_ha = rel_reprs.unsqueeze(-3).expand(
                -1, -1, ns, -1, -1
            )  # bs x ns x ns1 x no x dm
            b1_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
            b2_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b2_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
        elif self.factor_type in {"sibcopgp", "tersibcopgp"}:
            b1_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b1_hb = rel_reprs.unsqueeze(-3).expand(
                -1, -1, no, -1, -1
            )  # bs x ns x no1 x no x dm
            b2_ha = rel_reprs.unsqueeze(-3).expand(
                -1, -1, ns, -1, -1
            )  # bs x ns x ns1 x no x dm
            b2_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm
            b3_ha = rel_reprs.unsqueeze(-2).expand(
                -1, -1, -1, no, -1
            )  # bs x ns x no x no1 x dm
            b3_hb = rel_reprs.unsqueeze(-4).expand(
                -1, ns, -1, -1, -1
            )  # bs x ns1 x ns x no x dm

        # update hyperedge feature
        if self.args.factor_type in {"sib", "cop", "gp", "tersib", "tercop", "tergp"}:
            factor = self.layernorm1(self.dropout(self.factor_compose1(b1_ha, b1_hb)))
            return factor
        elif self.args.factor_type in {
            "sibcop",
            "sibgp",
            "copgp",
            "tersibcop",
            "tersibgp",
            "tercopgp",
        }:
            factor1 = self.layernorm1(self.dropout(self.factor_compose1(b1_ha, b1_hb)))
            factor2 = self.layernorm2(self.dropout(self.factor_compose2(b2_ha, b2_hb)))
            return (factor1, factor2)
        elif self.factor_type in {"sibcopgp", "tersibcopgp"}:
            factor1 = self.layernorm1(self.dropout(self.factor_compose1(b1_ha, b1_hb)))
            factor2 = self.layernorm2(self.dropout(self.factor_compose2(b2_ha, b2_hb)))
            factor3 = self.layernorm3(self.dropout(self.factor_compose3(b3_ha, b3_hb)))
            return (factor1, factor2, factor3)


class BiafEncoder(Module):
    def __init__(
        self,
        input_dim1,
        input_dim2,
        output_dim,
        rank=768,
        factorize=False,
        bias_1=True,
        bias_2=True,
    ):
        super().__init__()
        self.factorize = factorize
        if self.factorize:
            self.proj1 = Linear(input_dim1, rank)
            self.proj2 = Linear(input_dim2, rank)
            self.encoder = Linear(rank, output_dim)
        else:
            self.bias_1 = bias_1
            self.bias_2 = bias_2
            self.weight = Parameter(
                torch.Tensor(input_dim1 + bias_1, input_dim2 + bias_2, output_dim)
            )
            self.bias = Parameter(torch.Tensor(output_dim))
            self.reset_parameters()

    def reset_parameters(self):
        for w in [self.weight]:
            torch.nn.init.xavier_normal_(w)
        self.bias.data.fill_(0)
        return

    def forward(self, input1, input2):
        if self.factorize:
            input1 = self.proj1(input1)
            input2 = self.proj2(input2)
            repr = self.encoder(input1 * input2)
        else:
            if self.bias_1:
                input1 = torch.cat((input1, torch.ones_like(input1[..., :1])), -1)
            if self.bias_2:
                input2 = torch.cat((input2, torch.ones_like(input2[..., :1])), -1)
            if len(input1.shape) == 3:
                layer = torch.einsum(
                    "bnd,bne,deo->bno", input1, input2, self.weight.to(input1.dtype)
                )
            elif len(input1.shape) == 4:
                layer = torch.einsum(
                    "bnmd,bnme,deo->bnmo", input1, input2, self.weight.to(input1.dtype)
                )
            repr = layer + self.bias.to(layer.dtype)
        return repr


class HyperGNNBinaryAggregateLayer(Module):
    def __init__(self, rel_dim, dropout, args):
        super(HyperGNNBinaryAggregateLayer, self).__init__()

        self.factor_type = args.factor_type
        mem_dim = args.mem_dim
        self.args = args
        layernorm = args.layernorm
        self.dropout = Dropout(dropout)

        self.attn_combine = Linear(mem_dim + rel_dim, mem_dim)
        self.attn_combine = Sequential(self.attn_combine, GELU())
        self.v = Linear(mem_dim, 1, bias=False)
        self.fc = Linear(mem_dim, rel_dim)
        self.layernorm = LayerNorm(rel_dim, eps=1e-6) if layernorm else Identity()

    def update_single(self, ht, factor, ent_numbers):
        """
        ht is rel repr: bs x ns x no x dr, factor: bs x ni x nj x nk x dm
        if edgetype=='sib'
                factor: score for rij and rik; j,k are obj axis
                ha: ht in i,j axis; hb: ht in ik axis of factor
                ha: bs x ni x nj x nk x dm; hb:  bs x ni x nk x nj x dm
        if edgetype=='cop'
                factor: score of rik and rjk.
                ha: ht in ik axis of factor; hb: ht in jk axis of factor
                ha: bs x ni x nk x nj x d; hb: bs x nj x nk x ni x d
        if edgetype=='gp'
                factor: score for rij and rjk.
                bin_ha: bs x ni x nj x nk x d; bin_hb: bs x nj x nk x ni x d
        return updated ht
        """
        bs, ne, _, dr = ht.shape

        if self.factor_type == "sib":
            ha = factor
            hb = factor.permute(0, 1, 3, 2, 4)
        elif self.factor_type == "cop":
            ha = factor.permute(0, 1, 3, 2, 4)
            hb = factor.permute(0, 2, 3, 1, 4)
        elif self.factor_type == "gp":
            ha = factor
            hb = factor.permute(0, 2, 3, 1, 4)
        ht_new = torch.cat((ha, hb), dim=-2)

        res = ht

        total_h = ht_new
        ht = (
            ht.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ns x no x nc x dm
        comb = torch.cat([ht, total_h], dim=-1)  # bs x ns x no x nc x 2*dm

        energy = self.attn_combine(comb)  # bs x ns x no x nc x dm
        energy = self.v(energy).squeeze(-1)  # bs x ns x no x nc
        # attn_mask = torch.sum(ht, dim=-1) == 0		# bs x ns x no x nc
        batch_mask3d = get_ent_mask3d(ent_numbers)  # bs*ne*ne*ne
        if self.factor_type == "sib":
            m1 = (
                torch.eye(ne, dtype=torch.int, device=self.args.device)
                .unsqueeze(0)
                .repeat(bs, 1, 1)
            )  # bs x no x no1	, r(i,j) not att to r(i,j)
            m1 = (
                m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()
            )  # bs*ne*ne*ne, bs x ns x no x no1
            m1 = (m1 + ~batch_mask3d).bool()
            attn_mask = torch.stack((m1, m1), dim=-2).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x nc
        elif self.factor_type == "cop":
            m1 = (
                torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
            )  # bs x ns x ns1
            m1 = m1.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
            m1 = (m1 + ~batch_mask3d).bool()
            attn_mask = torch.stack((m1, m1), dim=-2).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2ns1
            # unsqueeze(-2).repeat(1,no,1).bool()	# ns x no x 2*ns1
        elif self.factor_type == "gp":
            m1 = ~batch_mask3d
            attn_mask = torch.stack((m1, m1), dim=-2).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2no
        else:
            raise ValueError("factor_type is not correct")
        energy = energy.masked_fill(attn_mask, -1e4)

        attention = energy.softmax(dim=-1)  # bs x ns x no x nc
        output = torch.einsum(
            "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
        )  # ns x no x dm

        output = self.dropout(self.fc(output)) + res
        output = self.layernorm(output)

        return output

    def update_double(self, ht, factors, ent_numbers):
        """
        factor_type: sibcop, sibgp, copgp
        ht: bs x ns x no x d
        if edgetype=='sib'
                factor: j,k are obj axis; ij, ik are two relations
                ha: ht in i,j axis; hb: ht in ik axis of factor
                ha: bs x ni x nj x nk x dm; hb:  bs x ni x nk x nj x dm
        if edgetype=='cop'
                factor: ik,jk are two relations.
                ha: ht in ik axis of factor; hb: ht in jk axis of factor
                ha: bs x ni x nk x nj x d; hb: bs x nj x nk x ni x d
        if edgetype=='gp'
                factor: ij, jk are two relations
                bin_ha: bs x ni x nj x nk x d; bin_hb: bs x nj x nk x ni x d
        """
        (factor1, factor2) = factors
        res = ht
        bs, ne, _, dm = ht.shape
        if self.factor_type in {"sibcop", "sibgp"}:
            f1_ha1 = factor1
            f1_hb1 = factor1.permute(0, 1, 3, 2, 4)
        elif self.factor_type == "copgp":
            f1_ha1 = factor1.permute(0, 1, 3, 2, 4)
            f1_hb1 = factor1.permute(0, 2, 3, 1, 4)
        f1_h = torch.cat([f1_ha1, f1_hb1], dim=-2)

        if self.factor_type in {"copgp", "sibgp"}:
            f2_ha1 = factor2  # bs x ns x no x no1 x d	gp
            f2_hb1 = factor2.permute(0, 2, 3, 1, 4)
        elif self.factor_type == "sibcop":
            f2_ha1 = factor2.permute(0, 1, 3, 2, 4)
            f2_hb1 = factor2.permute(0, 2, 3, 1, 4)
        f2_h = torch.cat([f2_ha1, f2_hb1], dim=-2)

        total_h = torch.cat([f1_h, f2_h], dim=-2)  # bs x ns x no x (3no1+ns1) x d
        ht = (
            ht.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ns x no x nc x dm,   (nc=(3no1+ns1)
        comb = torch.cat([ht, total_h], dim=-1)  # ns x no x nc x 2*dm

        energy = self.attn_combine(comb)  # bs x ns x no x nc x dm
        energy = self.v(energy).squeeze(-1)  # bs x ns x no x nc
        batch_mask3d = get_ent_mask3d(ent_numbers)  # bs x ns x no x no1

        if self.factor_type == "sibgp":
            m1 = (
                torch.eye(ne, dtype=torch.int, device=self.args.device)
                .unsqueeze(0)
                .repeat(bs, 1, 1)
            )  # bs x no x no1	, r(i,j) not att to r(i,j)
            m1 = m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()  # bs x ns x no x no1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # for sib,  bs x ns x no x 2no1
            m2 = ~batch_mask3d
            m2 = torch.cat((m2, m2), dim=-1).reshape(
                bs, ne, ne, -1
            )  # for gp, bs x ns x no x 2ns1
        elif self.factor_type == "sibcop":
            m1 = (
                torch.eye(ne, dtype=torch.int, device=self.args.device)
                .unsqueeze(0)
                .repeat(bs, 1, 1)
            )  # bs x no x no1	, r(i,j) not att to r(i,j)
            m1 = m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()  # bs x ns x no x no1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # for sib,  bs x ns x no x 2no1
            m2 = (
                torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
            )  # bs x ns x ns1
            m2 = m2.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
            m2 = (m2 + ~batch_mask3d).bool()
            m2 = torch.cat((m2, m2), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2ns1
        elif self.factor_type == "copgp":
            # cop
            m1 = (
                torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
            )  # bs x ns x ns1
            m1 = m1.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2ns1
            # gp
            m2 = ~batch_mask3d
            m2 = torch.cat((m2, m2), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2no
        else:
            raise ValueError("factor_type is not correct")
        attn_mask = torch.cat((m1, m2), dim=-1).reshape(bs, ne, ne, -1)

        energy = energy.masked_fill(attn_mask, -1e4)

        attention = energy.softmax(dim=-1)  # bs x ns x no x nc
        output = torch.einsum(
            "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
        )  # ns x no x dm
        # if self.args.attn_res:
        # 	output = self.dropout(self.fc(output)) + res
        # else:
        # 	output = self.dropout(self.fc(output))
        output = self.dropout(self.fc(output)) + res
        output = self.layernorm(output)

        return output

    def update_triple(self, ht, factors, ent_numbers):
        """
        self.factor_type == sibcopgp
        ht: bs x ns x no x d
        sib:
                ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
        cop:
                ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
        gp:
                ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
        """
        (factor1, factor2, factor3) = factors
        res = ht
        bs, ne, _, dm = ht.shape
        # sib
        f1_ha1 = factor1  # bs x ns x no x no1 x d	sib
        f1_hb1 = factor1.permute(0, 1, 3, 2, 4)  # bs x ns x no x no1 x d
        f1_h = torch.cat([f1_ha1, f1_hb1], dim=-2)
        # cop
        f2_ha1 = factor2.permute(0, 1, 3, 2, 4)  # bs x ns x no x ns1 x d	cop
        f2_hb1 = factor2.permute(0, 2, 3, 1, 4)  # bs x ns x no x ns1 x d
        f2_h = torch.cat([f2_ha1, f2_hb1], dim=-2)
        # gp
        f3_ha1 = factor3  # bs x ns x no x no1 x d	gp
        f3_hb1 = factor3.permute(0, 2, 3, 1, 4)  # bs x ns x no x ns1 x d
        f3_h = torch.cat([f3_ha1, f3_hb1], dim=-2)

        total_h = torch.cat(
            [f1_h, f2_h, f3_h], dim=-2
        )  # bs x ns x no x (3no1+3ns1) x d
        ht = (
            ht.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ns x no x nc x dm,   (nc=(3no1+ns1)
        comb = torch.cat([ht, total_h], dim=-1)  # bs x ns x no x nc x 2*dm

        energy = self.attn_combine(comb)  # bs x ns x no x nc x dm
        energy = self.v(energy).squeeze(-1)  # bs x reprsreprsns x no x nc
        batch_mask3d = get_ent_mask3d(ent_numbers)  # bs x ns x no x no1

        # sib
        m1 = (
            torch.eye(ne, dtype=torch.int, device=self.args.device)
            .unsqueeze(0)
            .repeat(bs, 1, 1)
        )  # bs x no x no1	, r(i,j) not att to r(i,j)
        m1 = m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()  # bs x ns x no x no1
        m1 = (m1 + ~batch_mask3d).bool()
        m1 = torch.cat((m1, m1), dim=-1).reshape(
            bs, ne, ne, -1
        )  # for sib,  bs x ns x no x 2no1
        # cop
        m2 = (
            torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        )  # bs x ns x ns1
        m2 = m2.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
        m2 = (m2 + ~batch_mask3d).bool()
        m2 = torch.cat((m2, m2), dim=-1).reshape(bs, ne, ne, -1)  # bs x ns x no x 2ns1
        # gp
        m3 = ~batch_mask3d
        m3 = torch.cat((m3, m3), dim=-2).reshape(
            bs, ne, ne, -1
        )  # bs x ns x no x (no1+ns1)

        attn_mask = torch.cat((m1, m2, m3), dim=-1).reshape(bs, ne, ne, -1)

        energy = energy.masked_fill(attn_mask, -1e4)

        attention = energy.softmax(dim=-1)  # bs x ns x no x nc
        output = torch.einsum(
            "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
        )  # ns x no x dm

        output = self.dropout(self.fc(output)) + res
        output = self.layernorm(output)

        return output

    def forward(self, *inputs):
        if self.factor_type in {"sib", "cop", "gp"}:
            return self.update_single(*inputs)
        elif self.factor_type in {"sibcop", "sibgp", "copgp"}:
            return self.update_double(*inputs)
        elif self.factor_type == "sibcopgp":
            return self.update_triple(*inputs)
        else:
            raise ValueError("factor_type is not correct")


class HyperGNNTernaryAggregateLayer(Module):
    def __init__(self, ent_dim, rel_dim, dropout, args):
        super(HyperGNNTernaryAggregateLayer, self).__init__()
        self.args = args
        mem_dim = args.mem_dim
        layernorm = args.layernorm
        self.dropout = Dropout(dropout)

        # for ablation study
        self.proj_s = Linear(ent_dim, mem_dim)
        self.proj_o = Linear(ent_dim, mem_dim)
        self.attn_combine_s = Linear(mem_dim + ent_dim, mem_dim)
        self.attn_combine_o = Linear(mem_dim + ent_dim, mem_dim)

        # if args.attn_encoder=='nonlinear':
        self.attn_combine_s = Sequential(self.attn_combine_s, GELU())
        self.attn_combine_o = Sequential(self.attn_combine_o, GELU())
        # self.attn_combine_r = Sequential(self.attn_combine_r, GELU())

        self.sv = Linear(mem_dim, 1, bias=False)
        self.ov = Linear(mem_dim, 1, bias=False)

        self.fc_s = Linear(mem_dim, ent_dim)
        self.fc_o = Linear(mem_dim, ent_dim)

        if self.args.attn_self:
            self.proj_r = Linear(rel_dim, mem_dim)
            self.attn_combine_r = Linear(mem_dim + rel_dim, mem_dim)
            self.attn_combine_r = Sequential(self.attn_combine_r, GELU())
            self.rv = Linear(mem_dim, 1, bias=False)
            self.fc_r = Linear(mem_dim, rel_dim)
        else:
            self.encode_r = LinearMessegePasser(mem_dim, rel_dim)

        self.layernorm_s = LayerNorm(ent_dim, eps=1e-6) if layernorm else Identity()
        self.layernorm_o = LayerNorm(ent_dim, eps=1e-6) if layernorm else Identity()
        self.layernorm_r = LayerNorm(rel_dim, eps=1e-6) if layernorm else Identity()

    def update_rel(self, rel_reprs, factor):
        """
        rel_reprs: bs x ns x no x dr
        factor: bs x ns x no x dm
        return updated rel_reprs
        """
        # ht = self._apply_mask(ht, mask)
        # ht_new = self._apply_mask(ht_new, mask)

        res = rel_reprs
        if self.args.attn_self:
            ht = self.proj_r(rel_reprs).unsqueeze(-2)  # bs x ns x no x 1 x dm
            total_h = torch.cat(
                [ht, factor.unsqueeze(-2)], dim=-2
            )  # bs x ns x no x 2 x dm
            ht = (
                rel_reprs.unsqueeze(-2)
                .repeat(1, 1, 1, total_h.shape[-2], 1)
                .contiguous()
            )
            comb = torch.cat([ht, total_h], dim=-1)  # bs x ns x no x 2 x (dr + dm)
            energy = self.attn_combine_r(comb)  # bs x ns x no x 2 x 1
            energy = self.rv(energy).squeeze(-1)  # bs x ns x no x 2
            attention = energy.softmax(dim=-1)  # bs x ns x no x 2
            output = torch.einsum(
                "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
            )
            output = self.dropout(self.fc_r(output)) + res
            output = self.layernorm_r(output)
        else:
            output = self.dropout(self.encode_r(factor, rel_reprs)) + res
            output = self.layernorm_r(output)
        # output = torch.max(total_h, dim=-2)[0]

        return output

    def update_sub(self, sub_reprs, factor, ent_numbers):
        """
        factor: bs x ns x no x dm
        sub_reprs: bs x ns x ds
        batch_mask: bs x ns x no
        """
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        res = sub_reprs
        bs, ne, _, dm = factor.shape
        ht = self.proj_s(res)
        # total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
        total_h = (
            torch.cat([ht.unsqueeze(-2), factor], dim=-2)
            if self.args.attn_self
            else factor
        )  # bs x ne x (1+ne) x dm or bs x ne x ne x dm

        ht = (
            sub_reprs.unsqueeze(-2).repeat(1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ne x ne x dm
        comb = torch.cat([ht, total_h], dim=-1)  # bs x ne x (1+ne) x dm+dr
        # pdb.set_trace()
        energy = self.attn_combine_s(comb)  # bs x ne x (1+ne) x dm
        energy = self.sv(energy).squeeze(-1)  # bs x ne x (1+ne)
        attn_mask = (
            torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        )  # bs x ne x ne
        attn_mask = (attn_mask + ~batch_mask).bool()  # bs x ne x ne
        if self.args.attn_self:
            attn_self = torch.zeros(
                (bs, ne, 1), device=self.args.device
            ).bool()  #  bs x ne x 1
            attn_mask = torch.cat((attn_self, attn_mask), axis=-1)  # bs x ne x (1+ne)

        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-1)  # ns x no
        output = torch.einsum(
            "bij,bijd->bid", attention, total_h.to(attention.dtype)
        )  # ns x dm
        output = self.dropout(self.fc_s(output)) + res
        output = self.layernorm_s(output)

        return output

    def update_obj(self, obj_reprs, factor, ent_numbers):
        """
        factor: bs x ns x no x dm
        obj_reprs: bs x no x de
        """
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        res = obj_reprs
        bs, ne, _, dm = factor.shape
        ht = self.proj_o(res)  # (b, no, dh)
        # total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
        total_h = (
            torch.cat([ht.unsqueeze(-3), factor], dim=-3)
            if self.args.attn_self
            else factor
        )  # bs*(1+ne)*ne*dm

        ht = (
            obj_reprs.unsqueeze(-3).repeat(1, total_h.shape[-3], 1, 1).contiguous()
        )  # bs x (1+ne) x ne x de
        comb = torch.cat([ht, total_h], dim=-1)  # bs x (1+ne) x ne x (de+dm)

        energy = self.attn_combine_o(comb)  # bs x (1+ne) x ne x dm
        energy = self.ov(energy).squeeze(-1)  # bs x (1+ne) x ne
        attn_mask = (
            torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        )  # bs x ne x ne
        attn_mask = (attn_mask + ~batch_mask).bool()  # bs x ne x ne
        if self.args.attn_self:
            attn_self = torch.zeros(
                (bs, 1, ne), device=self.args.device
            ).bool()  # bs x 1 x ne
            attn_mask = torch.cat((attn_self, attn_mask), axis=-2)  # bs x (1+ne) x ne

        # attn_mask = torch.sum(ht, dim=-1) == 0
        energy = energy.masked_fill(attn_mask, -1e4)

        attention = energy.softmax(dim=-2)  # bs x (1+ne) x ne
        output = torch.einsum(
            "bij,bijd->bjd", attention, total_h.to(attention.dtype)
        )  # bs x no x dm
        output = self.dropout(self.fc_o(output)) + res
        output = self.layernorm_o(output)

        return output

    def forward(self, sub_reprs, obj_reprs, rel_reprs, factor, ent_numbers):
        """
        rel_reprs: bs x ns x no x dr
        sub_reprs: bs x ns x de
        obj_reprs: no x de
        rel_h, sub_h, obj_h: ns x no x dx
        """
        sub_reprs = self.update_sub(sub_reprs, factor, ent_numbers)
        obj_reprs = self.update_obj(obj_reprs, factor, ent_numbers)
        rel_reprs = self.update_rel(rel_reprs, factor)
        return sub_reprs, obj_reprs, rel_reprs


class HyperGNNHybridGraph(Module):
    def __init__(self, ent_dim, rel_dim, dropout, args):
        super(HyperGNNHybridGraph, self).__init__()
        self.args = args
        self.iter = args.n_iter

        aggregator = HyperGNNHybridAggregateLayer

        self.hyperedgelayer1 = HyperGNNTernaryComposeLayer(
            ent_dim, rel_dim, dropout=dropout, args=args
        )
        self.hyperedgelayer2 = HyperGNNBinaryComposeLayer(
            rel_dim, dropout=dropout, args=args
        )
        self.aggregate = aggregator(ent_dim, rel_dim, dropout, args)

    def forward(self, sub_reprs, obj_reprs, rel_reprs, ent_numbers):
        """
        xx_reprs: node reprs
        """
        mask1d = get_ent_mask1d(ent_numbers)
        mask2d = get_ent_mask2d(ent_numbers)

        for i in range(self.iter):
            factor_ter = self.hyperedgelayer1(sub_reprs, obj_reprs, rel_reprs)
            if self.args.factor_type in {"tersib", "tercop", "tergp"}:
                factor_b1 = self.hyperedgelayer2(rel_reprs)
                factors = (factor_ter, factor_b1)
            elif self.args.factor_type in {"tersibcop", "tersibgp", "tercopgp"}:
                factor_b1, factor_b2 = self.hyperedgelayer2(rel_reprs)
                factors = (factor_ter, factor_b1, factor_b2)
            elif self.args.factor_type == "tersibcopgp":
                factor_b1, factor_b2, factor_b3 = self.hyperedgelayer2(rel_reprs)
                factors = (factor_ter, factor_b1, factor_b2, factor_b3)
            sub_reprs, obj_reprs, rel_reprs = self.aggregate(
                sub_reprs, obj_reprs, rel_reprs, factors, ent_numbers
            )
            sub_reprs *= mask1d.unsqueeze(-1)
            obj_reprs *= mask1d.unsqueeze(-1)
            rel_reprs *= mask2d.unsqueeze(-1)

        return sub_reprs, obj_reprs, rel_reprs


class HyperGNNTernaryComposeLayer(Module):
    """
    update hyperedge features
    """

    def __init__(self, ent_dim, rel_dim, dropout, args):
        super(HyperGNNTernaryComposeLayer, self).__init__()
        mem_dim = args.mem_dim
        self.dropout = Dropout(dropout)
        layernorm = args.layernorm
        dims = (ent_dim, ent_dim, rel_dim)
        if args.factor_encoder == "biaf":
            self.factor_compose = CPDTrilinear(
                ent_dim, ent_dim, rel_dim, mem_dim, mem_dim
            )
        elif args.factor_encoder == "cat":
            self.factor_compose = CatEncoder(
                input_dims=dims, output_dim=mem_dim, proj=True
            )

        self.layernorm = (
            LayerNorm(mem_dim, eps=1e-6) if layernorm else Identity()
        )  # for cell state of node1
        # self.layernorm_2 = nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
        # self.layernorm_3 = nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()

        # self.aggregate = HyperGNNTernaryAggregateLayer(ent_dim, rel_dim, dropout, args)

    def forward(self, sub_reprs, obj_reprs, rel_reprs):
        """
        input old vertex features, update hyperedge features, then update vertex features
        """
        bs, ne, _ = sub_reprs.shape

        # initial the node reprs in hyperedge
        sub_h = sub_reprs.unsqueeze(-2).expand(-1, -1, ne, -1)  # bs x ns x no x de
        obj_h = obj_reprs.unsqueeze(-3).expand(-1, ne, -1, -1)  # bs x ns x no x de
        rel_h = rel_reprs

        # update hyperedge feature
        factor = self.layernorm(self.dropout(self.factor_compose(sub_h, obj_h, rel_h)))

        # sub_reprs, obj_reprs, rel_reprs = self.aggregate(sub_reprs, obj_reprs, rel_reprs, factor, batch_mask)
        return factor


class CPDTrilinear(nn.Module):
    def __init__(self, input_dim1, input_dim2, input_dim3, rank, output_dim):
        super().__init__()
        """
		input three tensor with the same shape.
		"""
        self.proj1 = nn.Linear(input_dim1, rank)
        self.proj2 = nn.Linear(input_dim2, rank)
        self.proj3 = nn.Linear(input_dim3, rank)

        self.encode_proj = nn.Linear(rank, output_dim)

    def forward(self, input1, input2, input3):
        layer1 = self.proj1(input1)
        layer2 = self.proj2(input2)
        layer3 = self.proj3(input3)

        return self.encode_proj(layer1 * layer2 * layer3)


class HyperGNNHybridAggregateLayer(nn.Module):
    def __init__(self, ent_dim, rel_dim, dropout, args):
        super(HyperGNNHybridAggregateLayer, self).__init__()

        self.factor_type = args.factor_type
        mem_dim = args.mem_dim
        self.args = args
        layernorm = args.layernorm
        self.dropout = nn.Dropout(dropout)

        self.proj_s = nn.Linear(ent_dim, mem_dim)
        self.proj_o = nn.Linear(ent_dim, mem_dim)

        if self.args.attn_self:
            self.proj_r = nn.Linear(rel_dim, mem_dim)

        self.attn_combine_s = nn.Linear(mem_dim + ent_dim, mem_dim)
        self.attn_combine_o = nn.Linear(mem_dim + ent_dim, mem_dim)
        self.attn_combine_r = nn.Linear(mem_dim + rel_dim, mem_dim)
        # if args.attn_encoder=='nonlinear':
        self.attn_combine_s = nn.Sequential(self.attn_combine_s, nn.GELU())
        self.attn_combine_o = nn.Sequential(self.attn_combine_o, nn.GELU())
        self.attn_combine_r = nn.Sequential(self.attn_combine_r, nn.GELU())

        self.sv = nn.Linear(mem_dim, 1, bias=False)
        self.ov = nn.Linear(mem_dim, 1, bias=False)
        self.rv = nn.Linear(mem_dim, 1, bias=False)

        self.fc_s = nn.Linear(mem_dim, ent_dim)
        self.fc_o = nn.Linear(mem_dim, ent_dim)
        self.fc_r = nn.Linear(mem_dim, rel_dim)

        self.layernorm_s = (
            nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
        )
        self.layernorm_o = (
            nn.LayerNorm(ent_dim, eps=1e-6) if layernorm else nn.Identity()
        )
        self.layernorm_r = (
            nn.LayerNorm(rel_dim, eps=1e-6) if layernorm else nn.Identity()
        )

    def forward(self, *inputs):
        sub_reprs, obj_reprs, rel_reprs, factors, ent_numbers = inputs
        if self.factor_type in {"tersib", "tercop", "tergp"}:
            factor_ter = factors[0]
            sub_reprs = self.update_sub(sub_reprs, factor_ter, ent_numbers)
            obj_reprs = self.update_obj(obj_reprs, factor_ter, ent_numbers)
            rel_reprs = self.update_rel_single(rel_reprs, factors, ent_numbers)
        elif self.factor_type in {"tersibcop", "tersibgp", "tercopgp"}:
            factor_ter = factors[0]
            sub_reprs = self.update_sub(sub_reprs, factor_ter, ent_numbers)
            obj_reprs = self.update_obj(obj_reprs, factor_ter, ent_numbers)
            rel_reprs = self.update_rel_double(rel_reprs, factors, ent_numbers)
        elif self.factor_type == "tersibcopgp":
            factor_ter = factors[0]
            sub_reprs = self.update_sub(sub_reprs, factor_ter, ent_numbers)
            obj_reprs = self.update_obj(obj_reprs, factor_ter, ent_numbers)
            rel_reprs = self.update_rel_triple(rel_reprs, factors, ent_numbers)
        else:
            raise ValueError("factor_type is not correct")

        return sub_reprs, obj_reprs, rel_reprs

    def update_rel_single(self, rel_reprs, factors, ent_numbers):
        """
        rel_reprs: bs x ns x no x dr
        factor_ter: bs x ns x no x d
        sib:
                ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
        cop:
                ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
        gp:
                ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
        return updated ht
        """

        factor_ter, factor_b1 = factors
        bs, ne, _, dm = rel_reprs.shape
        res = rel_reprs

        ter_ht_new = factor_ter.unsqueeze(-2)  # bs x ns x no x 1 x dm

        # ablation
        if self.factor_type == "tersib":
            ha = factor_b1
            hb = factor_b1.permute(0, 1, 3, 2, 4)
        elif self.factor_type == "tercop":
            ha = factor_b1.permute(0, 1, 3, 2, 4)
            hb = factor_b1.permute(0, 2, 3, 1, 4)
        elif self.factor_type == "tergp":
            ha = factor_b1
            hb = factor_b1.permute(0, 2, 3, 1, 4)
        b1_ht_new = torch.cat((ha, hb), dim=-2)

        total_h = torch.cat(
            [ter_ht_new, b1_ht_new], dim=-2
        )  # bs x ns x no x nc x dm, for sib, nc=1+2no1, for cop, nc = 1+2ns1

        if self.args.attn_self:
            ht = self.proj_r(rel_reprs).unsqueeze(-2)  # bs x ns x no x 1 x dm
            total_h = torch.cat([ht, total_h], dim=-2)  # bs x ns x no x (1+nc) x dm

        ht = (
            rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        )  # ns x no x nc x dm
        comb = torch.cat([ht, total_h], dim=-1)  # ns x no x nc x 2*dm

        energy = self.attn_combine_r(comb)  # ns x no x nc x dm
        energy = self.rv(energy).squeeze(-1)  # ns x no x nc
        # attn_mask = torch.sum(ht, dim=-1) == 0		# ns x no x nc
        batch_mask2d = get_ent_mask2d(ent_numbers).to(self.args.device)
        batch_mask3d = get_ent_mask3d(ent_numbers)  # bs x ns x no x no1
        mask_ter = ~batch_mask2d.unsqueeze(-1)  # bs x ns x no x 1
        if self.factor_type == "tersib":
            m1 = (
                torch.eye(ne, dtype=torch.int, device=self.args.device)
                .unsqueeze(0)
                .repeat(bs, 1, 1)
            )  # bs x no x no1	, r(i,j) not att to r(i,j)
            m1 = m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()  # bs x ns x no x no1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2no1
        elif self.factor_type == "tercop":
            m1 = (
                torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
            )  # bs x ns x ns1
            m1 = m1.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2ns1
        elif self.factor_type == "tergp":
            m1 = ~batch_mask3d
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2no
        attn_mask = torch.cat((mask_ter, m1), dim=-1).reshape(
            bs, ne, ne, -1
        )  # bs x ns x no x (1+2ns1)

        if self.args.attn_self:
            attn_self = torch.zeros((bs, ne, ne, 1), device=self.args.device).bool()
            attn_mask = torch.cat(
                (attn_self, attn_mask), axis=-1
            )  # bs x ns x no x (2+2ns1)

        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-1)  # bs x ns x no x nc
        output = torch.einsum(
            "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
        )  # bs x ns x no x dm

        output = self.dropout(self.fc_r(output)) + res
        output = self.layernorm_r(output)

        return output

    def update_rel_double(self, rel_reprs, factors, ent_numbers):
        """
        rel_reprs: bs x ns x no x dr
        factor_ter: bs x ns x no x d
        factors for
        sib:
                ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
        cop:
                ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
        gp:
                ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
        """
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        factor_ter, factor_b1, factor_b2 = factors
        res = rel_reprs
        bs, ne, _, _ = rel_reprs.shape

        ter_ht_new = factor_ter.unsqueeze(-2)  # bs x ns x no x 1 x dm

        if self.factor_type in {"tersibcop", "tersibgp"}:
            f1_ha1 = factor_b1
            f1_hb1 = factor_b1.permute(0, 1, 3, 2, 4)
        elif self.factor_type == "tercopgp":
            f1_ha1 = factor_b1.permute(0, 1, 3, 2, 4)
            f1_hb1 = factor_b1.permute(0, 2, 3, 1, 4)
        f1_h = torch.cat([f1_ha1, f1_hb1], dim=-2)

        if self.factor_type in {"tercopgp", "tersibgp"}:
            f2_ha1 = factor_b2  # bs x ns x no x no1 x d	gp
            f2_hb1 = factor_b2.permute(0, 2, 3, 1, 4)
        elif self.factor_type == "tersibcop":
            f2_ha1 = factor_b2.permute(0, 1, 3, 2, 4)
            f2_hb1 = factor_b2.permute(0, 2, 3, 1, 4)
        f2_h = torch.cat([f2_ha1, f2_hb1], dim=-2)

        # b1_h = torch.cat((b1_ha1, b1_hb1), dim=-2)					# bs x ns x no x nc1	nc1=2no1 sib; 2ns1 cop; no1+ns1 gp;
        # b2_h = torch.cat((b2_ha1, b2_hb1), dim=-2)					# bs x ns x no x nc2	nc2=2ns1 cop; no1+ns1 gp;
        ht_new = torch.cat(
            [f1_h, f2_h], dim=-2
        )  # bs x ns x no x nc, nc= 2no1+2ns1 sibcop; 3no1+ns1 sibgp; 3ns1+no1 copgp

        total_h = torch.cat([ter_ht_new, ht_new], dim=-2)  # bs x ns x no x 1+nc x d

        if self.args.attn_self:
            ht = self.proj_r(rel_reprs).unsqueeze(-2)  # bs x ns x no x 1 x dm
            total_h = torch.cat([ht, total_h], dim=-2)  # bs x ns x no x (1+nc) x dm

        ht = (
            rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ns x no x nc x dm,
        comb = torch.cat([ht, total_h], dim=-1)  # bs x ns x no x nc x 2*dm

        energy = self.attn_combine_r(comb)  # ns x no x nc x dm
        energy = self.rv(energy).squeeze(-1)  # ns x no x nc

        # attn mask
        batch_mask3d = get_ent_mask3d(ent_numbers)  # bs x ns x no x no1
        mask_ter = ~batch_mask.unsqueeze(-1)  # bs x ns x no x 1

        if self.args.factor_type in {"tersibcop", "tersibgp"}:
            m1 = (
                torch.eye(ne, dtype=torch.int, device=self.args.device)
                .unsqueeze(0)
                .repeat(bs, 1, 1)
            )  # bs x no x no1	, r(i,j) not att to r(i,j)
            m1 = m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()  # bs x ns x no x no1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2no1
        elif self.args.factor_type == "tercopgp":
            m1 = (
                torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
            )  # bs x ns x ns1
            m1 = m1.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
            m1 = (m1 + ~batch_mask3d).bool()
            m1 = torch.cat((m1, m1), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2ns1

        if self.args.factor_type in {"tercopgp", "tersibgp"}:
            m2 = ~batch_mask3d
            m2 = torch.cat((m2, m2), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2no
        elif self.args.factor_type == "tersibcop":
            m2 = (
                torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
            )  # bs x ns x ns1
            m2 = m2.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
            m2 = (m2 + ~batch_mask3d).bool()
            m2 = torch.cat((m2, m2), dim=-1).reshape(
                bs, ne, ne, -1
            )  # bs x ns x no x 2ns1

        attn_mask = torch.cat((mask_ter, m1, m2), dim=-1).reshape(
            bs, ne, ne, -1
        )  # bs x ns x no x (1+nc)

        if self.args.attn_self:
            attn_self = torch.zeros((bs, ne, ne, 1), device=self.args.device).bool()
            attn_mask = torch.cat(
                (attn_self, attn_mask), axis=-1
            )  # bs x ns x no x (2+nc)

        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-1)  # bs x ns x no x nc

        output = torch.einsum(
            "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
        )  # bs x ns x no x dm

        output = self.dropout(self.fc_r(output)) + res
        output = self.layernorm_r(output)

        return output

    def update_rel_triple(self, rel_reprs, factors, ent_numbers):
        """
        rel_reprs: bs x ns x no x d
        factor_ter: bs x ns x no x d
        factors for
        sib:
                ha: bs x ns x no x no1 x d; hb: bs x ns x no1 x no x d
        cop:
                ha: bs x ns x ns1 x no x d; hb: bs x ns1 x ns x no x d
        gp:
                ha: bs x ns x no x no1 x d; hb: bs x ns1 x ns x no x d
        """
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)

        factor_ter, factor1, factor2, factor3 = factors
        res = rel_reprs
        bs, ne, _, _ = rel_reprs.shape

        ter_ht_new = factor_ter.unsqueeze(-2)  # bs x ns x no x 1 x dm
        # sib
        f1_ha1 = factor1  # bs x ns x no x no1 x d	sib
        f1_hb1 = factor1.permute(0, 1, 3, 2, 4)
        # cop
        f2_ha1 = factor2.permute(0, 1, 3, 2, 4)  # bs x ns x no x ns1 x d	cop
        f2_hb1 = factor2.permute(0, 2, 3, 1, 4)
        # gp
        f3_ha1 = factor3  # bs x ns x no x no1 x d	gp
        f3_hb1 = factor3.permute(0, 2, 3, 1, 4)
        # b1_ha1 = factor_b1							# bs x ns x no x no1 x d
        # b1_hb1 = factor_b1.permute(0,1,3,2,4)		# bs x ns x no x no1 x d
        # b2_ha1 = factor_b2.permute(0,1,3,2,4)		# bs x ns x no x ns1 x d
        # b2_hb1 = factor_b2.permute(0,2,3,1,4)		# bs x ns x no x ns1 x d
        # b3_ha1 = factor_b3							# bs x ns x no x no1 x d
        # b3_hb1 = factor_b3.permute(0,2,3,1,4)		# bs x ns x no x ns1 x d

        b1_h = torch.cat(
            (f1_ha1, f1_hb1), dim=-2
        )  # bs x ns x no x nc1	nc1=2no1 sib; 2ns1 cop; no1+ns1 gp;
        b2_h = torch.cat(
            (f2_ha1, f2_hb1), dim=-2
        )  # bs x ns x no x nc2	nc2=2ns1 cop; no1+ns1 gp;
        b3_h = torch.cat((f3_ha1, f3_hb1), dim=-2)
        ht_new = torch.cat(
            [b1_h, b2_h, b3_h], dim=-2
        )  # bs x ns x no x nc, nc= 2no1+2ns1 sibcop; 3no1+ns1 sibgp; 3ns1+no1 copgp

        total_h = torch.cat([ter_ht_new, ht_new], dim=-2)  # bs x ns x no x 1+nc x d

        if self.args.attn_self:
            ht = self.proj_r(rel_reprs).unsqueeze(-2)  # bs x ns x no x 1 x dm
            total_h = torch.cat([ht, total_h], dim=-2)  # bs x ns x no x (1+nc) x dm

        ht = (
            rel_reprs.unsqueeze(-2).repeat(1, 1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ns x no x nc x dm,
        comb = torch.cat([ht, total_h], dim=-1)  # bs x ns x no x nc x 2*dm

        energy = self.attn_combine_r(comb)  # ns x no x nc x dm
        energy = self.rv(energy).squeeze(-1)  # ns x no x nc

        # attn mask
        batch_mask3d = get_ent_mask3d(ent_numbers)  # bs x ns x no x no1
        mask_ter = ~batch_mask.unsqueeze(-1)  # bs x ns x no x 1

        # sib
        m1 = (
            torch.eye(ne, dtype=torch.int, device=self.args.device)
            .unsqueeze(0)
            .repeat(bs, 1, 1)
        )  # bs x no x no1	, r(i,j) not att to r(i,j)
        m1 = m1.unsqueeze(-3).repeat(1, ne, 1, 1).bool()  # bs x ns x no x no1
        m1 = (m1 + ~batch_mask3d).bool()
        m1 = torch.cat((m1, m1), dim=-1).reshape(bs, ne, ne, -1)  # bs x ns x no x 2no1
        # cop
        m2 = (
            torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        )  # bs x ns x ns1
        m2 = m2.unsqueeze(-2).repeat(1, 1, ne, 1).bool()  # bs x ns x no x ns1
        m2 = (m2 + ~batch_mask3d).bool()
        m2 = torch.cat((m2, m2), dim=-1).reshape(bs, ne, ne, -1)  # bs x ns x no x 2ns1
        # gp
        m3 = ~batch_mask3d
        m3 = torch.cat((m3, m3), dim=-1).reshape(bs, ne, ne, -1)  # bs x ns x no x 2no

        attn_mask = torch.cat((mask_ter, m1, m2, m3), dim=-1).reshape(
            bs, ne, ne, -1
        )  # bs x ns x no x (1+nc)

        if self.args.attn_self:
            attn_self = torch.zeros((bs, ne, ne, 1), device=self.args.device).bool()
            attn_mask = torch.cat(
                (attn_self, attn_mask), axis=-1
            )  # bs x ns x no x (2+nc)

        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-1)  # bs x ns x no x nc
        output = torch.einsum(
            "bijk,bijkd->bijd", attention, total_h.to(attention.dtype)
        )  # bs x ns x no x dm

        output = self.dropout(self.fc_r(output)) + res
        output = self.layernorm_r(output)

        return output

    def update_sub(self, sub_reprs, factor_ter, ent_numbers):
        """
        sub_reprs: bs x ns x dm
        factor_ter: bs x ns x no x dm
        """
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        res = sub_reprs
        bs, ne, _, dm = factor_ter.shape
        ht = self.proj_s(res)
        total_h = (
            torch.cat([ht.unsqueeze(-2), factor_ter], dim=-2)
            if self.args.attn_self
            else factor_ter
        )  # ns x (1+no) x dm or ns x no x dm

        # total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
        # total_h = factor_ter
        ht = (
            sub_reprs.unsqueeze(-2).repeat(1, 1, total_h.shape[-2], 1).contiguous()
        )  # bs x ns x no x dm
        comb = torch.cat([ht, total_h], dim=-1)  # bs x ns x no x dm+dr

        energy = self.attn_combine_s(comb)  # bs x ns x no x dm
        energy = self.sv(energy).squeeze(-1)  # bs x ns x no
        attn_mask = (
            torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        )  # bs x ns x ns
        attn_mask = (attn_mask + ~batch_mask).bool()
        if self.args.attn_self:
            attn_self = torch.zeros(
                (bs, ne, 1), device=self.args.device
            ).bool()  #  bs x ne x 1
            attn_mask = torch.cat((attn_self, attn_mask), axis=-1)
        # attn_mask = torch.sum(ht, dim=-1) == 0							# ns x (no+1)
        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-1)  # ns x no
        output = torch.einsum(
            "bij,bijd->bid", attention, total_h.to(attention.dtype)
        )  # ns x dm
        output = self.dropout(self.fc_s(output)) + res
        output = self.layernorm_s(output)

        return output

    def update_obj(self, obj_reprs, factor_ter, ent_numbers):
        """
        factor_ter: bs x ns x no x dm
        obj_reprs: bs x no x dm
        """
        batch_mask = get_ent_mask2d(ent_numbers).to(self.args.device)
        res = obj_reprs
        bs, ne, _, dm = factor_ter.shape
        # total_h = torch.cat([ht.unsqueeze(-2), ht_new], dim=-2)  		# ns x (no+1) x dm
        ht = self.proj_o(res)
        total_h = (
            torch.cat([ht.unsqueeze(-3), factor_ter], dim=-3)
            if self.args.attn_self
            else factor_ter
        )  # bs x (1+ne) x ne x dm or bs x ne x ne x dm

        ht = obj_reprs.unsqueeze(-3).repeat(1, total_h.shape[-3], 1, 1).contiguous()
        comb = torch.cat([ht, total_h], dim=-1)  # bs x nc x ne x dm+do

        energy = self.attn_combine_o(comb)  # bs x nc x ne x dm
        energy = self.ov(energy).squeeze(-1)  # bs x nc x ne
        attn_mask = (
            torch.eye(ne, device=self.args.device).unsqueeze(0).repeat(bs, 1, 1)
        )  # bs x ne x ne
        attn_mask = (attn_mask + ~batch_mask).bool()
        if self.args.attn_self:
            attn_self = torch.zeros(
                (bs, 1, ne), device=self.args.device
            ).bool()  # bs x 1 x ne
            attn_mask = torch.cat((attn_self, attn_mask), axis=-2)  # bs x (1+ne) x ne
        energy = energy.masked_fill(attn_mask, -1e4)
        attention = energy.softmax(dim=-2)  # bs x (1+ne) x ne
        output = torch.einsum(
            "bij,bijd->bjd", attention, total_h.to(attention.dtype)
        )  # bs x no x dm
        output = self.dropout(self.fc_o(output)) + res
        output = self.layernorm_o(output)  # no x dm

        return output


class LinearMessegePasser(Module):
    def __init__(self, sender_dim, receiver_dim):
        super().__init__()
        self.s_dim = sender_dim
        self.r_dim = receiver_dim
        self.net = Linear(self.s_dim + self.r_dim, self.r_dim)

    def forward(self, x_s, x_r):
        return self.net(torch.cat([x_s, x_r], dim=-1))


# ---------------------------------------------------------------------------
# Model registry — single source of truth for model_type → (config, model, tokenizer)
# ---------------------------------------------------------------------------

MODEL_CLASSES: dict[str, tuple] = {
    "hyper": (BertConfig, BertForHyperGNN, AutoTokenizer),
    "modernberthyper": (ModernBertConfig, ModernBertForHyperGNN, AutoTokenizer),
}
