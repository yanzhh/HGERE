# HGERE parameter reference

> **Auto-generated** from `hgere.hgere.config.HGERETrainConfig`.
> Do not edit by hand — run `uv run generate-pruner-docs --model hgere` to regenerate.

## Schema versioning

| Key | Value |
|-----|-------|
| Current version | `1.0` |
| Supported versions | `1.0` |

Add `schema_version: "1.0"` to your YAML config. An unsupported version raises a clear error at load time.

## Shared parameters

These fields live at the top level of the config and are used both at inference time (by the pipeline) and at training time.

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `label_set` | `--label_set` | `string` | **required** | Label set for entity/relation types (e.g. gsap, scier, scinlp). |
| `model_dir` | `--model_dir` | `string` | **required** | Directory where checkpoints will be written. |
| `base_model_name_or_path` | `--base_model_name_or_path` | `string` | **required** | Transformer model path or HuggingFace name. |
| `ner_prediction_dir` | `--ner_prediction_dir` | `string` | **required** | Input data directory containing pruner output files for HGERE. |
| `model_type` | `--model_type` | `string` | `"hyper"` | HGERE architecture key (hyper \| modernberthyper). |
| `do_lower_case` | `--do_lower_case` | `boolean` | `false` | Lowercase input tokens (true for uncased models). |
| `per_gpu_eval_batch_size` | `--per_gpu_eval_batch_size` | `integer` | `8` | Evaluation batch size per GPU (used during training eval). |
| `max_seq_length` | `--max_seq_length` | `integer` | `384` | Maximum tokenised sequence length. |
| `max_pair_length` | `--max_pair_length` | `integer` | `64` | Maximum number of span pairs per sequence. |
| `alpha` | `--alpha` | `float` | `1.0` | Loss scale alpha. |
| `no_sym` | `--no_sym` | `boolean` | `false` | Disable symmetric relation labels. |
| `ent_repr` | `--ent_repr` | `string` | `"mix"` | Entity representation source: sub \| obj \| mix. |
| `ent_enc` | `--ent_enc` | `string` | `"cat"` | Entity encoder type. |
| `ner_cls` | `--ner_cls` | `string` | `"cat"` | NER classifier type. |
| `rel_enc` | `--rel_enc` | `string` | `"cat"` | Relation encoder type. |
| `ent_dim` | `--ent_dim` | `integer` | `200` | Entity dimension. |
| `rel_dim` | `--rel_dim` | `integer` | `200` | Relation dimension. |
| `rel_rank` | `--rel_rank` | `integer` | `200` | Rank for biaffine factorisation. |
| `rel_factorize` | `--rel_factorize` | `boolean` | `false` | Use biaffine relation factorisation. |
| `factor_type` | `--factor_type` | `string` | `"ternary"` | Factor type for HyperGNN. |
| `mem_dim` | `--mem_dim` | `integer` | `200` | Memory dimension for HyperGNN. |
| `n_iter` | `--n_iter` | `integer` | `3` | Number of HyperGNN iterations. Maps to --iter in run_hgnn.py. |
| `factor_encoder` | `--factor_encoder` | `string` | `"cat"` | Factor encoder type. |
| `iter1` | `--iter1` | `integer` | `1` | Number of first-order iterations. |
| `lminit` | `--lminit` | `boolean` | `false` | Initialise span boundary embeddings from LM output. |
| `nocross` | `--nocross` | `boolean` | `false` | Disable cross-sentence span candidates. |
| `att_left` | `--att_left` | `boolean` | `false` | Use left attention. |
| `att_right` | `--att_right` | `boolean` | `false` | Use right attention. |
| `use_ner_results` | `--use_ner_results` | `boolean` | `false` | Use NER results during relation extraction. |
| `use_typemarker` | `--use_typemarker` | `boolean` | `false` | Use type markers in the input. |
| `layernorm` | `--layernorm` | `boolean` | `false` | Apply layer normalisation. |
| `layernorm_1st` | `--layernorm_1st` | `boolean` | `false` | Apply layer normalisation for first-order. |
| `attn_self` | `--attn_self` | `boolean` | `false` | Use self-attention. |
| `aggregate_type` | `--aggregate_type` | `string` | `"attn"` | Aggregation type: attn or test. |
| `aggregate_func` | `--aggregate_func` | `string` | `"max"` | Aggregation function: max or sum. |
| `agg_with_self` | `--agg_with_self` | `boolean` | `false` | Aggregate with self node. |
| `fix_obj` | `--fix_obj` | `boolean` | `false` | Fix object representation. |
| `edgetype` | `--edgetype` | `string` | `"sib"` | Edge type for HTNN. |
| `attn_scorer` | `--attn_scorer` | `string` | `"biaf"` | Attention scorer type. |
| `attn_res` | `--attn_res` | `boolean` | `false` | Use residual in attention scorer. |
| `n_head` | `--n_head` | `integer` | `8` | Number of attention heads. |
| `d_head` | `--d_head` | `integer` | `32` | Dimension per attention head. |
| `re_focal_loss` | `--re_focal_loss` | `boolean` | `false` | Use focal loss for relation classification. |
| `re_focal_gamma` | `--re_focal_gamma` | `float` | `2.0` | Focusing parameter γ for RE focal loss. |
| `ner_focal_loss` | `--ner_focal_loss` | `boolean` | `false` | Use focal loss for NER classification. |
| `ner_focal_gamma` | `--ner_focal_gamma` | `float` | `2.0` | Focusing parameter γ for NER focal loss. |
| `uni_ent` | `--uni_ent` | `boolean` | `false` | Use uniform entity representation (same repr for sub/obj). |
| `pred_sub` | `--pred_sub` | `boolean` | `false` | Predict subject. |
| `eval_logits` | `--eval_logits` | `boolean` | `false` | Decode with non-normalised logits. |
| `eval_logsoftmax` | `--eval_logsoftmax` | `boolean` | `false` | Decode with log-softmax. |
| `eval_softmax` | `--eval_softmax` | `boolean` | `false` | Decode with softmax. |
| `eval_unidirect` | `--eval_unidirect` | `boolean` | `false` | Evaluate with unidirectional relations. |
| `baseline` | `--baseline` | `string` | `"firstorder"` | Baseline method. |

## Training parameters (`train_params`)

These fields live under `train_params:` in the YAML and are ignored at inference time.  On the CLI they are prefixed with `--train_params__` (e.g. `--train_params__learning_rate`).

### Data

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `train_file` | `--train_params__train_file` | `string` | `"train.json"` | Training split filename inside ner_prediction_dir. |
| `dev_file` | `--train_params__dev_file` | `string` | `"dev.json"` | Dev split filename inside ner_prediction_dir. |
| `test_file` | `--train_params__test_file` | `string` | `"test.json"` | Test split filename inside ner_prediction_dir. |

### Optimisation

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `seed` | `--train_params__seed` | `integer` | `42` | Random seed for reproducibility. |
| `learning_rate` | `--train_params__learning_rate` | `float` | **required** | Learning rate for BERT layers. |
| `learning_rate_cls` | `--train_params__learning_rate_cls` | `float` | `-1` | Learning rate for layers beyond BERT. -1 = use learning_rate. |
| `num_train_epochs` | `--train_params__num_train_epochs` | `float` | **required** | Total number of training epochs to perform. |
| `per_gpu_train_batch_size` | `--train_params__per_gpu_train_batch_size` | `integer` | **required** | Training batch size per GPU. |
| `gradient_accumulation_steps` | `--train_params__gradient_accumulation_steps` | `integer` | `1` | Gradient accumulation steps before a weight update. |
| `adam_epsilon` | `--train_params__adam_epsilon` | `float` | `1e-08` | Epsilon for the Adam optimiser. |
| `weight_decay` | `--train_params__weight_decay` | `float` | `0.0` | Weight decay coefficient. |
| `max_grad_norm` | `--train_params__max_grad_norm` | `float` | `1.0` | Max gradient norm. |
| `max_steps` | `--train_params__max_steps` | `integer` | `-1` | If > 0: set total number of training steps. Overrides num_train_epochs. |
| `warmup_steps` | `--train_params__warmup_steps` | `integer` | `-1` | Linear warmup over warmup_steps. |
| `warmup_ratio` | `--train_params__warmup_ratio` | `float` | `0.1` | Linear warmup ratio (used if warmup_steps=-1). |
| `logging_steps` | `--train_params__logging_steps` | `integer` | `5` | Log every N update steps. |
| `save_steps` | `--train_params__save_steps` | `integer` | `1000` | Save a checkpoint every N update steps. |
| `eval_epochs` | `--train_params__eval_epochs` | `integer` | `-1` | Evaluate every N epochs. Set to -1 to use save_steps instead. |
| `save_total_limit` | `--train_params__save_total_limit` | `integer` | `1` | Limit total checkpoints; deletes older ones. |

### Hardware

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `no_cuda` | `--train_params__no_cuda` | `boolean` | `false` | Avoid using CUDA when available. |
| `fp16` | `--train_params__fp16` | `boolean` | `false` | Use mixed-precision (fp16) training. |
| `local_rank` | `--train_params__local_rank` | `integer` | `-1` | Local rank for distributed training (-1 = single GPU). |
| `server_ip` | `--train_params__server_ip` | `string` | `""` | For distant debugging. |
| `server_port` | `--train_params__server_port` | `string` | `""` | For distant debugging. |

### Loss

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `loss_re_weight_alpha` | `--train_params__loss_re_weight_alpha` | `float` | `0.5` | Weight the RE loss relative to NER loss. E.g. 0.7 => 0.7 RE loss + 0.3 NER loss. |
| `train_time_loss_weighting` | `--train_params__train_time_loss_weighting` | `boolean` | `false` | Enable dynamic NER→RE loss weighting over training. Alpha shifts via sigmoid. |
| `train_time_loss_turn` | `--train_params__train_time_loss_turn` | `float` | `0.5` | Fractional training progress [0, 1] at which the NER→RE weighting is at midpoint. |
| `train_time_loss_steepness` | `--train_params__train_time_loss_steepness` | `float` | `10.0` | Steepness of the sigmoid phase transition for dynamic loss weighting. |

### Evaluation & checkpointing

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `evaluate_during_training` | `--train_params__evaluate_during_training` | `boolean` | `false` | Run evaluation on dev set during training. |
| `eval_all_checkpoints` | `--train_params__eval_all_checkpoints` | `boolean` | `false` | Evaluate all saved checkpoints at the end of training. |
| `overwrite_output_dir` | `--train_params__overwrite_output_dir` | `boolean` | `false` | Allow overwriting an existing output directory. |
| `overwrite_cache` | `--train_params__overwrite_cache` | `boolean` | `false` | Overwrite the cached training and evaluation sets. |

### Run modes

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `do_train` | `--train_params__do_train` | `boolean` | `true` | Whether to run training. |
| `eval_train` | `--train_params__eval_train` | `boolean` | `false` | Run eval on the train set and save predictions. |
| `eval_dev` | `--train_params__eval_dev` | `boolean` | `true` | Run eval on the dev set and save predictions. |
| `eval_test` | `--train_params__eval_test` | `boolean` | `true` | Run eval on the test set and save predictions. |
| `no_test` | `--train_params__no_test` | `boolean` | `false` | Skip test set evaluation. |
| `save_results` | `--train_params__save_results` | `boolean` | `false` | Persist predictions to disk after evaluation. |

### Data loading

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `shuffle` | `--train_params__shuffle` | `boolean` | `false` | Shuffle training data. |
| `pre_filter_params` | `--train_params__pre_filter_params` | `typing.Annotated[typing.Union[hgere.span_classifier.config.ThresholdPreFilterParams, hgere.span_classifier.config.TopKPreFilterParams], FieldInfo(annotation=NoneType, required=True, discriminator='method')]` *(optional)* | `null` | Parameters for pre-filtering NER candidates. |
| `batch_by_size` | `--train_params__batch_by_size` | `boolean` | `false` | Sort sentences by entity count before batching. Mutually exclusive with shuffle. |
| `preload_dataset` | `--train_params__preload_dataset` | `boolean` | `false` | Preload dataset into memory. |

### Weights & Biases

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `project_name` | `--train_params__project_name` | `string` | `"hgere"` | Weights & Biases project name. |
| `run_name` | `--train_params__run_name` | `string` *(optional)* | `null` | Weights & Biases run name. |
| `log_wandb` | `--train_params__log_wandb` | `boolean` | `false` | Whether to log training in W&B. |
