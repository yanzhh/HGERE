# Pruner parameter reference

> **Auto-generated** from `hgere.span_classifier.config.PrunerTrainConfig`.
> Do not edit by hand — run `uv run generate-pruner-docs` to regenerate.

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
| `model_dir` | `--model_dir` | `string` | **required** | Directory where model checkpoints will be written. |
| `output_dir` | `--output_dir` | `string` *(optional)* | `null` | Directory where pruner prediction output (enriched dataset files) will be written. Pass this path as ner_prediction_dir when training HGERE. Defaults to model_dir if not set. |
| `base_model_name_or_path` | `--base_model_name_or_path` | `string` | **required** | Transformer model path or HuggingFace name. |
| `model_type` | `--model_type` | `string` | `"bertspanmarkerpruner"` | Pruner architecture key (bertspanmarkerpruner \| modernbertspanmarkerpruner). |
| `do_lower_case` | `--do_lower_case` | `boolean` | `true` | Lowercase input tokens (true for uncased models). |
| `per_gpu_eval_batch_size` | `--per_gpu_eval_batch_size` | `integer` | `32` | Evaluation batch size per GPU (used during training eval). |
| `max_seq_length` | `--max_seq_length` | `integer` | `256` | Maximum tokenised sequence length. |
| `max_pair_length` | `--max_pair_length` | `integer` | `64` | Maximum number of span pairs per sequence. |
| `max_mention_ori_length` | `--max_mention_ori_length` | `integer` | `12` | Maximum span width in tokens. |
| `alpha` | `--alpha` | `float` | `1.0` | Loss scale alpha. |

## Training parameters (`train_params`)

These fields live under `train_params:` in the YAML and are ignored at inference time.  On the CLI they are prefixed with `--train_params__` (e.g. `--train_params__learning_rate`).

### Data

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `data_dir` | `--train_params__data_dir` | `string` | **required** | Directory containing train/dev/test split files. |
| `train_file` | `--train_params__train_file` | `string` | `"train.jsonl"` | Training split filename inside data_dir. |
| `dev_file` | `--train_params__dev_file` | `string` | `"dev.jsonl"` | Dev split filename inside data_dir. |
| `test_file` | `--train_params__test_file` | `string` | `"test.jsonl"` | Test split filename inside data_dir. |
| `rulebased_pruner_file` | `--train_params__rulebased_pruner_file` | `string` *(optional)* | `null` | Path to a rule-based pruner JSON (from eval-rulebased-pruner). Spans matching its patterns are filtered before the neural pruner sees them. |

### Optimisation

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `seed` | `--train_params__seed` | `integer` | `42` | Random seed for reproducibility. |
| `learning_rate` | `--train_params__learning_rate` | `float` | **required** | Learning rate for BERT layers. |
| `learning_rate_span` | `--train_params__learning_rate_span` | `float` | `-1` | Learning rate for the span encoder. -1 = use learning_rate. |
| `num_train_epochs` | `--train_params__num_train_epochs` | `integer` | **required** | Total number of training epochs. |
| `eval_epochs` | `--train_params__eval_epochs` | `integer` | `1` | Evaluate every N epochs. Set to -1 to use save_steps instead. |
| `per_gpu_train_batch_size` | `--train_params__per_gpu_train_batch_size` | `integer` | **required** | Training batch size per GPU. |
| `gradient_accumulation_steps` | `--train_params__gradient_accumulation_steps` | `integer` | `1` | Gradient accumulation steps before a weight update. |
| `adam_epsilon` | `--train_params__adam_epsilon` | `float` | `1e-08` | Epsilon for the Adam optimiser. |
| `weight_decay` | `--train_params__weight_decay` | `float` | `0.0` | Weight decay coefficient. |
| `max_grad_norm` | `--train_params__max_grad_norm` | `float` | `1.0` | Max gradient norm. |
| `max_steps` | `--train_params__max_steps` | `integer` | `-1` | If > 0: set total number of training steps. Overrides num_train_epochs. |
| `warmup_steps` | `--train_params__warmup_steps` | `integer` | `-1` | Linear warmup over warmup_steps. |
| `logging_steps` | `--train_params__logging_steps` | `integer` | `5` | Log every N update steps. |
| `save_steps` | `--train_params__save_steps` | `integer` | `1000` | Save a checkpoint every N update steps. |
| `save_total_limit` | `--train_params__save_total_limit` | `integer` | `1` | Limit total checkpoints; deletes older ones. |
| `fp16` | `--train_params__fp16` | `boolean` | `false` | Use mixed-precision (fp16) training. |
| `local_rank` | `--train_params__local_rank` | `integer` | `-1` | Local rank for distributed training (-1 = single GPU). |

### Hardware

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `no_cuda` | `--train_params__no_cuda` | `boolean` | `false` | Avoid using CUDA when available. |
| `server_ip` | `--train_params__server_ip` | `string` | `""` | For distant debugging. |
| `server_port` | `--train_params__server_port` | `string` | `""` | For distant debugging. |
| `debug_overflow` | `--train_params__debug_overflow` | `boolean` | `false` | Enable DebugUnderflowOverflow to locate first NaN/Inf. |

### Loss

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `pruner_loss` | `--train_params__pruner_loss` | `"bce"` | `"focal"` | `"bce"` | Loss function: 'bce' = BCEWithLogitsLoss, 'focal' = focal loss. |
| `focal_gamma` | `--train_params__focal_gamma` | `float` | `2.0` | Focusing parameter γ for focal loss. |
| `focal_alpha` | `--train_params__focal_alpha` | `float` | `0.25` | Class-balance factor α for focal loss. |

### Candidate filtering (eval during training)

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `topk_ratio` | `--train_params__topk_ratio` | `float` | `0.5` | K = clamp(topk_ratio × sent_len, min_mentions_num, max_mentions_num). |
| `min_mentions_num` | `--train_params__min_mentions_num` | `integer` | `3` | Minimum spans to keep per sentence. |
| `max_mentions_num` | `--train_params__max_mentions_num` | `integer` | `18` | Maximum spans to keep per sentence. |

### Model flags

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `onedropout` | `--train_params__onedropout` | `boolean` | `false` | Share a single dropout mask across the span encoder. |
| `lminit` | `--train_params__lminit` | `boolean` | `false` | Initialise span boundary embeddings from LM output. |
| `nocross` | `--train_params__nocross` | `boolean` | `false` | Disable cross-sentence span candidates. |
| `biaf_span` | `--train_params__biaf_span` | `boolean` | `false` | Use biaffine span representation. |
| `biaf_mode` | `--train_params__biaf_mode` | `integer` | `3` | Biaffine span repr mode. |
| `biaf_factorize` | `--train_params__biaf_factorize` | `boolean` | `false` | Factorize the biaffine span matrix. |
| `span_hidden_size` | `--train_params__span_hidden_size` | `integer` | `768` | Hidden size for the span encoder. |
| `rank` | `--train_params__rank` | `integer` | `768` | Rank for biaffine factorization. |
| `span_size` | `--train_params__span_size` | `integer` | `256` | Output size of the span representation. |

### Evaluation & checkpointing

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `evaluate_during_training` | `--train_params__evaluate_during_training` | `boolean` | `true` | Run evaluation on dev set after every checkpoint. |
| `eval_all_checkpoints` | `--train_params__eval_all_checkpoints` | `boolean` | `false` | Evaluate all saved checkpoints at the end of training. |
| `overwrite_model_dir` | `--train_params__overwrite_model_dir` | `boolean` | `false` | Allow overwriting an existing model directory. |
| `overwrite_cache` | `--train_params__overwrite_cache` | `boolean` | `false` | Overwrite the cached training and evaluation sets. |

### Run modes

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `do_train` | `--train_params__do_train` | `boolean` | `true` | Whether to run training. |
| `do_test` | `--train_params__do_test` | `boolean` | `true` | Run evaluation on the test set after training. |
| `output_results` | `--train_params__output_results` | `boolean` | `true` | Persist predictions to disk after evaluation. |
| `shuffle` | `--train_params__shuffle` | `boolean` | `false` | Shuffle training data. |

### Eval settings

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `target_recall_diff` | `--train_params__target_recall_diff` | `float` | `0.01` | Gap below pool upper-bound recall used as analysis target during eval (e.g. 0.01 = 1%). Controls threshold/top-K search logged to wandb. |
| `prune_config` | `--train_params__prune_config` | `string` *(optional)* | `null` | Path to a best_config.json produced by evaluation/threshold_analysis.py. When set, the pruning parameters are loaded from this file instead of being estimated from the dev set at the end of training. |
| `use_full_layer` | `--train_params__use_full_layer` | `integer` | `-1` | If >= 0, use all hidden states up to this layer for span repr. |

### Weights & Biases

| Parameter | CLI flag | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `project_name` | `--train_params__project_name` | `string` | `"hgere-pruner"` | Weights & Biases project name. |
| `run_name` | `--train_params__run_name` | `string` *(optional)* | `null` | Weights & Biases run name. |
