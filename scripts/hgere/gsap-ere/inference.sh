GPU_ID=0
DATASET_DATE=2025-05-19
for seed in 45; do # 46 47 48 49
for epochs in 8; do # 6 8 10
for bs in 22 ; do # 10 14 18 22
# classifier above bert layer (common to set higher. The weights are randomly initialized.)
for lr_cls in 2e-5 ; do # 1e-4 5e-5 2e-5 1e-5
# bert layer
for lr in 2e-5 ; do # 1e-4 5e-5 2e-5 1e-5
for loss_weight in 0.9 ; do # 0.1 0.25 0.5.0.75 0.9
# best loss weight: 0.9
# Optimization made by @xxx  We could show, that the NER classifier learns much faster than the REL classifier. We emphasize the loss for the relations with a weighting factore. (1-L) * loss_ner + L * loss_re
for seq in 512; do
for entdim in 400; do
for reldim in 400; do
for memdim in 400; do
for facenc in biaf; do 
for factor in ternary ; do # ternary tersibcop
for iter in 3; do
for eps in 1e-8; do
NICK_NAME=pinguin-seed${seed}
#--do_train --do_eval \
# --shuffle \
#--do_train \
#--eval_train \
#--eval_dev \
#--train_file ent_pred_train.json \
#--dev_file {DATASET_DATent_pred_dev.json \
OUTPUT_DIR=/home/groups/gsap/gsap-ere/models/hgere/default
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  \
	--project_name gsap-rel-hgere \
	--run_name $NICK_NAME \
	--ner_prediction_dir  /home/groups/gsap/gsap-ere/models/pruner/default/ \
	--test_file ent_pred_${DATASET_DATE}_test_filtered_2.json  \
	--loss_re_weight_alpha $loss_weight \
	--output_dir $OUTPUT_DIR \
	--log_wandb \
	--shuffle \
	--eval_test \
	--preload_dataset \
	--model_name_or_path  pretrained_models/scibert_scivocab_uncased \
	--model_type hyper  \
	--do_lower_case  \
	--learning_rate $lr  \
	--learning_rate_cls $lr_cls \
	--num_train_epochs $epochs \
	--eval_epochs 1 \
	--per_gpu_train_batch_size $bs \
	--per_gpu_eval_batch_size $bs \
	--gradient_accumulation_steps 1  \
	--max_seq_length $seq  \
	--max_pair_length 18  \
	--adam_epsilon $eps \
	--evaluate_during_training \
       	--eval_all_checkpoints  \
	--seed $seed   \
	--overwrite_output_dir  \
	--factor_type $factor  \
	--iter $iter   \
	--factor_encoder $facenc \
	--ent_dim $entdim \
	--rel_dim $reldim  \
	--mem_dim $memdim  \
	--layernorm  \
	--layernorm_1st  \
	--attn_self \
	--local_rank -1 \
	--fp16
done;
done;
done;
done;
done;
done;
done;
done;
done;
done;
done;
done;
done;
done;
