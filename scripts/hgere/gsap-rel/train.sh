GPU_ID=0
DATASET_DATE=2025-05-19
ANNOTATOR_TRAIN=-suyash
ANNOTATOR_TRAIN=
ANNOTATOR_TEST=_Alica
ANNOTATOR_TEST=
for seed in 45 46 47 48 49; do
#for epoch in 10; do
for epochs in 8; do
for bs in 22 ; do
# classifier above bert layer (common to set higher. The weights are randomly initialized.)
for lr_cls in 2e-5 ; do
# bert layer
for lr in 2e-5 ; do 
# best loss weight: 0.9
# Optimization made by @ottowg (GSAP) We could show, that the NER classifier learns much faster than the REL classifier. We emphasize the loss for the relations with a weighting factore. (1-L) * loss_ner + L * loss_re
for loss_weight in 0.9 ; do
for seq in 512; do
for entdim in 400; do
for reldim in 400; do
for memdim in 400; do
for facenc in biaf; do
# original: tersibcop
for factor in ternary ; do
for iter in 3; do
for eps in 1e-8; do
NICK_NAME=iget-seed${seed}
#--do_train --do_eval \
# --shuffle \
# --model_name_or_path  pretrained_models/modernbert_base \
OUTPUT_DIR=saves/gsap/HGERE/scibert/${DATASET_DATE}-${NICK_NAME}${ANNOTATOR_TRAIN}
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  \
	--project_name gsap-rel-hgere \
	--run_name $NICK_NAME \
	--loss_re_weight_alpha $loss_weight \
	--output_dir $OUTPUT_DIR \
	--log_wandb \
	--shuffle \
	--do_train \
	--eval_train \
	--eval_dev \
	--eval_test \
	--preload_dataset \
	--ner_prediction_dir  saves/gsap/pruned_ner/$DATASET_DATE/ \
	--train_file ent_pred_${DATASET_DATE}${ANNOTATOR_TEST}_train.json \
	--dev_file ent_pred_${DATASET_DATE}${ANNOTATOR_TEST}_dev.json \
	--test_file ent_pred_${DATASET_DATE}${ANNOTATOR_TEST}_test.json  \
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
	--adam_epsilon $eps  \
	--evaluate_during_training   --eval_all_checkpoints  \
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
#--fp16  \
#--no_sym
