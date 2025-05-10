GPU_ID=3
DATASET_DATE=2025-04-15
NICK_NAME=qualle
ANNOTATOR_TRAIN=suyash
ANNOTATOR_TEST=_Alica
for seed in 42; do 
#for epoch in 10; do 
for epochs in 6; do 
for bs in 18; do
for lr in 2e-5; do
for lr1 in 2e-5; do
for seq in 512; do
for entdim in 400; do
for reldim in 400; do 
for memdim in 400; do
for facenc in biaf; do
for factor in tersibcop; do
for iter in 3; do
for eps in 1e-8; do
#--do_train --do_eval \
# --shuffle \
# --model_name_or_path  pretrained_models/modernbert_base \
# --loss_re_weight_alpha 0.9\
OUTPUT_DIR=saves/gsap/HGERE/scibert/${DATASET_DATE}-${NICK_NAME}-${ANNOTATOR_TRAIN}
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  \
	--project_name gsap-rel-hgere \
    --run_name $NICK_NAME \
    --output_dir $OUTPUT_DIR \
	--log_wandb \
	--shuffle \
    --do_test \
    --ner_prediction_dir  saves/gsap/pruned_ner/$DATASET_DATE/ \
    --train_file ent_pred_${DATASET_DATE}${ANNOTATOR_TEST}_train.json \
    --dev_file ent_pred_${DATASET_DATE}${ANNOTATOR_TEST}_dev.json \
    --test_file ent_pred_${DATASET_DATE}${ANNOTATOR_TEST}_test.json  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased \
    --model_type hyper  \
    --do_lower_case  \
    --learning_rate $lr  \
    --learning_rate_cls $lr1 \
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
    --local_rank -1
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
