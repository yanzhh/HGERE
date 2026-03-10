GPU_ID=0
NICK_NAME=fuchs
for seed in 43; do 
#for epoch in 10; do 
for epoch in 10; do 
for bs in 18; do
for lr in 2e-5; do
for lr_class in 0; do
for seq in 512; do
for weight_decay in 0.01; do
for entdim in 400; do
for reldim in 400; do 
for memdim in 400; do
for facenc in biaf; do
for factor in tersibcop; do
for iter in 3; do
for eps in 1e-8; do
for loss_re_weight_alpha in 0.7; do
#--do_train --do_eval \
NER_PREDICTION_DIR=saves/comparison/model_scier/data_scinlp/
OUTPUT_DIR=saves/scier/HGERE/${NICK_NAME}_${lr}
echo $OUTPUT_DIR
#--eval_test \
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  \
    --project_name scier-hgere \
    --run_name ${NICK_NAME}_${lr} \
    --output_dir $OUTPUT_DIR \
    --log_wandb \
    --loss_re_weight_alpha $loss_re_weight_alpha \
    --label_set scier \
    --shuffle \
    --eval_train \
    --eval_dev \
    --eval_test \
    --ner_prediction_dir  $NER_PREDICTION_DIR \
    --train_file ent_pred_de.json \
    --dev_file ent_pred_tes.json \
    --test_file ent_pred_trai.json \
    --model_type hyper  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased \
    --do_lower_case  \
    --weight_decay $weight_decay \
    --max_grad_norm 1.0 \
    --learning_rate $lr  \
    --learning_rate_cls $lr \
    --num_train_epochs $epoch \
    --eval_epochs 1 \
    --per_gpu_train_batch_size  $bs \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 1  \
    --max_seq_length $seq  \
    --max_pair_length 18  \
    --adam_epsilon $eps  \
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
	--no_sym \
	--fp16 \
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
done;
done;
#--fp16  \
