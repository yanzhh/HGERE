# SciERC HGERE — train on focal pruner predictions (focal-prefilter-2e-5)
GPU_ID=0
PRUNER_DIR=pruner_predictions/scierc/focal-prefilter-2e-5
for seed in 43; do
for bs in 18; do
for lr in 2e-5; do
for lr_cls in 1e-4; do
for seq in 512; do
for weight_decay in 0.01; do
for entdim in 400; do
for reldim in 400; do
for memdim in 400; do
for facenc in biaf; do
for factor in tersibcop; do
for iter in 1; do
for eps in 1e-8; do
for turn in -; do # 0.2
for steepness in -; do #20
for epoch in 8; do
NICK_NAME=scierc-re-focal-iter_1
OUTPUT_DIR=saves/scierc/HGERE/$NICK_NAME
mkdir -p $OUTPUT_DIR
cp $0 $OUTPUT_DIR/train_script.sh
#--train_time_loss_weighting \
#    --train_time_loss_turn $turn \
#    --train_time_loss_steepness $steepness \
# --shuffle \
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  \
    --re_focal_loss \
    --project_name scierc-hgere \
    --run_name $NICK_NAME \
    --output_dir $OUTPUT_DIR \
    --log_wandb \
    --label_set scierc \
    --shuffle \
    --do_train \
    --eval_dev \
    --eval_test \
    --ner_prediction_dir $PRUNER_DIR \
    --train_file ent_pred_trai.json \
    --dev_file ent_pred_de.json \
    --test_file ent_pred_tes.json \
    --model_type hyper \
    --model_name_or_path pretrained_models/scibert_scivocab_uncased \
    --do_lower_case \
    --weight_decay $weight_decay \
    --max_grad_norm 1.0 \
    --learning_rate $lr \
    --learning_rate_cls $lr_cls \
    --num_train_epochs $epoch \
    --eval_epochs 1 \
    --per_gpu_train_batch_size $bs \
    --per_gpu_eval_batch_size $bs \
    --gradient_accumulation_steps 1 \
    --max_seq_length $seq \
    --max_pair_length 18 \
    --adam_epsilon $eps \
    --evaluate_during_training --eval_all_checkpoints \
    --seed $seed \
    --overwrite_output_dir \
    --factor_type $factor \
    --iter $iter \
    --factor_encoder $facenc \
    --ent_dim $entdim \
    --rel_dim $reldim \
    --mem_dim $memdim \
    --layernorm \
    --layernorm_1st \
    --attn_self \
    --fp16 \
    --local_rank -1
done;
#--no_sym \
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
