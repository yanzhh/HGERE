# SciERC with scibert + rule-based pre-filter + focal loss
GPU_ID=0
DATADIR=datasets/scierc/
seed=43
for bs in 32; do
for minent in 1; do
for maxent in 30; do
for lr in 2e-5; do
#--do_train \

NICK_NAME=focal-prefilter-${lr}
MODEL_DIR=saves/scierc/pruner/${NICK_NAME}
OUTPUT_DIR=saves/scierc/pruner/${NICK_NAME}
# --prune_config ${OUTPUT_DIR} \
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_pruner.py  \
    --limit 10 \
    --pruner_loss focal \
    --rulebased-pruner-file saves/scierc/pruner_rulebased/scierc_rulebased.json \
    --project_name SciERC-Pruner \
    --run_name $NICK_NAME \
    --seed $seed \
    --data_dir $DATADIR  \
    --train_file train.json \
    --dev_file dev.json \
    --test_file test.json  \
    --label_set scierc \
    --output_dir $OUTPUT_DIR \
    --model_dir $MODEL_DIR \
    --overwrite_model_dir  \
    --output_results \
    --model_type bertspanmarkerpruner  \
    --base_model_name_or_path  pretrained_models/scibert_scivocab_uncased  \
    --do_lower_case  \
    --learning_rate $lr  \
    --num_train_epochs 3  \
    --eval_epochs 1   \
    --per_gpu_train_batch_size  $bs  \
    --per_gpu_eval_batch_size $bs \
    --gradient_accumulation_steps 1  \
    --max_seq_length 256  \
    --save_steps 1000  \
    --max_pair_length 64  \
    --max_mention_ori_length 12  \
    --topk_ratio 8 \
    --max_mentions_num $maxent \
    --min_mentions_num $minent \
    --do_train \
    --do_test  \
    --do_eval \
    --evaluate_during_training \
    --eval_all_checkpoints  \
    --onedropout  \
    --lminit  \
    --nocross \
    --biaf_span \
    --biaf_mode  2 \
    --biaf_factorize  \
    --span_hidden_size 768 \
    --rank 768  \
    --span_size 256 \
    --fp16 \
    --local_rank -1
done;
done;
done;
done;
