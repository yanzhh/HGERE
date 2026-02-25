# SciER with scibert
# set cuda visible devices here
GPU_ID=2
NICK_NAME=papagei
# directory of preprocessed dataset
DATADIR=/home/ottowg/projects/gsap/related_datasets/SciER/PLM/

for seed in 44; do 
for minent in 3; do
for maxent in 16; do
for lr in 2e-5; do
for epochs in 8; do
MODEL_DIR=saves/scier/pruner/${NICK_NAME}
OUTPUT_DIR=saves/scier/pruner/${NICK_NAME}
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_pruner.py  \
    --seed $seed \
    --data_dir $DATADIR  \
    --train_file train.jsonl \
    --dev_file dev.jsonl \
    --test_file test.jsonl  \
    --output_dir $OUTPUT_DIR \
    --model_dir $MODEL_DIR \
    --overwrite_output_dir  \
    --output_results \
    --model_type bertspanmarkerpruner  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased  \
    --do_lower_case  \
    --learning_rate $lr  \
    --num_train_epochs 8  \
    --eval_epochs 1   \
    --per_gpu_train_batch_size  9  \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 1  \
    --max_seq_length 256  \
    --save_steps 1000  \
    --max_pair_length 64  \
    --max_mention_ori_length 8  \
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
done;
#--fp16
