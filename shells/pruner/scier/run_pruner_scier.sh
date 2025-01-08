# SciERC with scibert
# set cuda visible devices here
GPU_ID=3
# directory of preprocessed dataset
DATADIR=/home/ottowg/projects/gsap/gsap-rel/preprocessing/gsap-rel-sentence-simple/  

for seed in 44; do 
for minent in 3; do
for maxent in 16; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  wolf_run_pruner.py  \
    --seed $seed \
    --data_dir $DATADIR  \
    --train_file train_debug.jsonl \
    --dev_file dev_debug.jsonl \
    --test_file test_debug.jsonl  \
    --output_dir saves/reproduce/gsap_models/pruner/biafencoder-spanlen12-rank768-hid768-span256-entnum$minent-$maxent-lr2e-5-epochs8/scierc_scibert-$seed  \
    --overwrite_output_dir  \
    --output_results \
    --model_type bertspanmarkerpruner  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased  \
    --do_lower_case  \
    --learning_rate 2e-5  \
    --num_train_epochs 8  \
    --eval_epochs 1   \
    --per_gpu_train_batch_size  8  \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 1  \
    --max_seq_length 256  \
    --save_steps 1000  \
    --max_pair_length 64  \
    --max_mention_ori_length 12  \
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
    --local_rank -1 \
    --fp16
done;
done;
done;
