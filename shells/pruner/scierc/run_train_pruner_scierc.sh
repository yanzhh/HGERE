# SciERC with scibert
GPU_ID=0

for seed in 42 43 44 45 46; do 
for minent in 3; do
for maxent in 18; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_pruner.py  --model_type bertspanmarkerpruner  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased  --do_lower_case  \
    --data_dir {directory of preprocessed dataset}  \
    --learning_rate 2e-5  \
    --num_train_epochs 8  --eval_epochs 1    --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
    --max_seq_length 256  --save_steps 1000  --max_pair_length 64  --max_mention_ori_length 12    \
    --max_mentions_num $maxent --min_mentions_num $minent \
    --do_train  --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --seed $seed  --onedropout  --lminit  --nocross\
    --train_file train.json --dev_file dev.json --test_file test.json  \
    --output_dir saves/sciner_models/pruner/biafencoder-spanlen12-rank768-hid768-span256-entnum$minent-$maxent-lr2e-5/scierc_scibert-$seed  --overwrite_output_dir  --output_results \
    --biaf_span  --biaf_mode  2  --biaf_factorize  --span_hidden_size 768   --rank 768  --span_size 256
done;
done;
done;