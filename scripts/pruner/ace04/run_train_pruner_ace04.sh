
GPU_ID=0



# train ace05 pruner, bert base
for fold in 0 1 2 3 4; do 
for minent in 3; do
for maxent in 18; do
for seq in 512; do
for spansize in 512; do
for mem in 1024; do
for span in 8; do
for extra in attn; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_pruner.py  --model_type bertspanmarkerpruner  \
    --model_name_or_path  pretrained_models/bert-base-uncased  --do_lower_case  \
    --data_dir {directory of preprocessed dataset}  \
    --learning_rate 1e-5  \
    --num_train_epochs 5  --eval_epochs 1    --per_gpu_train_batch_size  8  --per_gpu_eval_batch_size 8  --gradient_accumulation_steps 1  \
    --max_seq_length $seq  --save_steps 1000  --max_pair_length 30  --max_mention_ori_length $span    \
    --max_mentions_num $maxent --min_mentions_num $minent \
    --do_test   --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  --onedropout  --lminit  \
    --train_file train/$fold.json  --dev_file dev/$fold.json  --test_file test/$fold.json  \
    --output_dir saves/ace04ner_models/pruner/biafencoder_seq$seq-spanlen$span-rank$mem-hid$mem-span$spansize-entnum$minent-$maxent-lr1e-5-extra$extra/ace04-bert-$fold \
    --overwrite_output_dir  --output_results  \
    --biaf_span --rank $mem --span_hidden_size $mem  --span_size $spansize  --extra_repr $extra  
done;
done;
done;
done;
done;
done;
done;
done;



