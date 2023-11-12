
GPU_ID=0








for fold in 0 1 2 3 4; do 
for epoch in 15; do
for bs in 18; do
for lr in 2e-5; do
for lr1 in 5e-5; do
for seq in 384; do
for entdim in 400; do
for reldim in 400; do 
for memdim in 400; do
for facenc in biaf; do
for factor in tercop; do
for iter in 3; do
for eps in 1e-8; do
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  --model_type hyper  \
    --model_name_or_path  pretrained_models/bert-base-uncased  --do_lower_case  \
    --learning_rate $lr   --learning_rate_cls $lr1 \
    --num_train_epochs $epoch --eval_epochs 3 --per_gpu_train_batch_size  $bs --per_gpu_eval_batch_size 32  --gradient_accumulation_steps 1  \
    --max_seq_length $seq  --max_pair_length 18  --adam_epsilon $eps  \
    --do_train --do_eval  --evaluate_during_training   --eval_all_checkpoints  \
    --fp16  --seed 42  \
    --ner_prediction_dir  saves/ace04ner_models/pruner/biafencoder_seq512-spanlen8-rank1024-hid1024-span512-entnum3-18-lr1e-5-extraattn/ace04-bert-$fold  \
    --train_file ent_pred_train_$fold.json --dev_file ent_pred_dev_$fold.json --test_file ent_pred_test_$fold.json  \
    --output_dir saves/HGERE/ace04re_models/bert/$factor/facenc$facenc-seq$seq-mem$memdim-iter$iter-eps$eps-layernorm+_attnself/ent$entdim-rel$reldim-lr$lr-$lr1-bs$bs-ep$epoch/Hyper_ace04_bert-fold$fold  --overwrite_output_dir  \
    --factor_type $factor  --iter $iter   --factor_encoder $facenc \
    --ent_dim $entdim --rel_dim $reldim  --mem_dim $memdim  \
    --layernorm  --layernorm_1st  --attn_self
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