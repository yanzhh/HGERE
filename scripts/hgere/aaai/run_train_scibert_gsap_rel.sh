GPU_ID=7

for seed in 43; do 
#for epoch in 10; do 
for epoch in 10; do 
for bs in 18; do
for lr in 1e-4; do
for lr1 in 1e-4; do
for seq in 512; do
for entdim in 400; do
for reldim in 400; do 
for memdim in 400; do
for facenc in biaf; do
for factor in tersibcop; do
for iter in 3; do
for eps in 1e-8; do
#--do_train --do_eval \
OUTPUT_DIR=saves/gsap/HGERE/scibert/$factor/facenc$facenc-seq$seq-mem$memdim-iter$iter-layernorm+_attnself/ent$entdim-rel$reldim-lr$lr-$lr1-bs$bs-ep$epoch-eps$eps/hyper_gsap_scibert-2025-02-08_v0-ep$epochis-lr1$lr1-$seed
OUTPUT_DIR=saves/gsap/HGERE/scibert/2025-02-08-lr11e-4-ep10-43
echo $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=$GPU_ID  python  run_hgnn.py  \
    --output_dir $OUTPUT_DIR \
    --do_eval \
    --do_test \
    --ner_prediction_dir  saves/gsap/pruned_ner/2025-02-12/ \
    --test_file v0_ent_pred_2025-02-12_dev.json \
    --dev_file v0_ent_pred_2025-02-12_dev.json \
    --train_file v0_ent_pred_2025-02-12_test.json  \
    --model_type hyper  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased \
    --do_lower_case  \
    --learning_rate $lr  \
    --learning_rate_cls $lr1 \
    --num_train_epochs $epoch \
    --eval_epochs 2 \
    --per_gpu_train_batch_size  $bs \
    --per_gpu_eval_batch_size 32 \
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
