# SciERC with scibert
# set cuda visible devices here

GPU_ID=0
DATASET_DATE=2025-04-15
DATE_MODEL=2025-04-07
ANNOTATOR=
# directory of preprocessed dataset
DATADIR=/home/ottowg/projects/gsap/gsap-rel/preprocessing/gsap-rel-sentence-simple/  
DATADIR=/home/ottowg/projects/gsap/gsap-hub/data_plmarker/$DATASET_DATE/
for seed in 44; do 
for minent in 30; do
for maxent in 30; do
for epochs in 4; do
for lr in 1e-5; do
#--do_train \
OUTPUT_DIR=saves/gsap/pruner/biafencoder-spanlen12-rank768-hid768-span256-bs8-entnum$minent-$maxent-lr$lr-epochs$epochs/gsap_scibert_data2025-02-08-bs16-$lr-$epochs-$seed
OUTPUT_DIR=saves/gsap/pruner/lr$lr-epochs$epochs/gsap_scibert_data_${DATE_MODEL}-bs16-$seed  
echo $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_pruner.py  \
    --do_test  \
    --do_eval \
    --seed $seed \
    --data_dir $DATADIR \
    --train_file ${DATASET_DATE}${ANNOTATOR}_train.jsonl \
    --dev_file ${DATASET_DATE}${ANNOTATOR}_dev.jsonl \
    --test_file ${DATASET_DATE}${ANNOTATOR}_test.jsonl  \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_model_dir  \
    --output_results \
    --model_type bertspanmarkerpruner  \
    --model_name_or_path  pretrained_models/scibert_scivocab_uncased  \
    --do_lower_case  \
    --learning_rate $lr  \
    --num_train_epochs $epochs  \
    --eval_epochs 1   \
    --per_gpu_train_batch_size  16  \
    --per_gpu_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 256  \
    --save_steps 1000  \
    --max_pair_length 64  \
    --max_mention_ori_length 12  \
    --topk_ratio 100 \
    --max_mentions_num $maxent \
    --min_mentions_num $minent \
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
    --local_rank -1
done;
done;
done;
done;
done;
#--fp16
