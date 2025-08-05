# SciERC with scibert
# set cuda visible devices here

GPU_ID=0
DATASET_DATE=2025-05-19
DATE_MODEL=2025-05-09
ANNOTATOR=
RUN_NAME="first_try${ANNOTATOR}"
# directory of preprocessed dataset
DATADIR=/home/groups/gsap/projects/gsap/gsap-rel/preprocessing/gsap-rel-sentence-simple/
DATADIR=/home/groups/gsap/data/annotations/plmarker/$DATASET_DATE/ground_truth/clean/SentenceFootnote
for seed in 44; do
for minent in 30; do
for maxent in 30; do
for epochs in 4; do
for lr in 1e-5; do
#--do_train \
OUTPUT_DIR=saves/gsap/pruner/$RUN_NAME/${DATE_MODEL}
OUTPUT_DIR=saves/gsap/pruner/$RUN_NAME
echo $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=$GPU_ID  python3  run_pruner.py  \
    --run_name $RUN_NAME \
    --do_test  \
    --do_eval \
    --seed $seed \
    --data_dir $DATADIR \
    --train_file ${DATASET_DATE}${ANNOTATOR}_train.jsonl \
    --dev_file ${DATASET_DATE}${ANNOTATOR}_dev.jsonl \
    --test_file ${DATASET_DATE}${ANNOTATOR}_test.jsonl  \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir  \
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
