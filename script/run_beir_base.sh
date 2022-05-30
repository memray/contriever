#!/usr/bin/env bash

MODEL_NAME=$1
OUTPUT_BASE_DIR=$6
POOLING_TYPE=$7
BATCH_SIZE=$8

DATA_DIR=/export/home/data/beir/
OUTPUT_DIR=$OUTPUT_BASE_DIR/lr_"$LR"/"$TASK"/seed_"$SEED"/

mkdir -p $OUTPUT_DIR
echo "Source CKPT:" $CKPT_PATH
echo "Target DIR:" $OUTPUT_DIR

python eval_beir.py --model_name_or_path $MODEL_NAME --dataset $DATASET --beir_data_path $DATA_DIR --per_gpu_batch_size $BATCH_SIZE --output_dir $OUTPUT_DIR

echo "Task done: TASK=$TASK, LR=$LR, SEED=$SEED, CKPT_PATH=$CKPT_PATH"
