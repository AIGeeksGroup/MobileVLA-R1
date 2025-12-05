#!/bin/bash

MODEL_PATH=$1
TOTAL_CHUNKS=$2
IDX_START=$3
GPU_LIST=$4  # GPU list as a string (e.g., "0,2,4,6")

IFS=',' read -ra GPULIST <<< "$GPU_LIST"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    echo "Total Chunks: $TOTAL_CHUNKS, Local Chunks: $CHUNKS, Chunk Index: $CHUNK_IDX, GPU: ${GPULIST[$IDX]}"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run.py \
        --exp-config vlnce_baselines/config/r2r_baselines/navila.yaml \
        --run-type eval \
        --num-chunks $TOTAL_CHUNKS \
        --chunk-idx $CHUNK_IDX \
        EVAL_CKPT_PATH_DIR $MODEL_PATH &
done

wait
