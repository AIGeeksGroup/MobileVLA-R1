#!/bin/bash
# Explicitly invoked pipeline: Episode+Nav -> export -> Step subset -> export.
set -euo pipefail
: "${MODEL_PATH:?Set the initial NaVILA full model path}"
: "${COT_EPISODE_DATA:?Set the validated Episode JSONL path}"
: "${NAV_COT_DATA:?Set the validated Nav JSONL path}"
: "${COT_STEP_DATA:?Set the validated 10K Step JSONL path}"
: "${NAVCOT_IMAGE_ROOT:?Set the shared RGB root}"
: "${NAVCOT_DEPTH_ROOT:?Set the shared depth root}"
RUN_ROOT="${RUN_ROOT:-./checkpoints/cold-start}"
# Validate the second stage size without loading a model.
python3 -c 'from mobilevla.data import load_records; import os; n=len(load_records(os.environ["COT_STEP_DATA"])); assert n == 10000, f"Expected 10000 Step samples, got {n}"'
OUTPUT_DIR="$RUN_ROOT/episode-nav-adapter" bash scripts/train/sft_8frames.sh \
    "$@" --data_mixture cot_episode+nav_cot_vln --output_dir "$RUN_ROOT/episode-nav-adapter" --model_name_or_path "$MODEL_PATH"
python3 merge.py --base-model "$MODEL_PATH" --lora-path "$RUN_ROOT/episode-nav-adapter" \
    --output-dir "$RUN_ROOT/episode-nav-model"
MODEL_PATH="$RUN_ROOT/episode-nav-model" OUTPUT_DIR="$RUN_ROOT/step-adapter" \
    bash scripts/train/sft_8frames.sh "$@" --data_mixture cot_step --output_dir "$RUN_ROOT/step-adapter" --model_name_or_path "$RUN_ROOT/episode-nav-model"
python3 merge.py --base-model "$RUN_ROOT/episode-nav-model" --lora-path "$RUN_ROOT/step-adapter" \
    --output-dir "$RUN_ROOT/sft-model"
