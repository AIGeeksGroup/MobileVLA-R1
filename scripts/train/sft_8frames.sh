#!/bin/bash

OUTPUT="./checkpoints/navila-8b-8f-sft-lora/sft_new1"  # Change output dir name to track runs

# export NNODES=1
# export GPUS_PER_NODE=2
# export CURRENT_RANK=0
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=2950090-o+P
export WANDB_MODE=offline

torchrun --nnodes=1 --nproc_per_node=1 --master_port=29500 \
    --master_addr 127.0.0.1 --node_rank=0 \
    llava/train/train_mem.py \
    --longvila_sampler True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /root/autodl-tmp/model/finetune \
    --version llama_3 \
    --seed 10 \
    --data_mixture cot+nav_cot_vln \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    \
    `# LoRA configuration` \
    --lora_enable True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_llm True \
    --lora_vt False \
    --lora_bias none \
    \
    `# Fine-tuning strategy for LoRA` \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    \
    --use_depth_tower True \
    --use_point_tower True \
    --depth_tower depth_anything_v2 \
    --point_tower point_transformer \
    --depth_hidden_size 1024 \
    --point_hidden_size 1024 \
    \
    --navcot_use_depth True \
    --navcot_use_point True \
    --navcot_image_root /root/autodl-tmp/dataset/NavCoT/frames \
    --navcot_depth_root /root/autodl-tmp/dataset/NavCoT/depth \
    --navcot_depth_format png \
    --navcot_depth_scale 1000 \
    --navcot_pointcloud_points 2048 \
    --navcot_depth_frames 1 \
    \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir $OUTPUT \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 10000 \
    --fps 0.0 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb