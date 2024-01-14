#!/bin/bash
source ./scripts/env.sh

python multi_concepts/train.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
  --instance_data_dir examples/$1  \
  --output_dir results/$1 \
  --class_data_dir data/multi_concepts_data \
  --train_batch_size 4  \
  --phase1_train_steps 1000 \
  --phase2_train_steps 2000 \
  --learning_rate 3e-5 \
  --lambda_attention 1e-2 \
  --img_log_steps 500 \
  --checkpointing_steps 500 \
  --log_checkpoints \
  --boft_block_num=8 \
  --boft_block_size=0 \
  --boft_n_butterfly_factor=3 \
  --boft_dropout=0.1 \
  --use_boft \
  --boft_bias_fit \
  --boft_bias="boft_only" \
  --enable_xformers_memory_efficient_attention \