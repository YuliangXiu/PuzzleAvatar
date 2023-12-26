#!/bin/bash
source ./scripts/env.sh

python multi_concepts/train.py \
  --instance_data_dir examples/$1  \
  --output_dir results/$1 \
  --class_data_dir data/multi_concepts_data \
  --phase1_train_steps 1000 \
  --phase2_train_steps 2000 \
  --lambda_attention 1e-2 \
  --log_checkpoints \
  # --boft_block_num=32 \
  # --boft_block_size=0 \
  # --boft_n_butterfly_factor=5 \
  # --boft_dropout=0.1 \
  # --boft_bias_fit \
  # --boft_bias="boft_only" \
  # --use_boft \