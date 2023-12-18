#!/bin/bash
source ./scripts/env.sh

python multi_concepts/train.py \
  --instance_data_dir examples/$1  \
  --class_data_dir data/multi_concepts_data \
  --phase1_train_steps 500 \
  --phase2_train_steps 1000 \
  --lambda_attention 5e-2 \
  --output_dir results/$1