#!/bin/bash
source ./scripts/env.sh

rm -rf results/$1/output
python multi_concepts/inference.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
  --model_dir results/$1 \
  --instance_dir data/$1 \
  --num_samples 10 \
  --use_peft "lora" \
   # --use_shape_description \