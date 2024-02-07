#!/bin/bash
source ./scripts/env.sh

export BASE_MODEL=stabilityai/stable-diffusion-2-1
export project_name="multi_concepts"

python multi_concepts/train.py \
  --pretrained_model_name_or_path $BASE_MODEL \
  --project_name $project_name \
  --instance_data_dir data/$1  \
  --output_dir results/$1 \
  --class_data_dir data/multi_concepts_data \
  --train_batch_size 1  \
  --phase1_train_steps 1000 \
  --phase2_train_steps 5000 \
  --initial_learning_rate 5e-4 \
  --learning_rate 1e-5 \
  --prior_loss_weight 1.0 \
  --mask_loss_weight 1.0 \
  --lambda_attention 1e-1 \
  --img_log_steps 500 \
  --checkpointing_steps 1000 \
  --log_checkpoints \
  --boft_block_num=16 \
  --boft_block_size=0 \
  --boft_n_butterfly_factor=1 \
  --lora_r=16 \
  --enable_xformers_memory_efficient_attention \
  --use_peft "lora" \
  --wandb_mode "offline" \
  # --do_not_apply_masked_prior \
  # --no_prior_preservation \
  # --use_shape_description \

# rm -rf results/$1/output
# python multi_concepts/inference.py \
#   --pretrained_model_name_or_path $BASE_MODEL \
#   --model_dir results/$1 \
#   --instance_dir data/$1 \
#   --num_samples 10 \
#   --use_peft "lora" \
#   # --use_shape_description \