#!/bin/bash
# source ./scripts/env.sh

export INPUT_DIR=$1
export EXP_DIR=$2
export SUBJECT_NAME=$3  # project name
export BASE=$4

# if base==sd: 
#   basemodel = A
# elif base == mv:
#   basemodel = B

if [ "$BASE" == "sd" ]; then
  export BASE_MODEL=stabilityai/stable-diffusion-2-1-base
elif [ "$BASE" == "mv" ]; then
  export BASE_MODEL=ashawkey/mvdream-sd2.1-diffusers
fi
# # export BASE_MODEL=stabilityai/stable-diffusion-2-1-base
# export BASE_MODEL=ashawkey/mvdream-sd2.1-diffusers
export peft_type="none"

# Step 1: Run multi-concept DreamBooth training
# rm -rf ${EXP_DIR}/text_encoder
# rm -rf ${EXP_DIR}/unet
# rm -rf ${EXP_DIR}/img_logs

num=1000
num_2=4000
# 4000 \
python -m multi_concept_mv.train_puzzlebooth \
  --pretrained_model_name_or_path $BASE_MODEL \
  --project_name ${SUBJECT_NAME} \
  --instance_data_dir ${INPUT_DIR}  \
  --output_dir ${EXP_DIR} \
  --class_data_dir data/multi_concepts_data \
  --train_batch_size 1  \
  --phase1_train_steps ${num} \
  --phase2_train_steps ${num_2} \
  --lr_step_rules "1:2000,0.1" \
  --initial_learning_rate 5e-4 \
  --learning_rate 2e-6 \
  --prior_loss_weight 1.0 \
  --syn_loss_weight 2.0 \
  --mask_loss_weight 1.0 \
  --lambda_attention 1e-2 \
  --img_log_steps ${num} \
  --checkpointing_steps ${num} \
  --use_view_prompt \
  --log_checkpoints \
  --boft_block_num=8 \
  --boft_block_size=0 \
  --boft_n_butterfly_factor=1 \
  --lora_r=32 \
  --enable_xformers_memory_efficient_attention \
  --use_peft ${peft_type} \
  --do_not_apply_masked_prior 
  
  # --wandb_mode "offline" \
  # --use_shape_description \
  # --no_prior_preservation \

# # Step 2: Run multi-concept DreamBooth inference
# rm -rf ${EXP_DIR}/output
# python -m multi_concept_mv.inference \
#   --pretrained_model_name_or_path $BASE_MODEL \
#   --model_dir ${EXP_DIR} \
#   --instance_dir ${INPUT_DIR} \
#   --num_samples 10 \
#   --use_peft ${peft_type} \
#   --use_shape_description \