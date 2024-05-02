#!/bin/bash
source ./scripts/env.sh

export SUBJECT_OUTFIT_LST=$1
export EXP_DIR=results/multi/${SUBJECT_OUTFIT_LST}

export BASE_MODEL=stabilityai/stable-diffusion-2-1-base
export peft_type="none"

# Step 2: Run multi-concept DreamBooth inference
rm -rf ${EXP_DIR}/output
python multi_concepts/inference_multi.py \
  --pretrained_model_name_or_path $BASE_MODEL \
  --model_dir ${EXP_DIR} \
  --so_lst ${SUBJECT_OUTFIT_LST} \
  --num_samples 10 \
  --use_peft ${peft_type} \
  # --use_shape_description \