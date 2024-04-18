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


# # Step 2: Run multi-concept DreamBooth inference
# rm -rf ${EXP_DIR}/output
python -m multi_concept_mv.inference \
  --pretrained_model_name_or_path $BASE_MODEL \
  --model_dir ${EXP_DIR} \
  --instance_dir ${INPUT_DIR} \
  --num_samples 10 \
  --use_peft ${peft_type} \
  --use_shape_description \