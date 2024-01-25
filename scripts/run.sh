#!/bin/bash
source ./scripts/env.sh

export OPENAI_API_KEY=$(cat OPENAI_API_KEY)
export INPUT_DIR=$1
export EXP_DIR=$2
export SUBJECT_NAME=$3
export PYTHONPATH=$PYTHONPATH:$(pwd)
export BASE_MODEL=stabilityai/stable-diffusion-2-1

# Step 0: Run DINO+SAM
python multi_concepts/grounding_dino_sam.py --in_dir ${INPUT_DIR} --out_dir ${INPUT_DIR} --overwrite
python multi_concepts/islands_all.py --out_dir ${INPUT_DIR} --overwrite

# # Step 1: Run multi-concept DreamBooth training
# python multi_concepts/train.py \
#   --pretrained_model_name_or_path $BASE_MODEL \
#   --instance_data_dir ${INPUT_DIR}  \
#   --output_dir ${EXP_DIR} \
#   --class_data_dir data/multi_concepts_data \
#   --train_batch_size 2  \
#   --phase1_train_steps 10 \
#   --phase2_train_steps 2000 \
#   --initial_learning_rate 5e-4 \
#   --learning_rate 2e-6 \
#   --lambda_attention 1e-2 \
#   --img_log_steps 500 \
#   --checkpointing_steps 500 \
#   --log_checkpoints \
#   --boft_block_num=8 \
#   --boft_block_size=0 \
#   --boft_n_butterfly_factor=4 \
#   --boft_dropout=0.1 \
#   --use_boft \
#   --enable_xformers_memory_efficient_attention \

# # Step 2: Run multi-concept DreamBooth inference
# rm -rf $2/output
# python multi_concepts/inference.py \
#   --pretrained_model_name_or_path $BASE_MODEL \
#   --model_dir ${EXP_DIR} \
#   --instance_dir ${INPUT_DIR} \
#   --num_samples 10

# # Step 3: Run geometry stage (Run on a single GPU)
# rm -rf ${EXP_DIR}/geometry/checkpoints
# rm -rf ${EXP_DIR}/geometry/run
# rm -rf ${EXP_DIR}/geometry/validation

# python cores/main_mc.py \
#  --config configs/tech_mc_geometry.yaml \
#  --exp_dir ${EXP_DIR} \
#  --sub_name ${SUBJECT_NAME}

# python utils/body_utils/postprocess_mc.py \
#     --dir ${EXP_DIR} \
#     --name ${SUBJECT_NAME}

# # Step 4: Run texture stage (Run on a single GPU)
# rm -rf ${EXP_DIR}/texture
# python cores/main_mc.py \
#  --config configs/tech_mc_texture.yaml \
#  --exp_dir ${EXP_DIR} \
#  --sub_name ${SUBJECT_NAME}

# # # [Optional] export textured mesh with UV map, using atlas for UV unwraping.
# # python cores/main.py --config configs/tech_texture_export.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME --test
