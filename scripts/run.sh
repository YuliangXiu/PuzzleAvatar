#!/bin/bash
source ./scripts/env.sh

export OPENAI_API_KEY=$(cat OPENAI_API_KEY)
export INPUT_DIR=$1
export EXP_DIR=$2
export SUBJECT_NAME=$3
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 0: Run DINO+SAM
rm -rf examples/multi_concepts/human/yuliang/mask
python multi_concepts/grounding_dino_sam.py --in_dir ${INPUT_DIR} --out_dir ${INPUT_DIR}
python multi_concepts/islands_all.py


# Step 1: Run geometry stage (Run on a single GPU)
# rm -rf results/multi_concepts/human/yuliang/geometry
# python cores/main_mc.py \
#  --config configs/tech_mc_geometry.yaml \
#  --exp_dir ${EXP_DIR} \
#  --sub_name ${SUBJECT_NAME}

# python utils/body_utils/postprocess_mc.py \
#     --dir ${EXP_DIR} \
#     --name ${SUBJECT_NAME}

# Step 2: Run texture stage (Run on a single GPU)
# rm -rf results/multi_concepts/human/yuliang/texture
# python cores/main_mc.py \
#  --config configs/tech_mc_texture.yaml \
#  --exp_dir ${EXP_DIR} \
#  --sub_name ${SUBJECT_NAME}

# # [Optional] export textured mesh with UV map, using atlas for UV unwraping.
# python cores/main.py --config configs/tech_texture_export.yaml --exp_dir $EXP_DIR --sub_name $SUBJECT_NAME --test
