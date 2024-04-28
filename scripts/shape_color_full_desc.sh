#!/bin/bash
source ./scripts/env.sh

export peft_type="full"

# rm -rf results/${peft_type}_desc/$1/geometry/checkpoints
# rm -rf results/${peft_type}_desc/$1/geometry/run
# rm -rf results/${peft_type}_desc/$1/geometry/validation
# rm -rf results/${peft_type}_desc/$1/geometry/visualize

python cores/main_mc.py \
 --config configs/tech_mc_geometry.yaml \
 --exp_dir results/${peft_type}_desc/$1 \
 --sub_name $2 \
 --use_peft none \
 --use_shape_description \

python utils/body_utils/postprocess_mc.py \
    --dir results/${peft_type}_desc/$1 \
    --name $2

# rm -rf results/${peft_type}_desc/$1/texture

python cores/main_mc.py \
 --config configs/tech_mc_texture.yaml \
 --exp_dir results/${peft_type}_desc/$1 \
 --sub_name $2 \
 --use_peft none \
 --use_shape_description \