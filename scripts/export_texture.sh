#!/bin/bash
source ./scripts/env.sh

export peft_type="full"

python cores/main_mc.py \
 --config configs/tech_mc_texture_export.yaml \
 --exp_dir results/${peft_type}/$1 \
 --sub_name $2 \
 --use_peft none \
#  --use_shape_description \