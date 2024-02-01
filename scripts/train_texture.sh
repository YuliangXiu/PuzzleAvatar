#!/bin/bash
source ./scripts/env.sh

export peft_type="none"

rm -rf results/$1/texture
python cores/main_mc.py \
 --config configs/tech_mc_texture.yaml \
 --exp_dir results/$1 \
 --sub_name $2 \
 --use_peft ${peft_type} \
#  --use_shape_description \