#!/bin/bash
source ./scripts/env.sh

export peft_type="lora"

rm -rf results/$1/geometry/checkpoints
rm -rf results/$1/geometry/run
rm -rf results/$1/geometry/validation
rm -rf ${EXP_DIR}/geometry/visualize

python cores/main_mc.py \
 --config configs/tech_mc_geometry.yaml \
 --exp_dir results/$1 \
 --sub_name $2 \
 --use_peft ${peft_type} \
#  --use_shape_description \

python utils/body_utils/postprocess_mc.py \
    --dir results/$1 \
    --name $2