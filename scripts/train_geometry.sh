#!/bin/bash
source ./scripts/env.sh

rm -rf results/multi_concepts/human/yuliang/geometry
python cores/main_mc.py \
 --config configs/tech_mc_geometry.yaml \
 --exp_dir $1 \
 --sub_name $2

 python utils/body_utils/postprocess_mc.py \
    --dir $1 \
    --name $2