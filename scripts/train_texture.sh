#!/bin/bash
source ./scripts/env.sh

rm -rf results/multi_concepts/human/yuliang/texture
python cores/main_mc.py \
 --config configs/tech_mc_texture.yaml \
 --exp_dir $1 \
 --sub_name $2