#!/bin/bash
source ./scripts/env.sh

rm -rf results/$1/texture
python cores/main_mc.py \
 --config configs/tech_mc_texture.yaml \
 --exp_dir results/$1 \
 --sub_name $2