#!/bin/bash
source ./scripts/env.sh

rm -rf results/$1/geometry/checkpoints
rm -rf results/$1/geometry/run
rm -rf results/$1/geometry/validation

python cores/main_mc.py \
 --config configs/tech_mc_geometry.yaml \
 --exp_dir results/$1 \
 --sub_name $2

python utils/body_utils/postprocess_mc.py \
    --dir results/$1 \
    --name $2