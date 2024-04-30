#!/bin/bash
set +x 
# source ./scripts/env.sh

export peft_type="none"

# rm -rf results/$1/geometry/checkpoints
# rm -rf results/$1/geometry/run
# rm -rf results/$1/geometry/validation
# rm -rf results/$1/geometry/visualize
# rm -rf results/$1/geometry/tet

python -m cores.main_mc \
 --config configs/tech_mc_geometry_mv.yaml \
 --exp_dir results/$1 \
 --sub_name $2 \
 --use_peft ${peft_type} \
 --use_shape_description \
 --pretrain output/puzzle_int \
 --test \

# python -m utils.body_utils.postprocess_mc \
#     --dir results/$1 \
#     --name $2

# rm -rf results/$1/texture

# python -m cores.main_mc \
#  --config configs/tech_mc_texture.yaml \
#  --exp_dir results/$1 \
#  --sub_name $2 \
#  --use_peft ${peft_type} \
#  --use_shape_description \