#!/bin/bash
# set +x 
source ./scripts/env.sh

export EXP_DIR=$1
export SUB_NAME=$2
export DATA_DIR=$3

export peft_type="none"

# rm -rf results/$1/geometry/checkpoints
# rm -rf results/$1/geometry/run
# rm -rf results/$1/geometry/validation
# rm -rf results/$1/geometry/visualize
# rm -rf results/$1/geometry/tet

python -m cores.main_mc \
 --config configs/tech_mc_geometry_mv.yaml \
 --exp_dir ${EXP_DIR} \
 --data_dir data/${DATA_DIR} \
 --sub_name ${SUB_NAME} \
 --use_peft ${peft_type} \
 --use_shape_description \
 --pretrain ${EXP_DIR} \

python -m utils.body_utils.postprocess_mc \
    --dir ${EXP_DIR} \
    --data_dir data/${DATA_DIR} \
    --name ${SUB_NAME}

# # # rm -rf results/$1/texture

python -m cores.main_mc \
 --config configs/tech_mc_texture_mv.yaml \
 --exp_dir ${EXP_DIR} \
 --data_dir data/${DATA_DIR} \
 --sub_name ${SUB_NAME} \
 --use_peft ${peft_type} \
 --use_shape_description \
 --pretrain ${EXP_DIR}
