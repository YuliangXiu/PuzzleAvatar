#!/bin/bash
source ./scripts/env.sh

export INPUT_DIR=$1
export EXP_DIR=$2
export SUBJECT_NAME=$3

python -m multi_concept_mv.train_dreambooth_diffuser \
    instance_data_dir=${INPUT_DIR} \
    output_dir=${EXP_DIR} \
    expname=${SUBJECT_NAME}\