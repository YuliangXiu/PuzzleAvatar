#!/bin/bash
source ./scripts/env.sh

export INPUT_DIR=$1
export EXP_DIR=$2
export SUBJECT_NAME=$3

# python -m multi_concept_mv.train_dreambooth_diffuser \
#     instance_data_dir=data/${INPUT_DIR} \
#     expname= mvdream/${EXP_DIR}\

bash multi_concept_mv/train_puzzlebooth.sh \
        ${INPUT_DIR} \
        ${EXP_DIR} \
        ${SUBJECT_NAME} mv \
