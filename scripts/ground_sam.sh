#!/bin/bash
source ./scripts/env.sh

rm -rf $1/mask
python multi_concepts/grounding_dino_sam.py \
    --in_dir $1 \
    --out_dir $1
python multi_concepts/islands_all.py
