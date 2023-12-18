#!/bin/bash
source ./scripts/env.sh

rm -rf results/$1/output
python multi_concepts/inference.py \
  --model_dir results/$1 \
  --instance_dir examples/$1 \
  --num_samples 30