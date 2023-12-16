#!/bin/bash
source ./scripts/env.sh

cd thirdparties/GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -q -e .

# install segment-anything

pip install 'git+https://github.com/facebookresearch/segment-anything.git'

# setup DINO+SAM

mkdir -p weights && cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth