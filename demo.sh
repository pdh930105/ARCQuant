#!/bin/bash
# path to your model 
MODEL=${1}

dir=$(pwd)
export CUDA_VISIBLE_DEVICE="0"

python ${dir}/model/demo.py ${MODEL}




