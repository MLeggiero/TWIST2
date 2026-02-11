#!/bin/bash

# Add conda lib to LD_LIBRARY_PATH for IsaacGym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Script to convert student policy with future motion support to ONNX

# bash to_onnx.sh $YOUR_POLICY_PATH

ckpt_path=$1

cd legged_gym/legged_gym/scripts

# Run the correct ONNX conversion script
python save_onnx.py --ckpt_path ${ckpt_path}