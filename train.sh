#!/bin/bash

# Add conda lib to LD_LIBRARY_PATH for IsaacGym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Usage: bash train.sh <experiment_id> <device>

# bash train.sh 1103_twist2 cuda:0


cd legged_gym/legged_gym/scripts

robot_name="g1"
exptid=$1
device=$2

task_name="${robot_name}_stu_future"
proj_name="${robot_name}_stu_future"


# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${device}" \
                --teacher_exptid "None" \
                # --resume \
                # --debug \
