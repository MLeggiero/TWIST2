#!/bin/bash

# Add conda lib to LD_LIBRARY_PATH for IsaacGym
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# bash eval.sh 1002_twist2 cuda:1



# motion_file="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/v1_v2_v3_g1/0807_yanjie_walk_001.pkl"
motion_file="/home/yanjieze/projects/g1_wbc/TWIST-dev/motion_data/twist1_to_twist2/transitions_walksideways_walkbackwards.pkl"

task_name="g1_stu_future"
proj_name="g1_stu_future"
exptid=$1
device=$2

cd legged_gym/legged_gym/scripts

echo "Evaluating student policy with future motion support..."
echo "Task: ${task_name}"
echo "Project: ${proj_name}"
echo "Experiment ID: ${exptid}"
echo ""

# Run the evaluation script
python play.py --task "${task_name}" \
               --proj_name "${proj_name}" \
               --teacher_exptid "None" \
               --exptid "${exptid}" \
               --num_envs 1 \
               --record_video \
               --device "${device}" \
               --env.motion.motion_file "${motion_file}" \
               # --checkpoint 13000 \
               # --record_log \
               # --use_jit \
               # --teleop_mode