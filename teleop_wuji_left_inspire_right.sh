#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/bin/activate gmr
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/deploy_real"

redis_ip=${REDIS_IP:-localhost}
actual_human_height=${ACTUAL_HUMAN_HEIGHT:-1.6}

exec python xrobot_teleop_to_robot_w_hand.py \
    --robot unitree_g1 \
    --actual_human_height "${actual_human_height}" \
    --redis_ip "${redis_ip}" \
    --target_fps 100 \
    --measure_fps 1 \
    --hand_type inspire \
    --hand-output-mode wuji-left-inspire-right \
    --require-fresh-xr-tracking \
    "$@"
