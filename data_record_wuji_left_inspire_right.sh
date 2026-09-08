#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/bin/activate twist2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/deploy_real"

exec python server_data_record.py \
    --frequency "${DATA_FREQUENCY:-30}" \
    --robot_ip "${ROBOT_IP:-192.168.123.164}" \
    --redis-ip "${REDIS_IP:-localhost}" \
    --redis-port "${REDIS_PORT:-6379}" \
    --hand-backend wuji-left-inspire-right \
    "$@"
