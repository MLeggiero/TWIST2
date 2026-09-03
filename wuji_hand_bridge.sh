#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/bin/activate gmr
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/deploy_real"

bridge_args=()
[[ -n "${WUJI_LEFT_CONFIG:-}" ]] && bridge_args+=(--left-config "${WUJI_LEFT_CONFIG}")
[[ -n "${WUJI_RIGHT_CONFIG:-}" ]] && bridge_args+=(--right-config "${WUJI_RIGHT_CONFIG}")
[[ -n "${WUJI_LEFT_IP:-}" ]] && bridge_args+=(--left-ip "${WUJI_LEFT_IP}")
[[ -n "${WUJI_RIGHT_IP:-}" ]] && bridge_args+=(--right-ip "${WUJI_RIGHT_IP}")
bridge_args+=(--redis-ip "${REDIS_IP:-localhost}" --redis-port "${REDIS_PORT:-6379}")

exec python wuji_hand_bridge.py "${bridge_args[@]}" "$@"
