#!/usr/bin/env bash
set -euo pipefail

real_env=${TWIST2_REAL_ENV:-twist2_deploy}
source ~/miniconda3/bin/activate "${real_env}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ckpt_path=${TWIST2_POLICY:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx}
net=${G1_NET:-enp128s31f6}

if [[ -z "${INSPIRE_RIGHT_IP:-}" ]]; then
  echo "Set INSPIRE_RIGHT_IP before running hybrid real hardware mode" >&2
  exit 2
fi

cd "${SCRIPT_DIR}/deploy_real"
exec python server_low_level_g1_real.py \
    --policy "${ckpt_path}" \
    --net "${net}" \
    --device cuda \
    --use_hand \
    --hand_type inspire \
    --inspire-side right \
    --inspire_right_ip "${INSPIRE_RIGHT_IP}" \
    --inspire-action-timeout "${INSPIRE_ACTION_TIMEOUT:-0.5}" \
    --inspire-command-rate-limit "${INSPIRE_COMMAND_RATE_LIMIT:-250}" \
    --inspire-startup-interpolation-duration "${INSPIRE_STARTUP_DURATION:-1.75}" \
    --inspire-hold-on-close \
    --require-fresh-pico-body \
    --pico-body-timeout "${PICO_BODY_TIMEOUT:-0.5}" \
    --check_stale \
    "$@"
