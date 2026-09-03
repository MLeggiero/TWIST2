#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/bin/activate "${TWIST2_MUJOCO_ENV:-gmr}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
policy=${TWIST2_POLICY:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx}

sim_args=(
  --xml "${SCRIPT_DIR}/assets/g1/g1_sim2sim_29dof.xml"
  --policy "${policy}"
  --redis-ip "${REDIS_IP:-localhost}"
  --redis-port "${REDIS_PORT:-6379}"
  --device "${TWIST2_DEVICE:-cpu}"
  --require-fresh-pico-body
)

[[ -n "${WUJI_LEFT_CONFIG:-}" ]] && sim_args+=(--left-config "${WUJI_LEFT_CONFIG}")
[[ -n "${WUJI_RIGHT_CONFIG:-}" ]] && sim_args+=(--right-config "${WUJI_RIGHT_CONFIG}")

if [[ -z "${WUJI_LEFT_CONFIG:-}" && -z "${WUJI_RIGHT_CONFIG:-}" ]]; then
  echo "Set WUJI_LEFT_CONFIG and/or WUJI_RIGHT_CONFIG before running sim2sim_wuji.sh" >&2
  exit 2
fi

exec python "${SCRIPT_DIR}/deploy_real/server_low_level_g1_wuji_sim.py" \
  "${sim_args[@]}" \
  "$@"
