#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda3/bin/activate twist2_deploy
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ckpt_path=${TWIST2_POLICY:-${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx}
net=${G1_NET:-enp128s31f6}
cd "${SCRIPT_DIR}/deploy_real"

exec python server_low_level_g1_real.py \
    --policy "${ckpt_path}" \
    --net "${net}" \
    --device cuda \
    --ignore-hand-actions \
    "$@"
