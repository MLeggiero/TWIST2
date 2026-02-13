

source ~/miniconda3/bin/activate twist2

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
net=enp45s0
# net=enxa0cec85a586e

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    --use_hand \
    --hand_type inspire \
    # --smooth_body 0.5
    # --record_proprio \
