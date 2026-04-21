

source ~/miniconda3/bin/activate twist2_deploy

SCRIPT_DIR=$(dirname $(realpath $0))
ckpt_path=${SCRIPT_DIR}/assets/ckpts/twist2_1017_20k.onnx

# change the network interface name to your own that connects to the robot
net=enp128s31f6

cd deploy_real

python server_low_level_g1_real.py \
    --policy ${ckpt_path} \
    --net ${net} \
    --device cuda \
    #--use_hand \
    #--hand_type inspire \
    #--hand_side right \
    #--dex_finger_tracking \
    # --smooth_body 0.5
    # --record_proprio \
