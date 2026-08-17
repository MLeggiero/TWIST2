# TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System
By Yanjie Ze, Siheng Zhao, Weizhuo Wang, Angjoo Kanazawa†, Rocky Duan†, Pieter Abbeel†, Guanya Shi†, Jiajun Wu†, C. Karen Liu†, 2025 († Equal Advising)


[[Website]](https://yanjieze.com/TWIST2)
[[arXiv]](https://arxiv.org/abs/2511.02832)
[[Video]](https://youtu.be/lTtEvI0kUfo)

![Banner for TWIST](./assets/TWIST2.png)


# News
- **2025-12-02**. 1st successfuly reproducing of TWIST2 appears. Check [his bilibili video](https://www.bilibili.com/video/BV1UbSeBNETw/?share_source=copy_web&vd_source=c76e3ab14ac3f7219a9006b96b4b0f76)
- **2025-12-02**. I will give a video tutorial for TWIST2 next week to show you how to use TWIST2. Please stay tuned.
- **2025-12-02**. TWIST2 is open-sourced now. Give it a star on GitHub!
   - Disclamer 1: with current repo, you should be able to control Unitree G1 via cable connection in both sim and real with a PICO VR headset. 
   - Disclamer 2: I am still working on better documentation and cleaning some onboard streaming and inference code (as the teleop pipeline is complex and requires some hardware setup). Please stay tuned.
   - Disclamer 3: The high-level policy learning part will be released in a separate repo. It is modifed from [iDP3](https://github.com/YanjieZe/Improved-3D-Diffusion-Policy) and I am working on releasing it soon.
- **2025-11-05**. TWIST2 is released. Full code will be released within 1 month (mostly ready and under the internal process). Please stay tuned.




# Content Table

- [Installation](#installation)
- [Usage](#usage)
- [RealSense D435i Camera Integration](#realsense-d435i-camera-integration)
- [Wrist Camera Integration](#wrist-camera-integration)
- [Inspire Hand Control](#inspire-hand-control)
- [Pico 4 Ultra Finger Tracking](#pico-4-ultra-finger-tracking)
- [Citation and Contact](#citation-and-contact)


# Installation
We will have two conda environments for TWIST2. One is called `twist2`, which can be used for controller training, controller deployment, and teleop data collection. The other is called `gmr`, which can be used for online motion retargeting. This is because isaacgym requires python 3.8, but newest mujoco requires python 3.10.

**1**. Create conda environment:
```bash
conda env remove -n twist2
conda create -n twist2 python=3.8
conda activate twist2
```

**2**. Install isaacgym. Download from [official link](https://developer.nvidia.com/isaac-gym) and then install it:
```bash
cd isaacgym/python && pip install -e .
```

**3**. Install packages:
```bash
cd rsl_rl && pip install -e . && cd ..
cd legged_gym && pip install -e . && cd ..
cd pose && pip install -e . && cd ..
pip install "numpy==1.23.0" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown hydra-core imageio[ffmpeg] mujoco mujoco-python-viewer isaacgym-stubs pytorch-kinematics rich termcolor zmq
pip install redis[hiredis] # for redis communication
pip install pyttsx3 # for voice control
pip install onnx onnxruntime-gpu # for onnx model inference
pip install customtkinter # for gui
```

if this is your first time to use redis, install and start redis server:
```bash
# sudo apt install redis-server
# redis-server --daemonize yes

sudo apt update
sudo apt install -y redis-server

sudo systemctl enable redis-server
sudo systemctl start redis-server
```

edit `/etc/redis/redis.conf`:
```bash
sudo nano /etc/redis/redis.conf
```

modify to the following lines:
```bash
bind 0.0.0.0
protected-mode no
```

then restart redis-server:
```bash
sudo systemctl restart redis-server
```


**4**. if you wanna do sim2real with laptop, you also need to install my modified version of unitree sdk [here](https://github.com/YanjieZe/unitree_sdk2/tree/main/python_binding). (if you wanna do sim2real with onboard robot computer, no need to install unitree sdk on your laptop.)
```bash
# Clone the Unitree SDK2 repository
cd ..
git clone https://github.com/YanjieZe/unitree_sdk2.git
cd unitree_sdk2

# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev python3-pip pybind11-dev

# Install Python dependencies
pip install pybind11 pybind11-stubgen numpy

# Build Python SDK binding
cd python_binding
export UNITREE_SDK2_PATH=$(pwd)/..
bash build.sh --sdk-path $UNITREE_SDK2_PATH

# Install the compiled module to your conda environment
# Get the site-packages path for your current conda environment
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Installing to: $SITE_PACKAGES"

# Copy the compiled module (rename to remove version-specific suffix)
sudo cp build/lib/unitree_interface.cpython-*-linux-gnu.so $SITE_PACKAGES/unitree_interface.so

# Verify installation
python -c "import unitree_interface; print('✓ Unitree SDK Python binding installed successfully')"
python -c "import unitree_interface; print('Available robot types:', list(unitree_interface.RobotType.__members__.keys()))"

cd ../..
```


**4**. [If you want to train your own controller, download data; otherwise, skip this step] Download TWIST2 dataset from [my google drive](https://drive.google.com/file/d/1JbW_InVD0ji5fvsR5kz7nbsXSXZQQXpd/view?usp=sharing) [Small note: if you use this dataset in your project, please also add proper citation to this work]. Unzip it to anywhere you like, and specify the `root_path` in `legged_gym/motion_data_configs/twist2_dataset.yaml` to the unzipped folder.

**Note**: we also provide a small set of example motions in `assets/example_motions`. You can use them to test the system. It is recorded by myself so no license issue.

**Note**: We also provide our controller ckpt `assets/ckpts/twist2_1017_20k.onnx` for you to test the system directly.


**5**. Install GMR for online retargeting and teleop. We use a separate conda environment for GMR/online retargeting due to requiring python 3.10+.
```bash
conda create -n gmr python=3.10 -y
conda activate gmr

git clone https://github.com/YanjieZe/GMR.git

cd GMR

# install GMR
pip install -e .
cd ..

conda install -c conda-forge libstdcxx-ng -y

```

**6**. Install PICO SDK:
1. On your PICO, install PICO SDK: see [here](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/).
2. On your own PC, 
    - Download [deb package for ubuntu 22.04](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases/download/v1.0.0/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb), or build from the [repo source](https://github.com/XR-Robotics/XRoboToolkit-PC-Service).
    - To install, use command
        ```bash
        sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
        ```
        then you should see `xrobotoolkit-pc-service` in your APPs. remember to start this app before you do teleopperation.
    - Build PICO PC Service SDK and Python SDK for PICO streaming:
        ```bash
        conda activate gmr

        git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git
        cd XRoboToolkit-PC-Service-Pybind

        mkdir -p tmp
        cd tmp
        git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
        cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK 
        bash build.sh
        cd ../../../..
        

        mkdir -p lib
        mkdir -p include
        cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
        cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
        cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/
        # rm -rf tmp

        # Build the project
        conda install -c conda-forge pybind11
        pip uninstall -y xrobotoolkit_sdk
        python setup.py install
        ```


**7**. Ready for training & deployment!

# Usage
We have provided the trained student ckpt in `assets/ckpts/twist2_1017_20k.onnx`. You can directly use it for deployment. If you want to deploy our ckpt directly, go to **4** directly.

And we have also provided full motion datasets to ensure you can successfully train our teacher & student policy.


**1**. Training TWIST2 general motion tracker:
```bash
bash train.sh 1021_twist2 cuda:0
```
- arg 1: policy expid
- arg 2: cuda device id

**2**. Export policy to onnx model:
```bash
bash to_onnx.sh $YOUR_POLICY_PATH
```
- arg 1: your policy path. you should find the `.pt` file's path.


**3**. Sim2sim verification:

[If this is your first time to run this script] you need to warm up the redis server by running the high-level motion server.
```bash
bash run_motion_server.sh
```
You can also just select one motion file from our motion dataset by modifying the `motion_file` in `run_motion_server.sh`.

Then, you can run the low-level controller server in simulation.
```bash
bash sim2sim.sh
```
- This will start a simulation that runs the low-level controller only.
- This is because we separate the high-level control (i.e., teleop) from the low-level control (i.e., RL policy).
- You should now be able to see the robot stand still. The robot is standing still because we make redis server send stand pose by default.

You can also see the policy execution FPS in the terminal. It should be around 50 Hz. If your laptop's GPU/CPU is not strong enough, the FPS may be lower and hurts the policy execution.
```bash
=== Policy Execution FPS Results (steps 1-1000) ===
Average Policy FPS: 38.88
Max Policy FPS: 41.55
Min Policy FPS: 24.72
Std Policy FPS: 1.92
Expected FPS (from decimation): 50.00
```

And now you can control the robot via high-level motion streaming.

**Note**: you need to open another terminal to run the high-level motion streaming.

We have provided two choices for you:

1) for offline motion streaming:
```bash
bash run_motion_server.sh
```


2) for online PICO teleop:
```bash
bash teleop.sh
```



 

**4**. Sim2real verification. If you are not familiar with the deployment on physical robot, you can refer to [unitree_g1.md](./unitree_g1.md) or [unitree_g1.zh.md](./unitree_g1.zh.md) for more details.

More specifically, the pipeline for sim2real deploy is:
1. start the robot and connect the robot and your laptop via an Ethernet cable.
2. config the corresponding net interface on your laptop, by setting the IP address as `192.168.123.222` and the netmask as `255.255.255.0`.
3. now you should be able to ping the robot via `ping 192.168.123.164`.
4. then use Unitree G1's remote control to enter dev mode, i.e., press the `L2+R2` key combination.
5. now you should be able to see the robot joints in the damping state.
6. then you can run the low-level controller by:
```bash
bash sim2real.sh
```
- please set the network interface name to your own that connects to the robot in `sim2real.sh`.




Similarly, you run the low-level controller first and then control the robot via high-level motion server, i.e.,
1) offline motion streaming:
```bash
bash run_motion_server.sh
```
2) or online PICO teleop:
```bash
bash teleop.sh
```


**5**. GUI interface for everything. Check `gui.sh` for more details.
```bash
bash gui.sh
```
You should be able to
1) run the low-level controller in simulation
2) run the low-level controller on physical robot
3) run the high-level motion streaming in offline mode
4) run the high-level motion streaming in online PICO teleop mode
5) run the data collection script
6) run the neck controller script
7) run the ZED streaming script
all in this GUI. This GUI is also what I use for data collection and teleoperation.

# RealSense D435i Camera Integration

The G1 robot has an internal Intel RealSense D435i connected to the Orin NX via USB. This pipeline streams **RGB + aligned depth** at **424×240 @ 60 fps** over ZMQ to the workstation for data collection, alongside (and independently of) the existing ZED Mini pipeline.

### Quick Start

1. **Deploy the streamer to the Orin:**
```bash
bash deploy_real/onboard/deploy_to_robot.sh
```

2. **Start the streamer on the Orin:**
```bash
ssh unitree@192.168.123.164
bash ~/g1-onboard/start_realsense.sh
```

3. **Start the recorder on the workstation:**
```bash
bash data_record_d435.sh
```

4. **Toggle recording** with the PICO controller:
   - **Left controller Y button** — start/stop episode recording
   - **Left controller axis click** — quit recording

### Data Format
Episodes are saved to `deploy_real/twist2_demonstration/<task_name>/`, each containing `data.json` with per-frame state/action vectors, `rgb/` JPEG images, and `depth/` uint16 PNG depth maps.

### Validation
```bash
cd deploy_real
python validate_d435_data.py twist2_demonstration/<task_name>           # Check all episodes
python validate_d435_data.py twist2_demonstration/<task_name> --show    # Visual spot-check
```

For full details (wire format, CLI flags, troubleshooting), see [doc/REALSENSE_CAMERA.md](./doc/REALSENSE_CAMERA.md).

# Wrist Camera Integration

Two Arducam USB2.0 fisheye cameras are mounted on the G1's wrists, connected to the Orin via USB like the D435i. Each streams **RGB only** at **640x480 @ 30fps** over ZMQ, and both are merged into the **same synchronized episode** as the D435 recorder so all camera views + robot state/action stay frame-aligned for fine-manipulation policy training.

### Quick Start

1. **Deploy the streamers to the Orin:**
```bash
bash deploy_real/onboard/deploy_to_robot.sh
```

2. **Start both wrist streamers on the Orin:**
```bash
ssh unitree@192.168.123.164
bash ~/g1-onboard/start_wrist_left.sh &
bash ~/g1-onboard/start_wrist_right.sh &
```

3. **Start the recorder on the workstation**, with wrist ports enabled:
```bash
cd deploy_real
python server_data_record_d435.py --wrist_left_port 5556 --wrist_right_port 5558
```

4. **Toggle recording** with the PICO controller (same button drives all cameras):
   - **Left controller Y button** — start/stop episode recording
   - **Left controller axis click** — quit recording

### Data Format
Wrist frames are added to the same `episode_XXXX/data.json` as D435, under new `rgb_wrist_left/` and `rgb_wrist_right/` JPEG subdirectories.

### Validation
`validate_d435_data.py` checks the wrist directories automatically when present — no separate command needed.

For full details (udev setup for stable camera identity, wire format, CLI flags, troubleshooting), see [doc/WRIST_CAMERAS.md](./doc/WRIST_CAMERAS.md).


# Inspire Hand Control

TWIST2 supports the **Inspire RH56DFTP** dexterous hands (6 DOF per hand), connected via Modbus TCP.

### Hardware Setup
The Inspire hands connect to the workstation (or Orin) via Ethernet. Default IPs:
- Left hand: `192.168.123.210`
- Right hand: `192.168.123.211`

### Usage
Inspire hand support is enabled via the `--hand_type inspire` flag (already set in `teleop.sh` and `sim2real.sh`):

```bash
# In teleop.sh — already configured:
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 --hand_type inspire ...

# In sim2real.sh — already configured:
python server_low_level_g1_real.py --policy <ckpt> --use_hand --hand_type inspire
```

### DOF Mapping
| Index | Joint          | Range     |
|-------|----------------|-----------|
| 0     | Pinky          | 0–1000    |
| 1     | Ring           | 0–1000    |
| 2     | Middle         | 0–1000    |
| 3     | Index          | 0–1000    |
| 4     | Thumb bend     | 0–1000    |
| 5     | Thumb rotation | 0–1000    |

The value `0` = fully open, `1000` = fully closed. Commands are sent at the teleop loop rate through Redis → `InspireHandController.ctrl_dual_hand()`.

Custom Inspire hand IPs can be set via `--inspire_left_ip` and `--inspire_right_ip` on `server_low_level_g1_real.py`.

### Dependencies
```bash
pip install pymodbus  # Already included in twist2 env
```


# Pico 4 Ultra Finger Tracking

Instead of using controller buttons to open/close the hands, the **Pico 4 Ultra's hand tracking** can be used for per-finger dexterous control of the Inspire hands.

### How It Works
1. The Pico headset tracks 26 hand joints per hand (OpenXR format) — positions and orientations
2. The `PicoFingerTracker` class computes per-finger curl angles using geometric vector analysis (no external dependencies beyond numpy)
3. Curl values are mapped to Inspire 6-DOF commands (0–1000 per finger)
4. EMA smoothing reduces tracking jitter

### Operational Setup
- **VR controllers strapped to the user's wrists** — body tracking (SMPLX) continues working normally via controller poses
- **A separate operator at the keyboard** controls the state machine (start / pause / exit)
- **Finger tracking runs simultaneously** with controller tracking

### Usage
```bash
# Uncomment --finger_tracking in teleop.sh, or pass directly:
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1 \
    --hand_type inspire \
    --finger_tracking \
    --redis_ip localhost \
    --target_fps 100
```

### Keyboard Controls (separate operator)
| Key              | Action                                  |
|------------------|-----------------------------------------|
| `s` or `Enter`   | Cycle: idle → teleop → pause → teleop  |
| `q`              | Exit program                            |
| `e`              | Emergency stop                          |

### Tuning Parameters
The `PicoFingerTracker` class in `deploy_real/data_utils/finger_tracking.py` has configurable parameters:
- `smoothing_alpha` (default `0.3`) — EMA smoothing factor (higher = smoother but more latency)
- `curl_gain` (default `1.5`) — Sensitivity multiplier (higher = hand closes faster)
- `curl_deadzone` (default `0.05`) — Minimum curl below which output is 0 (reduces jitter at rest)

### Dry-Run Test
You can verify the finger tracking module without any hardware:
```bash
cd deploy_real
python3 data_utils/finger_tracking.py
```

### Architecture
```
Pico 4 Ultra
├── Body pose (SMPLX via controllers on wrists) → XRobotStreamer → GMR retargeting → 35D mimic obs
└── Hand tracking (26 joints per hand)           → PicoFingerTracker → 6-DOF Inspire angles
                                                                           ↓
Keyboard operator → State machine ──────────────────────────────────────► Redis
                                                                           ↓
                                                          server_low_level_g1_real.py
                                                                           ↓
                                                         InspireHandController (Modbus TCP)
```


# Citation and Contact
If you find this work useful, please cite:
```bibtex
@article{ze2025twist2,
title={TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System},
author= {Yanjie Ze and Siheng Zhao and Weizhuo Wang and Angjoo Kanazawa and Rocky Duan and Pieter Abbeel and Guanya Shi and Jiajun Wuand C. Karen Liu},
year= {2025},
journal= {arXiv preprint arXiv:2511.02832}
}
```
And also consider citing the related works:
```bibtex
@article{ze2025twist,
title={TWIST: Teleoperated Whole-Body Imitation System},
author= {Yanjie Ze and Zixuan Chen and João Pedro Araújo and Zi-ang Cao and Xue Bin Peng and Jiajun Wu and C. Karen Liu},
year= {2025},
journal= {arXiv preprint arXiv:2505.02833}
}

@article{joao2025gmr,
title={Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking},
author= {Joao Pedro Araujo and Yanjie Ze and Pei Xu and Jiajun Wu and C. Karen Liu},
year= {2025},
journal= {arXiv preprint arXiv:2510.02252}
}
```
If you have any questions, please contact me at `yanjieze@stanford.edu`.


# Acknowledgments
We use [AMASS](https://amass.is.tue.mpg.de/) and [OMOMO](https://arxiv.org/abs/2309.16237) motion datasets for research purposes only. Our code is built upon the [TWIST](https://github.com/YanjieZe/TWIST) repository.
