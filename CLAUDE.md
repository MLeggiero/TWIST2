# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TWIST2 (Teleoperated Whole-body Imitation System 2) is a humanoid robot data collection and teleoperation system for the Unitree G1 robot. It uses reinforcement learning to train motion tracking controllers that can be deployed in simulation or on real hardware, with PICO VR for teleoperation.

## Conda Environments

Two separate conda environments are required:
- `twist2` (Python 3.8): Controller training, deployment, and data collection. Required for IsaacGym compatibility.
- `gmr` (Python 3.10): Online motion retargeting and PICO teleoperation. Uses the GMR (General Motion Retargeting) library.

## Common Commands

### Training
```bash
# Train the motion tracking controller
bash train.sh <experiment_id> <cuda_device>
# Example: bash train.sh 1021_twist2 cuda:0
```

### Export to ONNX
```bash
# Convert trained policy to ONNX for deployment
bash to_onnx.sh <path_to_pt_checkpoint>
```

### Simulation Testing (Sim2Sim)
```bash
# Terminal 1: Start motion server (provides target poses via Redis)
bash run_motion_server.sh

# Terminal 2: Run low-level controller in MuJoCo simulation
bash sim2sim.sh
```

### Real Robot Deployment (Sim2Real)
```bash
# First configure network interface in sim2real.sh (default: eno1)
# Robot IP: 192.168.123.164, Workstation IP: 192.168.123.222

# Run low-level controller on physical robot
bash sim2real.sh
```

### Teleoperation with PICO VR
```bash
# Requires gmr conda environment
bash teleop.sh
```

### GUI Interface
```bash
# All-in-one interface for simulation, deployment, teleop, and data collection
bash gui.sh
```

### Data Recording
```bash
bash data_record.sh
```

### Evaluation
```bash
bash eval.sh <experiment_id> <cuda_device>
```

## Architecture

### Core Packages (all installed in editable mode via `pip install -e .`)

- **legged_gym/**: IsaacGym environments for humanoid motion tracking
  - `legged_gym/envs/g1/`: G1-specific environment implementations
    - `g1_mimic_future.py`: Main training environment with future motion prediction
    - `g1_mimic_future_config.py`: Configuration for training
  - `legged_gym/scripts/train.py`: Training entry point
  - `legged_gym/scripts/save_onnx.py`: ONNX export script
  - Task registry uses naming convention: `g1_stu_future` is the primary task for training

- **rsl_rl/**: RL algorithms (PPO-based) from ETH Zurich
  - `rsl_rl/algorithms/`: PPO implementation
  - `rsl_rl/modules/`: Neural network modules
  - `rsl_rl/runners/`: Training loop runners

- **pose/**: Motion retargeting utilities (poselib-based)

- **deploy_real/**: Deployment servers for simulation and real robot
  - `server_low_level_g1_sim.py`: MuJoCo simulation server
  - `server_low_level_g1_real.py`: Real robot control server
  - `server_motion_lib.py`: Motion playback server (sends poses via Redis)
  - `xrobot_teleop_to_robot_w_hand.py`: PICO VR teleoperation handler
  - `server_data_record.py`: Data collection server

### Communication

Redis is used for inter-process communication between the high-level motion server (teleop or motion playback) and the low-level controller. The low-level controller runs at ~50-100 Hz.

### Key Assets

- `assets/ckpts/twist2_1017_20k.onnx`: Pre-trained controller checkpoint
- `assets/g1/`: Robot URDF/XML files
- `assets/example_motions/`: Example motion files (.pkl format)

## Hardware Setup for Real Robot

1. Connect to robot via Ethernet cable
2. Configure workstation network: IP `192.168.123.222`, netmask `255.255.255.0`
3. Put robot in debug mode: Press `L2+R2` on remote control
4. Run the deployment script with correct network interface
