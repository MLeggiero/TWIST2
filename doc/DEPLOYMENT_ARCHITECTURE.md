# TWIST2 Deployment Architecture

This document describes the deployment architecture of TWIST2 to help with porting to new simulations or custom hardware.

## System Overview

TWIST2 uses a **two-process architecture** with Redis for inter-process communication:

```
┌─────────────────────────────┐     Redis      ┌─────────────────────────────┐
│   High-Level Server         │◄──────────────►│   Low-Level Controller      │
│   (Motion Source)           │                │   (Policy Execution)        │
│                             │                │                             │
│ • Teleop (GMR + PICO VR)    │   action_*     │ • MuJoCo simulation         │
│ • Motion playback (.pkl)    │ ────────────►  │ • Real robot (Unitree SDK)  │
│                             │                │                             │
│ Runs in: gmr env (Py 3.10)  │   state_*      │ Runs in: twist2 env (Py 3.8)│
│                             │ ◄────────────  │                             │
└─────────────────────────────┘                └─────────────────────────────┘
```

## Redis Communication Protocol

### High-Level → Low-Level (Target Poses)

| Key | Dimensions | Description |
|-----|------------|-------------|
| `action_body_unitree_g1_with_hands` | 35 | Target body pose (mimic observation) |
| `action_hand_left_unitree_g1_with_hands` | 7 | Left hand joint targets |
| `action_hand_right_unitree_g1_with_hands` | 7 | Right hand joint targets |
| `action_neck_unitree_g1_with_hands` | 2 | Neck joint targets (yaw, pitch) |
| `t_action` | 1 | Timestamp in milliseconds |

### Low-Level → High-Level (Robot State)

| Key | Dimensions | Description |
|-----|------------|-------------|
| `state_body_unitree_g1_with_hands` | 34 | Current body state |
| `state_hand_left_unitree_g1_with_hands` | 7 | Left hand state |
| `state_hand_right_unitree_g1_with_hands` | 7 | Right hand state |
| `state_neck_unitree_g1_with_hands` | 2 | Neck state |
| `t_state` | 1 | Timestamp in milliseconds |

### Mimic Observation Format (35 dims)

```python
mimic_obs = [
    root_vel_local_x,      # Local x velocity (1)
    root_vel_local_y,      # Local y velocity (1)
    root_pos_z,            # Height above ground (1)
    roll,                  # Roll angle (1)
    pitch,                 # Pitch angle (1)
    yaw_ang_vel_local,     # Local yaw angular velocity (1)
    dof_pos[0:29],         # Joint positions (29)
]  # Total: 35 dims
```

### State Body Format (34 dims)

```python
state_body = [
    ang_vel[0:3],          # Angular velocity in world frame (3)
    roll,                  # Roll angle (1)
    pitch,                 # Pitch angle (1)
    dof_pos[0:29],         # Joint positions (29)
]  # Total: 34 dims
```

## Policy Architecture

### Network Structure

```
Input: 1402 dimensions
├── Current observation (127 dims)
│   ├── Mimic obs (35 dims) - from Redis
│   └── Proprio obs (92 dims) - from robot sensors
├── History buffer (1270 dims) - 10 frames × 127 dims
└── Future obs (35 dims) - currently same as current mimic obs

                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ActorFuture Network                      │
├─────────────────────────────────────────────────────────────┤
│  Motion Encoder (1D Conv):  35 → 128 latent                 │
│  History Encoder (1D Conv): 1270 → 128 latent               │
│  Future Encoder (MLP):      35 → 128 latent                 │
│                                                             │
│  Concatenate: 35 + 92 + 128 + 128 + 128 = 511               │
│                                                             │
│  Actor MLP: [512, 512, 256, 128] with SiLU + LayerNorm      │
└─────────────────────────────────────────────────────────────┘
                    ▼
Output: 29 dimensions (joint position offsets)
```

### Observation Construction

#### Proprioceptive Observation (92 dims)

```python
obs_proprio = np.concatenate([
    ang_vel * 0.25,                    # Angular velocity scaled (3)
    rpy[:2],                           # Roll, pitch (2)
    dof_pos - default_dof_pos,         # DOF position offset (29)
    dof_vel * 0.05,                    # DOF velocity scaled (29)
    last_action,                       # Previous action (29)
])  # Total: 92 dims
```

**Important scaling factors:**
- Angular velocity: `× 0.25`
- DOF velocity: `× 0.05`
- Ankle DOF velocities (indices 4, 5, 10, 11): set to 0

#### History Buffer

```python
# Circular buffer of last 10 observations
self.proprio_history_buf = deque(maxlen=10)

# Each frame stores: mimic_obs + proprio_obs = 127 dims
obs_full = np.concatenate([mimic_obs, obs_proprio])
self.proprio_history_buf.append(obs_full)

# Flatten for policy input
obs_hist = np.array(self.proprio_history_buf).flatten()  # 1270 dims
```

#### Final Observation Assembly

```python
obs_buf = np.concatenate([
    obs_full,      # Current: 127 dims
    obs_hist,      # History: 1270 dims
    future_obs,    # Future: 35 dims (= current mimic_obs for now)
])  # Total: 1402 dims
```

## Policy Loading

### ONNX Model Loading

```python
import onnxruntime as ort

def load_onnx_policy(policy_path: str, device: str):
    providers = []
    if device.startswith('cuda'):
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(policy_path, providers=providers)
    input_name = session.get_inputs()[0].name
    return OnnxPolicyWrapper(session, input_name)

class OnnxPolicyWrapper:
    def __init__(self, session, input_name):
        self.session = session
        self.input_name = input_name

    def __call__(self, obs_tensor):
        obs_np = obs_tensor.detach().cpu().numpy()
        outputs = self.session.run(None, {self.input_name: obs_np})
        return torch.from_numpy(outputs[0].astype(np.float32))
```

### Pretrained Checkpoint

Location: `assets/ckpts/twist2_1017_20k.onnx`

## Control Loop Implementation

### Simulation (MuJoCo)

```python
# Control frequency
sim_dt = 0.001          # Physics timestep (1000 Hz)
policy_frequency = 50   # Policy runs at 50 Hz
decimation = 20         # Run policy every 20 sim steps

# Main loop
for i in range(steps):
    if i % decimation == 0:
        # 1. Read robot state
        dof_pos = data.qpos[7:7+29]
        dof_vel = data.qvel[6:6+29]
        quat = data.qpos[3:7]
        ang_vel = data.qvel[3:6]

        # 2. Get mimic obs from Redis
        mimic_obs = json.loads(redis.get("action_body_unitree_g1_with_hands"))

        # 3. Build observation
        obs_buf = build_observation(mimic_obs, dof_pos, dof_vel, quat, ang_vel)

        # 4. Run policy
        action = policy(obs_buf)
        action = np.clip(action, -10., 10.)

        # 5. Compute target position
        pd_target = action * action_scale + default_dof_pos

    # 6. PD control (runs at 1000 Hz)
    torque = (pd_target - dof_pos) * stiffness - dof_vel * damping
    torque = np.clip(torque, -torque_limits, torque_limits)

    # 7. Apply torque
    data.ctrl[:] = torque
    mujoco.mj_step(model, data)
```

### Real Robot (Unitree G1)

```python
# Control frequency: ~50-100 Hz (limited by communication)
control_dt = 0.01  # 100 Hz target

while True:
    t_start = time.time()

    # 1. Read robot state via Unitree SDK
    low_state = robot.read_low_state()
    dof_pos = [low_state.motor.q[idx] for idx in joint2motor_idx]
    dof_vel = [low_state.motor.dq[idx] for idx in joint2motor_idx]
    quat = low_state.imu.quat
    ang_vel = low_state.imu.omega

    # 2-4. Same observation building and policy inference
    obs_buf = build_observation(...)
    action = policy(obs_buf)

    # 5. Compute target position
    target_dof_pos = action * action_scale + default_dof_pos

    # 6. Send to robot (PD computed on motor controllers)
    cmd = robot.create_zero_command()
    cmd.q_target = target_dof_pos
    cmd.dq_target = np.zeros(29)
    cmd.kp = kp_gains
    cmd.kd = kd_gains
    cmd.tau_ff = np.zeros(29)
    robot.write_low_command(cmd)

    # 7. Rate limiting
    elapsed = time.time() - t_start
    if elapsed < control_dt:
        time.sleep(control_dt - elapsed)
```

## Robot Configuration (G1)

### Joint Order (29 DOF)

```python
# Index: Joint name
# 0-5:   Left leg (hip_roll, hip_yaw, hip_pitch, knee, ankle_pitch, ankle_roll)
# 6-11:  Right leg (same order)
# 12-14: Torso (waist_yaw, waist_roll, waist_pitch)
# 15-21: Left arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
# 22-28: Right arm (same order)
```

### Default Joint Positions

```python
default_dof_pos = np.array([
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # Left leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,   # Right leg
    0.0, 0.0, 0.0,                     # Torso
    0.0, 0.4, 0.0, 1.2, 0.0, 0.0, 0.0, # Left arm
    0.0, -0.4, 0.0, 1.2, 0.0, 0.0, 0.0 # Right arm
])
```

### PD Gains

```python
stiffness = np.array([
    100, 100, 100, 150, 40, 40,  # Left leg
    100, 100, 100, 150, 40, 40,  # Right leg
    150, 150, 150,               # Torso
    40, 40, 40, 40, 4.0, 4.0, 4.0,  # Left arm
    40, 40, 40, 40, 4.0, 4.0, 4.0   # Right arm
])

damping = np.array([
    2, 2, 2, 4, 2, 2,  # Left leg
    2, 2, 2, 4, 2, 2,  # Right leg
    4, 4, 4,           # Torso
    5, 5, 5, 5, 0.2, 0.2, 0.2,  # Left arm
    5, 5, 5, 5, 0.2, 0.2, 0.2   # Right arm
])
```

### Action Scaling

```python
action_scale = 0.5  # For all joints
# Final position = default_dof_pos + action * action_scale
```

## Porting to New Platforms

### New Simulation Engine

1. **Implement state reading:**
   - Extract: `dof_pos`, `dof_vel`, `quat` (wxyz), `ang_vel`
   - Match the 29-DOF joint order

2. **Implement control application:**
   - Option A: Apply torques directly (compute PD in Python)
   - Option B: Send position targets with PD gains

3. **Match timing:**
   - Physics: 1000 Hz recommended
   - Policy: 50-100 Hz
   - Use decimation to sync

4. **Connect to Redis:**
   - Read `action_body_*` keys for target poses
   - Write `state_body_*` keys for robot state

### New Robot Hardware

1. **Create robot wrapper** (similar to `g1_wrapper.py`):
   ```python
   class NewRobotEnv:
       def get_robot_state(self):
           # Return: (dof_pos, dof_vel, quat, ang_vel, ...)
           pass

       def send_robot_action(self, target_dof_pos, kp_scale, kd_scale):
           # Send position command with PD gains
           pass
   ```

2. **Map joint indices:**
   - Create `joint2motor_idx` mapping from TWIST2 order to your robot's motor indices

3. **Adjust configuration:**
   - `default_dof_pos`: Standing pose for your robot
   - `stiffness` / `damping`: Tune for your actuators
   - `action_scale`: May need adjustment based on joint ranges

4. **Handle IMU:**
   - Ensure quaternion convention matches (w, x, y, z)
   - Angular velocity in body frame

### Key Files to Reference

| Purpose | File |
|---------|------|
| Simulation deployment | `deploy_real/server_low_level_g1_sim.py` |
| Real robot deployment | `deploy_real/server_low_level_g1_real.py` |
| Robot wrapper template | `deploy_real/robot_control/g1_wrapper.py` |
| Robot config | `deploy_real/robot_control/configs/g1.yaml` |
| Motion server (offline) | `deploy_real/server_motion_lib.py` |
| Teleop server | `deploy_real/xrobot_teleop_to_robot_w_hand.py` |
| Policy network | `rsl_rl/rsl_rl/modules/actor_critic_future.py` |
| ONNX export | `legged_gym/legged_gym/scripts/save_onnx.py` |
| Default poses | `deploy_real/data_utils/params.py` |

## GMR (Motion Retargeting) Integration

GMR converts human body tracking to robot joint positions:

```python
from general_motion_retargeting import GeneralMotionRetargeting as GMR

# Initialize
retarget = GMR(
    src_human="xrobot",        # Input: PICO VR SMPL-X format
    tgt_robot="unitree_g1",    # Output: G1 joint positions
    actual_human_height=1.6,   # Scale factor
)

# Each frame
qpos = retarget.retarget(smplx_data, offset_to_ground=True)
# qpos: [root_pos(3), root_quat(4), joint_pos(29)] = 36 dims
```

To use a different motion source:
1. Convert your tracking data to SMPL-X format, or
2. Implement a new source adapter in GMR, or
3. Directly compute the 35-dim mimic observation and publish to Redis

## Debugging Tips

1. **Verify observation dimensions:**
   ```python
   assert obs_buf.shape[0] == 1402, f"Expected 1402, got {obs_buf.shape[0]}"
   ```

2. **Check Redis connectivity:**
   ```python
   redis_client.ping()  # Should not raise exception
   ```

3. **Monitor policy FPS:**
   - Target: 50 Hz minimum
   - If lower, check GPU utilization and Redis latency

4. **Verify quaternion convention:**
   - TWIST2 uses (w, x, y, z) for IMU
   - MuJoCo uses (w, x, y, z) internally
   - Some SDKs use (x, y, z, w) - convert if needed

5. **Test with motion playback first:**
   ```bash
   bash run_motion_server.sh  # Provides known-good target poses
   bash sim2sim.sh            # Test policy in simulation
   ```
