# Testing TWIST2 + Wuji Hand 2 in MuJoCo

This runbook tests the unchanged 29-DOF TWIST2 body policy together with one or
two 20-DOF Wuji Hand 2 models. It is a software-only gate before physical G1 or
Wuji operation. Nothing in this workflow imports `wuji_sdk`, enables motors, or
sends a Unitree command.

## What is simulated

```text
PICO / XRoboToolkit
        |
        v
teleop_wuji.sh
  |-- action_body_unitree_g1_with_hands (35)
  `-- pico_hand_{left,right}_mediapipe (21x3)
                                      |
                                      v
                         wuji_hand_bridge.sh --dry-run
                         Retargeter + URDF->MJCF permutation
                         + commanded velocity limiting
                                      |
                                      v
                         wuji_action_hand_{left,right} (20)
                                      |
                  +-------------------+------------------+
                  |                                      |
                  v                                      v
       TWIST2 ONNX body policy                  Wuji position targets
           29 body outputs                    20 actuators per hand
                  |                                      |
                  +-------------------+------------------+
                                      v
                            one combined MuJoCo model
```

`server_low_level_g1_wuji_sim.py` composes the model at runtime with MuJoCo's
`MjSpec.attach`. It loads:

- `assets/g1/g1_sim2sim_29dof.xml` unchanged;
- the exact left/right Hand 2 MJCF paths referenced by the retargeter YAML files.

This avoids copying the Wuji meshes, joint ranges, inertias, or actuator gains
into TWIST2. A dual-hand model has `nq=76`, `nv=75`, and `nu=69`: 29 body
actuators plus 20 actuators per hand. The TWIST2 policy remains 29-D.

The original `sim2sim.sh` and `server_low_level_g1_sim.py` remain the legacy
body/Dex3/Inspire path and are not changed by this integration.

## Environment

The launcher uses the `gmr` environment by default because it already contains
the Wuji retargeter and the MuJoCo version with `MjSpec.attach`. Install the
simulation-only runtime in that environment if needed:

```bash
conda activate gmr
python -m pip install "mujoco>=3.2" onnxruntime redis pyyaml
```

Use `onnxruntime-gpu` instead of `onnxruntime` only when CUDA execution is
deliberately configured. CPU inference is the default for the test launcher.

Set the configuration paths once:

```bash
cd ~/Projects/TWIST2

export WUJI_LEFT_CONFIG="$PWD/../wuji-hand/wuji-retargeting/example/config/adaptive_analytical_wuji_glove_wuji_hand_2_left.yaml"
export WUJI_RIGHT_CONFIG="$PWD/../wuji-hand/wuji-retargeting/example/config/adaptive_analytical_wuji_glove_wuji_hand_2_right.yaml"
```

For one-sided simulation, set only the corresponding variable.

## Test 1: compile the combined model

This test does not connect to Redis and does not load the ONNX policy:

```bash
conda activate gmr

python deploy_real/server_low_level_g1_wuji_sim.py \
  --left-config "$WUJI_LEFT_CONFIG" \
  --right-config "$WUJI_RIGHT_CONFIG" \
  --model-only
```

Expected dual-hand summary:

```text
nq=76, nv=75, nu=69
left=20, right=20
MODEL ONLY: no Redis connection, policy inference, or hardware access
```

To preserve the composed model as a reloadable MuJoCo binary:

```bash
python deploy_real/server_low_level_g1_wuji_sim.py \
  --left-config "$WUJI_LEFT_CONFIG" \
  --right-config "$WUJI_RIGHT_CONFIG" \
  --model-only \
  --save-composed-model /tmp/g1_wuji_hand2.mjb
```

The MJB contains the compiled model and can be reloaded with MuJoCo without
recomposing the source models. Keep the YAML/MJCF files as the source of truth.

## Test 2: run automated software checks

```bash
conda activate gmr
python -m unittest tests.test_wuji_mujoco_sim \
  tests.test_wuji_mujoco_controller -v
```

These checks verify model dimensions, actuator ordering against the official
standalone Wuji MJCFs, placeholder-hand removal, finite physics stepping, JSON
command validation, and hardware-free model-only operation.

## Test 3: validate PICO hand retargeting without hardware

Start the existing Redis service:

```bash
redis-cli ping
```

Start XRoboToolkit and connect PICO. In another terminal, start the producer:

```bash
cd ~/Projects/TWIST2
REDIS_IP=localhost ACTUAL_HUMAN_HEIGHT=<HEIGHT_METERS> bash teleop_wuji.sh
```

Start the Wuji bridge in dry-run mode:

```bash
cd ~/Projects/TWIST2

bash wuji_hand_bridge.sh \
  --dry-run \
  --velocity-limit-scale <VALIDATED_SCALE>
```

Dry-run must print `input=(21, 3) output=(20,)`. It publishes the limited target
that would be sent to hardware but does not import the SDK or discover, connect,
enable, or command a physical hand.

Inspect the Redis shapes if desired:

```bash
redis-cli GET pico_hand_left_mediapipe
redis-cli GET wuji_action_hand_left
```

Before continuing, verify that an open PICO hand resembles the working Wuji
`teleop_sim.py` result and that PIP joints are not incorrectly saturated near
2.094 rad. Such saturation is a warning sign for a Palm/Wrist ordering error.

## Test 4: live G1 + Wuji MuJoCo integration

Keep the producer and dry-run bridge running, then start the combined simulator:

```bash
cd ~/Projects/TWIST2
bash sim2sim_wuji.sh
```

The launcher reads `WUJI_LEFT_CONFIG`/`WUJI_RIGHT_CONFIG`, uses the provided
TWIST2 ONNX checkpoint, and opens one MuJoCo viewer. Expected behavior:

- the G1 stands under the existing 29-DOF policy;
- body motion follows the 35-D PICO/GMR target;
- each configured Wuji hand follows its independent 20-D Redis target;
- no Dex3 action key is required;
- `wuji_state_hand_left/right` contains simulated joint positions in official
  MJCF actuator/device order.

Useful overrides:

```bash
# Force CPU or request CUDA ONNX inference.
TWIST2_DEVICE=cpu bash sim2sim_wuji.sh
TWIST2_DEVICE=cuda bash sim2sim_wuji.sh

# Select another environment containing mujoco, onnxruntime, redis and PyYAML.
TWIST2_MUJOCO_ENV=<ENV_NAME> bash sim2sim_wuji.sh

# Run a bounded, headless integration test (5,000 physics steps).
bash sim2sim_wuji.sh --headless --no-realtime --max-steps 5000

# Hold commands sooner when their publisher exits.
bash sim2sim_wuji.sh \
  --body-command-timeout 0.25 \
  --hand-command-timeout 0.25
```

The simulator clips hand commands to the official MJCF actuator control ranges.
If a command or timestamp is absent, malformed, or stale, it holds the last valid
target and prints a throttled warning. It never substitutes an open or zero pose
after tracking loss.

## Test 5: tracking-loss and ownership checks

While the simulator is moving slowly:

1. Stop `teleop_wuji.sh`. The body must hold its last PD target after the body
   timestamp timeout.
2. Stop the dry-run bridge. Each hand must hold its most recent valid target
   after the hand action timestamp timeout.
3. Restart both and confirm that control resumes without a zero/open-hand jump.
4. Confirm `wuji_state_hand_left/right` remains exactly 20 values per configured
   side.
5. Confirm no physical Wuji device appears in bridge output; dry-run must always
   print its no-hardware banner.

Never run `sim2sim_wuji.sh` and physical G1/Wuji consumers against the same Redis
instance at the same time. They share command/state key names by design.

## Mount transform caveat

The default mount removes the G1 fixed rubber-hand visual and places the official
Wuji wrist model at the fixed-hand location. It rotates Wuji local `-Z` into the
G1 wrist's `+X` direction:

```text
left position:   0.0415,  0.003, 0
right position:  0.0415, -0.003, 0
quaternion WXYZ: 0.7071, 0, -0.7071, 0
```

This is suitable for software integration and visualization, but it is not a
measured CAD transform for a physical adapter. If a verified adapter transform
becomes available, override it without editing either vendor model:

```bash
bash sim2sim_wuji.sh \
  --left-mount-pos X Y Z \
  --left-mount-quat W X Y Z \
  --right-mount-pos X Y Z \
  --right-mount-quat W X Y Z
```

Validate orientation, wrist clearance, self-collision, fingertip contact, and
grasp geometry visually before using simulation results for mechanical claims.

## Gate before real hardware

Proceed to physical testing only after all of the following pass:

- standalone Wuji `teleop_sim.py`;
- bridge `--dry-run` with open-hand sanity checks;
- combined model compilation and automated tests;
- live G1 + Wuji MuJoCo operation;
- tracking-loss hold behavior;
- conservative task-specific velocity-scale selection.

Then follow the physical launch sequence in
[`WUJI_HAND_INTEGRATION.md`](WUJI_HAND_INTEGRATION.md#combined-launch). Stop the
MuJoCo processes first. The G1 and Wuji stop paths are independent; keep both
hardware stops accessible and do not rely on the legacy PICO process-kill handler
to stop `sim2real_wuji.sh` or `wuji_hand_bridge.py`.

## Troubleshooting

- `No module named onnxruntime`: install `onnxruntime` in the environment selected
  by `TWIST2_MUJOCO_ENV`.
- `MjSpec` missing: upgrade that environment to MuJoCo 3.2 or newer.
- `optimizer.mjcf_path is missing`: use the Hand 2 YAML that references both the
  official URDF and MJCF.
- missing meshes: preserve the adjacent Wuji description checkout and its
  relative directory structure.
- missing `t_action`: start `teleop_wuji.sh`; the simulator intentionally refuses
  untimestamped body targets.
- stale hand action: confirm `wuji_hand_bridge.sh --dry-run` is running and all
  processes use the same `REDIS_IP` and `REDIS_PORT`.
- incorrect finger motion: compare the printed actuator order with standalone
  `teleop_sim.py` and verify the corrected `Palm, Wrist, ...` PICO skeleton order.
