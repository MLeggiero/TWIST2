# Testing TWIST2 + Wuji Hand 2 in MuJoCo

The asymmetric left-Wuji/right-Inspire workflow is documented separately in
[`WUJI_INSPIRE_HYBRID.md`](WUJI_INSPIRE_HYBRID.md). This repository currently
has no Inspire MJCF, so the standard combined simulator can validate a left-only
Wuji command path but not right-Inspire articulation or accurate hybrid mass.

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
command validation, cached-frame rejection, and hardware-free model-only operation.

## PICO full-body setup and freshness preflight

Start the installed XRoboToolkit PC service in its own terminal:

```bash
/opt/apps/roboticsservice/run3D.sh
```

Keep the service and its UI running. This is the command used by the installed
`XRoboToolkit-PC-Service` desktop launcher; its wrapper configures the bundled
library and Qt paths, so do not run `RoboticsServiceProcess` directly.

For a headset, two hand controllers, and two ankle-mounted PICO Motion Trackers:

1. Wear one tracker on each ankle in the orientation shown by PICO.
2. Open the PICO Motion Tracker calibration flow and complete the quick full-body
   calibration while standing upright. PICO requires this again after each headset
   activation.
3. In the XRoboToolkit headset panel, confirm the PC connection says `WORKING`.
4. Enable Head, Controller, and Hand tracking as needed. Set **PICO Motion Tracker
   Mode** to **Full Body**, with tracker count 2, and turn **Send** on.
5. Turn **Switch w/ A Button** off. XRoboToolkit can use A to pause/resume Send,
   while TWIST2 also uses right-controller A to enter/pause teleoperation. Leaving
   both enabled can freeze the exact frame that TWIST2 then tries to track.

These settings follow the upstream
[XRoboToolkit Unity client panel](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client#unity-ui-main-panel-reference)
and PICO's
[body-tracking calibration sequence](https://github.com/picoxr/Pico-Body-Tracking-Demo#usage).

`teleop_wuji.sh` enables `--require-fresh-xr-tracking`. A cached frame is not
accepted merely because the SDK still reports body data available: its source
timestamp must change at least once. Only advancing frames refresh
`pico_body_timestamp` and the PICO hand timestamps.

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

The producer's MuJoCo window is a kinematic GMR preview, not the dynamically
balanced robot. Before pressing A or launching the combined simulator, stand
upright and move your head, controllers, and ankles. The preview must follow. If
it remains kneeling or frozen, stop here and repair the XR settings/calibration.
The console should otherwise warn that the XR body timestamp is not advancing.

Verify the source heartbeat from another terminal:

```bash
python - <<'PY'
import time
import redis
r = redis.Redis()
raw = r.get("pico_body_timestamp")
print("missing" if raw is None else f"age={time.time() - float(raw):.3f}s")
PY
```

An actively streaming body should report an age below the default 0.5-second
source timeout. This check tests source liveness, unlike `t_action`, which is the
producer loop timestamp.

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
TWIST2 ONNX checkpoint, enables `--require-fresh-pico-body`, and opens one MuJoCo
viewer. Start it while TWIST2 is still in idle; the G1 should stand. Once the GMR
preview is upright and moving, press right-controller A once to enter teleop.
At startup, verify that `TWIST2 ONNX checkpoint:` prints the expected resolved
checkpoint path. The default is `assets/ckpts/twist2_1017_20k.onnx`; set
`TWIST2_POLICY` to override it.
Expected behavior:

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
If a hand command or timestamp is absent, malformed, or stale, it holds the last
valid hand target. If PICO body input is missing or stale, the simulator keeps
running the closed-loop balance policy on the last valid 35-D target; before the
first proven-live PICO frame, that target is the normal upright default. It never
accepts the SDK's cached startup pose or substitutes an open/zero hand pose.
Warnings are throttled.

## Test 5: tracking-loss and ownership checks

While the simulator is moving slowly:

1. Stop `teleop_wuji.sh`. The simulator must continue balancing against its last
   valid high-level mimic target after the body timestamp timeout.
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
- missing `pico_body_timestamp` or frozen/kneeling preview: recalibrate the two
  ankle trackers, select Full Body, turn Send on, and disable XRoboToolkit's
  `Switch w/ A Button`; do not start body teleop until the preview moves.
- missing `t_action`: start `teleop_wuji.sh`; the simulator keeps balancing on its
  safe default/last valid target until a valid command arrives.
- stale hand action: confirm `wuji_hand_bridge.sh --dry-run` is running and all
  processes use the same `REDIS_IP` and `REDIS_PORT`.
- incorrect finger motion: compare the printed actuator order with standalone
  `teleop_sim.py` and verify the corrected `Palm, Wrist, ...` PICO skeleton order.
