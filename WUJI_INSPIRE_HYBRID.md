# Left Wuji Hand 2 + Right Inspire Hand

This optional hardware layout keeps the unchanged 29-DOF TWIST2 body policy,
controls a left Wuji Hand 2 through the Wuji SDK, and controls only the right
Inspire RH56 hand through Modbus TCP. Existing Dex3, dual-Inspire, and dual-Wuji
commands retain their original defaults.

## Architecture and ownership

```text
PICO body -> GMR -> action_body_* -> TWIST2 ONNX -> physical G1 body

PICO left hand -> MediaPipe 21x3 -> wuji_hand_bridge.py -> left Wuji Hand 2

PICO right controller -> 6-D Inspire target -> server_low_level_g1_real.py
                                             -> right Inspire hand
```

Only one process owns each physical side. Do not run the legacy
`sim2real.sh --use_hand`, a second Wuji bridge, or `test_inspire_hands.py` at
the same time as the hybrid real controller.

| Redis key | Shape | Producer | Consumer |
|---|---:|---|---|
| `action_body_unitree_g1_with_hands` | 35 | hybrid producer | TWIST2 body controller |
| `pico_body_timestamp` | scalar | hybrid producer | TWIST2 body freshness guard |
| `pico_hand_left_mediapipe` | 21x3 | hybrid producer | left Wuji bridge |
| `pico_hand_left_timestamp` | scalar | hybrid producer | left Wuji bridge, recorder |
| `wuji_action_hand_left` | 20 | left Wuji bridge | recorder/simulation |
| `wuji_state_hand_left` | 20 | Wuji SDK or simulation | recorder |
| `inspire_action_hand_right` | 6 | hybrid producer | real right-Inspire controller |
| `inspire_action_timestamp_right` | scalar | hybrid producer | real right-Inspire controller |
| `inspire_command_hand_right` | 6 | real controller | recorder |
| `inspire_state_hand_right` | 6 | real controller | recorder |
| `inspire_state_timestamp_right` | scalar | real controller | recorder |
| `inspire_force_hand_right` | 6 | real controller | recorder |
| `inspire_tactile_hand_right` | 1062 | real controller | recorder |

Wuji values are radians in Wuji device/MJCF order. Inspire values are device
counts ordered `pinky, ring, middle, index, thumb_bend, thumb_rotation`, with
`1000=open` and `0=closed`.

## PICO controls

- Right A: idle -> teleop -> pause -> teleop.
- Right grip: right Inspire pinky/ring/middle.
- Right trigger: right Inspire index finger.
- Right joystick: right Inspire thumb bend/rotation.
- Left joystick: G1 planar velocity. Yaw is deliberately zero in hybrid mode
  because the right joystick has one unambiguous owner.
- Left X: toggle Inspire precision mode.
- Left joystick click: request emergency termination of real G1 controllers.

Disable XRoboToolkit **Switch w/ A Button** so it does not consume the teleop
state button.

## Software-only preflight

First validate the producer and left Wuji retargeting without hardware:

```bash
export ACTUAL_HUMAN_HEIGHT=<HEIGHT_METERS>
bash teleop_wuji_left_inspire_right.sh
```

Press right A once and confirm the GMR preview follows the whole body. In another
terminal:

```bash
export WUJI_LEFT_CONFIG=/absolute/path/to/left.yaml
bash wuji_hand_bridge.sh --dry-run --velocity-limit-scale 0.2
```

The existing combined simulator can validate the body and left Wuji command path:

```bash
export WUJI_LEFT_CONFIG=/absolute/path/to/left.yaml
bash sim2sim_wuji.sh
```

The repository currently has no Inspire URDF/MJCF. With only the left config,
the simulator retains the right zero-mass rubber placeholder; it does not show
the right Inspire fingers or accurately model their mass. Obtain an Inspire
model or add a measured rigid mass/inertia proxy before treating simulation as
a validation of asymmetric balance dynamics. The existing
`g1_sim2sim_29dof_with_hands.xml` is a 7-DOF Unitree/Dex3-style model, not an
Inspire model.

## Right Inspire hand-only preflight

First inspect the dedicated hybrid command stream without hardware:

```bash
conda activate gmr
cd deploy_real
python test_inspire_hybrid_right.py
```

This default mode imports no Modbus/Wuji hardware backend and makes no device
connection. Once the values and directions look correct, explicitly opt into a
supported hand-only hardware test:

```bash
conda activate <REAL_ENV_WITH_PYMODBUS>
cd deploy_real
python test_inspire_hybrid_right.py \
  --hardware \
  --right-ip <RIGHT_INSPIRE_IP>
```

The hardware test uses the same measured-state startup interpolation,
application slew ceiling, timestamp timeout, strict Modbus behavior, and
hold-on-exit policy as the hybrid controller. Do not run it concurrently with
the hybrid real controller. The older `test_inspire_hands.py` consumes legacy
dual-hand Redis keys and is not the hybrid-stream preflight.

## Physical launch sequence

Do not proceed until the GMR preview is upright, left Wuji dry-run succeeds, the
right Inspire passes its one-sided test, and the asymmetric end-effector mass is
considered safe with a support/harness.

The real launch follows the existing repository convention and activates
`twist2_deploy`. That environment is not installed on the desktop inspected
during implementation, and its installed `twist2` environment currently lacks
`pymodbus`. On the real-G1 computer, use an environment containing the existing
Unitree/TWIST2 dependencies plus `onnxruntime`, `redis`, and `pymodbus`. If its
name differs, set `TWIST2_REAL_ENV=<NAME>`; the default remains
`twist2_deploy`.

1. Start Redis and XRoboToolkit.
2. Start the hybrid producer:

   ```bash
   ACTUAL_HUMAN_HEIGHT=<HEIGHT_METERS> \
     bash teleop_wuji_left_inspire_right.sh
   ```

3. Start only the left Wuji bridge in hardware mode:

   ```bash
   WUJI_LEFT_CONFIG=/absolute/path/to/left.yaml \
   WUJI_LEFT_IP=<LEFT_WUJI_ADDRESS> \
     bash wuji_hand_bridge.sh --velocity-limit-scale 0.2
   ```

4. Start the G1 plus right Inspire controller:

   ```bash
   G1_NET=<G1_INTERFACE> \
   INSPIRE_RIGHT_IP=<RIGHT_INSPIRE_IP> \
     bash sim2real_wuji_left_inspire_right.sh
   ```

The launch refuses to use an implicit right Inspire address. It runs the normal
G1 START/A initialization prompts, connects no left Inspire device, requires a
fresh PICO body timestamp, and uses the physical remote Select button to exit.

## Right Inspire safety behavior

Hybrid mode is additive; legacy dual-Inspire defaults are unchanged.

- Exact finite 6-D targets are required and clipped to `[0, 1000]`.
- Targets older than `0.5 s` are ignored and the last applied command is held.
- The first target starts at measured joint position and is interpolated for
  `1.75 s`.
- An application-level slew ceiling defaults to `250 counts/s`; delayed loops
  cannot issue a large catch-up step.
- Modbus bootstrap failures are strict and failed command writes are retried.
- Shutdown disconnects without commanding the right hand open.

Override the conservative operational settings only after a hand-only test:

```bash
INSPIRE_ACTION_TIMEOUT=0.5 \
INSPIRE_COMMAND_RATE_LIMIT=250 \
INSPIRE_STARTUP_DURATION=1.75 \
  bash sim2real_wuji_left_inspire_right.sh
```

These are application limits, not claimed native Inspire acceleration or jerk
limits. No automatic open command is issued on tracking loss or shutdown.

## Recording

```bash
bash data_record_wuji_left_inspire_right.sh
```

The logical episode fields remain `state_hand_left/right` and
`action_hand_left/right`, but their schemas are intentionally asymmetric:

- left state/action: 20-D Wuji radians;
- right state/action: 6-D Inspire device counts;
- human hand landmarks: optional MediaPipe 21x3 meters for either side.

Per-side timestamps are validated. Missing or stale data becomes `None`; it is
not reshaped or silently treated as the other hand backend.

## Remaining physical assumptions

- The right Inspire and its adapter mass/center of mass are not represented in
  the current MuJoCo hybrid model.
- The right Inspire IP does not prove handedness; verify the physical device and
  address before enabling the G1.
- The G1, Wuji, and Inspire processes have separate device connections. Keep the
  physical remote and terminal operators available during initial testing.
- Test with low arms and a support/harness before large asymmetric arm motions.
