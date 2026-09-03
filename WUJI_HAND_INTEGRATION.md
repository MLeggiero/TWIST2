# TWIST2 + PICO + Wuji Hand 2

## Architecture and compatibility

The existing TWIST2 body policy is unchanged: GMR publishes a 35-D mimic target and
the low-level policy commands the original 29 G1 body joints. Wuji fingers are never
part of the policy observation or action. A separate process retargets PICO hand
landmarks to the 20 Wuji joints.

Existing commands remain in legacy mode when no new flags are supplied. In
particular, `--hand-output-mode dex3` is the default and preserves both existing
Dex3 (7-D) and Inspire (6-D) behavior selected by `--hand_type`. The existing
`teleop.sh`, `sim2real.sh`, and `data_record.sh` are not used by Wuji mode.

```text
PICO -> XRobotStreamer/GMR -> 35-D body Redis -> TWIST2 -> 29 G1 joints
                         \-> 21x3 hand Redis -> Wuji bridge -> 20 hand joints
```

## Environments

- `gmr`: XRoboToolkit producer, `wuji_retargeting`, MuJoCo ordering helpers,
  `wuji_sdk`, and `wuji_hand_bridge.py`.
- `twist2_deploy`: physical G1 body controller.
- `twist2`: data recorder.

The bridge configuration paths may point into the adjacent `wuji-retargeting`
checkout. Current local Wuji Hand 2 configs are named
`adaptive_analytical_wuji_glove_wuji_hand_2_{left,right}.yaml`; despite the name,
they consume standard MediaPipe 21x3 landmarks.

## Redis schema

All new timestamps are Unix epoch seconds. The legacy `t_action` remains milliseconds.

| Key | Shape | Producer | Consumer |
|---|---:|---|---|
| `pico_hand_left_mediapipe` | 21x3 | PICO/GMR producer | bridge, recorder |
| `pico_hand_right_mediapipe` | 21x3 | PICO/GMR producer | bridge, recorder |
| `pico_hand_left_timestamp` | scalar | PICO/GMR producer | bridge, recorder |
| `pico_hand_right_timestamp` | scalar | PICO/GMR producer | bridge, recorder |
| `wuji_action_hand_left` | 20 | bridge | recorder |
| `wuji_action_hand_right` | 20 | bridge | recorder |
| `wuji_state_hand_left` | 20 | bridge | recorder |
| `wuji_state_hand_right` | 20 | bridge | recorder |
| `wuji_action_timestamp_left/right` | scalar | bridge | recorder |
| `wuji_state_timestamp_left/right` | scalar | bridge | recorder |

Legacy Dex3/Inspire action and state keys are unchanged and are not written by the
PICO producer in Wuji mode.

## PICO to MediaPipe mapping

The 21 points are Wrist; four Thumb points; four each for Index, Middle, Ring, and
Little, using MediaPipe order. Each result is `float32`, remains in metres, and is
made wrist-relative. The XRoboToolkit source skeleton must start with `Palm, Wrist`.
Point zero deliberately selects the actual `Wrist`, not `Palm`. Do not revert the
corrected GMR joint-name order.

## Wuji joint order

The retargeter returns URDF/Pinocchio order. The bridge resolves the config's MJCF,
matches joints by name, and permutes to MJCF/device finger-major order before sending
or recording: finger1 through finger5, joint1 through joint4. If an MJCF is declared
but names cannot be matched, the bridge aborts instead of assuming identity.

Measured state frames are variable-length SDK packets. The bridge maps entries by
explicit `nid` and publishes only complete 20-joint frames; packet order is ignored.

## Commanded velocity limits

Every hardware command and every dry-run action passes through a per-joint position
slew limiter. The bridge reads the 20 velocity ceilings from the configured Hand 2
URDF, applies the same URDF-to-device permutation as qpos, and limits each command
step to:

```text
abs(q_command - q_previous) <= max_velocity * elapsed_time
```

Elapsed time is capped at one nominal control period. A delayed Python/Redis loop
therefore slows down instead of issuing a large catch-up step. The limiter runs at
the bridge control rate even between new PICO frames. `wuji_action_hand_left/right`
contains the limited position actually sent (or that would be sent in dry-run), not
the unconstrained retargeter target.

`--velocity-limit-scale` may reduce every URDF ceiling and accepts only values in
`(0, 1]`. Its default is `1.0`, which enforces the URDF ceilings without claiming a
more conservative operational limit. For physical commissioning, choose a lower
scale approved for the task and hand setup, for example:

```bash
bash wuji_hand_bridge.sh --velocity-limit-scale <APPROVED_SCALE>
```

The bridge intentionally does not invent acceleration or jerk limits. Ruckig may
be used later when validated acceleration limits are available; it is not needed
for this velocity-only limiter. The limit applies to commanded positions and is
not a safety-rated guarantee of measured physical velocity.

## Dry-run and simulation

For the combined G1 plus attached Wuji Hand 2 workflow, see the
[Wuji MuJoCo test runbook](WUJI_MUJOCO_TESTING.md).

First confirm the upstream Wuji simulation works:

```bash
conda activate gmr
cd ~/Projects/wuji-hand/wuji-retargeting/example
python teleop_sim.py --hand left --config config/adaptive_analytical_wuji_glove_wuji_hand_2_left.yaml
```

Then run the TWIST2 bridge without hardware:

```bash
conda activate gmr
cd ~/Projects/TWIST2
python deploy_real/wuji_hand_bridge.py \
  --left-config ../wuji-hand/wuji-retargeting/example/config/adaptive_analytical_wuji_glove_wuji_hand_2_left.yaml \
  --dry-run
```

Dry-run connects only to Redis, performs retargeting/permutation, and publishes the
20-D action. It does not import the hardware SDK path, scan, connect, enable, or send
motor commands. `--max-frames 1` provides a bounded integration check.

Inspect an open-hand target before hardware testing. PIP joints unexpectedly near
2.094 rad commonly indicate the Palm/Wrist mapping is wrong.

## Physical hand testing

**Do not begin physical testing until both `teleop_sim.py` and bridge `--dry-run`
succeed. Keep an emergency stop accessible and start with conservative validated
gains/current limits.**

One hand:

```bash
export WUJI_LEFT_CONFIG=../wuji-hand/wuji-retargeting/example/config/adaptive_analytical_wuji_glove_wuji_hand_2_left.yaml
export WUJI_LEFT_IP=<ADDRESS>
bash wuji_hand_bridge.sh
```

For two hands, set both config and address pairs. Dual-hand hardware mode requires
explicit addresses. For one hand only, omitting its address discovers a device only
when exactly one Wuji Hand 2 is visible.

On the first fresh target, the bridge reads measured device-order q and interpolates
to the target over at least 1.75 seconds by default. Startup commands pass through
the same velocity limiter, so startup may take longer when required by a joint's
velocity ceiling. Dry-run does not interpolate.

## Combined launch

1. Start the existing Redis service.
2. Start XRoboToolkit and connect PICO.
3. Run `bash teleop_wuji.sh` in the `gmr` environment.
4. Run `bash wuji_hand_bridge.sh --dry-run` with config environment variables.
5. After validation, restart the bridge without `--dry-run` and with hand addresses.
6. Run `bash sim2real_wuji.sh` for the physical G1 body.
7. Run `bash data_record_wuji.sh` to record Wuji logical hand fields.

`sim2real_wuji.sh` passes `--ignore-hand-actions`, so the body controller neither
initializes nor reads legacy hand keys and never constructs Dex3/Inspire controllers.

## Recording

`--hand-backend dex3` is the recorder default and uses all original keys. Wuji mode
maps the new state/action keys into `state_hand_left/right` and
`action_hand_left/right`, adds `human_hand_left/right`, and records their timestamps.
Missing, malformed, or stale sides become `None`. Wuji episode metadata and schemas
declare 20-D radians and MediaPipe 21x3 metres; legacy metadata is unchanged.

## Tracking loss

The default freshness timeout is 0.25 seconds. On loss, no new target is calculated
or sent; the last target is held and warnings are throttled. Zeros or an open pose are
never substituted. `--disable-after-timeout 0` disables automatic motor shutdown.
If a positive long timeout is selected and expires, motors are disabled and the
bridge must be restarted before motion resumes.

## Troubleshooting

- `tracking stale`: confirm producer/bridge clocks and PICO timestamps.
- invalid `21x3`: verify all required XRoboToolkit keys and corrected Palm/Wrist order.
- URDF/MJCF alignment error: verify both assets in the YAML describe the same hand.
- partial measured state: inspect offline joints/diagnostics; incomplete frames are
  intentionally not published.
- multiple devices found: pass the correct explicit hand address.
- Redis connection failure: set `REDIS_IP`/`REDIS_PORT` consistently for all processes.
