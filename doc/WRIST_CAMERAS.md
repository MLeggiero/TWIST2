# Arducam Wrist Camera Integration

## Overview

Two Arducam USB2.0 fisheye cameras are mounted on the G1's wrists, one per
side, connected to the Orin via USB (like the internal D435i). Each camera
is streamed **RGB only** (no depth) at **640x480 @ 30fps** over ZMQ to the
workstation, where `server_data_record_d435.py` merges both wrist streams
into the *same* episode/`data.json` as the D435 RGB+depth and robot
state/action, so all camera views stay frame-aligned for fine-manipulation
policy training.

Fisheye distortion is **not** corrected — raw distorted frames are recorded
as-is, consistent with how raw depth is recorded today. Undistortion is out
of scope here and left as future work for whichever consumer needs it.

## Prerequisites

On the Orin (robot side): same `requirements.txt` as the D435/MID-360
streamers (`pyzmq`, `numpy`, `opencv-python-headless`) — no new dependencies.
Unlike the D435i, the wrist cameras are plain UVC devices, so `pyrealsense2`
is not needed for them.

```bash
pip install -r ~/g1-onboard/requirements.txt
```

## Hardware Check

SSH into the Orin and enumerate the USB video devices:

```bash
ssh unitree@192.168.123.164
sudo apt install -y v4l-utils   # if not already installed
v4l2-ctl --list-devices
```

You should see two entries for the Arducam modules (they will likely report
identical or near-identical names/vendor IDs since both are the same
model). Note their current `/dev/videoN` nodes — these are **not** stable
across reboots/replugs when two identical devices are on the bus, so do not
hardcode them; use the udev procedure below instead.

## udev Stable Device Symlinks

Because both cameras share the same USB `idVendor:idProduct` (and possibly
serial), `/dev/videoN` enumeration order can change across reboots or
replugs. This procedure creates persistent `/dev/video_wrist_left` /
`/dev/video_wrist_right` symlinks keyed on physical USB port topology, run
once on the actual robot:

1. With both cameras plugged in, find their current nodes:
   ```bash
   v4l2-ctl --list-devices
   ```
2. For each camera's `/dev/videoN`, find its sysfs device path and walk up
   to a stable, port-specific attribute:
   ```bash
   udevadm info -q path -n /dev/videoN
   udevadm info -a -p <path from above>
   ```
   Look for a `KERNELS=="..."` value tied to the physical USB port chain
   (e.g. something like `1-2.1`) rather than the device's serial, since
   identical units may not have distinguishable serials.
3. Check whether the device exposes multiple `/dev/video*` interfaces (some
   UVC cameras expose metadata/still-capture nodes alongside the video
   node) and confirm which one is the MJPG-capable capture node:
   ```bash
   v4l2-ctl -d /dev/videoN --list-formats-ext
   ```
4. Create `/etc/udev/rules.d/99-wrist-cameras.rules` on the Orin with two
   rules (fill in the real `KERNELS` values found in step 2 — they are
   hardware-specific and cannot be guessed):
   ```
   SUBSYSTEM=="video4linux", KERNELS=="<left-port-path>", SYMLINK+="video_wrist_left"
   SUBSYSTEM=="video4linux", KERNELS=="<right-port-path>", SYMLINK+="video_wrist_right"
   ```
5. Reload and verify:
   ```bash
   sudo udevadm control --reload-rules && sudo udevadm trigger
   v4l2-ctl --list-devices   # confirm /dev/video_wrist_left and _right exist
   ```
   Replug both cameras (or reboot) and re-run the check to confirm the
   symlinks persist.

## Deploy to Robot

From the workstation:

```bash
bash deploy_real/onboard/deploy_to_robot.sh
```

This now also copies `wrist_streamer.py`, `start_wrist_left.sh`, and
`start_wrist_right.sh` to `unitree@192.168.123.164:~/g1-onboard/` alongside
the existing RealSense/MID-360 files.

## Quick Start

### 1. Start both wrist streamers on the Orin

```bash
ssh unitree@192.168.123.164
bash ~/g1-onboard/start_wrist_left.sh &
bash ~/g1-onboard/start_wrist_right.sh &
```

Or from the GUI: click **START** on the "G1 Wrist Left" and "G1 Wrist
Right" panels.

### 2. Start the recorder on the workstation

The wrist streams are merged into the same recorder as D435. Pass the
wrist ports to enable them:

```bash
cd deploy_real
python server_data_record_d435.py --wrist_left_port 5556 --wrist_right_port 5558
```

Or uncomment the `--wrist_left_port`/`--wrist_right_port` lines in
`data_record_d435.sh` and run `bash data_record_d435.sh` /
click **START** on the "Record (D435)" GUI panel. Omitting both flags keeps
today's D435-only behavior unchanged.

### 3. Toggle recording with the PICO controller

Same as D435 — one button drives all cameras since they share one recorder:

- **Left controller Y button**: Start/stop episode recording
- **Left controller axis click**: Quit recording

## Wire Format

Identical to the D435 streamer's 16-byte header, with `depth_data_len`
always `0` and no depth payload appended:

| Offset | Size | Type  | Field            |
|--------|------|-------|------------------|
| 0      | 4B   | int32 | width            |
| 4      | 4B   | int32 | height           |
| 8      | 4B   | int32 | rgb_jpeg_len     |
| 12     | 4B   | int32 | depth_data_len (always 0) |
| 16     | var  | bytes | RGB JPEG data    |

Because the wire format matches, `data_utils/vision_client.py::VisionClient`
needs no changes to consume wrist frames — it's instantiated without depth
shared-memory args, same as any RGB-only stream.

## Data Format

Wrist frames land in the same episode directory as D435, under their own
subdirectories:

```
episode_0001/
  data.json
  rgb/                 (D435, 424x240 JPEG)
  depth/                (D435, uint16 depth)
  rgb_wrist_left/       (640x480 JPEG)
    000000.jpg
    000001.jpg
    ...
  rgb_wrist_right/      (640x480 JPEG)
    000000.jpg
    000001.jpg
    ...
```

Each frame in `data.json` gains four fields alongside the existing D435 and
state/action ones:

```json
{
  "idx": 0,
  "rgb": "rgb/000000.jpg",
  "depth": "depth/000000.npy",
  "t_img": 1707000000000,
  "rgb_wrist_left": "rgb_wrist_left/000000.jpg",
  "t_img_wrist_left": 1707000000001,
  "rgb_wrist_right": "rgb_wrist_right/000000.jpg",
  "t_img_wrist_right": 1707000000001,
  "state_body": [...],
  "action_body": [...]
}
```

`meta/modality.json`'s `"video"` block gains matching `rgb_wrist_left` /
`rgb_wrist_right` entries (see `data_utils/g1_schema.py::build_modality_json`).

**Known limitation**: `info.image` in `data.json` (width/height/fps) still
describes only the D435 stream — it predates multi-camera recording and
nothing else in the pipeline branches on it, so it wasn't extended to a
per-camera structure. Wrist resolution/fps are documented here and in the
CLI `--help` instead.

## Configuration

### Streamer CLI args (`wrist_streamer.py`)

| Flag             | Default | Description                              |
|------------------|---------|-------------------------------------------|
| `--device`       | (none)  | V4L2 device path, e.g. `/dev/video_wrist_left` (required unless `--test`) |
| `--port`         | 5556    | ZMQ PUB port                               |
| `--width`        | 640     | Stream width                               |
| `--height`       | 480     | Stream height                              |
| `--fps`          | 30      | Camera FPS                                 |
| `--jpeg-quality` | 80      | JPEG encode quality (0-100)                |
| `--test`         | off     | Publish synthetic frames, no hardware needed |

`start_wrist_left.sh` runs port 5556; `start_wrist_right.sh` runs port 5558
(5555 is D435, 5557 is MID-360).

### Recorder CLI args (`server_data_record_d435.py`, wrist-related)

| Flag                  | Default | Description                          |
|-----------------------|---------|---------------------------------------|
| `--wrist_left_port`   | (none)  | ZMQ port for left wrist streamer — omit to disable wrist recording entirely |
| `--wrist_right_port`  | (none)  | ZMQ port for right wrist streamer — omit to disable |
| `--wrist_width`       | 640     | Wrist image width                     |
| `--wrist_height`      | 480     | Wrist image height                    |

## Data Validation

`validate_d435_data.py` checks the wrist directories/frame fields
automatically when present, alongside its existing D435 checks:

```bash
cd deploy_real
python validate_d435_data.py twist2_demonstration/<task_name>
```

For ad-hoc live debugging of a single wrist stream (bypassing Redis and the
recorder entirely), reuse the existing D435 viewer — the wire format is
identical:

```bash
python visualize_d435.py --port 5556   # left wrist
python visualize_d435.py --port 5558   # right wrist
```

## Troubleshooting

### Camera not detected

- Run `v4l2-ctl --list-devices` and `lsusb` on the Orin to confirm both
  units enumerate.
- Unplug and replug the USB cable.

### Wrong camera mapped to wrong wrist

- Re-run the udev procedure above; verify with:
  ```bash
  v4l2-ctl -d /dev/video_wrist_left --info
  v4l2-ctl -d /dev/video_wrist_right --info
  ```

### USB 2.0 bandwidth

Arducam is USB 2.0 (~480 Mbps shared bus), and JPEG-compressed 640x480
frames are small (well under the bus limit even with D435 also attached).
If you see drops or stutter, check `lsusb -t`/`usb-devices` for bus
topology and fall back to a lower rate:

- Reduce resolution: `--width 320 --height 240`
- Reduce FPS: `--fps 15`

### Permission denied on `/dev/video*`

```bash
sudo chmod 666 /dev/video_wrist_left /dev/video_wrist_right
```
The udev rule from the stable-symlink procedure above is the persistent
fix — re-check it if permissions reset after a reboot.

### ZMQ connection issues

- Verify the Orin IP is reachable: `ping 192.168.123.164`
- Check firewall: `sudo ufw status` (disable if needed)
- Verify ports are not in use: `ss -tlnp | grep -E '5556|5558'`
