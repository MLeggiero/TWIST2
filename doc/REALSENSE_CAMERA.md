# RealSense D435i Camera Integration

## Overview

The G1 robot has an internal Intel RealSense D435i connected to the Orin via USB. This pipeline streams **RGB + aligned depth** at **424x240 @ 60fps** over ZMQ to the workstation for data collection.

The existing ZED Mini pipeline is preserved; both camera systems coexist independently.

## Prerequisites

On the Orin (robot side):

- Python 3 with pip
- `pyrealsense2`, `pyzmq`, `numpy`, `opencv-python-headless`

Install via:
```bash
pip install -r ~/g1-onboard/requirements.txt
```

## Hardware Check

SSH into the Orin and verify the D435i is detected:

```bash
ssh unitree@192.168.123.164
rs-enumerate-devices
```

You should see a D435i entry with serial number and firmware version.

If `rs-enumerate-devices` is not available, install the RealSense SDK:
```bash
sudo apt install librealsense2-utils
```

## Deploy to Robot

From the workstation, push the streamer files to the Orin:

```bash
bash deploy_real/onboard/deploy_to_robot.sh
```

This copies `realsense_streamer.py`, `start_realsense.sh`, and `requirements.txt` to `unitree@192.168.123.164:~/g1-onboard/` and installs Python dependencies.

## Quick Start

### 1. Start the streamer on the Orin

```bash
# SSH into the Orin
ssh unitree@192.168.123.164
bash ~/g1-onboard/start_realsense.sh
```

Or from the GUI: click **START** on the "G1 RealSense" panel.

### 2. Start the recorder on the workstation

```bash
bash data_record_d435.sh
```

Or from the GUI: click **START** on the "Record (D435)" panel.

### 3. Toggle recording with the PICO controller

- **Left controller Y button**: Start/stop episode recording
- **Left controller axis click**: Quit recording

## Wire Format

Each ZMQ message has a **16-byte header** followed by the payload:

| Offset | Size | Type  | Field            |
|--------|------|-------|------------------|
| 0      | 4B   | int32 | width            |
| 4      | 4B   | int32 | height           |
| 8      | 4B   | int32 | rgb_jpeg_len     |
| 12     | 4B   | int32 | depth_data_len   |
| 16     | var  | bytes | RGB JPEG data    |
| 16+J   | var  | bytes | depth raw uint16 |

- **RGB**: JPEG-encoded BGR image (quality configurable via `--jpeg-quality`).
- **Depth**: Raw `np.uint16` bytes, `width * height * 2` bytes. Depth values in millimeters.
- When `--no-depth` is set on the streamer, `depth_data_len = 0` and no depth bytes are appended.

## Data Format

Recorded episodes are stored under `deploy_real/twist2_demonstration/<task_name>/`:

```
episode_0001/
  data.json
  rgb/
    000000.jpg    (424x240 JPEG)
    000001.jpg
    ...
  depth/
    000000.png    (424x240 uint16 PNG, depth in mm)
    000001.png
    ...
```

Each frame in `data.json`:
```json
{
  "idx": 0,
  "rgb": "rgb/000000.jpg",
  "depth": "depth/000000.png",
  "t_img": 1707000000000,
  "state_body": [...],
  "state_hand_left": [...],
  "state_hand_right": [...],
  "state_neck": [...],
  "t_state": ...,
  "action_body": [...],
  "action_hand_left": [...],
  "action_hand_right": [...],
  "action_neck": [...],
  "t_action": ...
}
```

## Configuration

### Streamer CLI args (`realsense_streamer.py`)

| Flag             | Default | Description                          |
|------------------|---------|--------------------------------------|
| `--port`         | 5555    | ZMQ PUB port                         |
| `--width`        | 424     | Stream width                         |
| `--height`       | 240     | Stream height                        |
| `--fps`          | 60      | Camera FPS                           |
| `--jpeg-quality` | 80      | JPEG encode quality (0-100)          |
| `--no-depth`     | off     | Disable depth, stream RGB only       |

### Recorder CLI args (`server_data_record_d435.py`)

| Flag             | Default            | Description                          |
|------------------|--------------------|--------------------------------------|
| `--frequency`    | 60                 | Recording frequency in Hz            |
| `--robot_ip`     | 192.168.123.164    | Orin IP address                      |
| `--camera_port`  | 5555               | ZMQ port for streamer                |
| `--width`        | 424                | Image width                          |
| `--height`       | 240                | Image height                         |
| `--goal`         | "pick up the red cup" | Task goal description             |
| `--desc`         | ...                | Task description                     |
| `--steps`        | ...                | Task steps                           |
| `--data_folder`  | twist2_demonstration | Data output folder                 |
| `--task_name`    | `<timestamp>`      | Task subdirectory name               |

## Data Validation

After recording, validate all episodes in a task directory:

```bash
cd deploy_real
python validate_d435_data.py twist2_demonstration/<task_name>
```

Validate a single episode:
```bash
python validate_d435_data.py twist2_demonstration/<task_name> --episode 3
```

Visual spot-check (shows first frame of each episode):
```bash
python validate_d435_data.py twist2_demonstration/<task_name> --show
```

## Troubleshooting

### D435i not detected

```
RuntimeError: No device connected
```

- Unplug and replug the USB cable.
- Run `rs-enumerate-devices` to verify.
- Check USB 3.0 connection (USB 2.0 may not support the required bandwidth).

### Permission denied on `/dev/video*`

```bash
sudo chmod 666 /dev/video*
# Or add udev rules for persistent access:
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### Low FPS or frame drops

- Reduce resolution: `--width 320 --height 180`
- Reduce FPS: `--fps 30`
- Disable depth: `--no-depth`
- Check CPU/USB bandwidth on the Orin with `htop` and `lsusb -t`

### ZMQ connection issues

- Verify the Orin IP is reachable: `ping 192.168.123.164`
- Check firewall: `sudo ufw status` (disable if needed)
- Verify port is not in use: `ss -tlnp | grep 5555`
