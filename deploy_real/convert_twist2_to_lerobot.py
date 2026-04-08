"""
Convert TWIST2 demonstration data to LeRobot v2.0 dataset format.

Supports two action modes:
  - high_level: teleop target poses (action_body + optional hand/neck)
  - low_level:  motor commands from RL policy (action_low_level + optional hand)

Usage:
  python convert_twist2_to_lerobot.py \
      --data_dir twist2_demonstration/20260210_1017 \
      --output_dir /path/to/output \
      --repo_id "user/dataset_name" \
      --action_mode high_level
"""

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------- Dimension constants ----------
DIM_STATE_BODY = 34       # joint-first labelled order: dof_pos(29) + ang_vel(3) + rp(2)
DIM_STATE_HAND_DEX3 = 7   # per hand (Dex3)
DIM_STATE_HAND_INSPIRE = 6  # per hand (Inspire)
DIM_STATE_NECK = 2

DIM_ACTION_BODY = 35      # joint-first labelled order: dof_pos(29) + vel_xy(2) + z(1) + rp(2) + yaw_rate(1)
DIM_ACTION_HAND_DEX3 = 7  # per hand (Dex3)
DIM_ACTION_HAND_INSPIRE = 6  # per hand (Inspire)
DIM_ACTION_NECK = 2
DIM_ACTION_LOW_LEVEL = 29  # low-level motor commands

# Tactile (Inspire RH56DFTP only): 1062 uint16 touch points per hand,
# range 0-4095. Stored as int32 in the parquet because LeRobot's feature
# validation only accepts the standard numpy float/int dtypes -- int32 is
# the smallest universally-supported integer type that fits the range with
# room to spare.
DIM_TACTILE = 1062


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TWIST2 data to LeRobot format")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to a single session dir containing episode_XXXX/ folders")
    parser.add_argument("--task_name", type=str, default="g1 task",
                        help="Task description string for all episodes")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output LeRobot dataset root directory")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. user/dataset_name)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Recording frequency (default: 60)")
    parser.add_argument("--action_mode", type=str, required=True, choices=["high_level", "low_level"],
                        help="Action mode: high_level (teleop targets) or low_level (motor commands)")
    parser.add_argument("--hand_type", type=str, default="dex3", choices=["dex3", "inspire"],
                        help="Hand type: dex3 (7 DOF) or inspire (6 DOF)")

    # Hand inclusion defaults differ by mode — handled after parsing
    parser.add_argument("--include_hand", action="store_true", dest="include_hand", default=None,
                        help="Include hand/neck in action vector")
    parser.add_argument("--no_include_hand", action="store_false", dest="include_hand",
                        help="Exclude hand/neck from action vector")

    parser.add_argument("--use_videos", action="store_true", dest="use_videos", default=True,
                        help="Use video storage (default)")
    parser.add_argument("--no_videos", action="store_false", dest="use_videos",
                        help="Use image storage instead of video")

    # Tactile inclusion: only meaningful for Inspire hands. Default 'auto'
    # detects from the first frame of the first episode -- old recordings
    # without tactile fields are exported as before, new ones get the two
    # observation.tactile.* features.
    parser.add_argument("--include_tactile", type=str, default="auto",
                        choices=["auto", "yes", "no"],
                        help="Include Inspire tactile arrays as observation.tactile.* features")

    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push dataset to HuggingFace Hub")
    parser.add_argument("--filter_unsuccessful", action="store_true", default=False,
                        help="Skip episodes with 'unsuccessful' label in data.json")
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--image_writer_threads", type=int, default=4)

    args = parser.parse_args()

    # Set include_hand default based on action_mode
    if args.include_hand is None:
        args.include_hand = (args.action_mode == "high_level")

    return args


def get_action_dim(action_mode: str, include_hand: bool, hand_type: str) -> int:
    hand_dim = DIM_ACTION_HAND_INSPIRE if hand_type == "inspire" else DIM_ACTION_HAND_DEX3
    if action_mode == "high_level":
        dim = DIM_ACTION_BODY
        if include_hand:
            dim += hand_dim * 2 + DIM_ACTION_NECK  # Inspire: +14, Dex3: +16
        return dim
    else:  # low_level
        dim = DIM_ACTION_LOW_LEVEL
        if include_hand:
            dim += hand_dim * 2  # Inspire: +12, Dex3: +14
        return dim


def safe_array(value, expected_dim: int, field_name: str, frame_idx: int) -> np.ndarray:
    """Convert value to float32 array, zero-fill if None."""
    if value is None:
        warnings.warn(f"Frame {frame_idx}: '{field_name}' is None, zero-filling ({expected_dim}d)")
        return np.zeros(expected_dim, dtype=np.float32)
    arr = np.array(value, dtype=np.float32)
    if arr.shape[0] != expected_dim:
        warnings.warn(
            f"Frame {frame_idx}: '{field_name}' has dim {arr.shape[0]}, expected {expected_dim}. Zero-filling."
        )
        return np.zeros(expected_dim, dtype=np.float32)
    return arr


def get_state_dim(hand_type: str) -> int:
    """Get total state dimension based on hand type."""
    hand_dim = DIM_STATE_HAND_INSPIRE if hand_type == "inspire" else DIM_STATE_HAND_DEX3
    return DIM_STATE_BODY + hand_dim * 2 + DIM_STATE_NECK  # Inspire: 48, Dex3: 50

def build_state(frame: dict, idx: int, hand_type: str) -> np.ndarray:
    """Build observation state vector (48d for Inspire, 50d for Dex3)."""
    hand_dim = DIM_STATE_HAND_INSPIRE if hand_type == "inspire" else DIM_STATE_HAND_DEX3
    state_body = safe_array(frame.get("state_body"), DIM_STATE_BODY, "state_body", idx)
    hand_left = safe_array(frame.get("state_hand_left"), hand_dim, "state_hand_left", idx)
    hand_right = safe_array(frame.get("state_hand_right"), hand_dim, "state_hand_right", idx)
    neck = safe_array(frame.get("state_neck"), DIM_STATE_NECK, "state_neck", idx)
    return np.concatenate([state_body, hand_left, hand_right, neck])


def build_tactile(frame: dict, idx: int, side: str) -> np.ndarray:
    """Build a single-hand tactile vector as int32 with shape (1062,).

    ``side`` is "left" or "right". Missing or wrong-shape values are
    zero-filled with a one-line warning so old episodes (recorded before
    tactile sensing was added) still convert without aborting the run.
    """
    field_name = f"tactile_hand_{side}"
    value = frame.get(field_name)
    if value is None:
        warnings.warn(
            f"Frame {idx}: '{field_name}' is None, zero-filling ({DIM_TACTILE}d)"
        )
        return np.zeros(DIM_TACTILE, dtype=np.int32)
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != (DIM_TACTILE,):
        warnings.warn(
            f"Frame {idx}: '{field_name}' has shape {arr.shape}, "
            f"expected ({DIM_TACTILE},). Zero-filling."
        )
        return np.zeros(DIM_TACTILE, dtype=np.int32)
    return arr


def build_action(frame: dict, idx: int, action_mode: str, include_hand: bool, hand_type: str) -> np.ndarray:
    """Build action vector based on mode and hand flag."""
    hand_dim = DIM_ACTION_HAND_INSPIRE if hand_type == "inspire" else DIM_ACTION_HAND_DEX3
    if action_mode == "high_level":
        action = safe_array(frame.get("action_body"), DIM_ACTION_BODY, "action_body", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dim, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dim, "action_hand_right", idx)
            neck = safe_array(frame.get("action_neck"), DIM_ACTION_NECK, "action_neck", idx)
            action = np.concatenate([action, hand_left, hand_right, neck])
    else:  # low_level
        action = safe_array(frame.get("action_low_level"), DIM_ACTION_LOW_LEVEL, "action_low_level", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dim, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dim, "action_hand_right", idx)
            action = np.concatenate([action, hand_left, hand_right])
    return action


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    output_dir = Path(args.output_dir)

    # Discover episodes
    episode_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("episode_")
    ])
    if not episode_dirs:
        raise FileNotFoundError(f"No episode_XXXX directories found in {data_dir}")
    print(f"Found {len(episode_dirs)} episodes in {data_dir}")

    # Read first image to get actual dimensions
    first_episode_json = episode_dirs[0] / "data.json"
    with open(first_episode_json) as f:
        first_data = json.load(f)
    first_rgb_path = episode_dirs[0] / first_data["data"][0]["rgb"]
    first_img = cv2.imread(str(first_rgb_path))
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {first_rgb_path}")
    height, width = first_img.shape[:2]
    print(f"Image dimensions: {height}x{width}")

    # Compute action and state dims
    action_dim = get_action_dim(args.action_mode, args.include_hand, args.hand_type)
    state_dim = get_state_dim(args.hand_type)
    print(f"Action mode: {args.action_mode}, hand_type: {args.hand_type}, include_hand: {args.include_hand}")
    print(f"Action dim: {action_dim}, State dim: {state_dim}")

    # Decide whether to include tactile features. Only Inspire hands
    # produce tactile data; for Dex3 we always disable. In 'auto' mode we
    # peek at the first frame of the first episode and turn tactile on iff
    # the field is present and has the expected length.
    if args.hand_type != "inspire":
        include_tactile = False
        if args.include_tactile == "yes":
            print("Warning: --include_tactile=yes ignored because hand_type != inspire")
    elif args.include_tactile == "yes":
        include_tactile = True
    elif args.include_tactile == "no":
        include_tactile = False
    else:  # auto
        first_frame = first_data["data"][0]
        tac_left = first_frame.get("tactile_hand_left")
        include_tactile = (
            tac_left is not None and len(tac_left) == DIM_TACTILE
        )
    print(f"Include tactile: {include_tactile}")

    # Define features
    vision_dtype = "video" if args.use_videos else "image"
    features = {
        "observation.images.head_rgb": {
            "dtype": vision_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    if include_tactile:
        features["observation.tactile.left_hand"] = {
            "dtype": "int32",
            "shape": (DIM_TACTILE,),
            "names": ["tactile"],
        }
        features["observation.tactile.right_hand"] = {
            "dtype": "int32",
            "shape": (DIM_TACTILE,),
            "names": ["tactile"],
        }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=str(output_dir),
        robot_type="unitree_g1",
        fps=args.fps,
        features=features,
        use_videos=args.use_videos,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    total_frames = 0
    skipped_episodes = 0

    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        json_path = ep_dir / "data.json"
        with open(json_path) as f:
            ep_data = json.load(f)

        # Filter unsuccessful episodes if requested
        if args.filter_unsuccessful:
            label = ep_data.get("label", "")
            if label == "unsuccessful":
                tqdm.write(f"  Skipping {ep_dir.name}: labeled as unsuccessful")
                skipped_episodes += 1
                continue

        frames = ep_data["data"]
        num_frames = len(frames)

        for frame in frames:
            idx = frame["idx"]

            # Load RGB image (BGR -> RGB)
            rgb_path = ep_dir / frame["rgb"]
            img = cv2.imread(str(rgb_path))
            if img is None:
                warnings.warn(f"Cannot read image {rgb_path}, skipping frame {idx}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Build state and action
            state = build_state(frame, idx, args.hand_type)
            action = build_action(frame, idx, args.action_mode, args.include_hand, args.hand_type)

            frame_data = {
                "observation.images.head_rgb": img_rgb,
                "observation.state": state,
                "action": action,
                "task": args.task_name,
            }
            if include_tactile:
                frame_data["observation.tactile.left_hand"] = build_tactile(frame, idx, "left")
                frame_data["observation.tactile.right_hand"] = build_tactile(frame, idx, "right")
            dataset.add_frame(frame_data)

        dataset.save_episode()
        total_frames += num_frames
        tqdm.write(f"  Saved {ep_dir.name}: {num_frames} frames")

    print("Finalizing dataset...")
    dataset.finalize()

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(private=True)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Total episodes found: {len(episode_dirs)}")
    if args.filter_unsuccessful and skipped_episodes > 0:
        print(f"  Skipped (unsuccessful): {skipped_episodes}")
        print(f"  Converted episodes: {len(episode_dirs) - skipped_episodes}")
    else:
        print(f"  Converted episodes: {len(episode_dirs)}")
    print(f"  Frames:     {total_frames}")
    print(f"  State dim:  {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Image size: {height}x{width}")
    print(f"  Action mode: {args.action_mode}")
    print(f"  Hand type: {args.hand_type}")
    print(f"  Include hand: {args.include_hand}")
    print(f"  Include tactile: {include_tactile}")
    print(f"  Output:     {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
