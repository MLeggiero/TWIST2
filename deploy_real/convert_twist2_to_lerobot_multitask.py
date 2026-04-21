"""
Convert TWIST2 demonstration data to LeRobot v2.0 dataset format (multitask).

Task descriptions are inferred per-episode from the parent session folder's
postfix (e.g. ``20260413_1043_cup`` -> "place the cup into the box"). Input
must be the parent directory that holds multiple session subfolders; pass a
single session to the non-multitask variant instead.

Supports two action modes:
  - high_level: teleop target poses (action_body + optional hand/neck)
  - low_level:  motor commands from RL policy (action_low_level + optional hand)

Usage:
  python convert_twist2_to_lerobot_multitask.py \\
      --data_dir twist2_demonstration \\
      --output_dir /path/to/output \\
      --repo_id "user/dataset_name" \\
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
DIM_STATE_BODY = 34       # ang_vel(3) + roll_pitch(2) + dof_pos(29)
DIM_STATE_NECK = 2

DIM_ACTION_BODY = 35      # high-level teleop target
DIM_ACTION_NECK = 2
DIM_ACTION_LOW_LEVEL = 29  # low-level motor commands

HAND_DIM = {'dex3': 7, 'inspire': 6}
DIM_FORCE_HAND = {'dex3': 7, 'inspire': 6}  # motor current per finger
DIM_TACTILE = 1062  # Inspire RH56DFTP: 1062 uint16 touch points per hand


# ---------- Multitask postfix -> task description mapping ----------
POSTFIX_TO_TASK = {
    "red_ball": "red ball into the box",
    "pill":     "medicine into the box",
    "bottle":   "bottle into the box",
    "cup":      "cup into the box",
}


def extract_postfix(session_name: str) -> str:
    """Session names follow ``YYYYMMDD_HHMM_POSTFIX``; postfix may itself
    contain underscores (e.g. ``red_ball``). Drop the first two tokens."""
    parts = session_name.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Session '{session_name}' does not match 'YYYYMMDD_HHMM_POSTFIX'"
        )
    return "_".join(parts[2:])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TWIST2 data to LeRobot format with per-session task labels"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Parent directory containing session subfolders named "
                             "YYYYMMDD_HHMM_POSTFIX (each with episode_XXXX/ inside)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output LeRobot dataset root directory "
                             "(default: ~/.cache/huggingface/lerobot/<repo_id>)")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g. user/dataset_name)")
    parser.add_argument("--fps", type=int, default=60,
                        help="Recording frequency (default: 60)")
    parser.add_argument("--action_mode", type=str, required=True, choices=["high_level", "low_level"],
                        help="Action mode: high_level (teleop targets) or low_level (motor commands)")

    parser.add_argument("--include_hand", action="store_true", dest="include_hand", default=True,
                        help="Include hand in action vector (default: on)")
    parser.add_argument("--no_include_hand", action="store_false", dest="include_hand",
                        help="Exclude hand from action vector")

    parser.add_argument("--use_videos", action="store_true", dest="use_videos", default=True,
                        help="Use video storage (default)")
    parser.add_argument("--no_videos", action="store_false", dest="use_videos",
                        help="Use image storage instead of video")

    parser.add_argument("--push_to_hub", action="store_true", default=False,
                        help="Push dataset to HuggingFace Hub")
    parser.add_argument("--image_writer_processes", type=int, default=0)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    parser.add_argument("--hand_type", type=str, default="inspire", choices=["dex3", "inspire"],
                        help="Hand type: dex3 (7-DOF) or inspire (6-DOF). Default: inspire. Must match what was used during recording.")
    parser.add_argument("--include_force", action="store_true", default=False,
                        help="Include hand force/current data as observation.force (requires force-enabled recordings)")
    parser.add_argument("--include_tactile", type=str, default="auto",
                        choices=["auto", "on", "off"],
                        help="Include Inspire tactile arrays as observation.tactile.{left,right} "
                             "(auto: detect from first frame + hand_type=='inspire'; "
                             "on: force include; off: force exclude)")
    parser.add_argument("--skip_unsuccessful", action="store_true", dest="skip_unsuccessful", default=True,
                        help="Skip episodes whose label contains 'unsuccessful' (default: on)")
    parser.add_argument("--include_unsuccessful", action="store_false", dest="skip_unsuccessful",
                        help="Include unsuccessful episodes in the output dataset")
    parser.add_argument("--include_neck", action="store_true", default=False,
                        help="Include neck (2-DOF) in observation.state and, for high_level "
                             "action mode with --include_hand, in the action vector. Default: off.")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Validate session postfixes and print per-postfix episode counts, "
                             "then exit before creating the LeRobot dataset")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(
            Path.home() / ".cache" / "huggingface" / "lerobot" / args.repo_id
        )

    return args


def get_action_dim(action_mode: str, include_hand: bool, hand_dof: int, include_neck: bool) -> int:
    if action_mode == "high_level":
        dim = DIM_ACTION_BODY
        if include_hand:
            dim += hand_dof * 2
            if include_neck:
                dim += DIM_ACTION_NECK
        return dim
    else:  # low_level
        dim = DIM_ACTION_LOW_LEVEL
        if include_hand:
            dim += hand_dof * 2
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


def build_state(frame: dict, idx: int, hand_dof: int, include_neck: bool) -> np.ndarray:
    """Build observation state vector."""
    state_body = safe_array(frame.get("state_body"), DIM_STATE_BODY, "state_body", idx)
    hand_left = safe_array(frame.get("state_hand_left"), hand_dof, "state_hand_left", idx)
    hand_right = safe_array(frame.get("state_hand_right"), hand_dof, "state_hand_right", idx)
    parts = [state_body, hand_left, hand_right]
    if include_neck:
        parts.append(safe_array(frame.get("state_neck"), DIM_STATE_NECK, "state_neck", idx))
    return np.concatenate(parts)


def build_action(frame: dict, idx: int, action_mode: str, include_hand: bool, hand_dof: int, include_neck: bool) -> np.ndarray:
    """Build action vector based on mode and hand flag."""
    if action_mode == "high_level":
        action = safe_array(frame.get("action_body"), DIM_ACTION_BODY, "action_body", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dof, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dof, "action_hand_right", idx)
            parts = [action, hand_left, hand_right]
            if include_neck:
                parts.append(safe_array(frame.get("action_neck"), DIM_ACTION_NECK, "action_neck", idx))
            action = np.concatenate(parts)
    else:  # low_level
        action = safe_array(frame.get("action_low_level"), DIM_ACTION_LOW_LEVEL, "action_low_level", idx)
        if include_hand:
            hand_left = safe_array(frame.get("action_hand_left"), hand_dof, "action_hand_left", idx)
            hand_right = safe_array(frame.get("action_hand_right"), hand_dof, "action_hand_right", idx)
            action = np.concatenate([action, hand_left, hand_right])
    return action


def build_force(frame: dict, idx: int, hand_dof: int) -> np.ndarray:
    """Build force observation vector (motor current from both hands)."""
    force_left = safe_array(frame.get("force_hand_left"), hand_dof, "force_hand_left", idx)
    force_right = safe_array(frame.get("force_hand_right"), hand_dof, "force_hand_right", idx)
    return np.concatenate([force_left, force_right])


def build_tactile(frame: dict, idx: int, side: str, hand_type: str) -> np.ndarray:
    """Build a single-side tactile observation vector.

    Returns a zero vector if the hand type is not inspire or if the
    frame lacks the tactile field (e.g. pre-port episodes). Casts to
    int32 because parquet prefers signed integer types and the tactile
    values fit comfortably in 16 bits.
    """
    if hand_type != "inspire":
        return np.zeros(DIM_TACTILE, dtype=np.int32)
    value = frame.get(f"tactile_hand_{side}")
    if value is None:
        warnings.warn(
            f"Frame {idx}: 'tactile_hand_{side}' is None, zero-filling "
            f"({DIM_TACTILE}d)")
        return np.zeros(DIM_TACTILE, dtype=np.int32)
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape[0] != DIM_TACTILE:
        warnings.warn(
            f"Frame {idx}: 'tactile_hand_{side}' has dim {arr.shape[0]}, "
            f"expected {DIM_TACTILE}. Zero-filling.")
        return np.zeros(DIM_TACTILE, dtype=np.int32)
    return arr


def discover_episodes(data_dir: Path):
    """Walk a parent directory of session folders and build the ordered
    list of ``(session_dir, episode_dir, task_desc)`` triples plus the
    per-postfix episode counts. Rejects single-session input and fails
    loudly on any postfix absent from ``POSTFIX_TO_TASK``."""
    # Reject single-session input: a session dir contains episode_XXXX
    # children directly.
    if any(d.is_dir() and d.name.startswith("episode_") for d in data_dir.iterdir()):
        raise ValueError(
            f"--data_dir {data_dir} appears to be a single session folder "
            f"(contains episode_XXXX directly). The multitask script requires "
            f"a parent directory that holds session subfolders; use "
            f"convert_twist2_to_lerobot.py for a single session."
        )

    session_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if not session_dirs:
        raise FileNotFoundError(f"No session subdirectories under {data_dir}")

    unknown = []
    for sdir in session_dirs:
        postfix = extract_postfix(sdir.name)
        if postfix not in POSTFIX_TO_TASK:
            unknown.append((sdir.name, postfix))
    if unknown:
        known = sorted(POSTFIX_TO_TASK.keys())
        msg_lines = [f"Unknown session postfixes (known: {known}):"]
        msg_lines += [f"  {name}  (postfix='{pf}')" for name, pf in unknown]
        raise ValueError("\n".join(msg_lines))

    episodes = []
    per_postfix_counts = {}
    for sdir in session_dirs:
        postfix = extract_postfix(sdir.name)
        task_desc = POSTFIX_TO_TASK[postfix]
        eps = sorted([
            d for d in sdir.iterdir()
            if d.is_dir() and d.name.startswith("episode_")
        ])
        for ep in eps:
            episodes.append((sdir, ep, task_desc))
        per_postfix_counts[postfix] = per_postfix_counts.get(postfix, 0) + len(eps)

    if not episodes:
        raise FileNotFoundError(
            f"No episode_XXXX directories under any session in {data_dir}"
        )

    return episodes, session_dirs, per_postfix_counts


def main():
    args = parse_args()

    hand_dof = HAND_DIM[args.hand_type]
    dim_state = DIM_STATE_BODY + hand_dof * 2 + (DIM_STATE_NECK if args.include_neck else 0)
    print(f"Hand type: {args.hand_type} ({hand_dof}-DOF per hand)")

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path.cwd() / data_dir
    output_dir = Path(args.output_dir)

    episodes, session_dirs, per_postfix_counts = discover_episodes(data_dir)
    print(f"Found {len(episodes)} episodes across {len(session_dirs)} sessions in {data_dir}")
    print("Per-postfix episode counts:")
    for pf in sorted(per_postfix_counts):
        print(f"  {pf:10s} -> {per_postfix_counts[pf]:4d}  ({POSTFIX_TO_TASK[pf]})")

    if args.dry_run:
        print("[dry_run] Validation passed. Exiting before dataset creation.")
        return

    # Read first image to get actual dimensions
    first_episode_json = episodes[0][1] / "data.json"
    with open(first_episode_json) as f:
        first_data = json.load(f)
    first_rgb_path = episodes[0][1] / first_data["data"][0]["rgb"]
    first_img = cv2.imread(str(first_rgb_path))
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {first_rgb_path}")
    height, width = first_img.shape[:2]
    print(f"Image dimensions: {height}x{width}")

    # Resolve --include_tactile. "auto" means: enable iff hand_type is
    # inspire AND the first frame has a tactile_hand_left field.
    first_frame = first_data["data"][0]
    if args.include_tactile == "auto":
        include_tactile = (
            args.hand_type == "inspire"
            and first_frame.get("tactile_hand_left") is not None
        )
    elif args.include_tactile == "on":
        include_tactile = True
    else:
        include_tactile = False

    # Compute action dim
    action_dim = get_action_dim(args.action_mode, args.include_hand, hand_dof, args.include_neck)
    dim_force = hand_dof * 2  # left + right
    print(f"Action mode: {args.action_mode}, include_hand: {args.include_hand}, action_dim: {action_dim}")
    print(f"State dim: {dim_state}")
    if args.include_force:
        print(f"Force dim: {dim_force} (motor current, {hand_dof} per hand)")
    if include_tactile:
        print(f"Tactile dim: {DIM_TACTILE} per hand (int32)")

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
            "shape": (dim_state,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    if args.include_force:
        features["observation.force"] = {
            "dtype": "float32",
            "shape": (dim_force,),
            "names": ["force"],
        }

    if include_tactile:
        features["observation.tactile.left"] = {
            "dtype": "int32",
            "shape": (DIM_TACTILE,),
            "names": ["tactile_left"],
        }
        features["observation.tactile.right"] = {
            "dtype": "int32",
            "shape": (DIM_TACTILE,),
            "names": ["tactile_right"],
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
    converted_per_postfix = {pf: 0 for pf in per_postfix_counts}

    for session_dir, ep_dir, task_desc in tqdm(episodes, desc="Converting episodes"):
        json_path = ep_dir / "data.json"
        with open(json_path) as f:
            ep_data = json.load(f)

        # Skip unsuccessful episodes if requested
        label = ep_data.get("label", "")
        if args.skip_unsuccessful and "unsuccessful" in label:
            skipped_episodes += 1
            tqdm.write(f"  Skipped {session_dir.name}/{ep_dir.name}: label={label}")
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
            state = build_state(frame, idx, hand_dof, args.include_neck)
            action = build_action(frame, idx, args.action_mode, args.include_hand, hand_dof, args.include_neck)

            frame_data = {
                "observation.images.head_rgb": img_rgb,
                "observation.state": state,
                "action": action,
                "task": task_desc,
            }

            if args.include_force:
                frame_data["observation.force"] = build_force(frame, idx, hand_dof)
            if include_tactile:
                frame_data["observation.tactile.left"] = build_tactile(
                    frame, idx, "left", args.hand_type)
                frame_data["observation.tactile.right"] = build_tactile(
                    frame, idx, "right", args.hand_type)
            dataset.add_frame(frame_data)

        dataset.save_episode()
        total_frames += num_frames
        converted_per_postfix[extract_postfix(session_dir.name)] += 1
        tqdm.write(f"  Saved {session_dir.name}/{ep_dir.name}: {num_frames} frames")

    print("Finalizing dataset...")
    dataset.finalize()

    if args.push_to_hub:
        print("Pushing to HuggingFace Hub...")
        dataset.push_to_hub(private=True)

    # Summary
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"  Episodes:   {len(episodes) - skipped_episodes} (skipped {skipped_episodes} unsuccessful)")
    print(f"  Frames:     {total_frames}")
    print(f"  State dim:  {dim_state}")
    print(f"  Action dim: {action_dim}")
    print(f"  Hand type:  {args.hand_type} ({hand_dof}-DOF)")
    print(f"  Image size: {height}x{width}")
    print(f"  Action mode: {args.action_mode}")
    print(f"  Include hand: {args.include_hand}")
    print(f"  Include neck: {args.include_neck}")
    print(f"  Include force: {args.include_force}")
    print(f"  Include tactile: {include_tactile}")
    print(f"  Output:     {output_dir}")
    print("  Per-postfix episode counts (converted / total):")
    for pf in sorted(per_postfix_counts):
        print(f"    {pf:10s} -> {converted_per_postfix[pf]:4d} / {per_postfix_counts[pf]:<4d}  "
              f"({POSTFIX_TO_TASK[pf]})")
    print("=" * 60)


if __name__ == "__main__":
    main()
