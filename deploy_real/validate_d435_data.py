#!/usr/bin/env python3
"""Validate D435i recorded episode data for completeness and integrity."""

import argparse
import json
import os
import sys

import cv2
import numpy as np


class EpisodeResult:
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"

    def __init__(self, name):
        self.name = name
        self.status = self.OK
        self.n_frames = 0
        self.duration_s = 0.0
        self.rgb_count = 0
        self.depth_count = 0
        self.warnings = []
        self.errors = []

    def warn(self, msg):
        self.warnings.append(msg)
        if self.status == self.OK:
            self.status = self.WARN

    def fail(self, msg):
        self.errors.append(msg)
        self.status = self.FAIL

    def summary_line(self):
        base = f"{self.name}: {self.status}"
        detail = f"({self.n_frames} frames, {self.duration_s:.1f}s, rgb={self.rgb_count}, depth={self.depth_count})"
        issues = self.errors + self.warnings
        if issues:
            return f"{base}  {detail} — {'; '.join(issues)}"
        return f"{base}  {detail}"


def validate_episode(episode_dir, sample_interval=10):
    """Validate a single episode directory. Returns an EpisodeResult."""
    name = os.path.basename(episode_dir)
    result = EpisodeResult(name)

    # 1. Structure check
    data_json_path = os.path.join(episode_dir, "data.json")
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")

    if not os.path.isfile(data_json_path):
        result.fail("data.json missing")
        return result
    if not os.path.isdir(rgb_dir):
        result.fail("rgb/ directory missing")
        return result
    has_depth_dir = os.path.isdir(depth_dir)

    # 2. JSON parse
    try:
        with open(data_json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        result.fail(f"data.json parse error: {e}")
        return result

    for key in ("info", "text", "data"):
        if key not in manifest:
            result.fail(f"data.json missing top-level key '{key}'")
            return result

    info = manifest["info"]
    text = manifest["text"]
    frames = manifest["data"]

    # 3. Metadata
    img_info = info.get("image", {})
    width = img_info.get("width")
    height = img_info.get("height")
    fps = img_info.get("fps")

    for field, val in [("width", width), ("height", height), ("fps", fps)]:
        if val is None or not isinstance(val, (int, float)):
            result.fail(f"info.image.{field} missing or non-numeric")
            return result

    width = int(width)
    height = int(height)

    goal = text.get("goal", "")
    if not goal:
        result.warn("text.goal is empty")

    # 4. Frame count consistency
    result.n_frames = len(frames)
    rgb_files = os.listdir(rgb_dir)
    result.rgb_count = len(rgb_files)

    if result.n_frames != result.rgb_count:
        result.fail(f"rgb count mismatch (json={result.n_frames}, files={result.rgb_count})")

    if has_depth_dir:
        depth_files = os.listdir(depth_dir)
        result.depth_count = len(depth_files)
        if result.n_frames != result.depth_count:
            result.fail(f"depth count mismatch (json={result.n_frames}, files={result.depth_count})")
    else:
        result.depth_count = 0

    if result.n_frames == 0:
        result.warn("episode has 0 frames")
        return result

    # 5. Frame index continuity
    indices = [f.get("idx") for f in frames]
    if None in indices:
        result.fail("some frames missing 'idx' field")
    else:
        expected = list(range(len(frames)))
        if sorted(indices) != expected:
            if len(set(indices)) != len(indices):
                result.fail("duplicate idx values")
            else:
                result.fail(f"idx not contiguous 0..{len(frames)-1}")

    # 6. File existence
    missing_rgb = 0
    missing_depth = 0
    for frame in frames:
        rgb_path = frame.get("rgb")
        if rgb_path:
            full = os.path.join(episode_dir, rgb_path)
            if not os.path.isfile(full):
                missing_rgb += 1
        depth_path = frame.get("depth")
        if depth_path:
            full = os.path.join(episode_dir, depth_path)
            if not os.path.isfile(full):
                missing_depth += 1

    if missing_rgb:
        result.fail(f"{missing_rgb} rgb file(s) referenced in JSON but missing on disk")
    if missing_depth:
        result.fail(f"{missing_depth} depth file(s) referenced in JSON but missing on disk")

    # 7. Image readability & dimensions (sample every Nth frame)
    bad_rgb = 0
    bad_depth = 0
    for i in range(0, len(frames), sample_interval):
        frame = frames[i]
        rgb_path = frame.get("rgb")
        if rgb_path:
            full = os.path.join(episode_dir, rgb_path)
            if os.path.isfile(full):
                img = cv2.imread(full)
                if img is None:
                    bad_rgb += 1
                elif img.shape[0] != height or img.shape[1] != width:
                    bad_rgb += 1

        depth_path = frame.get("depth")
        if depth_path:
            full = os.path.join(episode_dir, depth_path)
            if os.path.isfile(full):
                dimg = cv2.imread(full, cv2.IMREAD_UNCHANGED)
                if dimg is None:
                    bad_depth += 1
                elif dimg.dtype != np.uint16:
                    bad_depth += 1
                elif dimg.shape[0] != height or dimg.shape[1] != width:
                    bad_depth += 1

    if bad_rgb:
        result.fail(f"{bad_rgb} sampled rgb image(s) unreadable or wrong dimensions")
    if bad_depth:
        result.fail(f"{bad_depth} sampled depth image(s) unreadable, wrong dtype, or wrong dimensions")

    # 8. State/action completeness
    missing_state = 0
    missing_action = 0
    for frame in frames:
        sb = frame.get("state_body")
        if sb is None or (isinstance(sb, list) and len(sb) == 0):
            missing_state += 1
        ab = frame.get("action_body")
        if ab is None or (isinstance(ab, list) and len(ab) == 0):
            missing_action += 1

    if missing_state:
        result.warn(f"{missing_state} frames missing state_body")
    if missing_action:
        result.warn(f"{missing_action} frames missing action_body")

    # 9. Timestamp monotonicity
    t_imgs = [f.get("t_img") for f in frames]
    if all(t is not None for t in t_imgs):
        for i in range(1, len(t_imgs)):
            if t_imgs[i] <= t_imgs[i - 1]:
                result.warn("t_img not strictly increasing")
                break
        # Compute duration
        result.duration_s = (t_imgs[-1] - t_imgs[0]) / 1000.0 if len(t_imgs) > 1 else 0.0
    else:
        result.warn("some frames missing t_img")
        result.duration_s = 0.0

    return result


def show_episodes(episode_dirs):
    """Display first frame of each episode (RGB + depth side-by-side)."""
    for ep_dir in episode_dirs:
        data_json = os.path.join(ep_dir, "data.json")
        if not os.path.isfile(data_json):
            continue
        try:
            with open(data_json, "r") as f:
                manifest = json.load(f)
        except Exception:
            continue

        frames = manifest.get("data", [])
        if not frames:
            continue

        frame = frames[0]
        name = os.path.basename(ep_dir)

        rgb_path = frame.get("rgb")
        rgb_img = None
        if rgb_path:
            full = os.path.join(ep_dir, rgb_path)
            rgb_img = cv2.imread(full)

        depth_path = frame.get("depth")
        depth_img = None
        if depth_path:
            full = os.path.join(ep_dir, depth_path)
            raw = cv2.imread(full, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                # Normalize to 8-bit and apply colormap
                normed = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX)
                depth_img = cv2.applyColorMap(normed.astype(np.uint8), cv2.COLORMAP_JET)

        if rgb_img is not None and depth_img is not None:
            # Resize depth to match rgb if needed
            if depth_img.shape[:2] != rgb_img.shape[:2]:
                depth_img = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]))
            combined = np.hstack([rgb_img, depth_img])
        elif rgb_img is not None:
            combined = rgb_img
        elif depth_img is not None:
            combined = depth_img
        else:
            print(f"  {name}: no images to display")
            continue

        cv2.imshow(f"Validate: {name}  (press any key / q to quit)", combined)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord("q"):
            break


def main():
    parser = argparse.ArgumentParser(
        description="Validate D435i recorded episode data."
    )
    parser.add_argument("task_dir", help="Path to the task directory containing episode_XXXX folders")
    parser.add_argument("--episode", type=int, default=None, help="Validate only a single episode by ID (e.g. --episode 3)")
    parser.add_argument("--show", action="store_true", help="Display first frame of each episode for visual spot-check")
    args = parser.parse_args()

    if not os.path.isdir(args.task_dir):
        print(f"Error: '{args.task_dir}' is not a directory")
        sys.exit(1)

    # Discover episode directories
    if args.episode is not None:
        ep_name = f"episode_{str(args.episode).zfill(4)}"
        ep_path = os.path.join(args.task_dir, ep_name)
        if not os.path.isdir(ep_path):
            print(f"Error: episode directory '{ep_path}' not found")
            sys.exit(1)
        episode_dirs = [ep_path]
    else:
        entries = sorted(
            e for e in os.listdir(args.task_dir)
            if e.startswith("episode_") and os.path.isdir(os.path.join(args.task_dir, e))
        )
        episode_dirs = [os.path.join(args.task_dir, e) for e in entries]

    if not episode_dirs:
        print("No episodes found.")
        sys.exit(0)

    # Validate
    results = []
    for ep_dir in episode_dirs:
        r = validate_episode(ep_dir)
        results.append(r)
        print(r.summary_line())

    # Summary
    n_ok = sum(1 for r in results if r.status == EpisodeResult.OK)
    n_warn = sum(1 for r in results if r.status == EpisodeResult.WARN)
    n_fail = sum(1 for r in results if r.status == EpisodeResult.FAIL)
    print(f"\nSummary: {len(results)} episodes, {n_ok} OK, {n_warn} WARN, {n_fail} FAIL")

    # Visual spot-check
    if args.show:
        show_episodes(episode_dirs)

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
