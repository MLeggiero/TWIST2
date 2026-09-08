"""Recording schema for left Wuji Hand 2 plus right Inspire hand."""

from copy import deepcopy
import time

import numpy as np

from data_utils.g1_schema import build_modality_json, build_schema_block


def build_hybrid_hand_recording_schema():
    modality = deepcopy(build_modality_json())
    for key in ("state_hand_left", "action_hand_left"):
        modality[key] = {"all": {"start": 0, "end": 20}}
    for key in ("state_hand_right", "action_hand_right"):
        modality[key] = {"all": {"start": 0, "end": 6}}
    for side in ("left", "right"):
        modality[f"human_hand_{side}"] = {
            "all": {"start": 0, "end": 63, "shape": [21, 3]}
        }

    schema = deepcopy(build_schema_block())
    schema["hand"] = {
        "left": {
            "backend": "wuji_hand_2",
            "dim": 20,
            "unit": "rad",
            "order": "device/MJCF finger-major: finger1..5, joint1..4",
        },
        "right": {
            "backend": "inspire_rh56",
            "dim": 6,
            "unit": "device_count",
            "range": [0, 1000],
            "order": "pinky, ring, middle, index, thumb_bend, thumb_rotation",
            "direction": "1000=open, 0=closed",
        },
    }
    schema["human_hand"] = {
        "shape": [21, 3],
        "format": "mediapipe_21x3",
        "units": "meters",
        "origin": "wrist-relative",
    }
    return modality, schema


def _fresh_value(data, value_key, timestamp_key, shape, max_age, now):
    value = data.get(value_key)
    timestamp = data.get(timestamp_key)
    try:
        timestamp = float(timestamp)
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if now - timestamp > max_age or timestamp - now > 1.0:
        return None
    if array.shape != shape or not np.all(np.isfinite(array)):
        return None
    return value


def validate_hybrid_hand_record(data, max_age, now=None):
    now = time.time() if now is None else now
    data["state_hand_left"] = _fresh_value(
        data, "state_hand_left", "t_state_hand_left", (20,), max_age, now
    )
    data["action_hand_left"] = _fresh_value(
        data, "action_hand_left", "t_action_hand_left", (20,), max_age, now
    )
    data["state_hand_right"] = _fresh_value(
        data, "state_hand_right", "t_state_hand_right", (6,), max_age, now
    )
    data["action_hand_right"] = _fresh_value(
        data, "action_hand_right", "t_action_hand_right", (6,), max_age, now
    )
    for side in ("left", "right"):
        data[f"human_hand_{side}"] = _fresh_value(
            data, f"human_hand_{side}", f"t_human_hand_{side}",
            (21, 3), max_age, now
        )
    return data
