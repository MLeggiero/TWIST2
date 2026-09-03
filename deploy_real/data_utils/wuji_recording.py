"""Backend-specific Wuji recording schema and validation helpers."""

from copy import deepcopy
import time

import numpy as np

from data_utils.g1_schema import build_modality_json, build_schema_block


def build_wuji_recording_schema():
    modality = deepcopy(build_modality_json())
    for key in ("state_hand_left", "state_hand_right",
                "action_hand_left", "action_hand_right"):
        modality[key] = {"all": {"start": 0, "end": 20}}
    modality["human_hand_left"] = {
        "all": {"start": 0, "end": 63, "shape": [21, 3]}
    }
    modality["human_hand_right"] = {
        "all": {"start": 0, "end": 63, "shape": [21, 3]}
    }

    schema = deepcopy(build_schema_block())
    schema["hand"] = {
        "backend": "wuji_hand_2",
        "dim": 20,
        "unit": "rad",
        "order": "device/MJCF finger-major: finger1..5, joint1..4",
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


def validate_wuji_record(data, max_age, now=None):
    now = time.time() if now is None else now
    for side in ("left", "right"):
        data[f"state_hand_{side}"] = _fresh_value(
            data, f"state_hand_{side}", f"t_state_hand_{side}", (20,), max_age, now)
        data[f"action_hand_{side}"] = _fresh_value(
            data, f"action_hand_{side}", f"t_action_hand_{side}", (20,), max_age, now)
        data[f"human_hand_{side}"] = _fresh_value(
            data, f"human_hand_{side}", f"t_human_hand_{side}", (21, 3), max_age, now)
    return data
