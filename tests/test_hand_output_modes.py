import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from pico_hand_utils import PICO_TO_MEDIAPIPE
from xrobot_teleop_to_robot_w_hand import (
    SourceTimestampWatchdog,
    XRobotTeleopToRobot,
    parse_arguments,
)


class FakePipeline:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def execute(self):
        return []


class FakeStateMachine:
    def __init__(self):
        self.calls = 0

    def get_hand_pose(self, _key):
        self.calls += 1
        size = 6 if _key == "unitree_g1_inspire" else 7
        return np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32)


def hand_tuple(side):
    return True, {
        f"{side}Hand{name}": [[float(i), float(i + 1), float(i + 2)], [1, 0, 0, 0]]
        for i, name in enumerate(PICO_TO_MEDIAPIPE)
    }


def producer(mode):
    value = XRobotTeleopToRobot.__new__(XRobotTeleopToRobot)
    value.redis_client = object()
    value.redis_pipeline = FakePipeline()
    value.hand_output_mode = mode
    value.state_machine = FakeStateMachine()
    value.hand_pose_key = "unitree_g1"
    return value


class HandOutputModeTest(unittest.TestCase):
    def test_default_mode_is_existing_dex3_path(self):
        with patch.object(sys, "argv", ["teleop"]):
            self.assertEqual(parse_arguments().hand_output_mode, "dex3")

    def test_source_watchdog_requires_timestamp_advancement(self):
        watchdog = SourceTimestampWatchdog(timeout=0.5)
        self.assertEqual(watchdog.observe(100, now=1.0), (False, False))
        self.assertEqual(watchdog.observe(100, now=1.1), (False, False))
        self.assertEqual(watchdog.observe(101, now=1.2), (True, True))
        self.assertEqual(watchdog.observe(101, now=1.6), (False, True))
        self.assertEqual(watchdog.observe(101, now=1.8), (False, False))

    def test_dex3_writes_original_7d_keys(self):
        value = producer("dex3")
        value.send_to_redis(np.zeros(35, dtype=np.float32))
        self.assertEqual(value.state_machine.calls, 1)
        left = json.loads(value.redis_pipeline.values["action_hand_left_unitree_g1_with_hands"])
        self.assertEqual(len(left), 7)

    def test_wuji_writes_21x3_and_never_legacy_keys(self):
        value = producer("wuji")
        value.send_to_redis(
            np.zeros(35, dtype=np.float32),
            left_hand_data=hand_tuple("Left"),
            right_hand_data=(False, {}),
        )
        self.assertEqual(value.state_machine.calls, 0)
        self.assertNotIn("action_hand_left_unitree_g1_with_hands", value.redis_pipeline.values)
        points = json.loads(value.redis_pipeline.values["pico_hand_left_mediapipe"])
        self.assertEqual(np.asarray(points).shape, (21, 3))
        self.assertNotIn("pico_hand_right_timestamp", value.redis_pipeline.values)

    def test_wuji_freshness_guard_does_not_refresh_cached_hand_timestamp(self):
        value = producer("wuji")
        value.require_fresh_xr_tracking = True
        value.send_to_redis(
            np.zeros(35, dtype=np.float32),
            left_hand_data=hand_tuple("Left"),
            hand_source_advanced=False,
        )
        self.assertNotIn("pico_hand_left_timestamp", value.redis_pipeline.values)

        value.send_to_redis(
            np.zeros(35, dtype=np.float32),
            left_hand_data=hand_tuple("Left"),
            hand_source_advanced=True,
            body_source_advanced=True,
            body_source_timestamp=123456789,
        )
        self.assertIn("pico_hand_left_timestamp", value.redis_pipeline.values)
        self.assertIn("pico_body_timestamp", value.redis_pipeline.values)
        self.assertEqual(
            value.redis_pipeline.values["pico_body_source_timestamp_ns"],
            "123456789",
        )

    def test_hybrid_writes_left_wuji_and_right_inspire_only(self):
        value = producer("wuji-left-inspire-right")
        value.hand_pose_key = "unitree_g1_inspire"
        value.send_to_redis(
            np.zeros(35, dtype=np.float32),
            left_hand_data=hand_tuple("Left"),
            right_hand_data=hand_tuple("Right"),
            hand_source_advanced=True,
        )
        written = value.redis_pipeline.values
        self.assertEqual(value.state_machine.calls, 1)
        self.assertEqual(
            np.asarray(json.loads(written["pico_hand_left_mediapipe"])).shape,
            (21, 3),
        )
        self.assertEqual(
            np.asarray(json.loads(written["inspire_action_hand_right"])).shape,
            (6,),
        )
        self.assertIn("inspire_action_timestamp_right", written)
        self.assertNotIn("action_hand_left_unitree_g1_with_hands", written)
        self.assertNotIn("action_hand_right_unitree_g1_with_hands", written)


if __name__ == "__main__":
    unittest.main()
