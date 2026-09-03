import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from pico_hand_utils import PICO_TO_MEDIAPIPE
from xrobot_teleop_to_robot_w_hand import XRobotTeleopToRobot, parse_arguments


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
        return np.arange(7, dtype=np.float32), np.arange(7, dtype=np.float32)


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


if __name__ == "__main__":
    unittest.main()
