import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from wuji_hand_bridge import device_order_state, parse_args, validate_keypoints


class WujiBridgePureTest(unittest.TestCase):
    def test_keypoint_validation(self):
        value = np.arange(63, dtype=np.float32).reshape(21, 3)
        parsed = validate_keypoints(json.dumps(value.tolist()))
        np.testing.assert_array_equal(parsed, value)
        self.assertIsNone(validate_keypoints(json.dumps([[0, 0, 0]])))

    def test_state_is_reordered_by_nid(self):
        entries = [SimpleNamespace(nid=nid, position=float(nid)) for nid in range(20)]
        frame = SimpleNamespace(joints=list(reversed(entries)))
        np.testing.assert_array_equal(device_order_state(frame), np.arange(20))

    def test_one_based_nids_and_partial_rejected(self):
        entries = [SimpleNamespace(nid=nid, position=float(nid)) for nid in range(1, 21)]
        state = device_order_state(SimpleNamespace(joints=entries))
        np.testing.assert_array_equal(state, np.arange(1, 21))
        self.assertIsNone(device_order_state(SimpleNamespace(joints=entries[:-1])))

    def test_dry_run_needs_no_address(self):
        args = parse_args(["--left-config", "left.yaml", "--dry-run"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.redis_port, 6379)
        self.assertEqual(args.velocity_limit_scale, 1.0)

    def test_velocity_limit_scale_may_only_reduce_urdf_limits(self):
        args = parse_args([
            "--left-config", "left.yaml", "--dry-run",
            "--velocity-limit-scale", "0.2",
        ])
        self.assertEqual(args.velocity_limit_scale, 0.2)
        with self.assertRaises(SystemExit):
            parse_args([
                "--left-config", "left.yaml", "--dry-run",
                "--velocity-limit-scale", "1.1",
            ])

    def test_dual_hardware_requires_addresses(self):
        with self.assertRaises(SystemExit):
            parse_args(["--left-config", "left.yaml", "--right-config", "right.yaml"])


if __name__ == "__main__":
    unittest.main()
