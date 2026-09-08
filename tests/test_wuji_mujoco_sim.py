import json
import sys
import unittest
from pathlib import Path

import numpy as np


DEPLOY_REAL = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY_REAL))

try:
    import mujoco

    from server_low_level_g1_wuji_sim import decode_vector, parse_args
    from wuji_mujoco_model import compose_g1_wuji_model, resolve_mjcf_path
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo is not installed in this environment")
class WujiMujocoSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[1]
        config_dir = (
            cls.repo.parent
            / "wuji-hand/wuji-retargeting/example/config"
        )
        cls.configs = {
            side: config_dir
            / f"adaptive_analytical_wuji_glove_wuji_hand_2_{side}.yaml"
            for side in ("left", "right")
        }
        missing = [str(path) for path in cls.configs.values() if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "adjacent Wuji Hand 2 configs are unavailable: " + ", ".join(missing)
            )
        cls.bundle = compose_g1_wuji_model(
            cls.repo / "assets/g1/g1_sim2sim_29dof.xml", cls.configs
        )

    def test_dual_hand_model_dimensions(self):
        self.assertEqual(self.bundle.model.nq, 76)
        self.assertEqual(self.bundle.model.nv, 75)
        self.assertEqual(self.bundle.model.nu, 69)
        self.assertEqual(set(self.bundle.hand_actuator_names), {"left", "right"})
        self.assertEqual(len(self.bundle.hand_actuator_names["left"]), 20)
        self.assertEqual(len(self.bundle.hand_actuator_names["right"]), 20)

    def test_hand_order_matches_official_mjcf_actuator_order(self):
        for side, config in self.configs.items():
            standalone = mujoco.MjModel.from_xml_path(str(resolve_mjcf_path(config)))
            expected = tuple(
                mujoco.mj_id2name(
                    standalone, mujoco.mjtObj.mjOBJ_ACTUATOR, index
                )
                for index in range(standalone.nu)
            )
            self.assertEqual(self.bundle.hand_actuator_names[side], expected)

    def test_fixed_hand_placeholders_are_removed(self):
        for side in ("left", "right"):
            identifier = mujoco.mj_name2id(
                self.bundle.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{side}_rubber_hand",
            )
            self.assertEqual(identifier, -1)

    def test_combined_model_steps_with_separate_control_groups(self):
        data = mujoco.MjData(self.bundle.model)
        data.ctrl[:] = 0.0
        for _ in range(10):
            mujoco.mj_step(self.bundle.model, data)
        self.assertTrue(np.all(np.isfinite(data.qpos)))
        self.assertTrue(np.all(np.isfinite(data.qvel)))

    def test_json_vector_validation(self):
        valid = np.arange(20, dtype=np.float64)
        np.testing.assert_array_equal(decode_vector(json.dumps(valid.tolist()), 20), valid)
        self.assertIsNone(decode_vector("[1, 2]", 20))
        self.assertIsNone(decode_vector("not json", 20))

    def test_fresh_pico_body_flag_is_opt_in_at_python_cli(self):
        base = ["--left-config", str(self.configs["left"]), "--model-only"]
        self.assertFalse(parse_args(base).require_fresh_pico_body)
        self.assertTrue(
            parse_args(base + ["--require-fresh-pico-body"]).require_fresh_pico_body
        )

    def test_model_only_does_not_require_policy(self):
        args = parse_args(
            ["--left-config", str(self.configs["left"]), "--model-only"]
        )
        self.assertTrue(args.model_only)
        self.assertIsNone(args.policy)


if __name__ == "__main__":
    unittest.main()
