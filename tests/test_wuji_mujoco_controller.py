import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


DEPLOY_REAL = Path(__file__).resolve().parents[1] / "deploy_real"
sys.path.insert(0, str(DEPLOY_REAL))

try:
    import mujoco

    import server_low_level_g1_wuji_sim as sim_server
    from server_low_level_g1_wuji_sim import (
        DEFAULT_BODY_MIMIC,
        G1WujiMujocoController,
    )
    from wuji_mujoco_model import compose_g1_wuji_model
except ImportError:
    mujoco = None


@unittest.skipIf(mujoco is None, "MuJoCo is not installed in this environment")
class WujiMujocoControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parents[1]
        config_dir = repo.parent / "wuji-hand/wuji-retargeting/example/config"
        configs = {
            side: config_dir
            / f"adaptive_analytical_wuji_glove_wuji_hand_2_{side}.yaml"
            for side in ("left", "right")
        }
        missing = [str(path) for path in configs.values() if not path.is_file()]
        if missing:
            raise unittest.SkipTest(
                "adjacent Wuji Hand 2 configs are unavailable: " + ", ".join(missing)
            )
        cls.bundle = compose_g1_wuji_model(
            repo / "assets/g1/g1_sim2sim_29dof.xml", configs
        )

    def test_headless_loop_with_mocked_policy_and_redis(self):
        class FakePipeline:
            def __init__(self, client):
                self.client = client
                self.pending = []

            def set(self, key, value):
                self.pending.append((key, value))
                return self

            def execute(self):
                for key, value in self.pending:
                    self.client.values[key] = value
                self.pending.clear()
                return []

        class FakeRedis:
            def __init__(self):
                now = __import__("time").time()
                self.values = {
                    "action_body_unitree_g1_with_hands": json.dumps([0.0] * 35),
                    "t_action": str(int(now * 1000)),
                    "wuji_action_hand_left": json.dumps([0.0] * 20),
                    "wuji_action_timestamp_left": str(now),
                    "wuji_action_hand_right": json.dumps([0.0] * 20),
                    "wuji_action_timestamp_right": str(now),
                }

            def ping(self):
                return True

            def mget(self, *keys):
                return [self.values.get(key) for key in keys]

            def pipeline(self):
                return FakePipeline(self)

        class FakePolicy:
            def __init__(self, *_args, **_kwargs):
                self.calls = 0

            def __call__(self, observation):
                self.calls += 1
                self.last_observation = observation
                return np.zeros(29, dtype=np.float64)

        fake_redis = FakeRedis()
        with patch.object(sim_server, "OnnxPolicy", FakePolicy), patch.object(
            sim_server.redis, "Redis", return_value=fake_redis
        ):
            controller = G1WujiMujocoController(
                bundle=self.bundle,
                policy_path="unused.onnx",
                redis_ip="unused",
                redis_port=0,
                device="cpu",
                policy_frequency=100.0,
                body_command_timeout=0.5,
                hand_command_timeout=0.5,
                realtime=False,
                headless=True,
                max_steps=20,
                print_every=0,
                require_fresh_pico_body=True,
            )
            controller.run()

        # The fake producer has no pico_body_timestamp. The simulator must
        # nevertheless keep invoking the closed-loop policy using its safe
        # upright startup target, rather than accepting cached all-zero input.
        self.assertEqual(controller.policy.calls, 2)
        np.testing.assert_allclose(
            controller.policy.last_observation[:35], DEFAULT_BODY_MIMIC
        )
        np.testing.assert_allclose(
            controller.policy.last_observation[-35:], DEFAULT_BODY_MIMIC
        )

        body = np.asarray(
            json.loads(fake_redis.values["state_body_unitree_g1_with_hands"])
        )
        low_level = np.asarray(
            json.loads(
                fake_redis.values["action_low_level_unitree_g1_with_hands"]
            )
        )
        self.assertEqual(body.shape, (34,))
        self.assertEqual(low_level.shape, (29,))
        for side in ("left", "right"):
            state = np.asarray(
                json.loads(fake_redis.values[f"wuji_state_hand_{side}"])
            )
            self.assertEqual(state.shape, (20,))
            self.assertTrue(np.all(np.isfinite(state)))


if __name__ == "__main__":
    unittest.main()
