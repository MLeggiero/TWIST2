import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from wuji_velocity_limiter import JointVelocityLimiter


class JointVelocityLimiterTest(unittest.TestCase):
    def test_per_joint_velocity_is_limited(self):
        limits = np.linspace(0.5, 2.0, 20)
        limiter = JointVelocityLimiter(limits, control_period=0.01)
        limiter.reset(np.zeros(20), now=1.0)

        command = limiter.step(np.full(20, 10.0), now=1.01)

        np.testing.assert_allclose(command, limits * 0.01, atol=1e-12)

    def test_loop_delay_never_creates_catch_up_jump(self):
        limiter = JointVelocityLimiter(np.ones(20), control_period=0.01)
        limiter.reset(np.zeros(20), now=1.0)

        command = limiter.step(np.ones(20), now=2.0)

        np.testing.assert_allclose(command, np.full(20, 0.01), atol=1e-12)

    def test_short_interval_uses_actual_elapsed_time(self):
        limiter = JointVelocityLimiter(np.ones(20), control_period=0.01)
        limiter.reset(np.zeros(20), now=1.0)

        command = limiter.step(np.ones(20), now=1.004)

        np.testing.assert_allclose(command, np.full(20, 0.004), atol=1e-12)

    def test_target_reversal_is_velocity_limited(self):
        limiter = JointVelocityLimiter(np.ones(20), control_period=0.01)
        limiter.reset(np.zeros(20), now=1.0)
        positive = limiter.step(np.ones(20), now=1.01)
        reversed_command = limiter.step(-np.ones(20), now=1.02)

        np.testing.assert_allclose(positive, np.full(20, 0.01), atol=1e-12)
        np.testing.assert_allclose(reversed_command, np.zeros(20), atol=1e-12)

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            JointVelocityLimiter(np.ones(19), control_period=0.01)
        with self.assertRaises(ValueError):
            JointVelocityLimiter(np.zeros(20), control_period=0.01)


if __name__ == "__main__":
    unittest.main()
