import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from inspire_hybrid_utils import (
    InspireCommandFilter,
    decode_inspire_target,
    timestamp_age,
)


class InspireHybridUtilsTest(unittest.TestCase):
    def test_decode_requires_six_finite_values_and_clips_range(self):
        decoded = decode_inspire_target(json.dumps([-1, 1, 2, 3, 4, 1001]))
        np.testing.assert_allclose(decoded, [0, 1, 2, 3, 4, 1000])
        self.assertIsNone(decode_inspire_target(json.dumps([1, 2])))
        self.assertIsNone(decode_inspire_target(json.dumps([1, 2, 3, 4, 5, np.nan])))

    def test_timestamp_age(self):
        self.assertAlmostEqual(timestamp_age("9.75", now=10.0), 0.25)
        self.assertIsNone(timestamp_age("bad", now=10.0))

    def test_filter_starts_at_measured_and_limits_slope(self):
        limiter = InspireCommandFilter(
            max_rate=100.0, control_period=0.02, startup_duration=1.0
        )
        np.testing.assert_allclose(limiter.reset(np.zeros(6), now=0.0), 0.0)
        np.testing.assert_allclose(limiter.step(np.full(6, 1000), now=0.0), 0.0)
        # 100 counts/s * 0.02 s = at most 2 counts per update.
        np.testing.assert_allclose(limiter.step(np.full(6, 1000), now=0.02), 2.0)

    def test_delayed_loop_does_not_catch_up(self):
        limiter = InspireCommandFilter(
            max_rate=100.0, control_period=0.02, startup_duration=0.0
        )
        limiter.reset(np.zeros(6), now=0.0)
        limiter.step(np.full(6, 1000), now=0.0)
        np.testing.assert_allclose(limiter.step(np.full(6, 1000), now=2.0), 2.0)

    def test_hold_does_not_advance_interpolation(self):
        limiter = InspireCommandFilter(
            max_rate=100.0, control_period=0.02, startup_duration=1.0
        )
        limiter.reset(np.zeros(6), now=0.0)
        limiter.step(np.full(6, 1000), now=0.0)
        before = limiter.step(np.full(6, 1000), now=0.02)
        np.testing.assert_allclose(limiter.hold(), before)


if __name__ == "__main__":
    unittest.main()
