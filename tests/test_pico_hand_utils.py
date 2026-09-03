import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from pico_hand_utils import PICO_TO_MEDIAPIPE, pico_hand_to_mediapipe


class PicoHandUtilsTest(unittest.TestCase):
    def hand_tuple(self, side="Left"):
        data = {}
        for index, name in enumerate(PICO_TO_MEDIAPIPE):
            data[f"{side}Hand{name}"] = [[index + 10.0, index + 20.0, index + 30.0], [1, 0, 0, 0]]
        return True, data

    def test_shape_dtype_order_and_wrist_origin(self):
        result = pico_hand_to_mediapipe(self.hand_tuple(), "Left")
        self.assertEqual(result.shape, (21, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.ones(3, dtype=np.float32))

    def test_unavailable_or_incomplete_returns_none(self):
        self.assertIsNone(pico_hand_to_mediapipe((False, {}), "Left"))
        active, data = self.hand_tuple("Right")
        del data["RightHandWrist"]
        self.assertIsNone(pico_hand_to_mediapipe((active, data), "Right"))

    def test_nonfinite_returns_none(self):
        active, data = self.hand_tuple()
        data["LeftHandWrist"][0][0] = float("nan")
        self.assertIsNone(pico_hand_to_mediapipe((active, data), "Left"))

    def test_bad_side_is_programmer_error(self):
        with self.assertRaises(ValueError):
            pico_hand_to_mediapipe(self.hand_tuple(), "left")


if __name__ == "__main__":
    unittest.main()
