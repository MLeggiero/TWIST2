import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy_real"))
from data_utils.wuji_recording import build_wuji_recording_schema, validate_wuji_record
from data_utils.hybrid_hand_recording import (
    build_hybrid_hand_recording_schema,
    validate_hybrid_hand_record,
)


class WujiRecordingTest(unittest.TestCase):
    def test_schema_is_20d_without_changing_default_builder(self):
        modality, schema = build_wuji_recording_schema()
        self.assertEqual(modality["action_hand_left"]["all"]["end"], 20)
        self.assertEqual(schema["hand"]["dim"], 20)
        self.assertEqual(schema["human_hand"]["shape"], [21, 3])

    def test_fresh_values_and_unavailable_side(self):
        now = 100.0
        data = {
            "state_hand_left": list(range(20)), "t_state_hand_left": now,
            "action_hand_left": list(range(20)), "t_action_hand_left": now,
            "human_hand_left": [[[0.0, 0.0, 0.0]][0] for _ in range(21)],
            "t_human_hand_left": now,
        }
        validate_wuji_record(data, 0.5, now)
        self.assertEqual(len(data["state_hand_left"]), 20)
        self.assertEqual(len(data["human_hand_left"]), 21)
        self.assertIsNone(data["state_hand_right"])

    def test_stale_value_becomes_none(self):
        data = {"action_hand_left": list(range(20)), "t_action_hand_left": 1.0}
        validate_wuji_record(data, 0.25, now=2.0)
        self.assertIsNone(data["action_hand_left"])

    def test_hybrid_schema_has_asymmetric_dimensions_and_units(self):
        modality, schema = build_hybrid_hand_recording_schema()
        self.assertEqual(modality["action_hand_left"]["all"]["end"], 20)
        self.assertEqual(modality["action_hand_right"]["all"]["end"], 6)
        self.assertEqual(schema["hand"]["left"]["unit"], "rad")
        self.assertEqual(schema["hand"]["right"]["unit"], "device_count")

    def test_hybrid_validation_accepts_20d_left_and_6d_right(self):
        now = 100.0
        data = {
            "state_hand_left": list(range(20)), "t_state_hand_left": now,
            "action_hand_left": list(range(20)), "t_action_hand_left": now,
            "state_hand_right": list(range(6)), "t_state_hand_right": now,
            "action_hand_right": list(range(6)), "t_action_hand_right": now,
        }
        validate_hybrid_hand_record(data, 0.5, now)
        self.assertEqual(len(data["state_hand_left"]), 20)
        self.assertEqual(len(data["state_hand_right"]), 6)

    def test_hybrid_validation_rejects_wrong_side_dimensions(self):
        now = 100.0
        data = {
            "state_hand_left": list(range(6)), "t_state_hand_left": now,
            "state_hand_right": list(range(20)), "t_state_hand_right": now,
        }
        validate_hybrid_hand_record(data, 0.5, now)
        self.assertIsNone(data["state_hand_left"])
        self.assertIsNone(data["state_hand_right"])


if __name__ == "__main__":
    unittest.main()
