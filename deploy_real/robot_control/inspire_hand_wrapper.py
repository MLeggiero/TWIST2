"""
Inspire Hand Controller for TWIST2 teleoperation.
Controls Inspire RH56DFTP dextrous hands via Modbus TCP.

The Inspire hand has 6 DOF per hand:
  Index 0: Pinky
  Index 1: Ring finger
  Index 2: Middle finger
  Index 3: Index finger
  Index 4: Thumb bend
  Index 5: Thumb rotation

Commands use angle_set values in range [0, 1000]:
  0 = fully open
  1000 = fully closed

Network defaults (on Unitree G1 internal network):
  Left hand:  192.168.123.210:6000
  Right hand: 192.168.123.211:6000

Dependencies:
  pip install pymodbus==3.6.9
"""
import numpy as np
import struct
import time
from enum import IntEnum

from pymodbus.client import ModbusTcpClient

from data_utils.params import DEFAULT_HAND_POSE


Inspire_Num_Motors = 6

# Modbus register addresses for Inspire hand
REG_CLEAR_ERROR = 1004
REG_POS_SET = 1474
REG_ANGLE_SET = 1486
REG_FORCE_SET = 1498
REG_SPEED_SET = 1522
REG_POS_ACT = 1534
REG_ANGLE_ACT = 1546
REG_FORCE_ACT = 1582
REG_CURRENT = 1594
REG_ERR = 1606       # 3 registers, byte-packed -> 6 values
REG_STATUS = 1612    # 3 registers, byte-packed -> 6 values
REG_TEMPERATURE = 1618  # 3 registers, byte-packed -> 6 values

DEFAULT_QPOS_LEFT = DEFAULT_HAND_POSE["unitree_g1_inspire"]["left"]["open"]
DEFAULT_QPOS_RIGHT = DEFAULT_HAND_POSE["unitree_g1_inspire"]["right"]["open"]


class InspireHandController:
    def __init__(self, left_ip='192.168.123.210', right_ip='192.168.123.211',
                 port=6000, device_id=1, re_init=True):
        """
        Initialize Inspire hand controller via Modbus TCP.

        Args:
            left_ip: IP address of the left Inspire hand
            right_ip: IP address of the right Inspire hand
            port: Modbus TCP port (default 6000)
            device_id: Modbus device ID (default 1)
            re_init: Whether to clear errors and move to default position
        """
        print("Initialize InspireHandController...")
        print(f"  Left hand IP: {left_ip}:{port}")
        print(f"  Right hand IP: {right_ip}:{port}")

        self.device_id = device_id

        self.left_client = ModbusTcpClient(left_ip, port=port)
        self.right_client = ModbusTcpClient(right_ip, port=port)

        if not self.left_client.connect():
            raise ConnectionError(
                f"Failed to connect to left Inspire hand at {left_ip}:{port}")
        print(f"  Left hand connected")

        if not self.right_client.connect():
            raise ConnectionError(
                f"Failed to connect to right Inspire hand at {right_ip}:{port}")
        print(f"  Right hand connected")

        # Clear errors on init
        if re_init:
            self.left_client.write_register(REG_CLEAR_ERROR, 1, slave=self.device_id)
            self.right_client.write_register(REG_CLEAR_ERROR, 1, slave=self.device_id)

        # State arrays
        self.left_hand_state_array = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.right_hand_state_array = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Lpos = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Rpos = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Ltemp = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Rtemp = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Ltau = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Rtau = np.zeros(Inspire_Num_Motors, dtype=np.float32)

        # Read initial state
        self.get_hand_state()
        print(f"  Left hand state: {self.left_hand_state_array}")
        print(f"  Right hand state: {self.right_hand_state_array}")

        if re_init:
            self.initialize()

        print("Initialize InspireHandController OK!\n")

    def _read_registers_signed(self, client, address, count):
        """Read Modbus registers and interpret as signed int16."""
        try:
            response = client.read_holding_registers(address, count, slave=self.device_id)
            if not response.isError():
                packed = struct.pack('>' + 'H' * count, *response.registers)
                return list(struct.unpack('>' + 'h' * count, packed))
            else:
                print(f"Error reading registers at {address}")
                return [0] * count
        except Exception as e:
            print(f"Exception reading registers at {address}: {e}")
            return [0] * count

    def _read_registers_bytes(self, client, address, count):
        """Read Modbus registers and unpack as individual bytes (2 bytes per register)."""
        try:
            response = client.read_holding_registers(address, count, slave=self.device_id)
            if not response.isError():
                byte_list = []
                for reg in response.registers:
                    byte_list.append((reg >> 8) & 0xFF)
                    byte_list.append(reg & 0xFF)
                return byte_list
            else:
                print(f"Error reading byte registers at {address}")
                return [0] * (count * 2)
        except Exception as e:
            print(f"Exception reading byte registers at {address}: {e}")
            return [0] * (count * 2)

    def get_hand_state(self):
        """Read current hand joint angles.

        Returns:
            (left_hand_state_6d, right_hand_state_6d): numpy arrays of angle values
        """
        # Read angle_act (signed int16, 6 values per hand)
        left_angles = self._read_registers_signed(self.left_client, REG_ANGLE_ACT, 6)
        right_angles = self._read_registers_signed(self.right_client, REG_ANGLE_ACT, 6)

        self.left_hand_state_array = np.array(left_angles, dtype=np.float32)
        self.right_hand_state_array = np.array(right_angles, dtype=np.float32)
        self.Lpos = self.left_hand_state_array.copy()
        self.Rpos = self.right_hand_state_array.copy()

        # Read motor current (proxy for torque estimation)
        left_current = self._read_registers_signed(self.left_client, REG_CURRENT, 6)
        right_current = self._read_registers_signed(self.right_client, REG_CURRENT, 6)
        self.Ltau = np.array(left_current, dtype=np.float32)
        self.Rtau = np.array(right_current, dtype=np.float32)

        # Read temperature (byte-packed: 3 registers -> 6 bytes)
        left_temp = self._read_registers_bytes(self.left_client, REG_TEMPERATURE, 3)
        right_temp = self._read_registers_bytes(self.right_client, REG_TEMPERATURE, 3)
        self.Ltemp = np.array(left_temp[:Inspire_Num_Motors], dtype=np.float32)
        self.Rtemp = np.array(right_temp[:Inspire_Num_Motors], dtype=np.float32)

        return self.left_hand_state_array.copy(), self.right_hand_state_array.copy()

    def get_hand_all_state(self):
        """Get complete hand telemetry.

        Returns:
            (Lpos, Rpos, Ltemp, Rtemp, Ltau, Rtau): 6-element arrays each
        """
        return (self.Lpos.copy(), self.Rpos.copy(),
                self.Ltemp.copy(), self.Rtemp.copy(),
                self.Ltau.copy(), self.Rtau.copy())

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """Send angle commands to both hands.

        Args:
            left_q_target: 6-element array/list of angle setpoints (0-1000 range)
            right_q_target: 6-element array/list of angle setpoints (0-1000 range)
        """
        left_angles = [int(np.clip(v, 0, 1000)) for v in left_q_target]
        right_angles = [int(np.clip(v, 0, 1000)) for v in right_q_target]

        try:
            self.left_client.write_registers(REG_ANGLE_SET, left_angles, slave=self.device_id)
            self.right_client.write_registers(REG_ANGLE_SET, right_angles, slave=self.device_id)
        except Exception as e:
            print(f"Error writing hand commands: {e}")

    def initialize(self):
        """Move hands to default open position."""
        print("Initializing Inspire hands with default open poses...")
        self.ctrl_dual_hand(DEFAULT_QPOS_LEFT, DEFAULT_QPOS_RIGHT)

    def close(self):
        """Send hands to open position and disconnect."""
        try:
            self.ctrl_dual_hand(DEFAULT_QPOS_LEFT, DEFAULT_QPOS_RIGHT)
            time.sleep(0.5)
            self.left_client.close()
            self.right_client.close()
            print("Inspire hand connections closed.")
        except Exception as e:
            print(f"Error closing Inspire hand connections: {e}")


class InspireLeftJointIndex(IntEnum):
    kPinky = 0
    kRing = 1
    kMiddle = 2
    kIndex = 3
    kThumbBend = 4
    kThumbRotation = 5


class InspireRightJointIndex(IntEnum):
    kPinky = 0
    kRing = 1
    kMiddle = 2
    kIndex = 3
    kThumbBend = 4
    kThumbRotation = 5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test Inspire hand controller')
    parser.add_argument('--left_ip', type=str, default='192.168.123.210',
                        help='Left hand IP address')
    parser.add_argument('--right_ip', type=str, default='192.168.123.211',
                        help='Right hand IP address')
    parser.add_argument('--port', type=int, default=6000,
                        help='Modbus TCP port')
    args = parser.parse_args()

    print("Testing InspireHandController...")
    hand_ctrl = InspireHandController(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        port=args.port
    )

    # Test: gradually close then open
    print("Running test sequence...")
    for i in range(11):
        angle = int(i * 100)  # 0 to 1000
        left_target = [angle] * 6
        right_target = [angle] * 6
        hand_ctrl.ctrl_dual_hand(left_target, right_target)
        left_state, right_state = hand_ctrl.get_hand_state()
        print(f"Step {i}: target={angle}, "
              f"Left={left_state[:3]}, Right={right_state[:3]}")
        time.sleep(0.3)

    # Return to open
    hand_ctrl.ctrl_dual_hand([0] * 6, [0] * 6)
    time.sleep(1.0)

    hand_ctrl.close()
    print("Test completed!")
