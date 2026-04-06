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

# Tactile sensor registers (PDF section 2.6.20).
# Address values are byte addresses; each Modbus holding register holds 2 bytes
# starting at the requested byte address. Each tactile point is a uint16 stored
# little-endian within the byte stream, range 0-4095.
REG_TACTILE_START = 3000
REG_TACTILE_END = 5123  # inclusive last byte
TACTILE_TOTAL_REGS = (REG_TACTILE_END - REG_TACTILE_START + 1) // 2  # 1062
MODBUS_MAX_REGS = 120  # safe margin under the 125-register Modbus limit

# (region_name, start_byte, n_points, (rows, cols)) — total points = 1062 per hand.
TACTILE_LAYOUT = [
    ("little_tip",     3000,   9, (3, 3)),
    ("little_nail",    3018,  96, (12, 8)),
    ("little_pad",     3210,  80, (10, 8)),
    ("ring_tip",       3370,   9, (3, 3)),
    ("ring_nail",      3388,  96, (12, 8)),
    ("ring_pad",       3580,  80, (10, 8)),
    ("middle_tip",     3740,   9, (3, 3)),
    ("middle_nail",    3758,  96, (12, 8)),
    ("middle_pad",     3950,  80, (10, 8)),
    ("index_tip",      4110,   9, (3, 3)),
    ("index_nail",     4128,  96, (12, 8)),
    ("index_pad",      4320,  80, (10, 8)),
    ("thumb_tip",      4480,   9, (3, 3)),
    ("thumb_nail",     4498,  96, (12, 8)),
    ("thumb_middle",   4690,   9, (3, 3)),
    ("thumb_pad",      4708,  96, (12, 8)),
    ("palm",           4900, 112, (8, 14)),
]


def slice_tactile(flat_buf):
    """Reshape a flat 1062-element tactile buffer into named region arrays.

    Args:
        flat_buf: 1-D array of length TACTILE_TOTAL_REGS (uint16 touch values),
            laid out by ascending byte address.

    Returns:
        dict mapping region name -> 2-D numpy array with the layout's shape.
    """
    out = {}
    for name, start_byte, n_points, shape in TACTILE_LAYOUT:
        offset = (start_byte - REG_TACTILE_START) // 2
        region = flat_buf[offset:offset + n_points]
        out[name] = region.reshape(shape)
    return out

DEFAULT_QPOS_LEFT = DEFAULT_HAND_POSE["unitree_g1_inspire"]["left"]["open"]
DEFAULT_QPOS_RIGHT = DEFAULT_HAND_POSE["unitree_g1_inspire"]["right"]["open"]


class InspireHandController:
    def __init__(self, left_ip='192.168.123.210', right_ip='192.168.123.211',
                 port=6000, device_id=1, re_init=True, read_current=False):
        """
        Initialize Inspire hand controller via Modbus TCP.

        Args:
            left_ip: IP address of the left Inspire hand
            right_ip: IP address of the right Inspire hand
            port: Modbus TCP port (default 6000)
            device_id: Modbus device ID (default 1)
            re_init: Whether to clear errors and move to default position
            read_current: If True, also read REG_CURRENT each cycle and expose
                it as Ltau/Rtau (motor current in mA, a noisy proxy for joint
                effort). Off by default to free Modbus bandwidth for the
                tactile sensor reads.
        """
        print("Initialize InspireHandController...")
        print(f"  Left hand IP: {left_ip}:{port}")
        print(f"  Right hand IP: {right_ip}:{port}")

        self.device_id = device_id
        self.read_current = read_current

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
        # Tactile buffers: flat uint16 arrays of length 1062 per hand. Use
        # slice_tactile() to reshape into named regions.
        self.Ltactile = np.zeros(TACTILE_TOTAL_REGS, dtype=np.uint16)
        self.Rtactile = np.zeros(TACTILE_TOTAL_REGS, dtype=np.uint16)

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

    def _read_tactile(self, client, prev_buf):
        """Read the full tactile sensor block from one hand.

        Reads TACTILE_TOTAL_REGS (1062) holding registers in chunks of
        MODBUS_MAX_REGS, decodes each register as two bytes (high, low) of the
        underlying byte stream, then reinterprets the byte stream as
        little-endian uint16 touch values per the PDF section 2.6.20.

        Args:
            client: pymodbus ModbusTcpClient for the target hand.
            prev_buf: previous tactile buffer (returned on transient error so
                consumers do not see momentary zeros mid-episode).

        Returns:
            np.ndarray of shape (TACTILE_TOTAL_REGS,) and dtype uint16.
        """
        try:
            byte_buf = bytearray(TACTILE_TOTAL_REGS * 2)
            n_remaining = TACTILE_TOTAL_REGS
            reg_addr = REG_TACTILE_START
            byte_offset = 0
            while n_remaining > 0:
                chunk = min(MODBUS_MAX_REGS, n_remaining)
                response = client.read_holding_registers(
                    reg_addr, chunk, slave=self.device_id)
                if response.isError():
                    print(f"Error reading tactile registers at {reg_addr}")
                    return prev_buf
                for reg in response.registers:
                    byte_buf[byte_offset] = (reg >> 8) & 0xFF
                    byte_buf[byte_offset + 1] = reg & 0xFF
                    byte_offset += 2
                reg_addr += chunk
                n_remaining -= chunk
            return np.frombuffer(bytes(byte_buf), dtype='<u2').copy()
        except Exception as e:
            print(f"Exception reading tactile registers: {e}")
            return prev_buf

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

        # Read tactile sensor block from each hand. This replaces motor current
        # as the default contact-sensing channel.
        self.Ltactile = self._read_tactile(self.left_client, self.Ltactile)
        self.Rtactile = self._read_tactile(self.right_client, self.Rtactile)

        # Optional: motor current (proxy for joint effort, in mA). Off by default
        # to keep the per-loop Modbus budget available for tactile reads.
        if self.read_current:
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
            tuple of (Lpos, Rpos, Ltemp, Rtemp, Ltau, Rtau, Ltactile, Rtactile)
              - Lpos/Rpos:     6-element float32 joint angles (0-1000)
              - Ltemp/Rtemp:   6-element float32 actuator temperatures (deg C)
              - Ltau/Rtau:     6-element float32 motor currents (mA), only
                               refreshed when read_current=True; otherwise
                               returns the last (zero) value.
              - Ltactile/Rtactile: 1062-element uint16 flat tactile buffers.
                               Use slice_tactile() to reshape into named regions.
        """
        return (self.Lpos.copy(), self.Rpos.copy(),
                self.Ltemp.copy(), self.Rtemp.copy(),
                self.Ltau.copy(), self.Rtau.copy(),
                self.Ltactile.copy(), self.Rtactile.copy())

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
        port=args.port,
        read_current=False,
    )

    # Sanity check the tactile buffer shape and dtype.
    assert hand_ctrl.Ltactile.shape == (TACTILE_TOTAL_REGS,), hand_ctrl.Ltactile.shape
    assert hand_ctrl.Ltactile.dtype == np.uint16, hand_ctrl.Ltactile.dtype
    print(f"Tactile buffer length: {hand_ctrl.Ltactile.shape[0]} touch points per hand")

    # Loop-rate sanity check: time 100 calls to get_hand_state().
    t0 = time.time()
    for _ in range(100):
        hand_ctrl.get_hand_state()
    elapsed = time.time() - t0
    print(f"100 get_hand_state() calls in {elapsed:.2f}s "
          f"({100.0 / elapsed:.1f} Hz achieved with tactile reads)")

    # Test: gradually close then open and report tactile activity.
    print("Running test sequence (press a fingertip / palm pad to see tactile change)...")
    for i in range(11):
        angle = int(i * 100)  # 0 to 1000
        left_target = [angle] * 6
        right_target = [angle] * 6
        hand_ctrl.ctrl_dual_hand(left_target, right_target)
        left_state, right_state = hand_ctrl.get_hand_state()
        print(f"Step {i}: target={angle}, "
              f"Left angles={left_state[:3]}, "
              f"Ltactile max={int(hand_ctrl.Ltactile.max())} sum={int(hand_ctrl.Ltactile.sum())}, "
              f"Rtactile max={int(hand_ctrl.Rtactile.max())} sum={int(hand_ctrl.Rtactile.sum())}")
        time.sleep(0.3)

    # Return to open
    hand_ctrl.ctrl_dual_hand([0] * 6, [0] * 6)
    time.sleep(1.0)

    hand_ctrl.close()
    print("Test completed!")
