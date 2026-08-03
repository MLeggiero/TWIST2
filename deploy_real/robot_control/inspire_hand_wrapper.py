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
  1000 = fully open
  0 = fully closed

Network defaults (on Unitree G1 internal network):
  Left hand:  192.168.123.210:6000
  Right hand: 192.168.123.211:6000

Dependencies:
  pip install pymodbus==3.6.9
"""
import numpy as np
import struct
import threading
import time
from enum import IntEnum

try:
    from pymodbus.client import ModbusTcpClient
except ImportError:
    from pymodbus.client.sync import ModbusTcpClient

# Detect pymodbus API flavour by inspecting function signatures.
# - pymodbus 2.x: unit= for device ID, count as positional arg
# - pymodbus 3.0-3.6: slave= for device ID, count as positional arg
# - pymodbus 3.7+: dev_id= for device ID, count= as keyword arg
import inspect as _inspect
_write_params = set(_inspect.signature(ModbusTcpClient.write_registers).parameters)
_read_params = set(_inspect.signature(ModbusTcpClient.read_holding_registers).parameters)

if 'dev_id' in _write_params:
    _DEVICE_ID_KEY = 'dev_id'
elif 'slave' in _write_params:
    _DEVICE_ID_KEY = 'slave'
elif 'unit' in _write_params:
    _DEVICE_ID_KEY = 'unit'
else:
    _DEVICE_ID_KEY = None

# In pymodbus 3.7+, count must be passed as keyword to read_holding_registers
_READ_COUNT_KEYWORD = 'count' in _read_params

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

# Tactile sensor registers (Inspire PDF §2.6.20).
# The full tactile block lives at byte addresses 3000-5123 inclusive.
# Each Modbus holding register holds 2 bytes, and every touch point is a
# little-endian uint16 spanning 2 bytes, so the register count equals the
# touch-point count (1062).
REG_TACTILE_START = 3000
REG_TACTILE_END = 5123  # inclusive last byte
TACTILE_TOTAL_REGS = (REG_TACTILE_END - REG_TACTILE_START + 1) // 2  # 1062
MODBUS_MAX_REGS = 120  # safe margin under the 125-register Modbus limit

# Tactile skin region layout: (region_name, start_byte, n_points, (rows, cols)).
# Byte addresses are offsets from REG_TACTILE_START; every region is a
# contiguous block of little-endian uint16 touch values in the tactile buffer.
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
        flat_buf: 1-D array of length TACTILE_TOTAL_REGS (dtype uint16).

    Returns:
        Dict mapping region name to a 2-D array of the shape given in
        TACTILE_LAYOUT. Each region is a view into `flat_buf`.
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
                 port=6000, device_id=1, re_init=True,
                 enable_left=True, enable_right=True):
        """
        Initialize Inspire hand controller via Modbus TCP.

        After initialization, one daemon worker thread per *enabled* hand
        takes over all Modbus I/O. The public methods (`ctrl_dual_hand`,
        `get_hand_state`, `get_hand_all_state`) become non-blocking: they
        buffer commands and return cached state snapshots. This keeps the
        50 Hz main control loop insulated from Modbus latency/jitter.

        Either hand may be disabled (e.g. for bench-testing a single hand).
        A disabled hand is never connected, gets no worker thread, and is
        silently ignored by `ctrl_dual_hand`; its cached state stays zeroed.

        Args:
            left_ip: IP address of the left Inspire hand
            right_ip: IP address of the right Inspire hand
            port: Modbus TCP port (default 6000)
            device_id: Modbus device ID (default 1)
            re_init: Whether to clear errors and move to default position
            enable_left: Connect to and control the left hand (default True)
            enable_right: Connect to and control the right hand (default True)
        """
        if not (enable_left or enable_right):
            raise ValueError("At least one of enable_left/enable_right must be True")

        self.enable_left = enable_left
        self.enable_right = enable_right

        print("Initialize InspireHandController...")
        print(f"  Left hand:  {left_ip}:{port} ({'enabled' if enable_left else 'DISABLED'})")
        print(f"  Right hand: {right_ip}:{port} ({'enabled' if enable_right else 'DISABLED'})")

        self.device_id = device_id
        self._dev_kwargs = {_DEVICE_ID_KEY: device_id} if _DEVICE_ID_KEY else {}

        self.left_client = ModbusTcpClient(left_ip, port=port) if enable_left else None
        self.right_client = ModbusTcpClient(right_ip, port=port) if enable_right else None

        if enable_left:
            if not self.left_client.connect():
                raise ConnectionError(
                    f"Failed to connect to left Inspire hand at {left_ip}:{port}")
            print(f"  Left hand connected")

        if enable_right:
            if not self.right_client.connect():
                raise ConnectionError(
                    f"Failed to connect to right Inspire hand at {right_ip}:{port}")
            print(f"  Right hand connected")

        # Clear errors on init
        if re_init:
            if enable_left:
                self.left_client.write_register(REG_CLEAR_ERROR, 1, **self._dev_kwargs)
            if enable_right:
                self.right_client.write_register(REG_CLEAR_ERROR, 1, **self._dev_kwargs)

        # State arrays (updated by worker threads under _state_lock)
        self.left_hand_state_array = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.right_hand_state_array = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Lpos = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Rpos = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Ltemp = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Rtemp = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Ltau = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        self.Rtau = np.zeros(Inspire_Num_Motors, dtype=np.float32)

        # Tactile buffers: flat uint16 arrays of length TACTILE_TOTAL_REGS
        # per hand. Use slice_tactile() to reshape into named skin regions.
        # Populated by the per-hand worker thread each tick.
        self.Ltactile = np.zeros(TACTILE_TOTAL_REGS, dtype=np.uint16)
        self.Rtactile = np.zeros(TACTILE_TOTAL_REGS, dtype=np.uint16)

        # --- threading primitives ---
        # Two independent locks: _cmd_lock protects the command buffer,
        # _state_lock protects the feedback cache. No method ever holds
        # both simultaneously, so deadlock is impossible.
        self._state_lock = threading.Lock()
        self._cmd_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Last-wins command buffer. Initialized to the default open pose so
        # the first write the worker sends matches _bootstrap_write_default_sync.
        self._target_left = np.array(DEFAULT_QPOS_LEFT, dtype=np.int32)
        self._target_right = np.array(DEFAULT_QPOS_RIGHT, dtype=np.int32)
        self._target_dirty_left = False
        self._target_dirty_right = False

        # Error bookkeeping (protected by _state_lock)
        self._consec_errors_left = 0
        self._consec_errors_right = 0
        self._last_err_log_left = 0.0
        self._last_err_log_right = 0.0

        # Worker cadence: 50 Hz matches the main control loop.
        self._worker_period_s = 0.02

        self._worker_left = None
        self._worker_right = None

        # Read initial state synchronously — must happen before workers start
        # so that consumers never see zeroed caches and TCP failures raise
        # from __init__ as they do today.
        self._bootstrap_read_sync()
        print(f"  Left hand state: {self.left_hand_state_array}")
        print(f"  Right hand state: {self.right_hand_state_array}")

        if re_init:
            self._bootstrap_write_default_sync()

        # Spawn per-hand workers for enabled hands only. Each owns its own
        # ModbusTcpClient exclusively — this avoids any need for a per-client
        # mutex and lets L and R hand I/O run truly in parallel (separate
        # sockets, separate IPs, GIL released during socket.recv).
        if enable_left:
            self._worker_left = threading.Thread(
                target=self._worker_loop,
                args=("left", self.left_client),
                name="InspireHandWorker-L",
                daemon=True,
            )
            self._worker_left.start()
        if enable_right:
            self._worker_right = threading.Thread(
                target=self._worker_loop,
                args=("right", self.right_client),
                name="InspireHandWorker-R",
                daemon=True,
            )
            self._worker_right.start()

        print("Initialize InspireHandController OK!\n")

    def _read_holding(self, client, address, count):
        """Read holding registers with correct API for installed pymodbus version."""
        if _READ_COUNT_KEYWORD:
            return client.read_holding_registers(address, count=count, **self._dev_kwargs)
        else:
            return client.read_holding_registers(address, count, **self._dev_kwargs)

    def _read_registers_signed(self, client, address, count):
        """Read Modbus registers and interpret as signed int16."""
        try:
            response = self._read_holding(client, address, count)
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
            response = self._read_holding(client, address, count)
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
        MODBUS_MAX_REGS (stays under the pymodbus 125-register cap),
        decodes each register as two bytes (high, low), and reinterprets
        the byte stream as little-endian uint16 touch values per the
        Inspire PDF §2.6.20.

        On any transient Modbus error the previously cached buffer is
        returned, so consumers never observe momentary zero-glitches
        mid-episode. No print/log is emitted here because the worker
        calls this at ~20 Hz — any chatty error would spam. The worker's
        `_note_error` path still fires via the enclosing try/except.
        """
        try:
            byte_buf = bytearray(TACTILE_TOTAL_REGS * 2)
            n_remaining = TACTILE_TOTAL_REGS
            reg_addr = REG_TACTILE_START
            byte_offset = 0
            while n_remaining > 0:
                chunk = min(MODBUS_MAX_REGS, n_remaining)
                response = self._read_holding(client, reg_addr, chunk)
                if response.isError():
                    return prev_buf
                for reg in response.registers:
                    byte_buf[byte_offset] = (reg >> 8) & 0xFF
                    byte_buf[byte_offset + 1] = reg & 0xFF
                    byte_offset += 2
                reg_addr += chunk
                n_remaining -= chunk
            return np.frombuffer(bytes(byte_buf), dtype='<u2').copy()
        except Exception:
            return prev_buf

    def _bootstrap_read_sync(self):
        """One-shot synchronous read of angle/current/temperature for both hands.

        Used only during __init__ before worker threads exist. Populates the
        feedback cache fields directly (no lock needed — no concurrent readers
        yet). If TCP is broken, the underlying read helpers return zeros and
        log an error, matching the legacy behavior.
        """
        if self.enable_left:
            left_angles = self._read_registers_signed(self.left_client, REG_ANGLE_ACT, 6)
            self.left_hand_state_array = np.array(left_angles, dtype=np.float32)
            self.Lpos = self.left_hand_state_array.copy()
            left_current = self._read_registers_signed(self.left_client, REG_CURRENT, 6)
            self.Ltau = np.array(left_current, dtype=np.float32)
            left_temp = self._read_registers_bytes(self.left_client, REG_TEMPERATURE, 3)
            self.Ltemp = np.array(left_temp[:Inspire_Num_Motors], dtype=np.float32)

        if self.enable_right:
            right_angles = self._read_registers_signed(self.right_client, REG_ANGLE_ACT, 6)
            self.right_hand_state_array = np.array(right_angles, dtype=np.float32)
            self.Rpos = self.right_hand_state_array.copy()
            right_current = self._read_registers_signed(self.right_client, REG_CURRENT, 6)
            self.Rtau = np.array(right_current, dtype=np.float32)
            right_temp = self._read_registers_bytes(self.right_client, REG_TEMPERATURE, 3)
            self.Rtemp = np.array(right_temp[:Inspire_Num_Motors], dtype=np.float32)

    def _bootstrap_write_default_sync(self):
        """One-shot synchronous write of the default open pose to both hands.

        Used only during __init__ (when re_init=True) and from close(). Must
        only be called from the main thread when workers are not running.
        """
        print("Initializing Inspire hands with default open poses...")
        left_angles = [int(np.clip(v, 0, 1000)) for v in DEFAULT_QPOS_LEFT]
        right_angles = [int(np.clip(v, 0, 1000)) for v in DEFAULT_QPOS_RIGHT]
        try:
            if self.enable_left:
                self.left_client.write_registers(REG_ANGLE_SET, left_angles, **self._dev_kwargs)
            if self.enable_right:
                self.right_client.write_registers(REG_ANGLE_SET, right_angles, **self._dev_kwargs)
        except Exception as e:
            print(f"Error writing default pose to hands: {e}")

    def _note_error(self, side, msg):
        """Record a Modbus I/O error for `side` and log at most once per 2s."""
        now = time.monotonic()
        with self._state_lock:
            if side == "left":
                self._consec_errors_left += 1
                count = self._consec_errors_left
                last_log = self._last_err_log_left
            else:
                self._consec_errors_right += 1
                count = self._consec_errors_right
                last_log = self._last_err_log_right
            should_log = (count == 1) or ((now - last_log) > 2.0)
            if should_log:
                if side == "left":
                    self._last_err_log_left = now
                else:
                    self._last_err_log_right = now

        if should_log:
            print(f"[InspireHandWorker-{side[0].upper()}] error (count={count}): {msg}")

    def _note_ok(self, side):
        """Reset the error counter for `side` after a successful I/O op."""
        with self._state_lock:
            if side == "left":
                if self._consec_errors_left > 0:
                    count = self._consec_errors_left
                    self._consec_errors_left = 0
                    print(f"[InspireHandWorker-L] recovered after {count} errors")
            else:
                if self._consec_errors_right > 0:
                    count = self._consec_errors_right
                    self._consec_errors_right = 0
                    print(f"[InspireHandWorker-R] recovered after {count} errors")

    def _worker_loop(self, side, client):
        """Background Modbus worker owning one hand's TCP client exclusively.

        Per tick (50 Hz):
            1. If the command buffer is dirty, write REG_ANGLE_SET.
            2. Read REG_ANGLE_ACT, REG_CURRENT, REG_TEMPERATURE.
            3. Commit the new feedback to the cache under _state_lock.
            4. Deadline-sleep until the next tick (cancellable via stop_event).

        No lock is ever held across a Modbus call.
        """
        is_left = (side == "left")
        period = self._worker_period_s
        next_tick = time.monotonic()

        while not self._stop_event.is_set():
            try:
                # --- 1. Write if dirty (last-wins, drop on error) ---
                target_to_write = None
                with self._cmd_lock:
                    if is_left:
                        dirty = self._target_dirty_left
                        if dirty:
                            target_to_write = self._target_left.tolist()
                            self._target_dirty_left = False
                    else:
                        dirty = self._target_dirty_right
                        if dirty:
                            target_to_write = self._target_right.tolist()
                            self._target_dirty_right = False

                if target_to_write is not None:
                    try:
                        client.write_registers(
                            REG_ANGLE_SET, target_to_write, **self._dev_kwargs)
                        self._note_ok(side)
                    except Exception as e:
                        self._note_error(side, f"write_registers: {e}")

                # --- 2a. Read angle_act ---
                pos = None
                try:
                    angles = self._read_registers_signed(client, REG_ANGLE_ACT, 6)
                    pos = np.array(angles, dtype=np.float32)
                    self._note_ok(side)
                except Exception as e:
                    self._note_error(side, f"read angle_act: {e}")

                # --- 2b. Read motor current ---
                tau = None
                try:
                    current = self._read_registers_signed(client, REG_CURRENT, 6)
                    tau = np.array(current, dtype=np.float32)
                except Exception as e:
                    self._note_error(side, f"read current: {e}")

                # --- 2c. Read temperature (byte-packed 3 registers -> 6 bytes) ---
                temp = None
                try:
                    temp_raw = self._read_registers_bytes(client, REG_TEMPERATURE, 3)
                    temp = np.array(temp_raw[:Inspire_Num_Motors], dtype=np.float32)
                except Exception as e:
                    self._note_error(side, f"read temperature: {e}")

                # --- 2d. Read tactile (dense 1062-uint16 block) ---
                # This read dominates the per-tick budget (~40 ms on the
                # 192.168.123.x LAN vs ~3 ms for the other reads), so the
                # worker effectively runs at ~19 Hz per hand under the
                # overrun-reset branch below. That is intentional; the
                # main 50 Hz control loop is unaffected because it reads
                # the cache, not the hardware.
                tactile = None
                try:
                    prev = self.Ltactile if is_left else self.Rtactile
                    tactile = self._read_tactile(client, prev)
                except Exception as e:
                    self._note_error(side, f"read tactile: {e}")

                # --- 3. Commit cache under _state_lock ---
                if (pos is not None or tau is not None
                        or temp is not None or tactile is not None):
                    with self._state_lock:
                        if is_left:
                            if pos is not None:
                                self.Lpos = pos
                                self.left_hand_state_array = pos
                            if tau is not None:
                                self.Ltau = tau
                            if temp is not None:
                                self.Ltemp = temp
                            if tactile is not None:
                                self.Ltactile = tactile
                        else:
                            if pos is not None:
                                self.Rpos = pos
                                self.right_hand_state_array = pos
                            if tau is not None:
                                self.Rtau = tau
                            if temp is not None:
                                self.Rtemp = temp
                            if tactile is not None:
                                self.Rtactile = tactile

            except Exception as e:
                # Catch-all: a bug in worker code must never silently kill
                # the thread. Log and continue to the next tick.
                print(f"[InspireHandWorker-{side[0].upper()}] unexpected error: {e}")

            # --- 4. Deadline sleep, cancellable by stop_event ---
            next_tick += period
            now = time.monotonic()
            remaining = next_tick - now
            if remaining > 0:
                if self._stop_event.wait(timeout=remaining):
                    break
            else:
                # Overrun: reset phase so we don't spiral trying to catch up.
                next_tick = now

    def get_hand_state(self):
        """Return the most recent cached hand joint angles (non-blocking).

        Returns:
            (left_hand_state_6d, right_hand_state_6d): float32 arrays, shape (6,).
        """
        with self._state_lock:
            return (self.left_hand_state_array.copy(),
                    self.right_hand_state_array.copy())

    def get_hand_all_state(self):
        """Return a snapshot of all cached hand telemetry (non-blocking).

        Returns:
            (Lpos, Rpos, Ltemp, Rtemp, Ltau, Rtau, Ltactile, Rtactile):
                - Lpos/Rpos/Ltemp/Rtemp/Ltau/Rtau: float32 arrays, shape (6,) each.
                - Ltactile/Rtactile: uint16 arrays, shape (TACTILE_TOTAL_REGS,) each.
        """
        with self._state_lock:
            return (self.Lpos.copy(), self.Rpos.copy(),
                    self.Ltemp.copy(), self.Rtemp.copy(),
                    self.Ltau.copy(), self.Rtau.copy(),
                    self.Ltactile.copy(), self.Rtactile.copy())

    def _coerce_shape(self, arr):
        """Fit a 1-D array to Inspire_Num_Motors by truncating or zero-padding.

        Upstream publishers may send a different-length array (e.g. the
        7-element dex3 command format) when the teleop side is configured
        for a different hand type. Coercing here keeps the main control
        loop alive regardless of upstream shape.
        """
        if arr.shape == (Inspire_Num_Motors,):
            return arr
        if arr.size >= Inspire_Num_Motors:
            return arr[:Inspire_Num_Motors]
        out = np.zeros(Inspire_Num_Motors, dtype=np.float32)
        out[:arr.size] = arr
        return out

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """Buffer the latest angle setpoints for both hands (non-blocking).

        The background worker threads pick up the most recent buffered
        command within one tick (~20 ms) and push it to the hardware.
        Identical repeats are filtered so we never put unnecessary traffic
        on the Modbus link.

        Inputs whose length does not match `Inspire_Num_Motors` are
        truncated or zero-padded (see `_coerce_shape`). This keeps the
        50 Hz main control loop from crashing when upstream publishes
        commands in a different hand format.

        Args:
            left_q_target: 6-element array/list of angle setpoints (0-1000 range)
            right_q_target: 6-element array/list of angle setpoints (0-1000 range)
        """
        left_arr = np.asarray(left_q_target, dtype=np.float32).flatten()
        right_arr = np.asarray(right_q_target, dtype=np.float32).flatten()

        left_fixed = self._coerce_shape(left_arr)
        right_fixed = self._coerce_shape(right_arr)

        left_clipped = np.clip(left_fixed, 0, 1000).astype(np.int32)
        right_clipped = np.clip(right_fixed, 0, 1000).astype(np.int32)

        with self._cmd_lock:
            if self.enable_left and not np.array_equal(left_clipped, self._target_left):
                self._target_left[:] = left_clipped
                self._target_dirty_left = True
            if self.enable_right and not np.array_equal(right_clipped, self._target_right):
                self._target_right[:] = right_clipped
                self._target_dirty_right = True

    def initialize(self):
        """Buffer the default open pose for both hands (API compatibility)."""
        print("Initializing Inspire hands with default open poses...")
        self.ctrl_dual_hand(DEFAULT_QPOS_LEFT, DEFAULT_QPOS_RIGHT)

    def close(self):
        """Stop workers, command default open pose, and disconnect TCP clients.

        Order matters: stop workers and join them before the main thread
        touches the Modbus clients, otherwise we'd race the worker on the
        same socket.
        """
        self._stop_event.set()

        for worker, label in ((self._worker_left, "L"), (self._worker_right, "R")):
            if worker is not None and worker.is_alive():
                worker.join(timeout=1.0)
                if worker.is_alive():
                    print(f"[InspireHandController] warning: "
                          f"worker {label} did not exit within 1s")

        # Workers stopped — main thread now owns both clients again.
        try:
            self._bootstrap_write_default_sync()
            time.sleep(0.5)  # let actuators physically start opening
        except Exception as e:
            print(f"[InspireHandController] warning: "
                  f"default-pose write on close failed: {e}")

        try:
            if self.left_client is not None:
                self.left_client.close()
            if self.right_client is not None:
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
    parser.add_argument('--hand', choices=['left', 'right', 'both'], default='both',
                        help='Which hand(s) to connect to and command')
    args = parser.parse_args()

    enable_left = args.hand in ('left', 'both')
    enable_right = args.hand in ('right', 'both')

    print(f"Testing InspireHandController (hand={args.hand})...")
    hand_ctrl = InspireHandController(
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        port=args.port,
        enable_left=enable_left,
        enable_right=enable_right,
    )

    # Test: gradually close then open. Disabled hands are ignored by
    # ctrl_dual_hand, so it is safe to always pass both targets.
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
