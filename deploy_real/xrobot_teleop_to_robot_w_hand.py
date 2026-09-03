"""
conda activate gmr
sudo ufw disable
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1

State Machine Controls:
- Right controller key_one (A): Cycle through idle -> teleop -> pause -> teleop...
- Right controller key_two (B): Hold ~1s to exit program from any state
- Left controller key_one (X): Toggle fine (precision) mode for Inspire hands
- Left controller axis_click: Emergency stop - kills sim2real.sh process
- Left controller axis: Control root xy velocity and yaw velocity
- Right controller axis: Fine-tune root xy velocity and yaw velocity
- Auto-transition: idle -> teleop when motion data is available

States:
- idle: Waiting for input or data
- teleop: Processing motion retargeting with velocity control
- pause: Data received but not processing
- exit: Program will terminate

Whole-Body Teleop Features:
- Sends whole-body mode information to Redis
- 35-dimensional mimic observations
- Uses retargeted motion directly from the teleoperation stream
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import cv2
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import redis
from data_utils.finger_tracking import PicoFingerTracker
from data_utils.fps_monitor import FPSMonitor
from data_utils.params import DEFAULT_HAND_POSE, DEFAULT_MIMIC_OBS
from data_utils.rot_utils import euler_from_quaternion_np, quat_diff_np, quat_rotate_inverse_np
from pico_hand_utils import pico_hand_to_mediapipe
from general_motion_retargeting import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    XRobotStreamer,
    draw_frame,
    human_head_to_robot_neck,
)
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from loop_rate_limiters import RateLimiter
from rich import print
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def start_interpolation(state_machine, start_obs, end_obs, duration=1.0):
    """Start interpolation from start_obs to end_obs over given duration"""
    state_machine.is_interpolating = True
    state_machine.interpolation_start_time = time.time()
    state_machine.interpolation_duration = duration
    state_machine.interpolation_start_obs = start_obs.copy() if start_obs is not None else None
    state_machine.interpolation_target_obs = end_obs.copy() if end_obs is not None else None
    

def get_interpolated_obs(state_machine):
    """Get current interpolated observation, returns None if interpolation complete"""
    if (not state_machine.is_interpolating or 
        state_machine.interpolation_start_obs is None or 
        state_machine.interpolation_target_obs is None or 
        state_machine.interpolation_start_time is None):
        return None
    elapsed_time = time.time() - state_machine.interpolation_start_time
    progress = min(elapsed_time / state_machine.interpolation_duration, 1.0)
    
    # Linear interpolation
    interp_obs = state_machine.interpolation_start_obs + (state_machine.interpolation_target_obs - state_machine.interpolation_start_obs) * progress
    
    # Check if interpolation is complete
    if progress >= 1.0:
        state_machine.is_interpolating = False
        return state_machine.interpolation_target_obs
    
    return interp_obs

def extract_mimic_obs_whole_body(qpos, last_qpos, dt=1/30):
    """Extract whole body mimic observations from robot joint positions (35 dims)"""
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()  # Make a copy to avoid modifying original
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    # convert root vel to local frame
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)
    
    # Standard mimic observation (35 dims)
    height = root_pos[2:3]
    # print("height: ", height)
    mimic_obs = np.concatenate([
        base_vel_local[:2],  # xy velocity (2 dims)
        root_pos[2:3],       # z position (1 dim)
        roll, pitch,         # roll, pitch (2 dims)
        base_ang_vel_local[2:3],  # yaw angular velocity (1 dim)
        robot_joints         # joint positions (29 dims)
    ])
    
    return mimic_obs


class SourceTimestampWatchdog:
    """Track whether an XR source timestamp is advancing locally.

    The first observed timestamp is deliberately not accepted as fresh. The
    XRoboToolkit SDK caches its last frame, so a nonzero timestamp alone does
    not prove that the headset is still sending data. A second, different
    timestamp establishes liveness; short duplicate runs are then tolerated
    for ``timeout`` seconds because the producer may run faster than PICO.
    """

    def __init__(self, timeout):
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("source timestamp timeout must be positive")
        self.last_timestamp = None
        self.last_advance_time = None

    def observe(self, timestamp, now=None):
        now = time.monotonic() if now is None else float(now)
        try:
            timestamp = int(timestamp)
        except (TypeError, ValueError, OverflowError):
            timestamp = None
        if timestamp is not None and timestamp <= 0:
            timestamp = None

        advanced = False
        if timestamp is not None:
            if self.last_timestamp is None:
                self.last_timestamp = timestamp
            elif timestamp != self.last_timestamp:
                self.last_timestamp = timestamp
                self.last_advance_time = now
                advanced = True

        fresh = (
            self.last_advance_time is not None
            and now - self.last_advance_time <= self.timeout
        )
        return advanced, fresh


class StateMachine:
    def __init__(self, enable_smooth=False, smooth_window_size=5, use_pinch=False,
                 use_finger_tracking=False, hand_type='dex3', grip_thumb=False,
                 fine_span=250.0, thumb_rate=500.0):
        """
        State process for teleoperation:
        idle -> teleop -> pause -> teleop ... -> idle -> exit
        """
        self.state = "idle"
        self.previous_state = "idle"
        self.right_key_one_was_pressed = False
        self.left_key_one_was_pressed = False
        self.left_axis_click_was_pressed = False
        # Exit (right key_two / B) requires a deliberate hold, not a tap - B sits
        # under the manipulating thumb next to A, so a stray press must not end the run.
        self.exit_hold_seconds = 1.0
        self._right_key_two_press_time = None
        # Interpolation state
        self.is_interpolating = False
        self.interpolation_start_time = None
        self.interpolation_duration = 2.0  # seconds
        self.interpolation_start_obs = None
        self.interpolation_target_obs = None
        self.current_mimic_obs = None
        self.last_mimic_obs = None
        self.current_neck_data = None
        self.last_neck_data = None

        # Hand type
        self.hand_type = hand_type

        # Hand state - interpolation values (0.0 = open, 1.0 = closed)
        self.hand_left_position = 0.0  # 0.0 = fully open, 1.0 = fully closed (Dex3)
        self.hand_right_position = 0.0
        # Inspire split control: fingers (3 DOF), index (1 DOF) and thumb (2 DOF)
        self.hand_left_finger_position = 0.0  # pinky, ring, middle
        self.hand_left_index_position = 0.0
        # Thumbs are rate-controlled accumulators, not absolute stick reads: 0.5 is
        # the mid pose the centred stick used to command, so rest behaviour is unchanged.
        self.hand_left_thumb_position = 0.5
        self.hand_left_thumb_bend_position = 0.5
        self.hand_right_finger_position = 0.0  # pinky, ring, middle
        self.hand_right_index_position = 0.0
        self.hand_right_thumb_position = 0.5
        self.hand_right_thumb_bend_position = 0.5

        # Fine (precision) mode - toggled by left key_one (X)
        self.fine_mode = False
        self.fine_span = float(fine_span)      # counts of finger travel per full lever sweep
        self.thumb_rate_coarse = float(thumb_rate)  # counts/sec at full stick deflection
        self.thumb_deadzone = 0.12             # rejects stick miscentring drift
        # Per-channel state for the analog levers (grip = 3 fingers, trigger = index)
        self._chan_pos = {}         # last commanded position, 0..1
        self._fine_anchor = {}      # (anchor_pos, anchor_raw) latched on entering fine mode
        self._takeover_pending = {} # after leaving fine mode, hold until coarse curve crosses
        self._prev_coarse = {}
        self._last_update_time = None
        self.use_pinch = use_pinch
        self.grip_thumb = grip_thumb  # Whether grip button also controls thumb rotation
        # Hand control parameters (Dex3 step-based)
        self.hand_movement_step = 0.05  # 5% movement per press/hold
        
        # Finger tracking
        self.use_finger_tracking = use_finger_tracking
        if use_finger_tracking:
            self.finger_tracker = PicoFingerTracker(
                smoothing_alpha=0.3,
                curl_gain=1.5,
                curl_deadzone=0.05
            )
            # Tracked finger poses (6-DOF Inspire angles per hand)
            self._tracked_left_hand_pose = None
            self._tracked_right_hand_pose = None
        
        # Velocity commands from joystick
        self.velocity_commands = np.array([0.0, 0.0, 0.0])  # [vx, vy, vyaw]
        
        # Smooth filtering
        self.enable_smooth = enable_smooth
        self.smooth_window_size = smooth_window_size
        self.smooth_history = []  # Store recent observations for sliding window

    def _set_fine_mode(self, enabled):
        """Toggle fine mode, arming the anchors (on) or soft takeover (off)."""
        if enabled == self.fine_mode:
            return
        self.fine_mode = enabled
        if enabled:
            # Anchors latch on the next sample, so engaging never moves a finger.
            self._fine_anchor = {}
        else:
            # Returning to the absolute curve would jump; hold each finger until the
            # coarse curve sweeps back through it (soft takeover).
            for ch in self._chan_pos:
                self._takeover_pending[ch] = True
                self._prev_coarse[ch] = None
        if enabled:
            print(f"[Hand] Fine mode ON (span {self.fine_span:.0f} counts)")
        else:
            print("[Hand] Fine mode OFF")

    def _map_lever(self, channel, raw):
        """Map a 0..1 grip/trigger reading to a 0..1 finger position.

        Coarse is the absolute curve (pressed -> closed). Fine anchors on the position
        and lever value at engagement and applies a reduced gain from there, so control
        resumes exactly where it stopped instead of snapping to a remapped range.
        """
        raw = float(raw)
        coarse = 1.0 - raw

        if self.fine_mode:
            if channel not in self._fine_anchor:
                self._fine_anchor[channel] = (self._chan_pos.get(channel, coarse), raw)
            anchor_pos, anchor_raw = self._fine_anchor[channel]
            gain = self.fine_span / 1000.0
            pos = min(max(anchor_pos - (raw - anchor_raw) * gain, 0.0), 1.0)
            # Ratchet: re-anchor at the lever end stops so travel isn't capped at one window.
            if raw <= 1e-3 or raw >= 1.0 - 1e-3:
                self._fine_anchor[channel] = (pos, raw)
        elif self._takeover_pending.get(channel, False):
            held = self._chan_pos.get(channel, coarse)
            prev = self._prev_coarse.get(channel)
            if prev is not None and (prev - held) * (coarse - held) <= 0.0:
                self._takeover_pending[channel] = False  # curve crossed - reattach
                pos = coarse
            else:
                pos = held
            self._prev_coarse[channel] = coarse
        else:
            pos = coarse

        self._chan_pos[channel] = pos
        return pos

    def _integrate_thumb(self, current, signed_input, dt):
        """Rate-control a thumb axis: deflection commands speed, centred holds position."""
        v = float(signed_input)
        if dt <= 0.0 or abs(v) < self.thumb_deadzone:
            return current
        # Rescale past the deadzone so motion eases in from zero rather than stepping.
        magnitude = (abs(v) - self.thumb_deadzone) / (1.0 - self.thumb_deadzone)
        rate = self.thumb_rate_coarse * (self.fine_span / 1000.0) if self.fine_mode \
            else self.thumb_rate_coarse
        delta = (1.0 if v > 0 else -1.0) * magnitude * (rate / 1000.0) * dt
        return min(max(current + delta, 0.0), 1.0)

    def update(self, controller_data):
        """Update state machine with controller data"""
        # Store previous state
        self.previous_state = self.state
        
        # Loop dt for rate-controlled axes (clamped so a stall can't fling the thumbs)
        now = time.time()
        dt = 0.0 if self._last_update_time is None \
            else min(max(now - self._last_update_time, 0.0), 0.1)
        self._last_update_time = now

        # Get current button states
        right_key_current = controller_data.get('RightController', {}).get('key_one', False)
        left_key_current = controller_data.get('LeftController', {}).get('key_one', False)
        right_key_two_current = controller_data.get('RightController', {}).get('key_two', False)
        
        # Hand control - Inspire: grip for 3 fingers, index_trig for index (Dex3: index_trig close, grip open)
        right_index_trig_current = controller_data.get('RightController', {}).get('index_trig', False)
        left_index_trig_current = controller_data.get('LeftController', {}).get('index_trig', False)
        right_grip_current = controller_data.get('RightController', {}).get('grip', False)
        left_grip_current = controller_data.get('LeftController', {}).get('grip', False)

        # Emergency stop - left controller axis_click
        left_axis_click_current = controller_data.get('LeftController', {}).get('axis_click', False)

        # Detect button presses
        right_key_just_pressed = right_key_current and not self.right_key_one_was_pressed
        left_key_just_pressed = left_key_current and not self.left_key_one_was_pressed
        left_axis_click_just_pressed = left_axis_click_current and not self.left_axis_click_was_pressed

        # Handle left axis click - emergency stop
        if left_axis_click_just_pressed:
            self._emergency_stop()

        # Handle right key_two (B) - exit, but only on a deliberate hold
        if right_key_two_current:
            if self._right_key_two_press_time is None:
                self._right_key_two_press_time = now
            elif now - self._right_key_two_press_time >= self.exit_hold_seconds:
                self.state = "exit"
        else:
            self._right_key_two_press_time = None

        # Handle left key press (X) - toggle fine mode
        if left_key_just_pressed:
            self._set_fine_mode(not self.fine_mode)

        # Handle right key press - cycle between idle, teleop, pause
        if self.state != "exit" and right_key_just_pressed:
            if self.state == "idle":
                self.state = "teleop"
            elif self.state == "teleop":
                self.state = "pause"
            elif self.state == "pause":
                self.state = "teleop"

        # Handle hand control
        if self.hand_type == 'inspire':
            # Inspire: proportional analog control
            # Side grip = pinky, ring, middle
            # Back trigger (index_trig) = index finger
            # Each joystick controls its respective hand's thumb:
            #   X axis = thumb rotation, Y axis = thumb bend
            # --grip_thumb flag: grip button also controls thumb rotation
            # Invert PICO input: pressed 1.0 → position 0.0 (closed).
            # Fine mode anchors each lever so precision control resumes where it stopped.
            self.hand_right_finger_position = self._map_lever('right_fingers', right_grip_current)
            self.hand_left_finger_position = self._map_lever('left_fingers', left_grip_current)
            self.hand_right_index_position = self._map_lever('right_index', right_index_trig_current)
            self.hand_left_index_position = self._map_lever('left_index', left_index_trig_current)

            # Right joystick → right thumb (rate control: centred stick holds position)
            right_axis = controller_data.get('RightController', {}).get('axis', [0.0, 0.0])
            if len(right_axis) >= 2:
                self.hand_right_thumb_bend_position = self._integrate_thumb(
                    self.hand_right_thumb_bend_position, -right_axis[1], dt)
                self.hand_right_thumb_position = self._integrate_thumb(
                    self.hand_right_thumb_position, right_axis[0], dt)
                if self.grip_thumb:
                    self.hand_right_thumb_position = max(
                        self.hand_right_thumb_position, 1.0 - float(right_grip_current))

            # Left joystick → left thumb (sign on rotation mirrors the right hand)
            left_axis = controller_data.get('LeftController', {}).get('axis', [0.0, 0.0])
            if len(left_axis) >= 2:
                self.hand_left_thumb_bend_position = self._integrate_thumb(
                    self.hand_left_thumb_bend_position, -left_axis[1], dt)
                self.hand_left_thumb_position = self._integrate_thumb(
                    self.hand_left_thumb_position, -left_axis[0], dt)
                if self.grip_thumb:
                    self.hand_left_thumb_position = max(
                        self.hand_left_thumb_position, 1.0 - float(left_grip_current))
        else:
            # Dex3: step-based interpolation (original behavior)
            # Right hand control
            if right_index_trig_current:  # Close right hand
                new_position = min(1.0, self.hand_right_position + self.hand_movement_step)
                if new_position != self.hand_right_position:
                    self.hand_right_position = new_position
                    print(f"Right hand closing: {self.hand_right_position:.1f}")
            elif right_grip_current:  # Open right hand
                new_position = max(0.0, self.hand_right_position - self.hand_movement_step)
                if new_position != self.hand_right_position:
                    self.hand_right_position = new_position
                    print(f"Right hand opening: {self.hand_right_position:.1f}")

            # Left hand control
            if left_index_trig_current:  # Close left hand
                new_position = min(1.0, self.hand_left_position + self.hand_movement_step)
                if new_position != self.hand_left_position:
                    self.hand_left_position = new_position
                    print(f"Left hand closing: {self.hand_left_position:.1f}")
            elif left_grip_current:  # Open left hand
                new_position = max(0.0, self.hand_left_position - self.hand_movement_step)
                if new_position != self.hand_left_position:
                    self.hand_left_position = new_position
                    print(f"Left hand opening: {self.hand_left_position:.1f}")
        
        # Extract velocity commands from controller axes
        self._update_velocity_commands(controller_data)
        
        # Update button state tracking
        self.right_key_one_was_pressed = right_key_current
        self.left_key_one_was_pressed = left_key_current
        self.left_axis_click_was_pressed = left_axis_click_current
    
    def _update_velocity_commands(self, controller_data):
        """Update velocity commands from controller axes"""
        left_axis = controller_data.get('LeftController', {}).get('axis', [0.0, 0.0])
        right_axis = controller_data.get('RightController', {}).get('axis', [0.0, 0.0])
        
        if self.hand_type == 'inspire':
            # Both joysticks used for thumb control in Inspire mode, no velocity commands
            self.velocity_commands[:] = 0.0
        elif len(left_axis) >= 2 and len(right_axis) >= 2:
            # Use left stick for xy movement, right stick X for yaw rotation
            xy_scale = 2.0  # m/s
            yaw_scale = 3.0  # rad/s
            self.velocity_commands[0] = left_axis[1] * xy_scale   # forward/backward
            self.velocity_commands[1] = -left_axis[0] * xy_scale  # left/right
            self.velocity_commands[2] = -right_axis[0] * yaw_scale  # yaw rotation
    
    def has_state_changed(self):
        """Check if state has changed since last update"""
        return self.state != self.previous_state
    
    
    def set_current_mimic_obs(self, mimic_obs):
        """Update current mimic obs"""
        self.current_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None
        
    def set_last_mimic_obs(self, mimic_obs):
        """Update last mimic obs (used when entering pause)"""
        self.last_mimic_obs = mimic_obs.copy() if mimic_obs is not None else None
        
    def set_last_neck_data(self, neck_data):
        """Update last neck data (used when entering pause)"""
        self.last_neck_data = neck_data[:] if neck_data is not None else None
        
    def set_current_neck_data(self, neck_data):
        """Update current neck data"""
        self.current_neck_data = neck_data[:] if neck_data is not None else None
    
    def get_current_state(self):
        return self.state
    

    def get_velocity_commands(self):
        return self.velocity_commands.copy()
        
    def is_teleop_active(self):
        """Return True if currently in teleop state"""
        return self.state == "teleop"
        
    def should_exit(self):
        """Return True if should exit the program"""
        return self.state == "exit"
        
    def should_process_data(self):
        """Return True if should process motion data"""
        return self.state == "teleop" and not self.is_interpolating
    
    def get_hand_state(self):
        return self.hand_left_position, self.hand_right_position
    
    def update_hand_from_tracking(self, left_hand_data, right_hand_data):
        """Update hand poses from Pico finger tracking data.
        
        Args:
            left_hand_data: Pico hand tracking array for left hand (26×7 or 27×7)
            right_hand_data: Pico hand tracking array for right hand (26×7 or 27×7)
        """
        if not self.use_finger_tracking:
            return
        
        if left_hand_data is not None:
            result = self.finger_tracker.pico_to_inspire_angles(left_hand_data, 'left')
            if result is not None:
                self._tracked_left_hand_pose = result
        
        if right_hand_data is not None:
            result = self.finger_tracker.pico_to_inspire_angles(right_hand_data, 'right')
            if result is not None:
                self._tracked_right_hand_pose = result
    
    def get_hand_pose(self, robot_name):
        """Get hand poses — from finger tracking, Inspire proportional, or Dex3 interpolation."""
        # If finger tracking is active and we have tracked data, use it directly
        if self.use_finger_tracking:
            # For Inspire hands, return the tracked 6-DOF angles directly
            left_default = DEFAULT_HAND_POSE.get(robot_name, {}).get('left', {}).get('open', np.zeros(6, dtype=np.float32))
            right_default = DEFAULT_HAND_POSE.get(robot_name, {}).get('right', {}).get('open', np.zeros(6, dtype=np.float32))

            left_pose = self._tracked_left_hand_pose if self._tracked_left_hand_pose is not None else left_default
            right_pose = self._tracked_right_hand_pose if self._tracked_right_hand_pose is not None else right_default
            return left_pose, right_pose

        # Inspire RH56DFTP: 1000 = open, 0 = closed
        # pressed (1.0) → close (0), released (0.0) → open (1000)
        if self.hand_type == 'inspire':
            l_finger = self.hand_left_finger_position * 1000
            l_index = self.hand_left_index_position * 1000
            l_thumb = self.hand_left_thumb_position * 1000
            l_thumb_bend = self.hand_left_thumb_bend_position * 1000
            r_finger = self.hand_right_finger_position * 1000
            r_index = self.hand_right_index_position * 1000
            r_thumb = self.hand_right_thumb_position * 1000
            r_thumb_bend = self.hand_right_thumb_bend_position * 1000
            left_pose = np.array([
                l_finger, l_finger, l_finger,  # pinky, ring, middle (side grip)
                l_index,                       # index (back trigger)
                l_thumb_bend, l_thumb,                       # thumb bend (joystick), thumb rotation
            ], dtype=np.float32)
            right_pose = np.array([
                r_finger, r_finger, r_finger,  # pinky, ring, middle (side grip)
                r_index,                       # index (back trigger)
                r_thumb_bend, r_thumb,                       # thumb bend (joystick), thumb rotation
            ], dtype=np.float32)
            return left_pose, right_pose

        # Dex3: button-based interpolation (original behavior)
        use_pinch = self.use_pinch

        if not use_pinch:
            left_open = DEFAULT_HAND_POSE[robot_name]['left']['open']
            left_closed = DEFAULT_HAND_POSE[robot_name]['left']['close']
            right_open = DEFAULT_HAND_POSE[robot_name]['right']['open']
            right_closed = DEFAULT_HAND_POSE[robot_name]['right']['close']
        else:
            left_fully_open = DEFAULT_HAND_POSE[robot_name]['left']['open_pinch']
            left_fully_closed = DEFAULT_HAND_POSE[robot_name]['left']['close_pinch']
            right_fully_open = DEFAULT_HAND_POSE[robot_name]['right']['open_pinch']
            right_fully_closed = DEFAULT_HAND_POSE[robot_name]['right']['close_pinch']

            ratio_open = 0.8
            ratio_closed = 0.0
            left_open =  left_fully_open * ratio_open + (1 - ratio_open) * left_fully_closed
            left_closed = left_fully_open * ratio_closed + (1 - ratio_closed) * left_fully_closed
            right_open = right_fully_open * ratio_open + (1 - ratio_open) * right_fully_closed
            right_closed = right_fully_open * ratio_closed + (1 - ratio_closed) * right_fully_closed

        # Interpolate between open and closed poses
        left_pose = left_open + (left_closed - left_open) * self.hand_left_position
        right_pose = right_open + (right_closed - right_open) * self.hand_right_position

        return left_pose, right_pose
    
    def apply_smooth(self, mimic_obs):
        """Apply sliding window smoothing to mimic observations"""
        if not self.enable_smooth or mimic_obs is None:
            return mimic_obs
            
        # Convert to numpy array if needed
        obs_array = np.array(mimic_obs) if not isinstance(mimic_obs, np.ndarray) else mimic_obs.copy()
        
        # Add current observation to history
        self.smooth_history.append(obs_array)
        
        # Keep only the recent window_size observations
        if len(self.smooth_history) > self.smooth_window_size:
            self.smooth_history.pop(0)
            
        # Apply sliding window average
        if len(self.smooth_history) >= 2:  # Need at least 2 observations for smoothing
            # Stack all observations in history
            history_stack = np.stack(self.smooth_history, axis=0)  # Shape: (history_len, obs_dim)
            # Compute mean across the time dimension
            smoothed_obs = np.mean(history_stack, axis=0)
            return smoothed_obs
        else:
            # Not enough history, return original observation
            return obs_array
    
    def reset_smooth_history(self):
        """Reset smooth history (call when transitioning states)"""
        self.smooth_history = []
        if self.use_finger_tracking:
            self.finger_tracker.reset()
    
    def _emergency_stop(self):
        """Emergency stop: kill sim2real.sh process (server_low_level_g1_real_future.py)"""
        try:
            print("[EMERGENCY STOP] Killing sim2real.sh process...")
            # Kill sim2real.sh which contains server_low_level_g1_real_future.py
            result = subprocess.run(['pkill', '-f', 'sim2real.sh'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed sim2real.sh process")
            else:
                print(f"[EMERGENCY STOP] pkill returned code {result.returncode}")

            # Also try to kill the specific server script directly as backup
            result2 = subprocess.run(['pkill', '-f', 'server_low_level_g1_real_future.py'], 
                                   capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed server_low_level_g1_real_future.py process")
            else:
                print(f"[EMERGENCY STOP] pkill for server script returned code {result2.returncode}")
                
        except subprocess.TimeoutExpired:
            print("[EMERGENCY STOP] pkill command timed out")
        except Exception as e:
            print(f"[EMERGENCY STOP] Error executing pkill: {e}")

class XRobotTeleopToRobot:
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        self.hand_type = getattr(args, 'hand_type', 'dex3')
        self.hand_output_mode = getattr(args, 'hand_output_mode', 'dex3')
        self.use_finger_tracking = getattr(args, 'finger_tracking', False)
        self.require_fresh_xr_tracking = getattr(
            args, 'require_fresh_xr_tracking', False
        )
        source_timeout = getattr(args, 'xr_source_timeout', 0.5)
        self.xr_frame_watchdog = SourceTimestampWatchdog(source_timeout)
        self.body_frame_watchdog = SourceTimestampWatchdog(source_timeout)
        self.latest_retarget_obs = None
        self.last_xr_warning_time = 0.0
        # Select hand pose configuration based on hand type
        if self.hand_type == 'inspire':
            self.hand_pose_key = "unitree_g1_inspire"
        else:
            self.hand_pose_key = args.robot

        print(f"Hand type: {self.hand_type}")
        print(f"Hand output mode: {self.hand_output_mode}")
        print(f"Hand pose config: {self.hand_pose_key}")
        print(f"Pinch mode: {self.args.pinch_mode}")
        print(f"Finger tracking: {self.use_finger_tracking}")
        if self.require_fresh_xr_tracking:
            print("Fresh XR source timestamps are required")
        # Initialize state tracking
        self.last_qpos = None
        self.last_time = time.time()
        self.target_fps = args.target_fps
        self.measured_dt = 1/ self.target_fps # default fallback dt

        # Initialize components
        self.teleop_data_streamer = None
        self.redis_client = None
        self.retarget = None
        self.model = None
        self.data = None
        self.state_machine = StateMachine(
            enable_smooth=args.smooth,
            smooth_window_size=args.smooth_window_size,
            use_pinch=args.pinch_mode,
            use_finger_tracking=self.use_finger_tracking,
            hand_type=self.hand_type,
            grip_thumb=args.grip_thumb,
            fine_span=getattr(args, 'fine_span', 250.0),
            thumb_rate=getattr(args, 'thumb_rate', 500.0),
        )
        self.rate = None
        
        # Video recording
        self.video_writer = None
        self.renderer = None
        
        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Teleop Loop"
        )
        
        # Keyboard control thread (for finger tracking mode)
        self._keyboard_thread = None
        self._keyboard_command = None


    def setup_teleop_data_streamer(self):
        """Initialize and start the teleop data streamer"""
        self.teleop_data_streamer = XRobotStreamer()
        print("Teleop data streamer initialized")
        
    def setup_redis_connection(self):
        """Setup Redis connection"""
        redis_ip = self.args.redis_ip
        self.redis_client = redis.Redis(host=redis_ip, port=6379, db=0)
        self.redis_pipeline = self.redis_client.pipeline()
        self.redis_client.ping()
        print("Redis connected successfully")

    def setup_retargeting_system(self):
        """Initialize the motion retargeting system"""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("Retargeting system initialized")
    
    def setup_mujoco_simulation(self):
        """Setup MuJoCo model and data"""
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        print("MuJoCo simulation initialized")
        
    def setup_video_recording(self):
        """Setup video recording if requested"""
        if not self.args.record_video:
            return
            
        self.video_writer = cv2.VideoWriter(
            'output.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, 
            (640, 480)
        )
        width, height = 640, 480
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        print("Video recording setup completed")
        
    def setup_rate_limiter(self):
        """Setup rate limiter for consistent FPS"""
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"Rate limiter setup for {self.target_fps} FPS")
        
    def get_teleop_data(self):
        """Get current teleop data from streamer"""
        if self.teleop_data_streamer is not None:
            return self.teleop_data_streamer.get_current_frame()
        return None, None, None, None, None

    @staticmethod
    def _controller_source_timestamp(controller_data):
        if not isinstance(controller_data, dict):
            return None
        return controller_data.get('timestamp')

    @staticmethod
    def _body_source_timestamp():
        """Read the body packet timestamp from the already initialized SDK."""
        try:
            import xrobotoolkit_sdk as xrt

            if not xrt.is_body_data_available():
                return None
            timestamp = int(xrt.get_body_timestamp_ns())
            return timestamp if timestamp > 0 else None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def observe_xr_sources(self, smplx_data, controller_data):
        """Return liveness for the overall XR packet and body substream."""
        source_timestamp = self._controller_source_timestamp(controller_data)
        hand_advanced, _ = self.xr_frame_watchdog.observe(source_timestamp)

        body_timestamp = self._body_source_timestamp()
        if body_timestamp is None:
            # Older bindings may not expose a usable body timestamp. The
            # top-level XR packet timestamp is still better than accepting a
            # cached frame while the headset is disconnected or Send is off.
            body_timestamp = source_timestamp
        body_advanced, body_fresh = self.body_frame_watchdog.observe(body_timestamp)
        body_fresh = body_fresh and smplx_data is not None

        if self.require_fresh_xr_tracking and not body_fresh:
            now = time.monotonic()
            if now - self.last_xr_warning_time >= 2.0:
                print(
                    "[bold yellow]WARNING: XR body timestamp is not advancing; "
                    "ignoring cached body poses. In XRoboToolkit select Full "
                    "Body, turn Send on, and turn 'Switch w/ A Button' off.[/bold yellow]"
                )
                self.last_xr_warning_time = now

        return hand_advanced, body_advanced, body_fresh, body_timestamp
        
    def process_retargeting(self, smplx_data):
        """Process motion retargeting and return observations"""
        if smplx_data is None or self.retarget is None:
            return None, None
            
        # Measure dt between retarget calls
        current_time = time.time()
        self.measured_dt = current_time - self.last_time
        self.last_time = current_time
        
        # Retarget till convergence
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        
        # Create mimic obs from retargeting
        if self.last_qpos is not None:
            current_retarget_obs = extract_mimic_obs_whole_body(qpos, self.last_qpos, dt=self.measured_dt)
        else:
            current_retarget_obs = DEFAULT_MIMIC_OBS[self.robot_name]
        
        self.last_qpos = qpos.copy()
        return qpos, current_retarget_obs
        
    def update_visualization(self, qpos, smplx_data, viewer):
        """Update MuJoCo visualization"""
        if qpos is None:
            return
            
        # Clean custom geometry
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            
        # Draw the task targets for reference
        if smplx_data is not None and self.retarget is not None:
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )
                
        # Update the simulation
        if qpos is not None:
            self.data.qpos[:] = qpos.copy()
            mj.mj_forward(self.model, self.data)
            
            # Camera follow the pelvis
            self._update_camera_position(viewer)
        
    def _update_camera_position(self, viewer):
        """Update camera to follow the robot"""
        FOLLOW_CAMERA = True
        if FOLLOW_CAMERA:
            robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            viewer.cam.lookat = robot_base_pos
            viewer.cam.distance = 3.0
            
    def handle_state_transitions(self, current_retarget_obs):
        """Handle state machine transitions and interpolations"""
        if not self.state_machine.has_state_changed():
            return
            
        current_state = self.state_machine.get_current_state()
        previous_state = self.state_machine.previous_state
        
        print(f"State changed: {previous_state} -> {current_state}")
        
        if current_state == "teleop":
            self._handle_enter_teleop(previous_state, current_retarget_obs)
        elif current_state == "pause":
            self._handle_enter_pause()
            
    def _handle_enter_teleop(self, previous_state, current_retarget_obs):
        """Handle entering teleop state"""
        if previous_state in ["idle", "pause"]:
            self.state_machine.reset_smooth_history()
            print("Reset smooth history on entering teleop")

        if previous_state == "idle":
            if current_retarget_obs is not None:
                default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
                start_interpolation(self.state_machine, default_obs, current_retarget_obs[:35])
                print("Interpolating from default to teleop...")
        elif previous_state == "pause":
            if (current_retarget_obs is not None and
                self.state_machine.last_mimic_obs is not None):
                last_obs_35d = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
                start_interpolation(self.state_machine, last_obs_35d, current_retarget_obs[:35])
                print("Interpolating from pause to teleop...")
    def _handle_enter_pause(self):
        """Handle entering pause state"""
        if self.state_machine.current_mimic_obs is not None:
            self.state_machine.set_last_mimic_obs(self.state_machine.current_mimic_obs)
            print("Entered pause mode, storing last obs")
        if self.state_machine.current_neck_data is not None:
            self.state_machine.set_last_neck_data(self.state_machine.current_neck_data)
            print("Entered pause mode, storing last neck data")
            
    def determine_mimic_obs_to_send(self, current_retarget_obs):
        """Determine which mimic observation to send based on current state"""
        current_state = self.state_machine.get_current_state()

        if current_state == "idle":
            obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "pause":
            if self.state_machine.last_mimic_obs is not None:
                obs = self.state_machine.last_mimic_obs[:35] if len(self.state_machine.last_mimic_obs) > 35 else self.state_machine.last_mimic_obs
            else:
                obs = DEFAULT_MIMIC_OBS[self.robot_name]
        elif current_state == "teleop":
            obs = self._get_teleop_mimic_obs(current_retarget_obs)
            obs = self.state_machine.apply_smooth(obs)
        else:
            obs = DEFAULT_MIMIC_OBS[self.robot_name]

        return obs
        
    def _get_teleop_mimic_obs(self, current_retarget_obs):
        """Get mimic obs for teleop state, handling interpolation"""
        if self.state_machine.is_interpolating:
            interp_obs = get_interpolated_obs(self.state_machine)
            if interp_obs is not None:
                self.state_machine.set_current_mimic_obs(interp_obs)
                return interp_obs
            return DEFAULT_MIMIC_OBS[self.robot_name]

        if current_retarget_obs is not None:
            obs_35d = current_retarget_obs[:35] if len(current_retarget_obs) > 35 else current_retarget_obs
            self.state_machine.set_current_mimic_obs(obs_35d)
            return obs_35d

        return DEFAULT_MIMIC_OBS[self.robot_name]
    
    def determine_neck_data_to_send(self, smplx_data):
        """Determine which neck data to send based on current state"""
       
        current_state = self.state_machine.get_current_state()
        
        # In non-teleop states, send default neck position [0, 0]
        if current_state in ["idle"]:
            return [0.0, 0.0]
        
        if current_state == "pause":
            # return [0.0, 0.0]
            # use last neck data
            if self.state_machine.last_neck_data is not None:
                return self.state_machine.last_neck_data
            else:
                return [0.0, 0.0]
            
        # In teleop state, extract neck data from smplx_data
        elif current_state == "teleop" and smplx_data is not None:
            scale = self.args.neck_retarget_scale
            neck_yaw, neck_pitch = human_head_to_robot_neck(smplx_data)
            return [neck_yaw * scale, neck_pitch * scale]
        
        # Default fallback
        return [0.0, 0.0]
            
    def send_to_redis(self, mimic_obs, neck_data=None,
                      left_hand_data=None, right_hand_data=None,
                      hand_source_advanced=True,
                      body_source_advanced=False,
                      body_source_timestamp=None):
        """Send mimic observations to Redis"""
        
        if self.redis_client is not None and mimic_obs is not None:
            # Expect 35D mimic observations
            assert len(mimic_obs) == 35, f"Expected 35 mimic obs dims, got {len(mimic_obs)}"
            # Send to both keys for compatibility
            self.redis_pipeline.set("action_body_unitree_g1_with_hands", json.dumps(mimic_obs.tolist()))
        
        # Send hand action to redis
        if self.redis_client is not None and self.hand_output_mode == "dex3":
            hand_left_pose, hand_right_pose = self.state_machine.get_hand_pose(self.hand_pose_key)
            self.redis_pipeline.set("action_hand_left_unitree_g1_with_hands", json.dumps(hand_left_pose.tolist()))
            self.redis_pipeline.set("action_hand_right_unitree_g1_with_hands", json.dumps(hand_right_pose.tolist()))
        elif self.redis_client is not None and self.hand_output_mode == "wuji":
            now = time.time()
            left_kp = pico_hand_to_mediapipe(left_hand_data, "Left")
            right_kp = pico_hand_to_mediapipe(right_hand_data, "Right")
            allow_hand_publish = (
                hand_source_advanced
                or not getattr(self, 'require_fresh_xr_tracking', False)
            )
            if left_kp is not None and allow_hand_publish:
                self.redis_pipeline.set("pico_hand_left_mediapipe", json.dumps(left_kp.tolist()))
                self.redis_pipeline.set("pico_hand_left_timestamp", str(now))
            if right_kp is not None and allow_hand_publish:
                self.redis_pipeline.set("pico_hand_right_mediapipe", json.dumps(right_kp.tolist()))
                self.redis_pipeline.set("pico_hand_right_timestamp", str(now))
            if body_source_advanced:
                self.redis_pipeline.set("pico_body_timestamp", str(now))
                if body_source_timestamp is not None:
                    self.redis_pipeline.set(
                        "pico_body_source_timestamp_ns",
                        str(int(body_source_timestamp)),
                    )
        
        # Send neck data to redis
        if neck_data is not None:
            self.redis_pipeline.set("action_neck_unitree_g1_with_hands", json.dumps(neck_data))
        
        # Send timestamp to redis
        t_action = int(time.time() * 1000) # current timestamp in ms
        self.redis_pipeline.set("t_action", t_action)

        # execute the pipeline once
        self.redis_pipeline.execute()

    
    def send_controller_data_to_redis(self, controller_data):
        """Send controller data to Redis"""
        if self.redis_client is not None and controller_data is not None:
            self.redis_client.set(
                f"controller_data", 
                json.dumps(controller_data)
            )
            
            
    def record_video_frame(self, viewer):
        """Record current frame to video if recording is enabled"""
        if not self.args.record_video or self.renderer is None:
            return
            
        self.renderer.update_scene(self.data, camera=viewer.cam)
        pixels = self.renderer.render()
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
        
    def handle_exit_sequence(self, viewer):
        """Handle graceful exit with interpolation to default pose"""
        if self.state_machine.current_mimic_obs is not None:
            default_obs = DEFAULT_MIMIC_OBS[self.robot_name]
            current_obs = self.state_machine.current_mimic_obs[:35] if len(self.state_machine.current_mimic_obs) > 35 else self.state_machine.current_mimic_obs
            start_interpolation(self.state_machine, current_obs, default_obs)
            print("Interpolating to default pose before exit...")
            
            # Wait for interpolation to complete
            while self.state_machine.is_interpolating:
                interp_obs = get_interpolated_obs(self.state_machine)
                if interp_obs is not None:
                    # During exit sequence, send default neck position [0, 0]
                    neck_data_to_send = self.determine_neck_data_to_send(None)
                    self.send_to_redis(interp_obs, neck_data_to_send)
                viewer.sync()
                self.rate.sleep()
                


    def _start_keyboard_control_thread(self):
        """Start a daemon thread that reads keyboard input for state machine control.
        
        Used when finger tracking is enabled and a separate operator
        controls start/pause/exit via keyboard.
        
        Keyboard commands:
          s / Enter : Cycle state (idle -> teleop -> pause -> teleop ...)
          q         : Exit program
          e         : Emergency stop
        """
        import threading
        
        def keyboard_loop():
            print("\n" + "=" * 50)
            print("KEYBOARD CONTROL ACTIVE (finger tracking mode)")
            print("  [s] or [Enter] : Start / Pause / Resume teleop")
            print("  [q]            : Exit program")
            print("  [e]            : Emergency stop")
            print("=" * 50 + "\n")
            
            while True:
                try:
                    cmd = input().strip().lower()
                    if cmd in ('s', ''):
                        self._keyboard_command = 'cycle'
                    elif cmd == 'q':
                        self._keyboard_command = 'exit'
                    elif cmd == 'e':
                        self._keyboard_command = 'emergency'
                except EOFError:
                    break
        
        self._keyboard_thread = threading.Thread(target=keyboard_loop, daemon=True)
        self._keyboard_thread.start()
    
    def _process_keyboard_commands(self):
        """Process any pending keyboard commands for state machine control."""
        cmd = self._keyboard_command
        if cmd is None:
            return
        
        self._keyboard_command = None
        
        if cmd == 'cycle':
            # Mimic right_key_one press: idle -> teleop -> pause -> teleop
            self.state_machine.previous_state = self.state_machine.state
            if self.state_machine.state == "idle":
                self.state_machine.state = "teleop"
                print("[Keyboard] State: idle -> teleop")
            elif self.state_machine.state == "teleop":
                self.state_machine.state = "pause"
                print("[Keyboard] State: teleop -> pause")
            elif self.state_machine.state == "pause":
                self.state_machine.state = "teleop"
                print("[Keyboard] State: pause -> teleop")
        elif cmd == 'exit':
            self.state_machine.previous_state = self.state_machine.state
            self.state_machine.state = "exit"
            print("[Keyboard] Exit requested")
        elif cmd == 'emergency':
            self.state_machine._emergency_stop()

    def initialize_all_systems(self):
        """Initialize all required systems"""
        print("Initializing teleop systems...")
        self.setup_teleop_data_streamer()
        self.setup_redis_connection()
        self.setup_retargeting_system()
        self.setup_mujoco_simulation()
        self.setup_video_recording()
        self.setup_rate_limiter()

        print("Teleop state machine initialized. Controls:")
        if self.use_finger_tracking:
            print("- FINGER TRACKING MODE: Hand poses from Pico hand tracking")
            print("- Keyboard control by separate operator (see terminal prompts)")
        else:
            print("- Right controller key_one (A): Cycle through idle -> teleop -> pause -> teleop...")
            print("- Right controller key_two (B): HOLD ~1s to exit program")
            print("- Left controller key_one (X): Toggle fine (precision) mode")
            print("- Left controller axis_click: Emergency stop - kills sim2real.sh process")
        print("- Left controller axis: Control root xy velocity")
        print("- Right controller axis: Control yaw velocity")
        print("- Publishes 35-dimensional mimic observations")
        print(f"Starting in state: {self.state_machine.get_current_state()}")

        if self.state_machine.enable_smooth:
            print(f"- Smooth filtering: ENABLED (window size: {self.state_machine.smooth_window_size} frames)")
        else:
            print("- Smooth filtering: DISABLED")
        
        if self.fps_monitor.enable_detailed_stats:
            print(f"- FPS measurement: ENABLED (detailed stats every {self.fps_monitor.detailed_print_interval} steps)")
        else:
            print(f"- FPS measurement: Quick stats only (every {self.fps_monitor.quick_print_interval} steps)")

        # Start keyboard control thread if using finger tracking
        if self.use_finger_tracking:
            self._start_keyboard_control_thread()

        print("Ready to receive teleop data.")

    def run(self):
        """Main execution loop"""
        self.initialize_all_systems()
        
        # Start the viewer
        with mjv.launch_passive(
            model=self.model, 
            data=self.data, 
            show_left_ui=False, 
            show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1
            
            while viewer.is_running():
                # Get current teleop data
                smplx_data, left_hand_data, right_hand_data, controller_data, headset_data = self.get_teleop_data()
                hand_source_advanced = True
                body_source_advanced = False
                body_source_timestamp = None
                if self.hand_output_mode == "wuji" or self.require_fresh_xr_tracking:
                    (hand_source_advanced, body_source_advanced,
                     _body_tracking_fresh, body_source_timestamp) = self.observe_xr_sources(
                        smplx_data, controller_data
                    )
                
                # Process keyboard commands (finger tracking mode)
                if self.use_finger_tracking:
                    self._process_keyboard_commands()
                
                # Update state machine from controller
                if controller_data is not None:
                    self.state_machine.update(controller_data)
                    self.send_controller_data_to_redis(controller_data)
                
                # Update hand poses from finger tracking
                if (self.hand_output_mode == "dex3" and self.use_finger_tracking
                        and self.state_machine.is_teleop_active()):
                    self.state_machine.update_hand_from_tracking(left_hand_data, right_hand_data)
                
                # Check if we should exit
                if self.state_machine.should_exit():
                    print("Exit requested via controller")
                    self.handle_exit_sequence(viewer)
                    break
                
                # Process retargeting if we have data
                qpos, current_retarget_obs = None, None
                should_process_body = smplx_data is not None and (
                    not self.require_fresh_xr_tracking or body_source_advanced
                )
                if should_process_body:
                    qpos, current_retarget_obs = self.process_retargeting(smplx_data)
                    self.latest_retarget_obs = current_retarget_obs
                    self.update_visualization(qpos, smplx_data, viewer)
                elif self.require_fresh_xr_tracking:
                    # Reuse the last genuinely received target between PICO
                    # frames. Never retarget the SDK's cached startup frame.
                    current_retarget_obs = self.latest_retarget_obs
                
                # Handle state transitions
                self.handle_state_transitions(current_retarget_obs)
                
                # Determine and send mimic observations
                mimic_obs_to_send = self.determine_mimic_obs_to_send(current_retarget_obs)
                neck_data_to_send = self.determine_neck_data_to_send(smplx_data)
                
                # Store current neck data in state machine for pause state handling
                if neck_data_to_send is not None:
                    self.state_machine.set_current_neck_data(neck_data_to_send)
                
                self.send_to_redis(
                    mimic_obs_to_send,
                    neck_data_to_send,
                    left_hand_data,
                    right_hand_data,
                    hand_source_advanced=hand_source_advanced,
                    body_source_advanced=body_source_advanced,
                    body_source_timestamp=body_source_timestamp,
                )
                
                # Update visualization and record video
                viewer.sync()
                self.record_video_frame(viewer)
                
                # FPS monitoring
                self.fps_monitor.tick()
                
                self.rate.sleep()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Whether to record the video.",
    )
    parser.add_argument(
        "--hand_type",
        type=str,
        default="dex3",
        choices=["dex3", "inspire"],
        help="Type of dextrous hand (dex3 or inspire).",
    )
    parser.add_argument(
        "--hand-output-mode",
        default="dex3",
        choices=["dex3", "wuji"],
        help=("Hand Redis output. 'dex3' preserves the existing Dex3/Inspire "
              "behavior selected by --hand_type; 'wuji' publishes PICO "
              "MediaPipe landmarks on separate keys."),
    )
    parser.add_argument(
        "--pinch_mode",
        action="store_true",
        help="Whether to use pinch mode for hand control.",
        default=False,
    )
    parser.add_argument(
        "--redis_ip",
        type=str,
        default="localhost",
        help="Redis IP",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height for retargeting.",
    )   
    parser.add_argument(
        "--neck_retarget_scale",
        type=float,
        default=1.5,
        help="Scale factor for neck data.",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth filtering for mimic observations in teleop mode.",
    )
    parser.add_argument(
        "--smooth_window_size",
        type=int,
        default=5,
        help="Window size for sliding window smoothing (default: 5 frames).",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Target FPS for the teleop system.",
    )
    parser.add_argument(
        "--measure_fps",
        type=int,
        default=0,
        help="Measure and print detailed FPS statistics (0=disabled, 1=enabled).",
    )
    parser.add_argument(
        "--finger_tracking",
        action="store_true",
        help="Use Pico 4 Ultra finger tracking for Inspire hand control instead of controller buttons.",
        default=False,
    )
    parser.add_argument(
        "--require-fresh-xr-tracking",
        action="store_true",
        help=("Reject cached XR body/hand frames until their source timestamp "
              "advances. Intended for the Wuji launch path; disabled by "
              "default to preserve existing teleop behavior."),
    )
    parser.add_argument(
        "--xr-source-timeout",
        type=float,
        default=0.5,
        help="Seconds an advancing XR source timestamp remains fresh (default: 0.5).",
    )
    parser.add_argument(
        "--fine_span",
        type=float,
        default=250.0,
        help="Inspire fine mode: counts of finger travel per full grip/trigger sweep "
             "(1000 = same as coarse). Lower is finer; below ~100 you hit the hand's "
             "own mechanical resolution. Toggled in-headset with left X.",
    )
    parser.add_argument(
        "--thumb_rate",
        type=float,
        default=500.0,
        help="Inspire thumb rate control: counts/sec at full stick deflection. "
             "Scaled by fine_span/1000 while fine mode is active.",
    )
    parser.add_argument(
        "--grip_thumb",
        action="store_true",
        help="Also use grip button for thumb rotation (combined with joystick). Note: for Inspire hands the grip already drives pinky/ring/middle, so this couples them to the thumb. Default: joystick-only thumb control.",
        default=False,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    teleop_robot = XRobotTeleopToRobot(args)
    teleop_robot.run()