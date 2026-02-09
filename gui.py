#!/usr/bin/env python3
"""
FAR-TWIST Teleop Control Center GUI with Multiple Theme Options
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import queue
import time
import os
import signal

class ThemeManager:
    """Theme management for different visual styles"""
    
    THEMES = {
        "Dark Blue": {"mode": "dark", "theme": "dark-blue"},
        "Blue": {"mode": "dark", "theme": "blue"}, 
        "Green": {"mode": "dark", "theme": "green"},
        "Light Blue": {"mode": "light", "theme": "blue"},
        "Light Green": {"mode": "light", "theme": "green"},
        "Cyberpunk": {"mode": "dark", "theme": "blue", "custom": True},
        "Neon": {"mode": "dark", "theme": "green", "custom": True},
        "Professional": {"mode": "light", "theme": "blue", "custom": True},
        "EVA Unit-01": {"mode": "dark", "theme": "green", "custom": True},
        "EVA Unit-02": {"mode": "dark", "theme": "blue", "custom": True},
        "EVA Unit-00": {"mode": "light", "theme": "blue", "custom": True},
        "NERV": {"mode": "dark", "theme": "blue", "custom": True},
    }
    
    @staticmethod
    def apply_theme(theme_name):
        """Apply selected theme"""
        theme_config = ThemeManager.THEMES.get(theme_name, ThemeManager.THEMES["Dark Blue"])
        
        ctk.set_appearance_mode(theme_config["mode"])
        ctk.set_default_color_theme(theme_config["theme"])
        
        return ThemeManager.get_custom_colors(theme_name)
    
    @staticmethod
    def get_custom_colors(theme_name):
        """Get custom color configuration"""
        color_schemes = {
            "Dark Blue": {
                "primary": "#1f538d",
                "success": "#4CAF50", 
                "danger": "#f44336",
                "warning": "#ff9800",
                "accent": "#81C784",
                "emergency": "#ff1744"
            },
            "Blue": {
                "primary": "#2196F3",
                "success": "#4CAF50",
                "danger": "#f44336", 
                "warning": "#ff9800",
                "accent": "#64B5F6",
                "emergency": "#ff1744"
            },
            "Green": {
                "primary": "#4CAF50",
                "success": "#8BC34A",
                "danger": "#f44336",
                "warning": "#ff9800", 
                "accent": "#81C784",
                "emergency": "#ff1744"
            },
            "Cyberpunk": {
                "primary": "#00ffff",
                "success": "#00ff41",
                "danger": "#ff0080",
                "warning": "#ffff00",
                "accent": "#ff6b00",
                "emergency": "#ff0040"
            },
            "Neon": {
                "primary": "#39ff14",
                "success": "#00ff00", 
                "danger": "#ff073a",
                "warning": "#ffff00",
                "accent": "#ff6600",
                "emergency": "#ff0066"
            },
            "Professional": {
                "primary": "#1976D2",
                "success": "#388E3C",
                "danger": "#D32F2F", 
                "warning": "#F57C00",
                "accent": "#7B1FA2",
                "emergency": "#C62828"
            },
            "EVA Unit-01": {
                "primary": "#4A148C",      # Deep purple (EVA-01 main color)
                "success": "#00E676",      # Green (EVA-01 green)
                "danger": "#FF1744",       # Red (warning color)
                "warning": "#FF6D00",      # Orange (AT field)
                "accent": "#E1BEE7",       # Light purple (accent)
                "emergency": "#B71C1C"     # Deep red (emergency)
            },
            "EVA Unit-02": {
                "primary": "#D32F2F",      # Red (EVA-02 main color)
                "success": "#FF5722",      # Orange-red (startup)
                "danger": "#B71C1C",       # Deep red (danger)
                "warning": "#FF9800",      # Orange (warning)
                "accent": "#FFCDD2",       # Light red (accent)
                "emergency": "#4A148C"     # Purple (emergency)
            },
            "EVA Unit-00": {
                "primary": "#1565C0",      # Blue (EVA-00 main color)
                "success": "#00BCD4",      # Cyan (system normal)
                "danger": "#F44336",       # Red (error)
                "warning": "#FFC107",      # Yellow (warning)
                "accent": "#BBDEFB",       # Light blue (accent)
                "emergency": "#FF1744"     # Red (emergency)
            },
            "NERV": {
                "primary": "#000000",      # Black (NERV main color)
                "success": "#4CAF50",      # Green (system normal)
                "danger": "#FF0000",       # Pure red (danger)
                "warning": "#FFFF00",      # Pure yellow (warning)
                "accent": "#FFFFFF",       # White (text)
                "emergency": "#FF0000"     # Red (emergency)
            }
        }
        
        return color_schemes.get(theme_name, color_schemes["Dark Blue"])

class TerminalPanel:
    """Modern terminal panel with enhanced styling"""
    
    def __init__(self, parent_frame, title: str, command: str, colors: dict, is_remote: bool = False, custom_kill_cmd: str = None):
        self.title = title
        self.command = command
        self.is_remote = is_remote
        self.custom_kill_cmd = custom_kill_cmd
        self.process = None
        self.output_queue = queue.Queue()
        self.is_running = False
        self.colors = colors
        
        # Create panel frame with gradient-like effect
        self.frame = ctk.CTkFrame(parent_frame, corner_radius=15, border_width=2, 
                                 border_color=colors["primary"])
        
        # Header with title and status
        self.header_frame = ctk.CTkFrame(self.frame, fg_color=colors["primary"], corner_radius=10)
        self.header_frame.pack(fill="x", padx=8, pady=(8, 4))
        
        # Title with icon - using safe standard emojis
        icons = {
            "G1 Neck Control": "Target",
            "G1 ZED Teleop": "Camera",
            "G1 ZED Policy": "Policy",
            "Onboard Policy": "Policy",
            "Offline Motion": "Control",
            "Online Teleop": "Remote",
            "Visuomotor Policy Deploy": "Vision",
            "Data Recording": "Record",
            "Sim2Sim Deploy": "Sync",
            "Sim2Real Deploy": "Launch"
        }
        
        icon = icons.get(title, "System")
        # Increased font size from 18 to 22 for better visibility
        self.title_label = ctk.CTkLabel(self.header_frame, text=title,
                                       font=ctk.CTkFont(size=22, weight="bold"),
                                       text_color="white")
        self.title_label.pack(side="left", padx=12, pady=10)
        
        # Animated status indicator - using text instead of emojis
        self.status_label = ctk.CTkLabel(self.header_frame, text="OFFLINE", 
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        text_color="#666666")
        self.status_label.pack(side="right", padx=12, pady=10)
        
        # Control buttons with custom styling
        self.control_frame = ctk.CTkFrame(self.frame, fg_color="transparent")
        self.control_frame.pack(fill="x", padx=8, pady=4)
        
        # Modern buttons with larger size
        self.start_btn = ctk.CTkButton(self.control_frame, text="START", 
                                      command=self.start, width=100, height=40,
                                      fg_color=colors["success"], 
                                      hover_color=self._darken_color(colors["success"]),
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self.start_btn.pack(side="left", padx=3)
        
        self.kill_btn = ctk.CTkButton(self.control_frame, text="KILL", 
                                     command=self.kill, width=100, height=40,
                                     fg_color=colors["danger"],
                                     hover_color=self._darken_color(colors["danger"]),
                                     font=ctk.CTkFont(size=14, weight="bold"))
        self.kill_btn.pack(side="left", padx=3)
        
        self.clear_btn = ctk.CTkButton(self.control_frame, text="CLEAR", 
                                      command=self.clear_output, width=100, height=40,
                                      fg_color=colors["warning"],
                                      hover_color=self._darken_color(colors["warning"]),
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self.clear_btn.pack(side="left", padx=3)
        
        # Command display with styling - increased font size from 10 to 12
        self.cmd_label = ctk.CTkLabel(self.frame, text=f"Command: {command}",
                                     font=ctk.CTkFont(size=12),
                                     text_color=colors["accent"])
        self.cmd_label.pack(fill="x", padx=10, pady=(0, 4))
        
        # Enhanced terminal output - keep terminal font size
        self.output_text = ctk.CTkTextbox(self.frame, height=100,
                                         font=ctk.CTkFont(family="Courier", size=10),
                                         fg_color="#0d1117" if ctk.get_appearance_mode() == "Dark" else "#f6f8fa",
                                         text_color="#c9d1d9" if ctk.get_appearance_mode() == "Dark" else "#24292f",
                                         border_width=1,
                                         border_color=colors["primary"])
        self.output_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        # Start output reader
        threading.Thread(target=self._output_reader, daemon=True).start()
    
    def _darken_color(self, hex_color):
        """Darken color for hover effect"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darkened = tuple(max(0, int(c * 0.8)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*darkened)
    
    def _update_status(self, status: str, color: str):
        """Update status indicator"""
        status_texts = {
            "stopped": "OFFLINE",
            "running": "ONLINE", 
            "error": "ERROR",
            "warning": "STARTING"
        }
        status_text = status_texts.get(status, "OFFLINE")
        self.status_label.configure(text=status_text, text_color=color)
    
    def _log_output(self, text: str):
        """Add output text"""
        self.output_queue.put(text)
    
    def _build_ssh_command(self, remote_command: str) -> list:
        """Build SSH command"""
        cmd = ["ssh"]
        if "sudo" in remote_command:
            cmd.append("-t")
        cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "LogLevel=ERROR", 
            "g1",
            remote_command
        ])
        return cmd
    
    def _output_reader(self):
        """Output reading thread"""
        while True:
            try:
                text = self.output_queue.get(timeout=0.1)
                self.output_text.after_idle(lambda t=text: self._insert_text(t))
            except queue.Empty:
                continue
    
    def _insert_text(self, text: str):
        """Insert text into output box"""
        self.output_text.insert("end", text)
        self.output_text.see("end")
    
    def start(self):
        """Start process"""
        if self.is_running:
            self._log_output("Process already running!\n")
            return
            
        self._log_output(f"Starting: {self.command}\n")
        self._update_status("warning", self.colors["warning"])
        
        if self.is_remote:
            self._start_remote()
        else:
            self._start_local()
    
    def _start_local(self):
        """Start local process"""
        try:
            cwd = os.path.dirname(os.path.abspath(__file__))
            
            self.process = subprocess.Popen(
                self.command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                universal_newlines=True,
                cwd=cwd,
                preexec_fn=os.setsid
            )
            
            self.is_running = True
            self._update_status("running", self.colors["success"])
            
            threading.Thread(target=self._monitor_output, daemon=True).start()
            
        except Exception as e:
            self._log_output(f"Error: {e}\n")
            self._update_status("error", self.colors["danger"])
    
    def _start_remote(self):
        """Start remote process"""
        try:
            self._log_output("Connecting to G1...\n")
            
            ssh_cmd = self._build_ssh_command(f"cd ~ && {self.command}")
            
            self.process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                preexec_fn=os.setsid
            )
            
            self.is_running = True
            self._update_status("running", self.colors["success"])
            self._log_output("Connected to G1\n")
            
            threading.Thread(target=self._monitor_output, daemon=True).start()
            
        except Exception as e:
            self._log_output(f"Connection error: {e}\n")
            self._update_status("error", self.colors["danger"])
    
    def _monitor_output(self):
        """Monitor process output"""
        try:
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self._log_output(line)
            
            if self.process:
                return_code = self.process.poll()
                self._log_output(f"\nProcess finished (code: {return_code})\n")
                
            self.is_running = False
            self._update_status("stopped", "#666666")
            
        except Exception as e:
            self._log_output(f"Monitor error: {e}\n")
            self.is_running = False
            self._update_status("error", self.colors["danger"])
    
    def kill(self):
        """Kill process"""
        if not self.is_running:
            self._log_output("No process running!\n")
            return
            
        self._log_output("Killing process...\n")
        
        try:
            if self.process:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                time.sleep(1)
                if self.process.poll() is None:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            
            if self.custom_kill_cmd:
                self._execute_cleanup_command()
                        
            self.is_running = False
            self._update_status("stopped", "#666666")
            self._log_output("Process killed\n")
            
        except Exception as e:
            self._log_output(f"Kill error: {e}\n")
    
    def _execute_cleanup_command(self):
        """Execute cleanup command"""
        self._log_output(f"Cleanup: {self.custom_kill_cmd}\n")
        
        try:
            if self.is_remote:
                cleanup_cmd = self._build_ssh_command(self.custom_kill_cmd)
            else:
                cleanup_cmd = self.custom_kill_cmd.split()
            
            result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self._log_output("Cleanup successful\n")
            else:
                self._log_output(f"Cleanup failed (code: {result.returncode})\n")
                    
        except Exception as e:
            self._log_output(f"Cleanup error: {e}\n")
    
    def clear_output(self):
        """Clear output"""
        self.output_text.delete("1.0", "end")
        self._log_output("Output cleared\n")


class TeleopControlCenter:
    """Main control interface"""
    
    def __init__(self):
        # self.current_theme = "Dark Blue"
        self.current_theme = "EVA Unit-01"
        # self.current_theme = "NERV"
        self.colors = ThemeManager.apply_theme(self.current_theme)
        
        self.root = ctk.CTk()
        self.root.title("FAR-TWIST Teleop Control Center")
        self.root.geometry("1800x1100")
        
        # Configure grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=2)
        
        self._create_widgets()
        self._test_g1_connection()
    
    def _create_widgets(self):
        """Create interface components"""
        
        # Top bar
        self._create_header()
        
        # Left panel - Remote G1
        self._create_left_panel()
        
        # Right panel - Local servers
        self._create_right_panel()
    
    def _create_header(self):
        """Create header bar"""
        header_frame = ctk.CTkFrame(self.root, fg_color=self.colors["primary"], corner_radius=0)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        
        # Choose title based on theme
        eva_titles = {
            "EVA Unit-01": "MAGI System - EVA Unit-01 Control",
            "EVA Unit-02": "MAGI System - EVA Unit-02 Control", 
            "EVA Unit-00": "MAGI System - EVA Unit-00 Control",
            "NERV": "NERV Command Center - EVA Control System"
        }
        
        title_text = eva_titles.get(self.current_theme, "FAR-TWIST Teleop Control Center")
        
        # Title - increased font size from 28 to 32
        title_label = ctk.CTkLabel(header_frame, text=title_text,
                                  font=ctk.CTkFont(size=32, weight="bold"),
                                  text_color="white")
        title_label.pack(side="left", padx=30, pady=20)
        
        # Theme selector
        theme_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        theme_frame.pack(side="left", padx=50)
        
        # Increased font size from 14 to 16
        theme_label = ctk.CTkLabel(theme_frame, text="Theme:",
                                  font=ctk.CTkFont(size=16, weight="bold"),
                                  text_color="white")
        theme_label.pack(side="left", padx=(0, 10))
        
        self.theme_selector = ctk.CTkOptionMenu(theme_frame,
                                               values=list(ThemeManager.THEMES.keys()),
                                               command=self._change_theme,
                                               width=160, height=35,
                                               font=ctk.CTkFont(size=14, weight="bold"))
        self.theme_selector.set(self.current_theme)
        self.theme_selector.pack(side="left")
        
        # Control buttons frame
        control_buttons_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_buttons_frame.pack(side="right", padx=30, pady=15)
        
        # Disable firewall button
        firewall_btn = ctk.CTkButton(control_buttons_frame, text="🔥 Disable Firewall",
                                    command=self._disable_firewall,
                                    font=ctk.CTkFont(size=14, weight="bold"),
                                    fg_color="#FF9800",
                                    hover_color="#E68900",
                                    width=180, height=45)
        firewall_btn.pack(side="left", padx=(0, 10))
        
        # Emergency stop button - increased font size
        emergency_btn = ctk.CTkButton(control_buttons_frame, text="🚨 EMERGENCY STOP",
                                     command=self._emergency_stop,
                                     font=ctk.CTkFont(size=18, weight="bold"),
                                     fg_color=self.colors["emergency"],
                                     hover_color="#b71c1c",
                                     width=250, height=55)
        emergency_btn.pack(side="left")
    
    def _create_left_panel(self):
        """Create left panel"""
        left_frame = ctk.CTkFrame(self.root, corner_radius=20)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=20)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Title - increased font size from 20 to 24
        left_title = ctk.CTkLabel(left_frame, text="Remote G1 Robot (SSH)",
                                 font=ctk.CTkFont(size=24, weight="bold"))
        left_title.grid(row=0, column=0, padx=20, pady=20)
        
        # Panel container
        panels_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        panels_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        panels_frame.grid_rowconfigure(0, weight=1)
        panels_frame.grid_rowconfigure(1, weight=1)
        panels_frame.grid_rowconfigure(2, weight=1)
        panels_frame.grid_rowconfigure(3, weight=1)
        panels_frame.grid_columnconfigure(0, weight=1)

        # G1 server panels
        self.neck_panel = TerminalPanel(panels_frame, "G1 Neck Control",
                                       "bash ~/g1-onboard/docker_neck.sh",
                                       self.colors, is_remote=True,
                                       custom_kill_cmd="pkill -f neck_teleop.py")
        self.neck_panel.frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        self.zed_panel = TerminalPanel(panels_frame, "G1 ZED Teleop",
                                      "bash ~/g1-onboard/docker_zed.sh",
                                      self.colors, is_remote=True,
                                      custom_kill_cmd="pkill -9 OrinVideoSender")
        self.zed_panel.frame.grid(row=1, column=0, sticky="nsew", pady=(5, 5))

        # New ZED Policy panel
        self.zed_policy_panel = TerminalPanel(panels_frame, "G1 ZED Policy",
                                             "bash ~/g1-onboard/docker_zed_policy.sh",
                                             self.colors, is_remote=True)
        self.zed_policy_panel.frame.grid(row=2, column=0, sticky="nsew", pady=(5, 5))

        # RealSense D435i streamer panel
        self.realsense_panel = TerminalPanel(panels_frame, "G1 RealSense",
                                            "bash ~/g1-onboard/start_realsense.sh",
                                            self.colors, is_remote=True,
                                            custom_kill_cmd="pkill -f realsense_streamer.py")
        self.realsense_panel.frame.grid(row=3, column=0, sticky="nsew", pady=(5, 0))
        
        # Onboard Policy panel
        # self.onboard_policy_panel = TerminalPanel(panels_frame, "Onboard Policy",
        #                                          "bash ~/g1-onboard/sim2real.sh",
        #                                          self.colors, is_remote=True)
        # self.onboard_policy_panel.frame.grid(row=3, column=0, sticky="nsew", pady=(5, 0))
        
        # All control buttons in one row
        buttons_frame = ctk.CTkFrame(panels_frame, fg_color="transparent")
        buttons_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        
        kill_port_btn = ctk.CTkButton(buttons_frame, text="Kill Port",
                                     command=self._execute_kill_port,
                                     font=ctk.CTkFont(size=12, weight="bold"),
                                     fg_color=self.colors["danger"],
                                     hover_color="#b71c1c",
                                     width=100, height=40)
        kill_port_btn.pack(side="left", padx=(0, 5), pady=10)
        
        test_zed_btn = ctk.CTkButton(buttons_frame, text="Test ZED",
                                    command=self._execute_test_zed,
                                    font=ctk.CTkFont(size=12, weight="bold"),
                                    fg_color=self.colors["primary"],
                                    hover_color="#1f4e8b",
                                    width=100, height=40)
        test_zed_btn.pack(side="left", padx=5, pady=10)
        
        g1_startup_btn = ctk.CTkButton(buttons_frame, text="🚀 Start Neck & ZED Teleop",
                                      command=self._start_g1_servers,
                                      font=ctk.CTkFont(size=12, weight="bold"),
                                      fg_color="#FF6B00",
                                      hover_color="#E55A00",
                                      width=180, height=40)
        g1_startup_btn.pack(side="left", padx=(5, 0), pady=10)
        
        # Connection status
        self._create_connection_status(left_frame)
    
    def _create_connection_status(self, parent):
        """Create connection status display"""
        status_frame = ctk.CTkFrame(parent, fg_color=self.colors["primary"], corner_radius=15)
        status_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))
        
        # Increased font size from 16 to 18
        status_title = ctk.CTkLabel(status_frame, text="Connection Status",
                                   font=ctk.CTkFont(size=18, weight="bold"),
                                   text_color="white")
        status_title.pack(side="left", padx=20, pady=15)
        
        # Increased font size from 14 to 16
        self.g1_status_label = ctk.CTkLabel(status_frame, text="G1 OFFLINE",
                                           font=ctk.CTkFont(size=16, weight="bold"),
                                           text_color="white")
        self.g1_status_label.pack(side="left", padx=20)
        
        test_btn = ctk.CTkButton(status_frame, text="Test SSH",
                                command=self._test_g1_connection,
                                width=130, height=40,
                                fg_color="white", text_color=self.colors["primary"],
                                hover_color="#f0f0f0",
                                font=ctk.CTkFont(size=14, weight="bold"))
        test_btn.pack(side="right", padx=20, pady=10)
    
    def _create_right_panel(self):
        """Create right panel"""
        right_frame = ctk.CTkFrame(self.root, corner_radius=20)
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=20)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_columnconfigure(1, weight=1)
        right_frame.grid_columnconfigure(2, weight=1)
        
        # Title - increased font size from 20 to 24
        right_title = ctk.CTkLabel(right_frame, text="Local Servers",
                                  font=ctk.CTkFont(size=24, weight="bold"))
        right_title.grid(row=0, column=0, columnspan=3, padx=20, pady=20)
        
        # Server panels
        self._create_server_panels(right_frame)
    
    def _create_server_panels(self, parent):
        """Create server panels"""
        # Column titles - increased font size from 18 to 24
        titles = [
            ("Low Level", self.colors["primary"]),
            ("High Level", self.colors["success"]), 
            ("Record", "#9C27B0")
        ]
        
        for i, (title, color) in enumerate(titles):
            title_frame = ctk.CTkFrame(parent, fg_color=color, corner_radius=10)
            title_frame.grid(row=1, column=i, sticky="ew", padx=10, pady=(0, 10))
            
            title_label = ctk.CTkLabel(title_frame, text=title,
                                      font=ctk.CTkFont(size=24, weight="bold"),
                                      text_color="white")
            title_label.pack(pady=15)
        
        # Server container
        servers_frame = ctk.CTkFrame(parent, fg_color="transparent")
        servers_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=20, pady=(0, 20))
        servers_frame.grid_rowconfigure(0, weight=1)
        servers_frame.grid_columnconfigure(0, weight=1)
        servers_frame.grid_columnconfigure(1, weight=1)
        servers_frame.grid_columnconfigure(2, weight=1)
        
        # Low level servers
        low_frame = ctk.CTkFrame(servers_frame, fg_color="transparent")
        low_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        low_frame.grid_rowconfigure(0, weight=1)
        low_frame.grid_rowconfigure(1, weight=1)
        low_frame.grid_columnconfigure(0, weight=1)
        
        self.sim2sim_panel = TerminalPanel(low_frame, "Sim2Sim Deploy",
                                          "bash sim2sim.sh", self.colors)
        self.sim2sim_panel.frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        
        self.sim2real_panel = TerminalPanel(low_frame, "Sim2Real Deploy", 
                                           "bash sim2real.sh", self.colors,
                                           custom_kill_cmd="pkill -f server_low_level_g1_real_future.py")
        self.sim2real_panel.frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        
        # High level servers
        high_frame = ctk.CTkFrame(servers_frame, fg_color="transparent")
        high_frame.grid(row=0, column=1, sticky="nsew", padx=7)
        high_frame.grid_rowconfigure(0, weight=1)
        high_frame.grid_rowconfigure(1, weight=1)
        high_frame.grid_rowconfigure(2, weight=1)
        high_frame.grid_columnconfigure(0, weight=1)
        
        self.motion_panel = TerminalPanel(high_frame, "Offline Motion",
                                         "bash run_motion_server.sh", self.colors)
        self.motion_panel.frame.grid(row=0, column=0, sticky="nsew", pady=(0, 3))
        
        self.teleop_panel = TerminalPanel(high_frame, "Online Teleop",
                                         "bash teleop.sh", self.colors)
        self.teleop_panel.frame.grid(row=1, column=0, sticky="nsew", pady=(3, 3))
        
        self.visuomotor_panel = TerminalPanel(high_frame, "Visuomotor Policy Deploy",
                                             "bash /home/ANT.AMAZON.COM/yanjieze/lab42/src/Improved-3D-Diffusion-Policy/deploy_policy.sh", self.colors)
        self.visuomotor_panel.frame.grid(row=2, column=0, sticky="nsew", pady=(3, 0))
        
        # Record servers
        record_frame = ctk.CTkFrame(servers_frame, fg_color="transparent")
        record_frame.grid(row=0, column=2, sticky="nsew", padx=(7, 0))
        record_frame.grid_rowconfigure(0, weight=1)
        record_frame.grid_rowconfigure(1, weight=1)
        record_frame.grid_columnconfigure(0, weight=1)

        self.record_panel = TerminalPanel(record_frame, "Data Recording",
                                         "bash data_record.sh", self.colors,
                                         custom_kill_cmd="pkill -f server_data_record.py")
        self.record_panel.frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))

        self.record_d435_panel = TerminalPanel(record_frame, "Record (D435)",
                                              "bash data_record_d435.sh", self.colors,
                                              custom_kill_cmd="pkill -f server_data_record_d435.py")
        self.record_d435_panel.frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        
        # One-click local server startup button
        local_startup_frame = ctk.CTkFrame(servers_frame, fg_color="transparent")
        local_startup_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        local_startup_btn = ctk.CTkButton(local_startup_frame, text="🚀 Start Sim2Real Deploy & Teleop & Record",
                                         command=self._start_local_servers,
                                         font=ctk.CTkFont(size=16, weight="bold"),
                                         fg_color="#FF6B00",
                                         hover_color="#E55A00",
                                         width=500, height=50)
        local_startup_btn.pack(pady=15)
        
        # Store all panels
        self.all_panels = [
            self.neck_panel, self.zed_panel, self.zed_policy_panel,
            self.realsense_panel,
            self.motion_panel,
            self.teleop_panel, self.visuomotor_panel, self.record_panel,
            self.record_d435_panel, self.sim2sim_panel,
            self.sim2real_panel
        ]
    
    def _change_theme(self, theme_name):
        """Change theme"""
        self.current_theme = theme_name
        messagebox.showinfo("Theme Change", 
                           f"Theme changed to {theme_name}!\nRestart the application to see the changes.")
    
    def _test_g1_connection(self):
        """Test G1 connection"""
        def test_connection():
            try:
                ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
                          "-o", "LogLevel=ERROR", "g1", "echo 'SSH test successful'"]
                
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
                
                connected = result.returncode == 0
                self.root.after(0, lambda: self._update_g1_status(connected))
                
            except Exception as e:
                self.root.after(0, lambda: self._update_g1_status(False))
        
        threading.Thread(target=test_connection, daemon=True).start()
    
    def _update_g1_status(self, connected: bool):
        """Update G1 status"""
        if connected:
            self.g1_status_label.configure(text="G1 ONLINE", text_color="#4CAF50")
        else:
            self.g1_status_label.configure(text="G1 OFFLINE", text_color="#f44336")
    
    def _execute_kill_port(self):
        """Execute kill_port.sh via SSH"""
        def run_kill_port():
            try:
                ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR", 
                          "-t", "g1", "echo '123' | sudo -S bash ~/g1-onboard/kill_port.sh"]
                
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.root.after(0, lambda: messagebox.showinfo("Kill Port", "kill_port.sh executed successfully!"))
                else:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    self.root.after(0, lambda: messagebox.showerror("Kill Port Error", f"Failed to execute kill_port.sh:\n{error_msg}"))
                    
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: messagebox.showerror("Kill Port Error", "Command timed out"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Kill Port Error", f"Error: {str(e)}"))
        
        threading.Thread(target=run_kill_port, daemon=True).start()
    
    def _execute_test_zed(self):
        """Execute test_zed.sh via SSH"""
        def run_test_zed():
            try:
                ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR", 
                          "g1", "bash ~/g1-onboard/test_zed.sh"]
                
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    output = result.stdout or "Command executed successfully"
                    self.root.after(0, lambda: messagebox.showinfo("Test ZED", f"test_zed.sh executed successfully!\n\nOutput:\n{output}"))
                else:
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    self.root.after(0, lambda: messagebox.showerror("Test ZED Error", f"Failed to execute test_zed.sh:\n{error_msg}"))
                    
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: messagebox.showerror("Test ZED Error", "Command timed out"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Test ZED Error", f"Error: {str(e)}"))
        
        threading.Thread(target=run_test_zed, daemon=True).start()
    
    def _disable_firewall(self):
        """Disable system firewall"""
        # if messagebox.askyesno("Disable Firewall", "This will disable the system firewall using sudo. Continue?"):
        if True:
            def disable_firewall():
                try:
                    # Use echo to pipe password to sudo
                    cmd = 'echo "Zyj20011113*" | sudo -S ufw disable'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                    
                    # Update UI in main thread
                    if result.returncode == 0:
                        self.root.after(0, lambda: messagebox.showinfo("Firewall", "Firewall disabled successfully!"))
                    else:
                        error_msg = result.stderr or result.stdout or "Unknown error"
                        self.root.after(0, lambda: messagebox.showerror("Firewall Error", f"Failed to disable firewall:\n{error_msg}"))
                        
                except subprocess.TimeoutExpired:
                    self.root.after(0, lambda: messagebox.showerror("Firewall Error", "Command timed out"))
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Firewall Error", f"Error: {str(e)}"))
            
            # Run in background thread
            threading.Thread(target=disable_firewall, daemon=True).start()
    
    def _start_g1_servers(self):
        """Start G1 neck and ZED servers"""
        try:
            # Start neck server
            if not self.neck_panel.is_running:
                self.neck_panel.start()
                time.sleep(1)  # Small delay between starts
            
            # Start ZED server  
            if not self.zed_panel.is_running:
                self.zed_panel.start()
                
            messagebox.showinfo("G1 Servers", "Starting G1 Neck and ZED servers...")
            
        except Exception as e:
            messagebox.showerror("G1 Servers Error", f"Failed to start G1 servers: {str(e)}")
    
    def _start_local_servers(self):
        """Start Sim2Real Deploy, Teleop, and Data Record servers"""
        try:
            # Start Sim2Real Deploy server
            if not self.sim2real_panel.is_running:
                self.sim2real_panel.start()
                time.sleep(1)  # Small delay between starts
            
            # Start Teleop server
            if not self.teleop_panel.is_running:
                self.teleop_panel.start()
                time.sleep(1)  # Small delay between starts
            
            # Start Data Record server
            if not self.record_panel.is_running:
                self.record_panel.start()
                
            messagebox.showinfo("Local Servers", "Starting Sim2Real Deploy, Teleop, and Data Record servers...")
            
        except Exception as e:
            messagebox.showerror("Local Servers Error", f"Failed to start local servers: {str(e)}")
    
    def _emergency_stop(self):
        """Emergency stop"""
        if messagebox.askyesno("Emergency Stop", "Kill all running processes?"):
            for panel in self.all_panels:
                if panel.is_running:
                    panel.kill()
    
    def run(self):
        """Run application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = TeleopControlCenter()
    app.run()
