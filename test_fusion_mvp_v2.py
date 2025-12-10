#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP Fusion Test v2: Load Cell + Vision Integration with Event Queue
Sensor Fusion 프로토타입 - Enhanced with Event Queue System

개선사항 (v2):
- Event Queue: 연속 이벤트 처리
- Cancellation Detection: 가져갔다 바로 돌려놓기 감지
- Event Timeout: 오래된 이벤트 자동 삭제
- Start Weight Tracking: 이벤트 시작 무게 기록

전략:
- Top Camera (ID 0): 실시간 hand detection (상시 모니터링)
- Side Camera (ID 2): 무게 변화 감지 → 프레임 저장 → 배치 추론
- Load Cell: 무게 변화 감지 → Event Trigger
- Fusion: Vision CLASS + Weight COUNT
"""

import sys
import os
import time
import yaml
import cv2
import json
import threading
import tkinter as tk
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_loadcell'))

from ultralytics import YOLO
from loadcell_serial import LoadCellSerial
from loadcell_protocol import LoadCellProtocol
from filters.kalman_filter import LoadCellKalmanFilter


class SystemState(Enum):
    """System state machine"""
    IDLE = 0
    EVENT_TRIGGER = 1
    INTERACTION = 2
    VERIFICATION = 3


class OperationMode(Enum):
    """Operation mode for the vending machine"""
    NORMAL = 0      # Customer mode: detect product removal
    RESTOCKING = 1  # Owner mode: add products (zeroing enabled)


@dataclass
class WeightEvent:
    """Weight change event"""
    timestamp: float
    start_weight: float  # Weight when event triggered
    captured_frames: List  # Frames captured from side camera
    event_id: int


class EventQueue:
    """
    Event queue for managing weight change events

    Features:
    - Sequential event processing
    - Timeout management (discard old events)
    - Event statistics
    """

    def __init__(self, timeout: float = 10.0):
        """
        Initialize event queue

        Args:
            timeout: Event timeout in seconds (default: 10s)
        """
        self.queue = deque()
        self.timeout = timeout
        self.next_event_id = 1

        # Statistics
        self.total_events = 0
        self.processed_events = 0
        self.cancelled_events = 0
        self.timeout_events = 0

    def push(self, start_weight: float, captured_frames: List) -> int:
        """
        Push new event to queue

        Args:
            start_weight: Weight when event was triggered
            captured_frames: List of captured frames from side camera

        Returns:
            Event ID
        """
        event = WeightEvent(
            timestamp=time.time(),
            start_weight=start_weight,
            captured_frames=captured_frames.copy() if captured_frames else [],
            event_id=self.next_event_id
        )

        self.queue.append(event)
        self.total_events += 1

        event_id = self.next_event_id
        self.next_event_id += 1

        return event_id

    def pop(self) -> Optional[WeightEvent]:
        """
        Pop oldest event from queue

        Returns:
            WeightEvent or None if queue is empty
        """
        if not self.queue:
            return None

        return self.queue.popleft()

    def peek(self) -> Optional[WeightEvent]:
        """
        Peek at oldest event without removing

        Returns:
            WeightEvent or None if queue is empty
        """
        if not self.queue:
            return None

        return self.queue[0]

    def clear_expired(self):
        """Remove expired events from queue"""
        current_time = time.time()

        while self.queue:
            event = self.queue[0]

            if current_time - event.timestamp > self.timeout:
                # Event expired
                self.queue.popleft()
                self.timeout_events += 1
                print(f"[EventQueue] Event {event.event_id} expired (timeout)")
            else:
                # Events are ordered by time, so we can stop
                break

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.queue) == 0

    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)

    def get_stats(self):
        """Get queue statistics"""
        return {
            'total_events': self.total_events,
            'processed_events': self.processed_events,
            'cancelled_events': self.cancelled_events,
            'timeout_events': self.timeout_events,
            'pending_events': len(self.queue)
        }


class VendingMachineUI:
    """
    Real-time UI for vending machine status display

    States:
    - READY: Idle, waiting for customer
    - PROCESSING: Item being detected
    - RESULT: Display detected item and count
    """
    def __init__(self, fusion_system=None):
        self.fusion_system = fusion_system  # Reference to main system for keyboard commands

        self.root = tk.Tk()
        self.root.title("AI Smart Vending Machine")
        self.root.geometry("900x750")
        self.root.configure(bg='#2c3e50')

        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Status display (large)
        self.status_label = tk.Label(
            main_frame,
            text="READY",
            font=("Arial", 72, "bold"),
            fg="#2ecc71",
            bg="#2c3e50"
        )
        self.status_label.pack(pady=30)

        # Product info display
        self.product_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 36),
            fg="#ecf0f1",
            bg="#2c3e50"
        )
        self.product_label.pack(pady=20)

        # Control panel (keyboard shortcuts) - ABOVE info panel
        control_frame = tk.Frame(main_frame, bg="#34495e", relief=tk.RIDGE, bd=2)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        control_title = tk.Label(
            control_frame,
            text="⌨️  Keyboard Controls",
            font=("Arial", 14, "bold"),
            fg="#ecf0f1",
            bg="#34495e"
        )
        control_title.pack(pady=8)

        # Buttons for keyboard controls
        button_frame = tk.Frame(control_frame, bg="#34495e")
        button_frame.pack(pady=10)

        # R button - Restocking mode
        self.restock_button = tk.Button(
            button_frame,
            text="[R] Restocking",
            font=("Arial", 11, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            width=18,
            height=2,
            command=self.on_restock_pressed,
            cursor="hand2"
        )
        self.restock_button.pack(side=tk.LEFT, padx=8)

        # Z button - Manual Zero
        self.zero_button = tk.Button(
            button_frame,
            text="[Z] Manual Zero",
            font=("Arial", 11, "bold"),
            bg="#e67e22",
            fg="white",
            activebackground="#d35400",
            activeforeground="white",
            width=18,
            height=2,
            command=self.on_zero_pressed,
            cursor="hand2"
        )
        self.zero_button.pack(side=tk.LEFT, padx=8)

        # Q button - Quit
        self.quit_button = tk.Button(
            button_frame,
            text="[Q] Quit",
            font=("Arial", 11, "bold"),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
            activeforeground="white",
            width=18,
            height=2,
            command=self.on_quit_pressed,
            cursor="hand2"
        )
        self.quit_button.pack(side=tk.LEFT, padx=8)

        # Info panel (bottom)
        info_frame = tk.Frame(main_frame, bg="#34495e", relief=tk.RIDGE, bd=2)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Current weight
        self.weight_label = tk.Label(
            info_frame,
            text="Current Weight: 0.0g",
            font=("Arial", 18),
            fg="#bdc3c7",
            bg="#34495e"
        )
        self.weight_label.pack(pady=10)

        # Session info
        self.session_label = tk.Label(
            info_frame,
            text="Total Transactions: 0",
            font=("Arial", 16),
            fg="#95a5a6",
            bg="#34495e"
        )
        self.session_label.pack(pady=5)

        # Mode indicator
        self.mode_label = tk.Label(
            info_frame,
            text="Mode: NORMAL",
            font=("Arial", 16),
            fg="#95a5a6",
            bg="#34495e"
        )
        self.mode_label.pack(pady=5)

        # Bind keyboard events to window
        self.root.bind('<KeyPress-r>', lambda e: self.on_restock_pressed())
        self.root.bind('<KeyPress-R>', lambda e: self.on_restock_pressed())
        self.root.bind('<KeyPress-z>', lambda e: self.on_zero_pressed())
        self.root.bind('<KeyPress-Z>', lambda e: self.on_zero_pressed())
        self.root.bind('<KeyPress-q>', lambda e: self.on_quit_pressed())
        self.root.bind('<KeyPress-Q>', lambda e: self.on_quit_pressed())

        self.running = True

    def update_status(self, status: str):
        """Update main status display"""
        color_map = {
            'READY': '#2ecc71',       # Green
            'PROCESSING': '#f39c12',  # Orange
            'DETECTED': '#3498db',    # Blue
            'ERROR': '#e74c3c',       # Red
            'RESTOCKING': '#9b59b6'   # Purple
        }

        self.status_label.config(
            text=status,
            fg=color_map.get(status, '#ecf0f1')
        )
        self.root.update()

    def update_product(self, product_name: str, count: int, action: str = "REMOVED"):
        """Update product display"""
        if product_name and count:
            text = f"{action}: {product_name} × {abs(count)}"
            self.product_label.config(text=text)
        else:
            self.product_label.config(text="")
        self.root.update()

    def update_weight(self, weight: float):
        """Update current weight display"""
        self.weight_label.config(text=f"Current Weight: {weight:.1f}g")
        self.root.update()

    def update_session_info(self, total_transactions: int):
        """Update session statistics"""
        self.session_label.config(text=f"Total Transactions: {total_transactions}")
        self.root.update()

    def update_mode(self, mode: str):
        """Update operation mode"""
        self.mode_label.config(text=f"Mode: {mode}")
        self.root.update()

    def clear_product(self):
        """Clear product display"""
        self.product_label.config(text="")
        self.root.update()

    def on_restock_pressed(self):
        """Handle R key press - Toggle restocking mode"""
        if self.fusion_system is None:
            print("[UI] Fusion system not connected")
            return

        try:
            from enum import Enum
            # Check current mode and toggle
            if hasattr(self.fusion_system, 'operation_mode'):
                # Assuming OperationMode.NORMAL = 0, RESTOCKING = 1
                if self.fusion_system.operation_mode.value == 0:  # NORMAL
                    print("[UI] R key pressed - Entering RESTOCKING mode")
                    self.fusion_system.enter_restocking_mode()
                else:  # RESTOCKING
                    print("[UI] R key pressed - Exiting RESTOCKING mode")
                    self.fusion_system.exit_restocking_mode()
        except Exception as e:
            print(f"[UI] Error toggling restocking mode: {e}")

    def on_zero_pressed(self):
        """Handle Z key press - Manual zero adjustment"""
        if self.fusion_system is None:
            print("[UI] Fusion system not connected")
            return

        try:
            print("[UI] Z key pressed - Manual zero adjustment")
            self.fusion_system.manual_zero()
        except Exception as e:
            print(f"[UI] Error performing manual zero: {e}")

    def on_quit_pressed(self):
        """Handle Q key press - Quit application"""
        print("[UI] Q key pressed - Quitting application")
        if self.fusion_system is not None:
            self.fusion_system.top_running = False
            self.fusion_system.side_running = False
        self.stop()

    def start(self):
        """Start UI main loop in separate thread"""
        def ui_loop():
            while self.running:
                try:
                    self.root.update()
                    time.sleep(0.05)  # 20 FPS
                except:
                    break

        ui_thread = threading.Thread(target=ui_loop, daemon=True)
        ui_thread.start()

    def stop(self):
        """Stop UI"""
        self.running = False
        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


class FusionMVP_v2:
    """
    MVP Sensor Fusion System v2 with Event Queue

    Components:
    - Top Camera: Hand detection (real-time)
    - Side Camera: Product detection (on-demand)
    - Load Cell: Weight monitoring (continuous)
    - Event Queue: Sequential event processing
    - Fusion Engine: Combine CLASS + COUNT
    """

    def __init__(self, config_path: str = 'config.yaml', label_path: str = '13subset_label.json'):
        """
        Initialize Fusion MVP v2

        Args:
            config_path: Path to configuration file
            label_path: Path to label JSON file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load labels
        with open(label_path, 'r') as f:
            label_data = json.load(f)
            if isinstance(label_data, list):
                self.labels = {item['id']: item['name'] for item in label_data}
            else:
                self.labels = {int(k): v for k, v in label_data.items()}

        print("=" * 70)
        print("MVP Fusion Test v2: Event Queue System")
        print("=" * 70)

        # Configuration
        vision_config = self.config['vision']
        hw_config = self.config['hardware']
        kalman_config = self.config['kalman']
        event_config = self.config['event_detection']
        product_config = self.config['products']
        operation_config = self.config.get('operation', {})
        fusion_config = self.config.get('fusion', {})

        # Vision settings
        self.model_path = vision_config['model_path']
        self.device = vision_config.get('device', '0')
        self.confidence_threshold = vision_config['confidence_threshold']
        self.top_camera_id = hw_config['top_camera_id']
        self.side_camera_id = hw_config['side_camera_id']
        self.lookback_time = vision_config['lookback_time']

        # Camera settings
        self.camera_width = vision_config.get('camera_width', 640)
        self.camera_height = vision_config.get('camera_height', 480)
        self.camera_fps = vision_config.get('camera_fps', 30)

        # Product database (weight information)
        self.product_weights = {}
        for pid, pdata in product_config.items():
            if isinstance(pdata, dict) and pdata.get('is_product', False):
                self.product_weights[int(pid)] = pdata['weight']

        # State
        self.state = SystemState.IDLE

        # Operation mode (NORMAL or RESTOCKING)
        initial_mode = operation_config.get('initial_mode', 'NORMAL')
        self.operation_mode = OperationMode.NORMAL if initial_mode == 'NORMAL' else OperationMode.RESTOCKING

        # Auto drift correction settings
        drift_config = operation_config.get('auto_drift_correction', {})
        self.drift_correction_enabled = drift_config.get('enabled', True)
        self.drift_stability_time = drift_config.get('stability_time', 600.0)  # 10 minutes
        self.drift_threshold = drift_config.get('drift_threshold', 50.0)  # grams
        self.drift_stability_variance = drift_config.get('stability_variance', 5.0)

        # Drift tracking
        self.drift_stable_start = None  # When stability started
        self.drift_stable_weight = 0.0  # Weight during stability period

        # Event queue
        self.event_queue = EventQueue(timeout=10.0)
        self.current_event = None  # Event being processed

        # Vision components
        self.model = None
        self.top_camera = None
        self.side_camera = None
        self.top_running = False
        self.side_running = False

        # Side camera frame buffer (circular buffer for recent frames)
        self.side_buffer = deque(maxlen=int(self.lookback_time * self.camera_fps))
        self.side_buffer_lock = threading.Lock()

        # Load cell components
        self.serial = LoadCellSerial()
        self.is_connected = False
        self.filter = LoadCellKalmanFilter(
            process_noise=kalman_config['process_noise'],
            measurement_noise=kalman_config['measurement_noise'],
            dead_zone=kalman_config.get('dead_zone', 3.0)
        )

        # Event detection
        self.weight_change_threshold = event_config['weight_change_threshold']
        self.settling_time = event_config['settling_time']
        self.variance_threshold = event_config['variance_threshold']

        # Fusion settings
        self.inference_frames = fusion_config.get('inference_frames', 60)  # Default: 60 frames

        # Debug frame saving
        self.save_debug_frames = fusion_config.get('save_debug_frames', False)
        self.debug_frames_path = fusion_config.get('debug_frames_path', 'debug_frames')
        self.save_annotated = fusion_config.get('save_annotated', True)

        # Weight tracking
        self.raw_weight = 0.0
        self.filtered_weight = 0.0
        self.zero_offset = 0.0
        self.last_stable_weight = 0.0
        self.weight_history = []

        # Hand detection tracking
        self.hand_detected = False
        self.hand_confidence = 0.0

        # Transaction logging
        self.log_file = "fusion_mvp_v2_log.txt"
        self.clear_log()

        # UI initialization (pass self reference for keyboard commands)
        self.ui = VendingMachineUI(fusion_system=self)
        self.ui.start()
        self.total_transactions = 0

        print(f"\n[Configuration]")
        print(f"Model Path: {self.model_path}")
        print(f"Device: {self.device}")
        print(f"Top Camera ID: {self.top_camera_id} (Hand detection)")
        print(f"Side Camera ID: {self.side_camera_id} (Product detection)")
        print(f"Inference Frames: {self.inference_frames} frames ({self.inference_frames/self.camera_fps:.1f}s @ {self.camera_fps}fps)")
        print(f"Debug Frame Saving: {'Enabled' if self.save_debug_frames else 'Disabled'}")
        if self.save_debug_frames:
            print(f"  Save Path: {self.debug_frames_path}/")
            print(f"  Save Annotated: {self.save_annotated}")
        print(f"Weight Change Threshold: {self.weight_change_threshold}g")
        print(f"Event Queue Timeout: {self.event_queue.timeout}s")
        print(f"Loaded {len(self.product_weights)} products")
        print("=" * 70)

    def load_model(self):
        """Load YOLOv8 model"""
        print(f"\n[Loading Model]")
        print(f"Path: {self.model_path}")

        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model file not found: {self.model_path}")
            return False

        try:
            self.model = YOLO(self.model_path)
            print("[SUCCESS] Model loaded")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def open_camera(self, camera_id: int):
        """
        Open camera

        Args:
            camera_id: Camera device ID

        Returns:
            cv2.VideoCapture object or None
        """
        print(f"\n[Opening Camera {camera_id}]")

        try:
            cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    print(f"[ERROR] Failed to open camera {camera_id}")
                    return None

            # Set properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

            # Test read
            ret, test_frame = cap.read()
            if not ret:
                print(f"[ERROR] Camera {camera_id} cannot read frames")
                cap.release()
                return None

            print(f"[SUCCESS] Camera {camera_id} opened")
            return cap

        except Exception as e:
            print(f"[ERROR] Failed to open camera: {e}")
            return None

    def connect_loadcell(self, port: str = None, baudrate: int = 115200):
        """
        Connect to load cell

        Args:
            port: Serial port (auto-detect if None)
            baudrate: Baud rate

        Returns:
            True if connected successfully
        """
        print(f"\n[Connecting Load Cell]")

        # List available ports
        ports = LoadCellSerial.list_ports()

        if not ports:
            print("[ERROR] No serial ports found")
            return False

        print("Available Ports:")
        for i, (port_name, desc) in enumerate(ports):
            print(f"  {i + 1}. {port_name} - {desc}")

        # Auto-select
        if port is None:
            port = self.config['hardware'].get('loadcell_port', '/dev/ttyUSB0')
            port_names = [p[0] for p in ports]
            if port not in port_names:
                port = ports[0][0]

        print(f"Using: {port}")

        if self.serial.connect(port, baudrate):
            self.is_connected = True
            print("[SUCCESS] Load cell connected")
            return True
        else:
            print("[ERROR] Failed to connect")
            return False

    def calibrate_zero(self):
        """Calibrate zero point"""
        print("\n[Zero Calibration]")
        print("Make sure load cell is empty...")
        time.sleep(2)

        # Read current weight
        raw = self.read_weight()
        self.zero_offset = raw

        # Reset filter
        kalman_config = self.config['kalman']
        self.filter = LoadCellKalmanFilter(
            process_noise=kalman_config['process_noise'],
            measurement_noise=kalman_config['measurement_noise'],
            dead_zone=kalman_config.get('dead_zone', 3.0),
            initial_state=0.0
        )

        # Reset tracking
        self.weight_history.clear()
        self.last_stable_weight = 0.0

        print(f"Zero offset set: {self.zero_offset:.1f}g")
        self.log_event("ZERO_CALIBRATION", self.zero_offset)

    def enter_restocking_mode(self):
        """
        Enter restocking mode for adding products

        Process:
        1. Switch to RESTOCKING mode
        2. Calibrate zero (tare)
        3. Wait for owner to add products
        4. Call exit_restocking_mode() when done
        """
        print("\n" + "=" * 70)
        print("[RESTOCKING MODE]")
        print("=" * 70)

        self.operation_mode = OperationMode.RESTOCKING

        # Update UI
        self.ui.update_mode("RESTOCKING")
        self.ui.update_status("RESTOCKING")

        # Pause event processing
        print("Pausing event detection...")

        # Calibrate zero
        self.calibrate_zero()

        print("\n[Instructions]")
        print("1. Add products to the load cell")
        print("2. Press 'r' to complete restocking")
        print("=" * 70)

        self.log_event("ENTER_RESTOCKING_MODE", 0.0)

    def exit_restocking_mode(self):
        """
        Exit restocking mode and return to normal operation

        Process:
        1. Set current weight as new baseline
        2. Switch to NORMAL mode
        3. Resume event detection
        """
        print("\n" + "=" * 70)
        print("[EXIT RESTOCKING MODE]")
        print("=" * 70)

        # Update baseline weight
        self.last_stable_weight = self.filtered_weight

        print(f"New baseline weight: {self.last_stable_weight:.1f}g")

        # Switch to normal mode
        self.operation_mode = OperationMode.NORMAL

        # Update UI
        self.ui.update_mode("NORMAL")
        self.ui.update_status("READY")

        # Reset drift tracking
        self.drift_stable_start = None
        self.drift_stable_weight = 0.0

        print("Resuming normal operation...")
        print("=" * 70)

        self.log_event("EXIT_RESTOCKING_MODE", self.last_stable_weight)

    def manual_zero(self):
        """
        Manual zero adjustment (emergency use)

        This is for emergency situations only.
        Use enter_restocking_mode() for normal product additions.
        """
        print("\n[MANUAL ZERO ADJUSTMENT]")
        print("WARNING: This will reset the baseline weight!")

        self.calibrate_zero()

        # Reset drift tracking
        self.drift_stable_start = None
        self.drift_stable_weight = 0.0

        self.log_event("MANUAL_ZERO", self.zero_offset)

    def check_auto_drift_correction(self):
        """
        Check and apply automatic drift correction

        Auto-correction triggers when:
        1. Weight is stable for drift_stability_time (default: 10 minutes)
        2. Weight change is small (< drift_threshold, default: 50g)
        3. Variance is low (< drift_stability_variance)

        This helps compensate for sensor drift over time.
        """
        if not self.drift_correction_enabled:
            return

        if self.operation_mode != OperationMode.NORMAL:
            return

        # Check if weight is stable
        if len(self.weight_history) < 10:
            return

        recent_weights = list(self.weight_history)[-10:]
        variance = sum((w - self.filtered_weight) ** 2 for w in recent_weights) / len(recent_weights)

        # If variance is too high, reset stability tracking
        if variance > self.drift_stability_variance:
            self.drift_stable_start = None
            self.drift_stable_weight = 0.0
            return

        # Start tracking stability
        current_time = time.time()

        if self.drift_stable_start is None:
            self.drift_stable_start = current_time
            self.drift_stable_weight = self.filtered_weight
            return

        # Check if stable for required duration
        stable_duration = current_time - self.drift_stable_start

        if stable_duration < self.drift_stability_time:
            return

        # Check if weight change is within threshold (small drift)
        weight_change = abs(self.filtered_weight - self.last_stable_weight)

        if weight_change > self.drift_threshold:
            # Large change - not a drift, reset tracking
            self.drift_stable_start = None
            self.drift_stable_weight = 0.0
            return

        # Apply drift correction
        print(f"\n[AUTO DRIFT CORRECTION]")
        print(f"Baseline: {self.last_stable_weight:.1f}g → {self.filtered_weight:.1f}g")
        print(f"Drift: {weight_change:.1f}g (Stable for {stable_duration/60:.1f} min)")

        self.last_stable_weight = self.filtered_weight

        # Reset tracking
        self.drift_stable_start = None
        self.drift_stable_weight = 0.0

        self.log_event("AUTO_DRIFT_CORRECTION", weight_change)

    def read_weight(self) -> float:
        """Read weight from load cell"""
        if not self.is_connected:
            return 0.0

        self.serial.clear_rx_buffer()
        time.sleep(0.001)

        cmd = LoadCellProtocol.create_weight_read_command()
        self.serial.send_command(cmd)

        time.sleep(0.05)

        rx_buffer = self.serial.get_rx_buffer()

        if len(rx_buffer) >= 8:
            weight_data = LoadCellProtocol.parse_weight_response(rx_buffer)
            if weight_data:
                return weight_data['weight']

        return 0.0

    def update_weight(self):
        """Update weight reading and apply Kalman filter"""
        self.raw_weight = self.read_weight()
        zeroed_weight = self.raw_weight - self.zero_offset
        self.filtered_weight = self.filter.filter(zeroed_weight)

        # Add to history
        self.weight_history.append(self.filtered_weight)
        if len(self.weight_history) > 40:
            self.weight_history = self.weight_history[-40:]

    def get_variance(self) -> float:
        """Calculate variance of recent weight readings"""
        if len(self.weight_history) < 2:
            return float('inf')

        recent = self.weight_history[-10:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)

        return variance

    def is_weight_stable(self) -> bool:
        """Check if weight is stable"""
        if len(self.weight_history) < 10:
            return False

        variance = self.get_variance()
        return variance < self.variance_threshold

    def top_camera_thread(self):
        """Top camera thread: Real-time hand detection only"""
        print("[Thread] Top camera started")

        while self.top_running:
            ret, frame = self.top_camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Run hand detection (lightweight, real-time)
            results = self.model.predict(frame, device=self.device, conf=self.confidence_threshold, verbose=False)

            # Check for hand (class_id = 0)
            hand_detected = False
            max_confidence = 0.0

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == 0:  # hand
                        hand_detected = True
                        max_confidence = max(max_confidence, confidence)

            self.hand_detected = hand_detected
            self.hand_confidence = max_confidence

            time.sleep(0.033)  # ~30 FPS

        print("[Thread] Top camera stopped")

    def side_camera_thread(self):
        """Side camera thread: Frame buffering (no inference)"""
        print("[Thread] Side camera started")

        while self.side_running:
            ret, frame = self.side_camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Add to circular buffer
            with self.side_buffer_lock:
                self.side_buffer.append({
                    'timestamp': time.time(),
                    'frame': frame.copy()
                })

            time.sleep(0.033)  # ~30 FPS

        print("[Thread] Side camera stopped")

    def capture_side_frames(self, num_frames: int = None) -> List:
        """
        Capture frames in real-time AFTER weight change detection

        This captures frames while the user is picking/placing the item,
        not before the interaction started.

        Args:
            num_frames: Number of frames to capture (default: self.inference_frames)

        Returns:
            List of captured frames
        """
        if num_frames is None:
            num_frames = self.inference_frames

        captured_frames = []

        print(f"[Capture] Capturing {num_frames} frames in real-time...")
        print(f"[Capture] Please complete the interaction (pick/place item)...")

        # Capture frames in real-time for the next N frames
        start_time = time.time()
        frame_interval = 1.0 / self.camera_fps  # Expected interval between frames

        for i in range(num_frames):
            # Wait for new frame to be available in buffer
            with self.side_buffer_lock:
                if len(self.side_buffer) > 0:
                    # Get the most recent frame
                    frame_data = self.side_buffer[-1]
                    captured_frames.append(frame_data['frame'])

            # Wait for next frame (approximately)
            time.sleep(frame_interval)

            # Progress indicator every 30 frames (1 second)
            if (i + 1) % 30 == 0:
                elapsed = time.time() - start_time
                print(f"[Capture] Progress: {i+1}/{num_frames} frames ({elapsed:.1f}s)")

        elapsed_time = time.time() - start_time
        print(f"[Capture] {len(captured_frames)} frames captured in {elapsed_time:.1f}s")

        return captured_frames

    def save_inference_frames(self, frames: List, event_id: int, camera_id: int, results_list: List = None):
        """
        Save inference frames to disk for debugging

        Directory structure: debug_frames/cam{camera_id}/event_{event_id}_{timestamp}/

        Args:
            frames: List of frames to save
            event_id: Event ID for organizing files
            camera_id: Camera ID (e.g., 2 for side camera)
            results_list: Optional list of YOLO results for annotated frames
        """
        if not self.save_debug_frames:
            return

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory: debug_frames/cam{camera_id}/event_{event_id}_{timestamp}/
        save_dir = os.path.join(
            self.debug_frames_path,
            f"cam{camera_id}",
            f"event_{event_id}_{timestamp}"
        )

        os.makedirs(save_dir, exist_ok=True)

        print(f"[Debug] Saving {len(frames)} frames to {save_dir}")

        # Save each frame
        for idx, frame in enumerate(frames):
            # Save original frame
            frame_filename = f"frame_{idx:03d}.jpg"
            frame_path = os.path.join(save_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Save annotated frame if results provided and enabled
            if self.save_annotated and results_list and idx < len(results_list):
                result = results_list[idx]

                # Draw bounding boxes on frame
                annotated_frame = frame.copy()

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.labels.get(class_id, f'Class {class_id}')

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Choose color based on class (hand = red, products = green)
                    color = (0, 0, 255) if class_id == 0 else (0, 255, 0)

                    # Draw box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 4),
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Save annotated frame
                annotated_filename = f"frame_{idx:03d}_annotated.jpg"
                annotated_path = os.path.join(save_dir, annotated_filename)
                cv2.imwrite(annotated_path, annotated_frame)

        print(f"[Debug] Frames saved successfully")

    def calculate_bbox_movement(self, bbox_history: List) -> float:
        """
        Calculate movement score of bounding box across frames

        Args:
            bbox_history: List of (x1, y1, x2, y2) coordinates

        Returns:
            Movement score (0 = stationary, higher = more movement)
        """
        if len(bbox_history) < 2:
            return 0.0

        total_movement = 0.0
        for i in range(1, len(bbox_history)):
            x1_prev, y1_prev, x2_prev, y2_prev = bbox_history[i-1]
            x1_curr, y1_curr, x2_curr, y2_curr = bbox_history[i]

            # Calculate center movement
            center_prev = ((x1_prev + x2_prev) / 2, (y1_prev + y2_prev) / 2)
            center_curr = ((x1_curr + x2_curr) / 2, (y1_curr + y2_curr) / 2)

            # Euclidean distance
            dx = center_curr[0] - center_prev[0]
            dy = center_curr[1] - center_prev[1]
            movement = (dx**2 + dy**2) ** 0.5

            total_movement += movement

        # Average movement per frame
        avg_movement = total_movement / (len(bbox_history) - 1)
        return avg_movement

    def infer_frames_with_motion(self, frames: List, camera_name: str = "side") -> List:
        """
        Run YOLO inference with motion filtering

        Enhanced: Tracks only frames where bbox is detected (not null)
        to handle partially occluded products correctly.

        Args:
            frames: List of frames to process
            camera_name: Camera name for logging ("side" or "top")

        Returns:
            List of (class_id, votes, movement_score) tuples for moving objects
        """
        if not frames:
            return []

        print(f"[Inference {camera_name.upper()}] Processing {len(frames)} frames with motion filtering...")

        # Track per-frame detections: frame_index -> {class_id: bbox}
        frame_detections = defaultdict(dict)
        all_detected_classes = set()

        # Collect all detections per frame
        for frame_idx, frame in enumerate(frames):
            results = self.model.predict(frame, device=self.device, conf=self.confidence_threshold, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])

                    # Filter out hand (class_id = 0)
                    if class_id == 0:
                        continue

                    # Store bbox coordinates for this frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    frame_detections[frame_idx][class_id] = (x1, y1, x2, y2)
                    all_detected_classes.add(class_id)

        # Build bbox history per class (only frames where detected, excluding null frames)
        class_bbox_history = defaultdict(list)

        for class_id in all_detected_classes:
            for frame_idx in sorted(frame_detections.keys()):
                if class_id in frame_detections[frame_idx]:
                    bbox = frame_detections[frame_idx][class_id]
                    class_bbox_history[class_id].append(bbox)

        # Calculate movement and return only moving objects
        motion_threshold = 10.0  # pixels average movement per frame
        moving_candidates = []

        for class_id, bbox_history in class_bbox_history.items():
            movement_score = self.calculate_bbox_movement(bbox_history)

            # Only include if object is moving
            if movement_score >= motion_threshold:
                votes = len(bbox_history)
                moving_candidates.append((class_id, votes, movement_score))

                print(f"[Motion {camera_name.upper()}] {self.labels.get(class_id, class_id)}: "
                      f"{votes} detections, movement={movement_score:.1f}px/frame (MOVING ✓)")
            else:
                print(f"[Motion {camera_name.upper()}] {self.labels.get(class_id, class_id)}: "
                      f"{len(bbox_history)} detections, movement={movement_score:.1f}px/frame (STATIC ✗)")

        return moving_candidates

    def infer_side_frames(self, frames: List, return_candidates: bool = True, event_id: int = None):
        """
        Run YOLO inference on captured frames with motion filtering

        Enhanced with motion filtering: Only count objects with moving bounding boxes
        to avoid false positives from static background products.

        Args:
            frames: List of frames from side camera
            return_candidates: If True, return list of candidates sorted by votes
            event_id: Optional event ID for debug frame saving

        Returns:
            If return_candidates=True: List of (class_id, votes) tuples sorted by votes
            If return_candidates=False: Dominant class ID
        """
        if not frames:
            return [] if return_candidates else None

        # Process side camera with motion filtering
        moving_candidates = self.infer_frames_with_motion(frames, camera_name="side")

        # Convert to final format (class_id, votes)
        final_candidates = [(class_id, votes) for class_id, votes, _ in moving_candidates]

        # Save debug frames if enabled (side camera only for now)
        if self.save_debug_frames and event_id is not None:
            # Re-run inference to get results_list for saving
            results_list = []
            for frame in frames:
                results = self.model.predict(frame, device=self.device, conf=self.confidence_threshold, verbose=False)
                results_list.append(results[0] if results else None)

            self.save_inference_frames(
                frames=frames,
                event_id=event_id,
                camera_id=self.side_camera_id,
                results_list=results_list
            )

        if not final_candidates:
            print("[Inference] No moving product detected")
            return [] if return_candidates else None

        # Sort by votes (descending)
        final_candidates.sort(key=lambda x: x[1], reverse=True)

        if return_candidates:
            # Print final candidates
            print(f"\n[Final Candidates] Top moving objects:")
            for idx, (class_id, votes) in enumerate(final_candidates[:5]):  # Show top 5
                class_name = self.labels.get(class_id, f'Class {class_id}')
                print(f"  {idx+1}. {class_name} (ID {class_id}) - {votes} votes")

            return final_candidates
        else:
            # Backward compatibility: return dominant class
            dominant_class = final_candidates[0][0]
            class_name = self.labels.get(dominant_class, f'Class {dominant_class}')
            print(f"[Inference] Dominant class: {class_name} (ID {dominant_class})")
            return dominant_class

    def estimate_count(self, class_id: int, weight_delta: float) -> int:
        """
        Estimate product count from weight change

        Args:
            class_id: Detected product class
            weight_delta: Weight change (positive = added, negative = removed)

        Returns:
            Estimated count (signed)
        """
        if class_id not in self.product_weights:
            print(f"[Warning] Unknown product weight for class {class_id}")
            return 0

        unit_weight = self.product_weights[class_id]

        if unit_weight == 0:
            return 0

        # Calculate count
        count = round(weight_delta / unit_weight)

        return int(count)

    def verify_transaction(self, class_id: int, count: int, weight_delta: float) -> bool:
        """
        Verify transaction using weight tolerance check

        Args:
            class_id: Detected product class
            count: Estimated count
            weight_delta: Actual weight change

        Returns:
            True if transaction is valid
        """
        if class_id not in self.product_weights:
            return False

        unit_weight = self.product_weights[class_id]
        expected_weight = count * unit_weight

        # Tolerance: 10% of total weight
        tolerance = abs(expected_weight * 0.10)

        error = abs(weight_delta - expected_weight)

        is_valid = error < tolerance

        print(f"[Verification]")
        print(f"  Expected: {expected_weight:.1f}g ({count} × {unit_weight:.1f}g)")
        print(f"  Actual: {weight_delta:.1f}g")
        print(f"  Error: {error:.1f}g (Tolerance: {tolerance:.1f}g)")
        print(f"  Result: {'VALID ✓' if is_valid else 'INVALID ✗'}")

        return is_valid

    def find_best_match_from_candidates(self, candidates: List, weight_delta: float):
        """
        Find the best matching product from candidate list using weight validation

        Multi-tier decision logic:
        1. Primary: Side camera candidates (sorted by votes)
        2. Secondary: Weight-based validation
        3. Future: Top camera before/after comparison

        Args:
            candidates: List of (class_id, votes) tuples from vision
            weight_delta: Actual weight change

        Returns:
            Tuple of (class_id, count, is_valid) or (None, 0, False) if no match
        """
        print(f"\n[Multi-Tier Validation] Testing {len(candidates)} candidates...")

        best_match = None
        best_error = float('inf')

        for idx, (class_id, votes) in enumerate(candidates):
            class_name = self.labels.get(class_id, f'Class {class_id}')

            # Skip if unknown product
            if class_id not in self.product_weights:
                print(f"  {idx+1}. {class_name}: SKIP (unknown weight)")
                continue

            unit_weight = self.product_weights[class_id]

            # Estimate count based on weight
            count = self.estimate_count(class_id, weight_delta)

            if count == 0:
                print(f"  {idx+1}. {class_name}: SKIP (count = 0)")
                continue

            # Calculate expected weight
            expected_weight = count * unit_weight
            tolerance = abs(expected_weight * 0.10)
            error = abs(weight_delta - expected_weight)

            is_valid = error < tolerance

            # Print candidate evaluation
            print(f"  {idx+1}. {class_name} (ID {class_id}):")
            print(f"      Expected: {expected_weight:.1f}g ({count} × {unit_weight:.1f}g)")
            print(f"      Error: {error:.1f}g / Tolerance: {tolerance:.1f}g")
            print(f"      Result: {'✓ VALID' if is_valid else '✗ INVALID'}")

            # Track best match (even if invalid, for fallback)
            if error < best_error:
                best_error = error
                best_match = (class_id, count, is_valid, error, votes)

            # If valid match found, return immediately
            if is_valid:
                print(f"\n[Match Found] {class_name} × {abs(count)} (Rank #{idx+1}, Votes: {votes})")
                return (class_id, count, True)

        # No valid match found
        if best_match:
            class_id, count, _, error, votes = best_match
            class_name = self.labels.get(class_id, f'Class {class_id}')
            print(f"\n[Best Guess] {class_name} × {abs(count)} (Error: {error:.1f}g, Votes: {votes}) - INVALID ✗")
            return (class_id, count, False)
        else:
            print(f"\n[No Match] Could not find any valid candidate")
            return (None, 0, False)

    def log_event(self, event_type: str, value):
        """Log event to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {event_type}: {value}\n")

    def clear_log(self):
        """Clear log file"""
        with open(self.log_file, 'w') as f:
            f.write("")

    def keyboard_input_thread(self):
        """
        Keyboard input handler thread

        Controls:
        - 'r': Enter/Exit RESTOCKING mode
        - 'z': Manual zero adjustment (emergency)
        - 'q': Quit
        """
        import sys
        import select

        print("\n[Keyboard Controls]")
        print("  'r' - Enter/Exit RESTOCKING mode")
        print("  'z' - Manual zero adjustment (emergency)")
        print("  'q' - Quit")
        print("-" * 70)

        while self.top_running:
            # Check if input is available (non-blocking)
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()

                if key == 'r':
                    if self.operation_mode == OperationMode.NORMAL:
                        self.enter_restocking_mode()
                    else:
                        self.exit_restocking_mode()

                elif key == 'z':
                    self.manual_zero()

                elif key == 'q':
                    print("\nQuitting...")
                    self.top_running = False
                    break

            time.sleep(0.1)

    def run_fusion_loop(self):
        """Main fusion loop with state machine and event queue"""
        print("\n[Fusion Loop Started]")
        print("Press Ctrl+C to stop")
        print("-" * 70)

        # Start camera threads
        self.top_running = True
        self.side_running = True

        top_thread = threading.Thread(target=self.top_camera_thread, daemon=True)
        side_thread = threading.Thread(target=self.side_camera_thread, daemon=True)
        keyboard_thread = threading.Thread(target=self.keyboard_input_thread, daemon=True)

        top_thread.start()
        side_thread.start()
        keyboard_thread.start()

        # Wait for buffers to fill
        print("[Waiting for camera buffers to fill...]")
        time.sleep(2)

        last_print_time = 0

        try:
            while True:
                # Update weight
                self.update_weight()

                # Update UI with current weight
                self.ui.update_weight(self.filtered_weight)

                # Check auto drift correction (background process)
                self.check_auto_drift_correction()

                # Skip event processing if in RESTOCKING mode
                if self.operation_mode == OperationMode.RESTOCKING:
                    # Display current weight for restocking
                    current_time = time.time()
                    if current_time - last_print_time > 1.0:
                        print(f"\r[RESTOCKING] Current weight: {self.filtered_weight:.1f}g", end='', flush=True)
                        last_print_time = current_time
                    time.sleep(0.05)
                    continue

                # Clear expired events
                self.event_queue.clear_expired()

                # State machine
                if self.state == SystemState.IDLE:
                    # Update UI to READY state
                    self.ui.update_status("READY")
                    self.ui.clear_product()

                    # Check if there are pending events in queue
                    if not self.event_queue.is_empty():
                        # Process next event from queue
                        self.current_event = self.event_queue.pop()
                        print(f"\n>>> [QUEUE] Processing event {self.current_event.event_id} from queue")
                        self.state = SystemState.INTERACTION
                        continue

                    # Monitor for new weight change
                    delta = abs(self.filtered_weight - self.last_stable_weight)

                    if delta > self.weight_change_threshold:
                        # New event triggered
                        print(f"\n>>> [EVENT TRIGGER] Weight change detected: {delta:.1f}g")
                        print(f"[EVENT TRIGGER] Starting frame capture...")

                        # Create placeholder event (frames will be captured in INTERACTION state)
                        event_id = self.event_queue.push(
                            start_weight=self.last_stable_weight,
                            captured_frames=[]  # Empty initially, will be filled in INTERACTION
                        )

                        print(f"[EventQueue] Event {event_id} created (queue size: {self.event_queue.size()})")

                        # Process immediately
                        self.current_event = self.event_queue.pop()
                        self.state = SystemState.INTERACTION

                        # Update UI to PROCESSING
                        self.ui.update_status("PROCESSING")

                elif self.state == SystemState.INTERACTION:
                    # Capture frames in real-time AFTER weight change detection
                    if not self.current_event.captured_frames:
                        # First time in INTERACTION state - capture frames from side camera
                        print(f"\n[INTERACTION] Capturing frames during user interaction...")

                        # Capture from side camera only
                        captured_frames = self.capture_side_frames(num_frames=self.inference_frames)
                        self.current_event.captured_frames = captured_frames

                        print(f"[INTERACTION] Captured {len(captured_frames)} side frames")

                    # Wait for weight to stabilize
                    if self.is_weight_stable():
                        print("[INTERACTION] Weight stabilized")
                        self.state = SystemState.VERIFICATION

                elif self.state == SystemState.VERIFICATION:
                    # Calculate weight change from event start
                    weight_delta = self.filtered_weight - self.current_event.start_weight

                    print(f"[Event {self.current_event.event_id}] Weight change: {weight_delta:+.1f}g (from {self.current_event.start_weight:.1f}g to {self.filtered_weight:.1f}g)")

                    # Check for cancellation (weight returned to start)
                    if abs(weight_delta) < self.weight_change_threshold:
                        # Event cancelled
                        print(f">>> [CANCELLED] Event {self.current_event.event_id} - Weight returned to start")
                        self.log_event("EVENT_CANCELLED", f"Event {self.current_event.event_id}")
                        self.event_queue.cancelled_events += 1

                        # Don't update last_stable_weight
                        self.current_event = None
                        self.state = SystemState.IDLE
                        continue

                    # Run inference on captured frames (side camera only)
                    candidates = self.infer_side_frames(
                        frames=self.current_event.captured_frames,
                        return_candidates=True,
                        event_id=self.current_event.event_id
                    )

                    if candidates:
                        # Multi-tier validation: try all candidates until valid match found
                        detected_class, count, is_valid = self.find_best_match_from_candidates(candidates, weight_delta)

                        if detected_class is not None:
                            # Log result
                            class_name = self.labels.get(detected_class, f'Class {detected_class}')
                            action = "ADDED" if weight_delta > 0 else "REMOVED"

                            status = "VALID ✓" if is_valid else "INVALID ✗ (best guess)"
                            print(f"\n>>> [{action}] {class_name} × {abs(count)} - {status}")

                            # Update UI with detected product
                            self.ui.update_status("DETECTED")
                            self.ui.update_product(class_name, count, action)

                            # Update transaction counter
                            if is_valid:
                                self.total_transactions += 1
                                self.ui.update_session_info(self.total_transactions)

                            # Log even if invalid (for review)
                            log_prefix = f"{action}_{class_name}" if is_valid else f"INVALID_{action}_{class_name}"
                            self.log_event(log_prefix, f"{count} items, {weight_delta:.1f}g")

                            # Update last stable weight
                            self.last_stable_weight = self.filtered_weight

                            # Mark event as processed
                            self.event_queue.processed_events += 1

                            # Wait 2 seconds to display result
                            time.sleep(2.0)

                        else:
                            print("[VERIFICATION] No valid match found - ignoring event")
                            self.log_event("NO_MATCH", f"Event {self.current_event.event_id}, weight: {weight_delta:.1f}g")

                            # Show error on UI
                            self.ui.update_status("ERROR")
                            time.sleep(1.0)

                    else:
                        print("[VERIFICATION] No product detected - ignoring event")
                        self.log_event("NO_PRODUCT", f"Event {self.current_event.event_id}")

                        # Show error on UI
                        self.ui.update_status("ERROR")
                        time.sleep(1.0)

                    # Clear current event
                    self.current_event = None

                    # Return to IDLE
                    self.state = SystemState.IDLE

                # Print status periodically
                current_time = time.time()
                if current_time - last_print_time > 1.0:
                    hand_status = f"HAND ({self.hand_confidence:.2f})" if self.hand_detected else "No hand"
                    queue_status = f"Queue: {self.event_queue.size()}"
                    print(f"\r[{self.state.name:15s}] Weight: {self.filtered_weight:7.1f}g | Top: {hand_status:20s} | {queue_status:15s} | Side: {len(self.side_buffer):3d} frames",
                          end='', flush=True)

                    last_print_time = current_time

                # Sleep
                time.sleep(0.05)  # 20Hz

        except KeyboardInterrupt:
            print("\n\n[Fusion Loop Stopped]")

        finally:
            # Stop threads
            self.top_running = False
            self.side_running = False

            top_thread.join(timeout=2)
            side_thread.join(timeout=2)

            # Print queue statistics
            stats = self.event_queue.get_stats()
            print("\n[Event Queue Statistics]")
            print(f"Total Events: {stats['total_events']}")
            print(f"Processed: {stats['processed_events']}")
            print(f"Cancelled: {stats['cancelled_events']}")
            print(f"Timeout: {stats['timeout_events']}")
            print(f"Pending: {stats['pending_events']}")

    def cleanup(self):
        """Cleanup resources"""
        if self.top_camera:
            self.top_camera.release()

        if self.side_camera:
            self.side_camera.release()

        if self.is_connected:
            self.serial.disconnect()

        # Stop UI
        self.ui.stop()

        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='MVP Fusion Test v2 (Event Queue)')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--labels',
        default='13subset_label.json',
        help='Path to label JSON file'
    )
    parser.add_argument(
        '--port',
        help='Serial port (auto-detect if not specified)'
    )

    args = parser.parse_args()

    # Check files
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found")
        sys.exit(1)

    if not os.path.exists(args.labels):
        print(f"Error: {args.labels} not found")
        sys.exit(1)

    # Initialize fusion system
    fusion = FusionMVP_v2(config_path=args.config, label_path=args.labels)

    # Load model
    if not fusion.load_model():
        print("Failed to load model")
        sys.exit(1)

    # Open cameras
    fusion.top_camera = fusion.open_camera(fusion.top_camera_id)
    if fusion.top_camera is None:
        print("Failed to open top camera")
        sys.exit(1)

    fusion.side_camera = fusion.open_camera(fusion.side_camera_id)
    if fusion.side_camera is None:
        print("Failed to open side camera")
        fusion.top_camera.release()
        sys.exit(1)

    # Connect load cell
    if not fusion.connect_loadcell(port=args.port):
        print("Failed to connect load cell")
        fusion.cleanup()
        sys.exit(1)

    # Calibrate zero
    fusion.calibrate_zero()

    try:
        # Run fusion loop
        fusion.run_fusion_loop()

    finally:
        fusion.cleanup()

    print("\n" + "=" * 70)
    print("Fusion Test Complete")
    print(f"Log saved to: {fusion.log_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
