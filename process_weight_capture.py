#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process 1: Weight Detection + Image Capture

독립 프로세스로 실행되며:
- 무게 변화 감지 (실시간)
- 감지 시점 기준 앞 0.5초 + 뒤 2.5초 = 총 3초 (90프레임) 저장
- 완료되면 YOLO 프로세스에 신호 전송 (파일 기반 큐)
- 영점 조절값 저장/불러오기 (프로그램 재시작해도 유지)

신호 방식: JSON 파일을 event_queue/ 폴더에 생성

조작법:
  z - 영점 조절 (현재 무게를 0으로 설정, config에 저장됨)
  q - 종료
"""

import cv2
import os
import sys
import time
import json
import yaml
import threading
import select
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

# Add loadcell module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_loadcell'))
from loadcell_serial import LoadCellSerial
from loadcell_protocol import LoadCellProtocol

# ============================================================
# Configuration
# ============================================================
CONFIG_PATH = "config.yaml"
EVENT_QUEUE_DIR = "event_queue"  # 신호 파일 저장 폴더
CAPTURE_DIR = "captured_events"  # 이미지 저장 폴더
COMMAND_FILE = "event_queue/command.json"  # UI로부터 받는 명령

# Capture settings
PRE_CAPTURE_SEC = 1.0   # 감지 전 1초
POST_CAPTURE_SEC = 2.0  # 감지 후 2초 (연속 픽업 시 연장됨)
EXTENSION_SEC = 2.0     # 연속 픽업 시 추가 시간
TARGET_FPS = 30

# Buffer size - must hold pre-capture + max post-capture (with extensions)
# 1s pre + 2s base + 5*2s extensions = 13s max = 390 frames
BUFFER_SIZE = int((PRE_CAPTURE_SEC + POST_CAPTURE_SEC + EXTENSION_SEC * 5) * TARGET_FPS) + 30


@dataclass
class CaptureEvent:
    """Capture event data"""
    event_id: str
    timestamp: float
    start_weight: float
    end_weight: float
    weight_delta: float
    side_frames_dir: str
    top_frames_dir: str
    num_side_frames: int
    num_top_frames: int


class WeightCaptureProcess:
    """
    Process 1: Weight detection and image capture
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Camera settings (from hardware or vision section)
        hardware = self.config.get('hardware', {})
        vision = self.config.get('vision', {})

        self.top_camera_id = hardware.get('top_camera_id', vision.get('top_camera_id', 0))
        self.side_camera_id = hardware.get('side_camera_id', vision.get('side_camera_id', 2))
        self.camera_width = vision.get('camera_width', 640)
        self.camera_height = vision.get('camera_height', 480)
        self.camera_fps = vision.get('camera_fps', 30)

        # Weight settings (from event_detection section)
        event_detection = self.config.get('event_detection', {})
        self.weight_change_threshold = event_detection.get('weight_change_threshold', 5.0)
        self.settling_time = event_detection.get('settling_time', 2.0)
        self.stability_threshold = event_detection.get('variance_threshold', 2.0)

        # Serial port (from hardware section)
        self.serial_port = hardware.get('loadcell_port', '/dev/ttyUSB0')
        self.serial_baudrate = hardware.get('loadcell_baudrate', 115200)

        # Zero offset (저장된 영점값 불러오기)
        zero_offset_config = self.config.get('zero_offset', {})
        self.zero_offset = zero_offset_config.get('value', 0.0)
        last_updated = zero_offset_config.get('last_updated')
        if last_updated:
            print(f"[Config] Loaded zero offset: {self.zero_offset:.1f}g (saved at {last_updated})")

        # Frame buffers (circular)
        self.side_buffer = deque(maxlen=BUFFER_SIZE)
        self.top_buffer = deque(maxlen=BUFFER_SIZE)
        self.side_buffer_lock = threading.Lock()
        self.top_buffer_lock = threading.Lock()

        # Weight state
        self.current_weight = 0.0
        self.filtered_weight = 0.0
        self.last_stable_weight = 0.0
        self.weight_history = deque(maxlen=20)

        # Serial connection (using LoadCellSerial)
        self.serial = LoadCellSerial()

        # Cameras
        self.cap_top = None
        self.cap_side = None

        # Control flags
        self.running = False
        self.capturing = False  # True when actively capturing post-event frames
        self.is_paused = False  # UI에서 일시 정지
        self.is_restocking = False  # 물건 쌓기 모드

        # Command tracking
        self.last_command_time = None

        # Event counter
        self.event_counter = 0

        # Create directories
        os.makedirs(EVENT_QUEUE_DIR, exist_ok=True)
        os.makedirs(CAPTURE_DIR, exist_ok=True)

        # Clear old command file to prevent stale commands
        if os.path.exists(COMMAND_FILE):
            os.remove(COMMAND_FILE)
            print("[Init] Cleared old command file")

        print(f"[Config] Top Camera: {self.top_camera_id}, Side Camera: {self.side_camera_id}")
        print(f"[Config] Weight threshold: {self.weight_change_threshold}g")
        print(f"[Config] Capture: {PRE_CAPTURE_SEC}s before + {POST_CAPTURE_SEC}s after (extendable)")

    def connect_serial(self) -> bool:
        """Connect to load cell via serial"""
        # Try configured port first
        if self.serial.connect(self.serial_port, self.serial_baudrate):
            print(f"[Serial] Connected to {self.serial_port}")
            return True

        # Try to find available ports
        ports = LoadCellSerial.list_ports()
        print(f"[Serial] Available ports: {ports}")

        for port, desc in ports:
            if 'USB' in port or 'ttyUSB' in port:
                if self.serial.connect(port, self.serial_baudrate):
                    print(f"[Serial] Connected to {port}")
                    return True

        print(f"[ERROR] Serial connection failed")
        return False

    def open_cameras(self) -> bool:
        """Open both cameras"""
        try:
            # Top camera
            self.cap_top = cv2.VideoCapture(self.top_camera_id, cv2.CAP_V4L2)
            if not self.cap_top.isOpened():
                self.cap_top = cv2.VideoCapture(self.top_camera_id)

            if not self.cap_top.isOpened():
                print(f"[ERROR] Failed to open top camera {self.top_camera_id}")
                return False

            # Side camera
            self.cap_side = cv2.VideoCapture(self.side_camera_id, cv2.CAP_V4L2)
            if not self.cap_side.isOpened():
                self.cap_side = cv2.VideoCapture(self.side_camera_id)

            if not self.cap_side.isOpened():
                print(f"[ERROR] Failed to open side camera {self.side_camera_id}")
                return False

            # Set properties (resolution/fps only, don't touch exposure)
            for cap in [self.cap_top, self.cap_side]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                cap.set(cv2.CAP_PROP_FPS, self.camera_fps)

            print(f"[Camera] Top ({self.top_camera_id}) and Side ({self.side_camera_id}) opened")
            return True

        except Exception as e:
            print(f"[ERROR] Camera open failed: {e}")
            return False

    def read_weight(self) -> Optional[float]:
        """Read weight from load cell using protocol"""
        if not self.serial.is_connected:
            return None

        try:
            # Clear buffer and send weight read command
            self.serial.clear_rx_buffer()
            cmd = LoadCellProtocol.create_weight_read_command()
            self.serial.send_command(cmd)

            # Wait for response
            time.sleep(0.05)

            # Parse response
            rx_buffer = self.serial.get_rx_buffer()
            if rx_buffer:
                weight_data = LoadCellProtocol.parse_weight_response(rx_buffer)
                if weight_data and weight_data.get('weight') is not None:
                    return weight_data['weight']

        except Exception as e:
            print(f"[Serial] Read error: {e}")

        return None

    def update_weight(self):
        """Update weight with filtering and zero offset"""
        raw_weight = self.read_weight()

        if raw_weight is not None:
            # Apply zero offset
            self.current_weight = raw_weight - self.zero_offset
            self.weight_history.append(self.current_weight)

            # Simple moving average filter
            if len(self.weight_history) >= 3:
                self.filtered_weight = sum(list(self.weight_history)[-5:]) / min(5, len(self.weight_history))
            else:
                self.filtered_weight = self.current_weight

    def save_zero_offset(self):
        """Save zero offset to config file"""
        try:
            # Update config
            self.config['zero_offset'] = {
                'value': self.zero_offset,
                'last_updated': datetime.now().isoformat()
            }

            # Write to file
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

            print(f"[Zero] Saved zero offset: {self.zero_offset:.1f}g to config.yaml")
        except Exception as e:
            print(f"[ERROR] Failed to save zero offset: {e}")

    def manual_zero(self):
        """Set current weight as zero point"""
        # Read raw weight directly
        raw_weight = self.read_weight()
        if raw_weight is not None:
            self.zero_offset = raw_weight
            self.filtered_weight = 0.0
            self.last_stable_weight = 0.0
            self.weight_history.clear()

            # Save to config
            self.save_zero_offset()

            print(f"\n[Zero] Zero point set! Offset: {self.zero_offset:.1f}g")
            print(f"[Zero] Current weight is now 0.0g")
        else:
            print("[ERROR] Cannot read weight for zeroing")

    def is_weight_stable(self) -> bool:
        """Check if weight is stable"""
        if len(self.weight_history) < 5:
            return False

        recent = list(self.weight_history)[-5:]
        variance = max(recent) - min(recent)
        return variance < self.stability_threshold

    def check_commands(self):
        """Check for commands from UI (Process 2)"""
        try:
            if os.path.exists(COMMAND_FILE):
                # Read command
                with open(COMMAND_FILE, 'r') as f:
                    cmd_data = json.load(f)

                cmd_time = cmd_data.get('timestamp')

                # Skip if already processed
                if cmd_time == self.last_command_time:
                    return

                self.last_command_time = cmd_time

                command = cmd_data.get('command')
                value = cmd_data.get('value')

                if command == 'zero':
                    print("\n[Command] Zero adjust from UI")
                    self.manual_zero()

                elif command == 'pause':
                    self.is_paused = value
                    status = "PAUSED" if value else "RESUMED"
                    print(f"\n[Command] {status}")

                elif command == 'restocking':
                    self.is_restocking = value
                    if value:
                        print("\n[Command] RESTOCKING mode - weight changes ignored")
                    else:
                        # 물건 쌓기 완료 - 현재 무게를 새 baseline으로 설정 (영점 조절 X)
                        print("\n[Command] Restocking complete - updating baseline")
                        self.last_stable_weight = self.filtered_weight
                        print(f"[Command] New baseline: {self.last_stable_weight:.1f}g")

                elif command == 'stop':
                    print("\n[Command] STOP from UI")
                    self.running = False

        except Exception as e:
            # Ignore file read errors
            pass

    def top_camera_thread(self):
        """Thread for capturing top camera frames"""
        print("[Thread] Top camera started")

        while self.running:
            ret, frame = self.cap_top.read()
            if ret:
                with self.top_buffer_lock:
                    self.top_buffer.append({
                        'frame': frame.copy(),
                        'timestamp': time.time()
                    })
            time.sleep(1.0 / self.camera_fps)

        print("[Thread] Top camera stopped")

    def side_camera_thread(self):
        """Thread for capturing side camera frames"""
        print("[Thread] Side camera started")

        while self.running:
            ret, frame = self.cap_side.read()
            if ret:
                with self.side_buffer_lock:
                    self.side_buffer.append({
                        'frame': frame.copy(),
                        'timestamp': time.time()
                    })
            time.sleep(1.0 / self.camera_fps)

        print("[Thread] Side camera stopped")

    def capture_event_frames(self, start_weight: float) -> Optional[CaptureEvent]:
        """
        Capture frames for an event with continuous pickup support

        - Pre-capture: 1 second before detection (from buffer)
        - Post-capture: 2 seconds after detection (continue buffering)
        - Cancel if weight returns to original within capture period

        Returns:
            CaptureEvent with saved frame info, or None if cancelled
        """
        self.capturing = True

        # Generate event ID
        self.event_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        event_id = f"event_{self.event_counter:04d}_{timestamp}"

        print(f"\n[CAPTURE] Starting event {event_id}")
        print(f"[CAPTURE] Buffer: {len(self.side_buffer)} side, {len(self.top_buffer)} top frames")

        # Create event directory
        event_dir = os.path.join(CAPTURE_DIR, event_id)
        side_dir = os.path.join(event_dir, "side")
        top_dir = os.path.join(event_dir, "top")
        os.makedirs(side_dir, exist_ok=True)
        os.makedirs(top_dir, exist_ok=True)

        # Record the detection time - we'll grab frames from 1s before to 2s after
        detection_time = time.time()
        pre_frame_count = int(PRE_CAPTURE_SEC * TARGET_FPS)  # 30 frames before

        # ========== Wait for post-capture period ==========
        capture_end_time = detection_time + POST_CAPTURE_SEC  # Initial 2 seconds after
        last_detected_weight = self.filtered_weight
        total_extensions = 0
        max_extensions = 5

        print(f"[CAPTURE] Waiting {POST_CAPTURE_SEC}s for post-capture (extendable)...")

        last_print_time = detection_time

        while time.time() < capture_end_time:
            current_time = time.time()

            # Update weight
            self.update_weight()

            # Check for weight changes during capture
            weight_change = abs(self.filtered_weight - last_detected_weight)

            if weight_change > self.weight_change_threshold:
                # Check if weight returned to original (cancel condition)
                weight_from_start = abs(self.filtered_weight - start_weight)

                if weight_from_start < self.weight_change_threshold:
                    # Weight returned to original - cancel event
                    print(f"\n[CAPTURE] CANCELLED - Weight returned to original")
                    print(f"[CAPTURE] Start: {start_weight:.1f}g, Current: {self.filtered_weight:.1f}g")

                    # Cleanup - remove created directories
                    import shutil
                    if os.path.exists(event_dir):
                        shutil.rmtree(event_dir)

                    self.capturing = False
                    return None

                # Additional weight change detected - extend capture time
                if total_extensions < max_extensions:
                    total_extensions += 1
                    capture_end_time = current_time + EXTENSION_SEC
                    last_detected_weight = self.filtered_weight

                    print(f"\n[CAPTURE] +{EXTENSION_SEC}s extended (change: {weight_change:+.1f}g, extensions: {total_extensions})")

            # Progress every 0.5 second
            if current_time - last_print_time >= 0.5:
                elapsed = current_time - detection_time
                remaining = capture_end_time - current_time
                print(f"\r[CAPTURE] {elapsed:.1f}s elapsed, {remaining:.1f}s remaining   ", end='', flush=True)
                last_print_time = current_time

            time.sleep(0.02)  # 50Hz check rate

        # ========== Now grab all frames from buffer ==========
        # Total frames needed: pre (1s) + post (2s+) = 3s+ at 30fps = 90+ frames
        total_capture_time = time.time() - detection_time + PRE_CAPTURE_SEC
        total_frame_count = int(total_capture_time * TARGET_FPS)

        print(f"\n[CAPTURE] Grabbing {total_frame_count} frames from buffer ({total_capture_time:.1f}s)...")

        with self.side_buffer_lock:
            buffer_list = list(self.side_buffer)
            all_frames_side = buffer_list[-total_frame_count:] if len(buffer_list) >= total_frame_count else buffer_list[:]

        with self.top_buffer_lock:
            buffer_list = list(self.top_buffer)
            all_frames_top = buffer_list[-total_frame_count:] if len(buffer_list) >= total_frame_count else buffer_list[:]

        print(f"[CAPTURE] Got: {len(all_frames_side)} side, {len(all_frames_top)} top frames")
        if total_extensions > 0:
            print(f"[CAPTURE] Total extensions: {total_extensions} (+{total_extensions * EXTENSION_SEC}s)")

        # ========== Save frames ==========
        # Save side frames
        for i, frame_data in enumerate(all_frames_side):
            filename = os.path.join(side_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(filename, frame_data['frame'])

        # Save top frames
        for i, frame_data in enumerate(all_frames_top):
            filename = os.path.join(top_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(filename, frame_data['frame'])

        print(f"[CAPTURE] Saved: {len(all_frames_side)} side + {len(all_frames_top)} top frames to {event_dir}")

        # Final weight reading (no long wait for stability)
        end_weight = self.filtered_weight
        weight_delta = end_weight - start_weight

        self.capturing = False

        # Create event object
        event = CaptureEvent(
            event_id=event_id,
            timestamp=time.time(),
            start_weight=start_weight,
            end_weight=end_weight,
            weight_delta=weight_delta,
            side_frames_dir=side_dir,
            top_frames_dir=top_dir,
            num_side_frames=len(all_frames_side),
            num_top_frames=len(all_frames_top)
        )

        return event

    def send_signal_to_yolo(self, event: CaptureEvent):
        """
        Send signal to YOLO process by creating a JSON file in event_queue/
        """
        signal_data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp,
            'start_weight': event.start_weight,
            'end_weight': event.end_weight,
            'weight_delta': event.weight_delta,
            'side_frames_dir': event.side_frames_dir,
            'top_frames_dir': event.top_frames_dir,
            'num_side_frames': event.num_side_frames,
            'num_top_frames': event.num_top_frames,
            'created_at': datetime.now().isoformat()
        }

        # Write signal file
        signal_file = os.path.join(EVENT_QUEUE_DIR, f"{event.event_id}.json")
        with open(signal_file, 'w') as f:
            json.dump(signal_data, f, indent=2)

        print(f"[SIGNAL] Sent to YOLO: {signal_file}")
        print(f"[SIGNAL] Weight delta: {event.weight_delta:+.1f}g")

    def keyboard_input_thread(self):
        """Thread for handling keyboard input"""
        import sys
        import tty
        import termios

        print("[Keyboard] Thread started")
        print("[Keyboard] Press 'z' to zero, 'q' to quit")

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setcbreak(sys.stdin.fileno())

            while self.running:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()

                    if key == 'z':
                        print("\n[Keyboard] Zero requested...")
                        self.manual_zero()
                    elif key == 'q':
                        print("\n[Keyboard] Quit requested...")
                        self.running = False
                        break

        except Exception as e:
            print(f"[Keyboard] Error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        print("[Keyboard] Thread stopped")

    def run(self):
        """Main loop"""
        print("\n" + "=" * 60)
        print("Process 1: Weight Detection + Image Capture")
        print("=" * 60)

        # Initialize
        if not self.connect_serial():
            print("[ERROR] Cannot start without serial connection")
            return

        if not self.open_cameras():
            print("[ERROR] Cannot start without cameras")
            return

        # Start camera threads
        self.running = True

        top_thread = threading.Thread(target=self.top_camera_thread, daemon=True)
        side_thread = threading.Thread(target=self.side_camera_thread, daemon=True)
        keyboard_thread = threading.Thread(target=self.keyboard_input_thread, daemon=True)

        top_thread.start()
        side_thread.start()
        keyboard_thread.start()

        # Wait for buffers to fill
        print("[Init] Waiting for buffers to fill...")
        time.sleep(1.0)

        # Initial weight reading
        print("[Init] Reading initial weight...")
        for _ in range(30):
            self.update_weight()
            time.sleep(0.05)

        self.last_stable_weight = self.filtered_weight
        print(f"[Init] Initial weight: {self.last_stable_weight:.1f}g")
        print(f"[Init] Zero offset: {self.zero_offset:.1f}g")

        print("\n[Ready] Monitoring for weight changes...")
        print("[Ready] Controls: 'z' = zero adjust, 'q' = quit")
        print("-" * 60)

        last_print_time = 0

        try:
            while self.running:
                # Check for UI commands
                self.check_commands()

                # Update weight
                self.update_weight()

                # Skip if already capturing
                if self.capturing:
                    time.sleep(0.01)
                    continue

                # Skip if paused or restocking
                if self.is_paused or self.is_restocking:
                    # Status output every second
                    current_time = time.time()
                    if current_time - last_print_time > 1.0:
                        mode = "PAUSED" if self.is_paused else "RESTOCKING"
                        print(f"\r[{mode}] Weight: {self.filtered_weight:7.1f}g | Offset: {self.zero_offset:7.1f}g",
                              end='', flush=True)
                        last_print_time = current_time
                    time.sleep(0.02)
                    continue

                # Check for weight change
                delta = abs(self.filtered_weight - self.last_stable_weight)

                if delta > self.weight_change_threshold:
                    print(f"\n>>> [WEIGHT CHANGE] {delta:.1f}g detected!")
                    print(f">>> From {self.last_stable_weight:.1f}g to {self.filtered_weight:.1f}g")

                    # Capture event
                    event = self.capture_event_frames(self.last_stable_weight)

                    if event:
                        # Send signal to YOLO process
                        self.send_signal_to_yolo(event)

                        # Update baseline
                        self.last_stable_weight = event.end_weight
                        print(f"[Update] New baseline weight: {self.last_stable_weight:.1f}g")
                    else:
                        # Event was cancelled (weight returned to original)
                        print(f"[Update] Event cancelled, baseline unchanged: {self.last_stable_weight:.1f}g")

                    print("\n[Ready] Monitoring for weight changes...")
                    print("-" * 60)

                # Status output every second
                current_time = time.time()
                if current_time - last_print_time > 1.0:
                    side_buf = len(self.side_buffer)
                    top_buf = len(self.top_buffer)
                    print(f"\r[Monitor] Weight: {self.filtered_weight:7.1f}g | Baseline: {self.last_stable_weight:7.1f}g | Offset: {self.zero_offset:7.1f}g | Buf: S:{side_buf:2d} T:{top_buf:2d}",
                          end='', flush=True)
                    last_print_time = current_time

                time.sleep(0.02)  # 50Hz monitoring

        except KeyboardInterrupt:
            print("\n\n[Stop] Shutting down...")

        finally:
            self.running = False

            if self.cap_top:
                self.cap_top.release()
            if self.cap_side:
                self.cap_side.release()
            if self.serial.is_connected:
                self.serial.disconnect()

            print("[Stop] Process 1 stopped")


def main():
    process = WeightCaptureProcess()
    process.run()


if __name__ == '__main__':
    main()
