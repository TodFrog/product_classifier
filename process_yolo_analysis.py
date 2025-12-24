#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process 2: YOLO Analysis

독립 프로세스로 실행되며:
- event_queue/ 폴더를 모니터링
- 새 신호 파일 발견 시 이미지 분석
- Hand-Product IoU + Motion 기반 상품 판별
- 결과 출력 및 UI 업데이트

UI도 이 프로세스에서 관리
"""

# Imports in same order as working test_fusion_mvp_v3.py
import sys
import os
import time
import yaml
import cv2
import json
import glob
import threading
import tkinter as tk
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================
CONFIG_PATH = "config.yaml"
EVENT_QUEUE_DIR = "event_queue"
PROCESSED_DIR = "event_queue/processed"  # 처리 완료된 신호 파일 이동
COMMAND_FILE = "event_queue/command.json"  # Process 1에 보내는 명령

# Hand class ID (보통 0번)
HAND_CLASS_ID = 3  # config에서 읽어올 수도 있음

# Motion threshold
MOTION_THRESHOLD = 10.0  # pixels per frame


@dataclass
class AnalysisResult:
    """Analysis result"""
    event_id: str
    weight_delta: float
    detected_products: Dict[int, int]  # class_id -> count
    is_valid: bool
    action: str  # "ADDED" or "REMOVED"


class VendingMachineUI:
    """
    UI for displaying results
    """

    def __init__(self, send_command_callback=None):
        self.root = tk.Tk()
        self.root.title("AI Smart Vending Machine - YOLO Process")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')

        # Callback for sending commands to Process 1
        self.send_command_callback = send_command_callback

        # Transaction log
        self.transaction_log = []

        # Mode state
        self.is_restocking_mode = False
        self.is_paused = False

        # ========== Layout ==========
        container = tk.Frame(self.root, bg='#2c3e50')
        container.pack(expand=True, fill='both', padx=10, pady=10)

        # Left panel - Status
        left_frame = tk.Frame(container, bg='#2c3e50')
        left_frame.pack(side=tk.LEFT, expand=True, fill='both', padx=10)

        # Right panel - Transaction log
        right_frame = tk.Frame(container, bg='#34495e', relief=tk.RIDGE, bd=2, width=350)
        right_frame.pack(side=tk.RIGHT, fill='y', padx=10, pady=5)
        right_frame.pack_propagate(False)

        # ========== Right: Transaction Log ==========
        log_title = tk.Label(right_frame, text="Transaction Log",
                             font=("Arial", 16, "bold"), fg="#ecf0f1", bg="#34495e")
        log_title.pack(pady=10)

        log_container = tk.Frame(right_frame, bg="#2c3e50")
        log_container.pack(expand=True, fill='both', padx=5, pady=5)

        self.log_canvas = tk.Canvas(log_container, bg="#2c3e50", highlightthickness=0)
        self.log_scrollbar = tk.Scrollbar(log_container, orient="vertical", command=self.log_canvas.yview)
        self.log_scrollable_frame = tk.Frame(self.log_canvas, bg="#2c3e50")

        self.log_scrollable_frame.bind("<Configure>",
            lambda e: self.log_canvas.configure(scrollregion=self.log_canvas.bbox("all")))

        self.log_canvas.create_window((0, 0), window=self.log_scrollable_frame, anchor="nw")
        self.log_canvas.configure(yscrollcommand=self.log_scrollbar.set)

        self.log_canvas.pack(side="left", fill="both", expand=True)
        self.log_scrollbar.pack(side="right", fill="y")

        clear_btn = tk.Button(right_frame, text="Clear Log", font=("Arial", 10),
                              bg="#7f8c8d", fg="white", command=self.clear_transaction_log)
        clear_btn.pack(pady=5)

        # ========== Left: Status ==========
        self.status_label = tk.Label(left_frame, text="WAITING",
                                     font=("Arial", 72, "bold"), fg="#95a5a6", bg="#2c3e50")
        self.status_label.pack(pady=30)

        self.product_label = tk.Label(left_frame, text="",
                                      font=("Arial", 36), fg="#ecf0f1", bg="#2c3e50")
        self.product_label.pack(pady=20)

        # Info panel
        info_frame = tk.Frame(left_frame, bg="#34495e", relief=tk.RIDGE, bd=2)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.weight_label = tk.Label(info_frame, text="Weight Delta: --",
                                     font=("Arial", 18), fg="#bdc3c7", bg="#34495e")
        self.weight_label.pack(pady=10)

        self.queue_label = tk.Label(info_frame, text="Queue: 0 events",
                                    font=("Arial", 16), fg="#95a5a6", bg="#34495e")
        self.queue_label.pack(pady=5)

        self.session_label = tk.Label(info_frame, text="Total Transactions: 0",
                                      font=("Arial", 16), fg="#95a5a6", bg="#34495e")
        self.session_label.pack(pady=5)

        # ========== Control Buttons ==========
        control_frame = tk.Frame(left_frame, bg='#2c3e50')
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Mode label
        self.mode_label = tk.Label(control_frame, text="Mode: NORMAL",
                                   font=("Arial", 14, "bold"), fg="#3498db", bg="#2c3e50")
        self.mode_label.pack(pady=5)

        # Button frame
        btn_frame = tk.Frame(control_frame, bg='#2c3e50')
        btn_frame.pack(pady=10)

        # Restocking button (물건 쌓기)
        self.restock_btn = tk.Button(btn_frame, text="Restocking",
                                     font=("Arial", 12, "bold"), width=14, height=2,
                                     bg="#27ae60", fg="white", activebackground="#2ecc71",
                                     command=self.toggle_restocking_mode)
        self.restock_btn.pack(side=tk.LEFT, padx=5)

        # Zero adjust button (영점 조절)
        self.zero_btn = tk.Button(btn_frame, text="Zero Adjust",
                                  font=("Arial", 12, "bold"), width=14, height=2,
                                  bg="#3498db", fg="white", activebackground="#5dade2",
                                  command=self.request_zero_adjust)
        self.zero_btn.pack(side=tk.LEFT, padx=5)

        # Pause button (일시 정지)
        self.pause_btn = tk.Button(btn_frame, text="Pause",
                                   font=("Arial", 12, "bold"), width=14, height=2,
                                   bg="#f39c12", fg="white", activebackground="#f5b041",
                                   command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        # Stop button (중단)
        self.stop_btn = tk.Button(btn_frame, text="Stop",
                                  font=("Arial", 12, "bold"), width=14, height=2,
                                  bg="#e74c3c", fg="white", activebackground="#ec7063",
                                  command=self.request_stop)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.total_transactions = 0
        self.running = True

    def update_status(self, status: str):
        """Update status display"""
        color_map = {
            'WAITING': '#95a5a6',
            'PROCESSING': '#f39c12',
            'DETECTED': '#3498db',
            'ERROR': '#e74c3c',
            'RESTOCKING': '#27ae60',
            'PAUSED': '#f39c12'
        }
        self.status_label.config(text=status, fg=color_map.get(status, '#ecf0f1'))
        self.root.update()

    def update_product(self, text: str):
        """Update product display"""
        self.product_label.config(text=text)
        self.root.update()

    def update_weight(self, delta: float):
        """Update weight delta display"""
        sign = "+" if delta > 0 else ""
        self.weight_label.config(text=f"Weight Delta: {sign}{delta:.1f}g")
        self.root.update()

    def update_queue(self, count: int):
        """Update queue count"""
        self.queue_label.config(text=f"Queue: {count} events")
        self.root.update()

    def add_transaction_log(self, product_name: str, count: int, action: str):
        """Add to transaction log"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if action == "ADDED":
            color = "#2ecc71"
            symbol = "+"
        elif action == "REMOVED":
            color = "#e74c3c"
            symbol = "-"
        else:
            color = "#f39c12"
            symbol = "!"

        entry_frame = tk.Frame(self.log_scrollable_frame, bg="#3d566e", relief=tk.FLAT, bd=1)
        entry_frame.pack(fill='x', padx=2, pady=2)

        time_label = tk.Label(entry_frame, text=timestamp, font=("Arial", 9),
                              fg="#95a5a6", bg="#3d566e")
        time_label.pack(side=tk.LEFT, padx=5)

        info_text = f"{symbol} {product_name} × {abs(count)}"
        info_label = tk.Label(entry_frame, text=info_text, font=("Arial", 11, "bold"),
                              fg=color, bg="#3d566e")
        info_label.pack(side=tk.LEFT, padx=5)

        self.transaction_log.append(entry_frame)
        self.log_canvas.update_idletasks()
        self.log_canvas.yview_moveto(1.0)

        self.total_transactions += 1
        self.session_label.config(text=f"Total Transactions: {self.total_transactions}")
        self.root.update()

    def clear_transaction_log(self):
        """Clear log"""
        for entry in self.transaction_log:
            entry.destroy()
        self.transaction_log.clear()
        self.root.update()

    def toggle_restocking_mode(self):
        """Toggle restocking mode (물건 쌓기)"""
        self.is_restocking_mode = not self.is_restocking_mode

        if self.is_restocking_mode:
            self.mode_label.config(text="Mode: RESTOCKING", fg="#e74c3c")
            self.restock_btn.config(text="End Restocking", bg="#c0392b")
            self.update_status("RESTOCKING")
            self.update_product("물건을 쌓아주세요...")
        else:
            self.mode_label.config(text="Mode: NORMAL", fg="#3498db")
            self.restock_btn.config(text="Restocking", bg="#27ae60")
            self.update_status("WAITING")
            self.update_product("")

        # Send command to Process 1
        if self.send_command_callback:
            self.send_command_callback("restocking", self.is_restocking_mode)

        self.root.update()

    def request_zero_adjust(self):
        """Request zero adjustment (영점 조절)"""
        # Send command to Process 1
        if self.send_command_callback:
            self.send_command_callback("zero", True)

        # Visual feedback
        self.zero_btn.config(bg="#2980b9")
        self.root.update()

        # Add log entry
        self.add_system_log("영점 조절 요청됨")

        # Restore button after delay
        self.root.after(500, lambda: self.zero_btn.config(bg="#3498db"))

    def toggle_pause(self):
        """Toggle pause state (일시 정지)"""
        self.is_paused = not self.is_paused

        if self.is_paused:
            self.pause_btn.config(text="Resume", bg="#16a085")
            self.mode_label.config(text="Mode: PAUSED", fg="#f39c12")
            self.update_status("PAUSED")
        else:
            self.pause_btn.config(text="Pause", bg="#f39c12")
            if self.is_restocking_mode:
                self.mode_label.config(text="Mode: RESTOCKING", fg="#e74c3c")
            else:
                self.mode_label.config(text="Mode: NORMAL", fg="#3498db")
            self.update_status("WAITING")

        # Send command to Process 1
        if self.send_command_callback:
            self.send_command_callback("pause", self.is_paused)

        self.root.update()

    def request_stop(self):
        """Request stop (중단)"""
        # Confirmation
        from tkinter import messagebox
        if messagebox.askyesno("확인", "프로그램을 종료하시겠습니까?"):
            # Send command to Process 1
            if self.send_command_callback:
                self.send_command_callback("stop", True)

            self.running = False
            self.root.quit()

    def add_system_log(self, message: str):
        """Add system message to transaction log"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        entry_frame = tk.Frame(self.log_scrollable_frame, bg="#2c3e50", relief=tk.FLAT, bd=1)
        entry_frame.pack(fill='x', padx=2, pady=2)

        time_label = tk.Label(entry_frame, text=timestamp, font=("Arial", 9),
                              fg="#95a5a6", bg="#2c3e50")
        time_label.pack(side=tk.LEFT, padx=5)

        info_label = tk.Label(entry_frame, text=f"[SYS] {message}", font=("Arial", 10, "italic"),
                              fg="#9b59b6", bg="#2c3e50")
        info_label.pack(side=tk.LEFT, padx=5)

        self.transaction_log.append(entry_frame)
        self.log_canvas.update_idletasks()
        self.log_canvas.yview_moveto(1.0)
        self.root.update()

    def update(self):
        """Update UI"""
        self.root.update()


class YOLOAnalysisProcess:
    """
    Process 2: YOLO analysis and product detection
    """

    def __init__(self, config_path: str = CONFIG_PATH):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # YOLO settings
        self.model_path = self.config['vision'].get('model_path', 'model/best.engine')
        self.confidence_threshold = self.config['vision'].get('confidence_threshold', 0.5)

        # Build labels and product_weights from products section
        products = self.config.get('products', {})
        self.labels = {}
        self.product_weights = {}
        self.product_tolerances = {}

        for class_id, product_info in products.items():
            class_id_str = str(class_id)
            product_name = product_info.get('name', f'Class {class_id}')
            self.labels[class_id_str] = product_name

            if product_info.get('is_product', False):
                self.product_weights[product_name] = product_info.get('weight', 0)
                self.product_tolerances[product_name] = product_info.get('tolerance', 5.0)

        print(f"[Config] Loaded {len(self.labels)} labels, {len(self.product_weights)} products with weights")

        # Hand class ID - find from products or use default
        self.hand_class_id = HAND_CLASS_ID
        for class_id, product_info in products.items():
            if product_info.get('name', '').lower() == 'hand' and not product_info.get('is_product', True):
                self.hand_class_id = int(class_id)
                break
        print(f"[Config] Hand class ID: {self.hand_class_id}")

        # Model
        self.model = None

        # CLAHE object - create once, reuse for speed
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Batch size for YOLO inference (lower for Jetson memory constraints)
        # Jetson has limited GPU memory, use batch_size=1 for stability
        self.batch_size = 1

        # UI - will be created in run()
        self.ui = None

        # Control
        self.running = False
        self.is_paused = False
        self.is_restocking = False

        # Create processed directory
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        print(f"[Config] Model: {self.model_path}")
        print(f"[Config] Confidence: {self.confidence_threshold}")
        print(f"[Config] Hand class ID: {self.hand_class_id}")

    def send_command(self, command: str, value):
        """Send command to Process 1 via file"""
        try:
            command_data = {
                'command': command,
                'value': value,
                'timestamp': datetime.now().isoformat()
            }

            with open(COMMAND_FILE, 'w') as f:
                json.dump(command_data, f)

            print(f"[Command] Sent: {command}={value}")

            # Update local state
            if command == "pause":
                self.is_paused = value
            elif command == "restocking":
                self.is_restocking = value
            elif command == "stop":
                self.running = False

        except Exception as e:
            print(f"[ERROR] Failed to send command: {e}")

    def clear_gpu_memory(self):
        """Clear GPU memory cache (for Jetson memory management)"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass  # Ignore if torch not available or no CUDA

    def load_model(self) -> bool:
        """Load YOLO model"""
        try:
            print(f"[Model] Loading {self.model_path}...")
            self.model = YOLO(self.model_path)

            # Warmup
            import numpy as np
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, conf=self.confidence_threshold, verbose=False)

            # Clear memory after warmup
            self.clear_gpu_memory()

            print("[Model] Loaded and warmed up")
            return True
        except Exception as e:
            print(f"[ERROR] Model load failed: {e}")
            return False

    def preprocess_frame(self, frame):
        """Apply CLAHE preprocessing"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        # Reuse CLAHE object for speed (created once in __init__)
        l_channel_clahe = self.clahe.apply(l_channel)
        lab_clahe = cv2.merge([l_channel_clahe, a_channel, b_channel])
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    def preprocess_batch(self, frames):
        """Apply CLAHE preprocessing to multiple frames"""
        return [self.preprocess_frame(f) for f in frames]

    def draw_detections(self, frame, result) -> np.ndarray:
        """Draw YOLO detections on frame for debugging"""
        annotated = frame.copy()

        if result.boxes is None or len(result.boxes) == 0:
            return annotated

        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Get class name
            class_name = self.labels.get(str(class_id), f'Class {class_id}')

            # Color: hand=blue, products=green
            if class_id == self.hand_class_id:
                color = (255, 100, 0)  # Blue for hand
            else:
                color = (0, 255, 0)  # Green for products

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def calculate_bbox_movement(self, bbox_history: List[Tuple]) -> float:
        """Calculate average movement from bbox history"""
        if len(bbox_history) < 2:
            return 0.0

        movements = []
        for i in range(1, len(bbox_history)):
            prev = bbox_history[i - 1]
            curr = bbox_history[i]

            prev_center = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
            curr_center = ((curr[0] + curr[2]) / 2, (curr[1] + curr[3]) / 2)

            dist = ((curr_center[0] - prev_center[0]) ** 2 +
                    (curr_center[1] - prev_center[1]) ** 2) ** 0.5
            movements.append(dist)

        return sum(movements) / len(movements) if movements else 0.0

    def calculate_hand_overlap_ratio(self, hand_bbox: Tuple, product_bbox: Tuple) -> float:
        """Calculate overlap ratio between hand and product"""
        x1 = max(hand_bbox[0], product_bbox[0])
        y1 = max(hand_bbox[1], product_bbox[1])
        x2 = min(hand_bbox[2], product_bbox[2])
        y2 = min(hand_bbox[3], product_bbox[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        product_area = (product_bbox[2] - product_bbox[0]) * (product_bbox[3] - product_bbox[1])

        if product_area <= 0:
            return 0.0

        return intersection / product_area

    def analyze_frames(self, side_frames_dir: str, top_frames_dir: str, save_debug: bool = True) -> List[Tuple]:
        """
        Analyze frames and return candidates

        Args:
            side_frames_dir: Directory containing side camera frames
            top_frames_dir: Directory containing top camera frames
            save_debug: Whether to save YOLO annotated frames for debugging

        Returns:
            List of (class_id, votes, movement_score, hand_overlap)
        """
        # Load frames
        side_files = sorted(glob.glob(os.path.join(side_frames_dir, "*.jpg")))
        top_files = sorted(glob.glob(os.path.join(top_frames_dir, "*.jpg")))

        print(f"[Analysis] Side: {len(side_files)}, Top: {len(top_files)} frames")

        if not side_files and not top_files:
            print("[ERROR] No frames to analyze")
            return []

        # Create debug directories if saving
        debug_side_dir = None
        debug_top_dir = None
        if save_debug:
            event_dir = os.path.dirname(side_frames_dir)  # Parent of side/ is event_xxx/
            debug_side_dir = os.path.join(event_dir, "yolo_side")
            debug_top_dir = os.path.join(event_dir, "yolo_top")
            os.makedirs(debug_side_dir, exist_ok=True)
            os.makedirs(debug_top_dir, exist_ok=True)
            print(f"[Analysis] Saving YOLO debug frames to {event_dir}")

        # Per-frame data collection
        side_frame_data = defaultdict(lambda: {'hand': None, 'products': {}})
        top_frame_data = defaultdict(lambda: {'hand': None, 'products': {}})
        all_detected_classes = set()

        # Process side frames with batch inference
        print(f"[Analysis] Processing side frames (batch_size={self.batch_size})...")
        start_time = time.time()

        # Load and preprocess all side frames first
        side_frames = []
        side_originals = []
        side_indices = []
        for frame_idx, frame_path in enumerate(side_files):
            frame = cv2.imread(frame_path)
            if frame is not None:
                side_originals.append(frame)
                side_frames.append(self.preprocess_frame(frame))
                side_indices.append(frame_idx)

        # Batch inference for side frames
        for batch_start in range(0, len(side_frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(side_frames))
            batch_frames = side_frames[batch_start:batch_end]
            batch_indices = side_indices[batch_start:batch_end]
            batch_originals = side_originals[batch_start:batch_end]

            # Run batch prediction
            results_list = self.model.predict(batch_frames, conf=self.confidence_threshold, verbose=False)

            # Process results
            for i, (result, frame_idx, original) in enumerate(zip(results_list, batch_indices, batch_originals)):
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        bbox = tuple(box.xyxy[0].cpu().numpy())

                        if class_id == self.hand_class_id:
                            side_frame_data[frame_idx]['hand'] = bbox
                        else:
                            side_frame_data[frame_idx]['products'][class_id] = bbox
                            all_detected_classes.add(class_id)

                # Save annotated frame for debug (always save, even if no detections)
                if save_debug and debug_side_dir:
                    annotated = self.draw_detections(original, result)
                    debug_path = os.path.join(debug_side_dir, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(debug_path, annotated)

        side_time = time.time() - start_time
        print(f"[Analysis] Side frames done in {side_time:.2f}s ({len(side_files)/(side_time+0.001):.1f} fps)")

        # Process top frames with batch inference
        print(f"[Analysis] Processing top frames (batch_size={self.batch_size})...")
        start_time = time.time()

        # Load and preprocess all top frames first
        top_frames = []
        top_originals = []
        top_indices = []
        for frame_idx, frame_path in enumerate(top_files):
            frame = cv2.imread(frame_path)
            if frame is not None:
                top_originals.append(frame)
                top_frames.append(self.preprocess_frame(frame))
                top_indices.append(frame_idx)

        # Batch inference for top frames
        for batch_start in range(0, len(top_frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(top_frames))
            batch_frames = top_frames[batch_start:batch_end]
            batch_indices = top_indices[batch_start:batch_end]
            batch_originals = top_originals[batch_start:batch_end]

            # Run batch prediction
            results_list = self.model.predict(batch_frames, conf=self.confidence_threshold, verbose=False)

            # Process results
            for i, (result, frame_idx, original) in enumerate(zip(results_list, batch_indices, batch_originals)):
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        bbox = tuple(box.xyxy[0].cpu().numpy())

                        if class_id == self.hand_class_id:
                            top_frame_data[frame_idx]['hand'] = bbox
                        else:
                            top_frame_data[frame_idx]['products'][class_id] = bbox
                            all_detected_classes.add(class_id)

                # Save annotated frame for debug (always save, even if no detections)
                if save_debug and debug_top_dir:
                    annotated = self.draw_detections(original, result)
                    debug_path = os.path.join(debug_top_dir, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(debug_path, annotated)

        top_time = time.time() - start_time
        print(f"[Analysis] Top frames done in {top_time:.2f}s ({len(top_files)/(top_time+0.001):.1f} fps)")

        if save_debug:
            print(f"[Analysis] Saved YOLO debug frames: side={len(side_frame_data)}, top={len(top_frame_data)}")

        # Clear GPU memory after processing all frames
        self.clear_gpu_memory()

        # Clear frame lists to free memory
        del side_frames, side_originals, top_frames, top_originals

        print(f"[Analysis] Detected {len(all_detected_classes)} product classes")

        # Calculate scores for each product
        class_scores = defaultdict(lambda: {
            'side_bbox_history': [],
            'top_bbox_history': [],
            'side_hand_overlaps': [],
            'top_hand_overlaps': [],
            'side_detection_count': 0,
            'top_detection_count': 0
        })

        # Gather from side camera
        for frame_idx in sorted(side_frame_data.keys()):
            frame_info = side_frame_data[frame_idx]
            hand_bbox = frame_info['hand']

            for class_id, product_bbox in frame_info['products'].items():
                class_scores[class_id]['side_bbox_history'].append(product_bbox)
                class_scores[class_id]['side_detection_count'] += 1

                if hand_bbox:
                    overlap = self.calculate_hand_overlap_ratio(hand_bbox, product_bbox)
                    class_scores[class_id]['side_hand_overlaps'].append(overlap)

        # Gather from top camera
        for frame_idx in sorted(top_frame_data.keys()):
            frame_info = top_frame_data[frame_idx]
            hand_bbox = frame_info['hand']

            for class_id, product_bbox in frame_info['products'].items():
                class_scores[class_id]['top_bbox_history'].append(product_bbox)
                class_scores[class_id]['top_detection_count'] += 1

                if hand_bbox:
                    overlap = self.calculate_hand_overlap_ratio(hand_bbox, product_bbox)
                    class_scores[class_id]['top_hand_overlaps'].append(overlap)

        # Calculate final scores
        candidates = []

        for class_id in all_detected_classes:
            scores = class_scores[class_id]
            class_name = self.labels.get(str(class_id), f'Class {class_id}')

            # Motion score (Side 50%, Top 50%)
            side_movement = self.calculate_bbox_movement(scores['side_bbox_history'])
            top_movement = self.calculate_bbox_movement(scores['top_bbox_history'])

            if scores['side_bbox_history'] and scores['top_bbox_history']:
                combined_movement = (side_movement * 0.5) + (top_movement * 0.5)
            elif scores['side_bbox_history']:
                combined_movement = side_movement
            elif scores['top_bbox_history']:
                combined_movement = top_movement
            else:
                combined_movement = 0.0

            side_is_moving = side_movement >= MOTION_THRESHOLD
            top_is_moving = top_movement >= MOTION_THRESHOLD
            is_moving = side_is_moving or top_is_moving or combined_movement >= MOTION_THRESHOLD

            # Hand overlap
            side_overlaps = scores['side_hand_overlaps']
            top_overlaps = scores['top_hand_overlaps']

            avg_side_overlap = sum(side_overlaps) / len(side_overlaps) if side_overlaps else 0.0
            avg_top_overlap = sum(top_overlaps) / len(top_overlaps) if top_overlaps else 0.0
            max_side_overlap = max(side_overlaps) if side_overlaps else 0.0
            max_top_overlap = max(top_overlaps) if top_overlaps else 0.0

            combined_overlap = (avg_side_overlap * 0.5) + (avg_top_overlap * 0.5)
            max_combined_overlap = max(max_side_overlap, max_top_overlap)

            # Votes
            votes = scores['side_detection_count'] + scores['top_detection_count']

            # Log
            side_status = "✓" if side_is_moving else "✗"
            top_status = "✓" if top_is_moving else "✗"

            print(f"  [{class_name}]")
            print(f"    Motion - Side: {side_movement:.1f} ({side_status}) | Top: {top_movement:.1f} ({top_status})")
            print(f"    Hand Overlap: Side={max_side_overlap:.1%}, Top={max_top_overlap:.1%}")
            print(f"    Votes: {votes} (Side:{scores['side_detection_count']}, Top:{scores['top_detection_count']})")

            if is_moving:
                hand_bonus = 1.0 + combined_overlap
                final_score = votes * hand_bonus

                candidates.append({
                    'class_id': class_id,
                    'votes': votes,
                    'movement_score': combined_movement,
                    'hand_overlap': combined_overlap,
                    'final_score': final_score
                })
                print(f"    → ✓ CANDIDATE (Score: {final_score:.1f})")
            else:
                print(f"    → ✗ REJECTED (static)")

        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Convert to result format
        result = [(c['class_id'], c['votes'], c['movement_score'], c['hand_overlap'])
                  for c in candidates]

        return result

    def find_best_match(self, candidates: List[Tuple], weight_delta: float) -> Tuple[Optional[Dict], bool]:
        """
        Find best product match based on weight

        Returns:
            (products_dict, is_valid)
        """
        if not candidates:
            return None, False

        abs_weight = abs(weight_delta)
        sign = 1 if weight_delta > 0 else -1

        # Build valid candidates with weight info
        valid_candidates = []
        for class_id, votes, movement, overlap in candidates:
            class_name = self.labels.get(str(class_id), f'Class {class_id}')
            unit_weight = self.product_weights.get(class_name, 0)

            if unit_weight > 0:
                valid_candidates.append((class_id, votes, unit_weight))
                print(f"  {class_name}: unit_weight={unit_weight}g")

        if not valid_candidates:
            print("[Warning] No candidates with known weight")
            return None, False

        # Try single product match first
        print("\n[Matching] Single product:")
        for class_id, votes, unit_weight in valid_candidates:
            class_name = self.labels.get(str(class_id), f'Class {class_id}')
            error = abs(abs_weight - unit_weight)
            # Use tolerance from config, or fallback to 5% of weight
            tolerance = self.product_tolerances.get(class_name, unit_weight * 0.05)

            if error < tolerance:
                print(f"  [Match] {class_name} x 1 (error: {error:.1f}g, tol: {tolerance:.1f}g)")
                return {class_id: 1 * sign}, True

            print(f"  {class_name}: expected={unit_weight:.1f}g, error={error:.1f}g, tol={tolerance:.1f}g")

        # Try x2
        print("\n[Matching] Single product x 2:")
        for class_id, votes, unit_weight in valid_candidates:
            class_name = self.labels.get(str(class_id), f'Class {class_id}')
            expected = unit_weight * 2
            error = abs(abs_weight - expected)
            # Use tolerance * 2 for double quantity
            base_tolerance = self.product_tolerances.get(class_name, unit_weight * 0.05)
            tolerance = base_tolerance * 2

            if error < tolerance:
                print(f"  [Match] {class_name} x 2 (error: {error:.1f}g, tol: {tolerance:.1f}g)")
                return {class_id: 2 * sign}, True

        # Best guess (first candidate)
        if valid_candidates:
            class_id, votes, unit_weight = valid_candidates[0]
            count = max(1, round(abs_weight / unit_weight))
            return {class_id: count * sign}, False

        return None, False

    def process_event(self, signal_data: Dict) -> Optional[AnalysisResult]:
        """Process a single event"""
        event_id = signal_data['event_id']
        weight_delta = signal_data['weight_delta']
        side_dir = signal_data['side_frames_dir']
        top_dir = signal_data['top_frames_dir']

        print(f"\n{'=' * 60}")
        print(f"[Processing] Event: {event_id}")
        print(f"[Processing] Weight delta: {weight_delta:+.1f}g")
        print(f"{'=' * 60}")

        # Update UI
        self.ui.update_status("PROCESSING")
        self.ui.update_weight(weight_delta)

        # Analyze frames
        candidates = self.analyze_frames(side_dir, top_dir)

        if not candidates:
            print("[ERROR] No candidates detected")
            self.ui.update_status("ERROR")
            self.ui.update_product("감지 실패")
            return None

        # Find match
        products_dict, is_valid = self.find_best_match(candidates, weight_delta)

        if products_dict:
            action = "ADDED" if weight_delta > 0 else "REMOVED"

            # Build display string
            display_parts = []
            for class_id, count in products_dict.items():
                class_name = self.labels.get(str(class_id), f'Class {class_id}')
                display_parts.append(f"{class_name} × {abs(count)}")

            display_str = " + ".join(display_parts)

            # Update UI
            self.ui.update_status("DETECTED")
            self.ui.update_product(f"{action}: {display_str}")

            # Add to log
            for class_id, count in products_dict.items():
                class_name = self.labels.get(str(class_id), f'Class {class_id}')
                self.ui.add_transaction_log(class_name, count, action)

            status = "✓ VALID" if is_valid else "✗ INVALID (best guess)"
            print(f"\n>>> [{action}] {display_str} - {status}")

            return AnalysisResult(
                event_id=event_id,
                weight_delta=weight_delta,
                detected_products=products_dict,
                is_valid=is_valid,
                action=action
            )

        else:
            print("[ERROR] No match found")
            self.ui.update_status("ERROR")
            self.ui.update_product("인식 실패")
            return None

    def get_pending_signals(self) -> List[str]:
        """Get list of pending signal files"""
        pattern = os.path.join(EVENT_QUEUE_DIR, "event_*.json")
        return sorted(glob.glob(pattern))

    def move_to_processed(self, signal_path: str):
        """Move processed signal file"""
        filename = os.path.basename(signal_path)
        dest = os.path.join(PROCESSED_DIR, filename)
        os.rename(signal_path, dest)

    def run(self):
        """Main loop"""
        print("\n" + "=" * 60)
        print("Process 2: YOLO Analysis")
        print("=" * 60)

        # Create UI first (must be done in main thread)
        print("[UI] Creating UI...")
        self.ui = VendingMachineUI(send_command_callback=self.send_command)
        print("[UI] UI created")

        # Load model
        if not self.load_model():
            print("[ERROR] Cannot start without model")
            return

        self.running = True

        print("\n[Ready] Waiting for events from Process 1...")
        print(f"[Ready] Monitoring: {EVENT_QUEUE_DIR}/")
        print("-" * 60)

        try:
            while self.running:
                # Update UI
                self.ui.update()

                # Check for pending signals
                pending = self.get_pending_signals()

                if pending:
                    self.ui.update_queue(len(pending))

                    # Process first signal
                    signal_path = pending[0]

                    try:
                        with open(signal_path, 'r') as f:
                            signal_data = json.load(f)

                        # Process event
                        result = self.process_event(signal_data)

                        # Move to processed
                        self.move_to_processed(signal_path)

                        # Brief display time
                        time.sleep(1.0)

                        # Return to waiting
                        self.ui.update_status("WAITING")
                        self.ui.update_product("")

                    except Exception as e:
                        print(f"[ERROR] Processing failed: {e}")
                        # Move anyway to avoid infinite loop
                        self.move_to_processed(signal_path)

                else:
                    self.ui.update_queue(0)

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n[Stop] Shutting down...")

        finally:
            self.running = False
            self.ui.root.destroy()
            print("[Stop] Process 2 stopped")


def main():
    process = YOLOAnalysisProcess()
    process.run()


if __name__ == '__main__':
    main()
