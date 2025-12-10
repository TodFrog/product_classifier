#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 Test: Vision System (YOLOv8)
카메라와 YOLOv8 모델만 단독으로 테스트하는 스크립트
"""

import sys
import os
import time
import yaml
import cv2
import json
from collections import defaultdict, deque
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ultralytics import YOLO


class VisionTester:
    """
    Phase 2 Vision System Test

    기능:
    - 카메라 연결 (Top/Side)
    - YOLOv8 모델 로드
    - 실시간 객체 감지
    - Dominant class 추출
    - Frame buffer 관리
    """

    def __init__(self, config_path: str = 'config.yaml', label_path: str = '13subset_label.json'):
        """
        Initialize Vision Tester

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
            # Convert list to dict {id: name}
            if isinstance(label_data, list):
                self.labels = {item['id']: item['name'] for item in label_data}
            else:
                # If dict format
                self.labels = {int(k): v for k, v in label_data.items()}

        print("=" * 60)
        print("Phase 2 Test: Vision System (YOLOv8)")
        print("=" * 60)

        # Vision configuration
        vision_config = self.config['vision']
        hw_config = self.config['hardware']

        self.model_path = vision_config['model_path']
        self.device = vision_config.get('device', '0')  # CUDA device
        self.confidence_threshold = vision_config['confidence_threshold']
        self.top_camera_id = hw_config['top_camera_id']
        self.side_camera_id = hw_config['side_camera_id']
        self.lookback_time = vision_config['lookback_time']

        # Camera settings
        self.camera_width = vision_config.get('camera_width', 640)
        self.camera_height = vision_config.get('camera_height', 480)
        self.camera_fps = vision_config.get('camera_fps', 30)

        # Model
        self.model = None

        # Camera
        self.camera = None
        self.current_camera_id = None

        # Frame buffer (for dominant class calculation)
        self.frame_buffer = deque(maxlen=int(self.lookback_time * self.camera_fps))

        # Statistics
        self.total_frames = 0
        self.total_detections = 0
        self.class_counts = defaultdict(int)

        # FPS tracking
        self.fps_start_time = 0
        self.fps_frame_count = 0
        self.current_fps = 0

        print(f"\n[Configuration]")
        print(f"Model Path: {self.model_path}")
        print(f"Device: {self.device} (CUDA)")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Top Camera ID: {self.top_camera_id}")
        print(f"Side Camera ID: {self.side_camera_id}")
        print(f"Lookback Time: {self.lookback_time}s")
        print(f"Loaded {len(self.labels)} classes")
        print("=" * 60)

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
            True if opened successfully
        """
        print(f"\n[Opening Camera {camera_id}]")
        print(f"Trying to open /dev/video{camera_id}...")

        try:
            # Try different backends for better compatibility
            # V4L2 backend is preferred for Linux
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

            if not self.camera.isOpened():
                print(f"[ERROR] Failed to open camera {camera_id} with V4L2 backend")
                print("Trying default backend...")
                self.camera = cv2.VideoCapture(camera_id)

                if not self.camera.isOpened():
                    print(f"[ERROR] Failed to open camera {camera_id}")
                    print("\nAvailable cameras:")
                    for i in range(10):
                        test_cam = cv2.VideoCapture(i)
                        if test_cam.isOpened():
                            print(f"  - Camera {i} is available")
                            test_cam.release()
                    return False

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)

            # Get actual properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))

            # Test read one frame
            ret, test_frame = self.camera.read()
            if not ret:
                print(f"[ERROR] Camera {camera_id} opened but cannot read frames")
                self.camera.release()
                return False

            print(f"[SUCCESS] Camera opened")
            print(f"Resolution: {actual_width}x{actual_height}")
            print(f"FPS: {actual_fps}")
            print(f"Test frame shape: {test_frame.shape}")

            self.current_camera_id = camera_id

            return True

        except Exception as e:
            print(f"[ERROR] Failed to open camera: {e}")
            import traceback
            traceback.print_exc()
            return False

    def close_camera(self):
        """Close camera"""
        if self.camera:
            self.camera.release()
            self.camera = None
            print("\n[Camera Closed]")

    def detect_objects(self, frame):
        """
        Detect objects in frame

        Args:
            frame: Input frame (BGR)

        Returns:
            List of detections [{class_id, confidence, bbox}, ...]
        """
        if self.model is None:
            return []

        # Run inference with CUDA acceleration
        results = self.model.predict(frame, device=self.device, conf=self.confidence_threshold, verbose=False)

        detections = []

        # Parse results
        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox
                })

        return detections

    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for det in detections:
            class_id = det['class_id']
            confidence = det['confidence']
            bbox = det['bbox']

            # Get label
            label_name = self.labels.get(class_id, f'Class {class_id}')

            # Draw bbox
            x1, y1, x2, y2 = map(int, bbox)

            # Color based on class (hand = red, others = green)
            color = (0, 0, 255) if class_id == 0 else (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{label_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            # Background for text
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Text
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

        return annotated

    def update_buffer(self, detections):
        """
        Update frame buffer with detections

        Args:
            detections: List of detections
        """
        self.frame_buffer.append({
            'timestamp': time.time(),
            'detections': detections
        })

    def get_dominant_class(self, filter_hand: bool = True):
        """
        Get dominant class from recent frames

        Args:
            filter_hand: Whether to filter out hand class (ID 0)

        Returns:
            Dominant class ID or None
        """
        if not self.frame_buffer:
            return None

        # Count votes for each class
        class_votes = defaultdict(int)

        cutoff_time = time.time() - self.lookback_time

        for frame_data in self.frame_buffer:
            if frame_data['timestamp'] < cutoff_time:
                continue

            for det in frame_data['detections']:
                class_id = det['class_id']

                # Filter hand if requested
                if filter_hand and class_id == 0:
                    continue

                class_votes[class_id] += 1

        if not class_votes:
            return None

        # Return class with most votes
        dominant_class = max(class_votes, key=class_votes.get)

        return dominant_class

    def draw_stats(self, frame):
        """
        Draw statistics overlay on frame

        Args:
            frame: Input frame

        Returns:
            Frame with stats overlay
        """
        overlay = frame.copy()

        # Get dominant class
        dominant_class = self.get_dominant_class(filter_hand=True)

        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text
        y_offset = 30

        # Dominant class
        if dominant_class is not None:
            dominant_name = self.labels.get(dominant_class, f'Class {dominant_class}')
            text = f"Dominant: {dominant_name} (ID {dominant_class})"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Dominant: None", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y_offset += 30

        # Buffer size
        text = f"Buffer: {len(self.frame_buffer)} frames"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 25

        # Total frames/detections
        text = f"Total Frames: {self.total_frames}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 25

        text = f"Total Detections: {self.total_detections}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 25

        # Camera ID
        text = f"Camera: {self.current_camera_id}"
        cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y_offset += 30

        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(frame, "Press 's' to switch camera", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def run_streaming(self):
        """Run real-time streaming with detection"""
        print("\n[Streaming Started]")
        print("Press 'q' to quit, 's' to switch camera")
        print("-" * 60)

        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()

                if not ret:
                    print("[ERROR] Failed to read frame")
                    break

                # Detect objects
                detections = self.detect_objects(frame)

                # Update buffer
                self.update_buffer(detections)

                # Update statistics
                self.total_frames += 1
                self.total_detections += len(detections)

                for det in detections:
                    self.class_counts[det['class_id']] += 1

                # Draw detections
                annotated = self.draw_detections(frame, detections)

                # Draw stats
                annotated = self.draw_stats(annotated)

                # Calculate FPS
                self.fps_frame_count += 1
                if time.time() - self.fps_start_time > 1.0:
                    self.current_fps = self.fps_frame_count / (time.time() - self.fps_start_time)
                    self.fps_frame_count = 0
                    self.fps_start_time = time.time()

                # Draw FPS
                cv2.putText(
                    annotated,
                    f"FPS: {int(self.current_fps)}",
                    (annotated.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                # Show frame
                cv2.imshow('Phase 2 Vision Test', annotated)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    # Quit
                    break
                elif key == ord('s'):
                    # Switch camera
                    self.close_camera()
                    new_camera_id = self.side_camera_id if self.current_camera_id == self.top_camera_id else self.top_camera_id
                    if self.open_camera(new_camera_id):
                        # Clear buffer when switching
                        self.frame_buffer.clear()
                        # Reset FPS counter
                        self.fps_start_time = time.time()
                        self.fps_frame_count = 0
                        self.current_fps = 0
                    else:
                        print("[ERROR] Failed to switch camera")
                        break

        except KeyboardInterrupt:
            print("\n\n[Streaming Stopped]")

        finally:
            cv2.destroyAllWindows()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        print(f"\nTotal Frames Processed: {self.total_frames}")
        print(f"Total Detections: {self.total_detections}")

        if self.total_frames > 0:
            print(f"Average Detections per Frame: {self.total_detections / self.total_frames:.2f}")

        print("\nClass Distribution:")
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts[class_id]
            label_name = self.labels.get(class_id, f'Class {class_id}')
            percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
            print(f"  {class_id:2d} ({label_name:20s}): {count:5d} ({percentage:5.1f}%)")

        print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 2 Vision Test')
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
        '--camera',
        type=int,
        help='Camera ID to use (default: top camera from config)'
    )

    args = parser.parse_args()

    # Check files
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found")
        sys.exit(1)

    if not os.path.exists(args.labels):
        print(f"Error: {args.labels} not found")
        sys.exit(1)

    # Initialize tester
    tester = VisionTester(config_path=args.config, label_path=args.labels)

    # Load model
    if not tester.load_model():
        print("Failed to load model")
        sys.exit(1)

    # Open camera
    camera_id = args.camera if args.camera is not None else tester.top_camera_id

    if not tester.open_camera(camera_id):
        print("Failed to open camera")
        sys.exit(1)

    try:
        # Run streaming
        tester.run_streaming()

    finally:
        tester.close_camera()
        tester.print_summary()

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
