"""
Object Classifier: YOLOv8 Wrapper for Product Detection

Manages camera streams and YOLO inference for class identification
"""

import cv2
import logging
from ultralytics import YOLO
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import deque
import threading
import time


class ObjectClassifier:
    """
    Vision-based Product Classifier using YOLOv8

    Features:
    - Multi-camera support (Top + Side cameras)
    - Frame buffering for temporal analysis
    - Dominant class voting
    - Hand detection filtering
    """

    def __init__(self, model_path: str, camera_ids: List[int],
                 conf_threshold: float = 0.5,
                 buffer_size: int = 60):
        """
        Initialize Object Classifier

        Args:
            model_path: Path to trained YOLOv8 model (.pt)
            camera_ids: List of camera device IDs [top_cam, side_cam]
            conf_threshold: Confidence threshold for detections
            buffer_size: Frame buffer size for temporal analysis
        """
        self.logger = logging.getLogger(__name__)

        # Load YOLO model
        self.logger.info("Loading YOLO model from: %s", model_path)
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # Camera setup
        self.camera_ids = camera_ids
        self.cameras = {}
        self.frame_buffers = {}
        self.buffer_size = buffer_size

        # Threading control
        self.running = False
        self.threads = {}
        self.locks = {}

        # Initialize cameras
        self._init_cameras()

    def _init_cameras(self):
        """Initialize all cameras and frame buffers"""
        for cam_id in self.camera_ids:
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                self.logger.error("Failed to open camera %d", cam_id)
                continue

            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            self.cameras[cam_id] = cap
            self.frame_buffers[cam_id] = deque(maxlen=self.buffer_size)
            self.locks[cam_id] = threading.Lock()

            self.logger.info("Camera %d initialized: 640x480 @ 30fps", cam_id)

    def start_streaming(self):
        """Start camera streaming threads"""
        if self.running:
            self.logger.warning("Streaming already running")
            return

        self.running = True

        for cam_id in self.cameras.keys():
            thread = threading.Thread(
                target=self._stream_camera,
                args=(cam_id,),
                daemon=True
            )
            thread.start()
            self.threads[cam_id] = thread

        self.logger.info("Camera streaming started for %d cameras", len(self.cameras))

    def stop_streaming(self):
        """Stop camera streaming threads"""
        self.running = False

        for thread in self.threads.values():
            thread.join(timeout=2.0)

        self.threads.clear()
        self.logger.info("Camera streaming stopped")

    def _stream_camera(self, cam_id: int):
        """
        Camera streaming thread function

        Args:
            cam_id: Camera device ID
        """
        cap = self.cameras[cam_id]

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.logger.warning("Failed to read from camera %d", cam_id)
                time.sleep(0.1)
                continue

            # Add timestamp
            timestamp = time.time()
            frame_data = {
                'frame': frame,
                'timestamp': timestamp
            }

            # Add to buffer
            with self.locks[cam_id]:
                self.frame_buffers[cam_id].append(frame_data)

    def get_latest_frame(self, cam_id: int) -> Optional[np.ndarray]:
        """
        Get latest frame from camera buffer

        Args:
            cam_id: Camera device ID

        Returns:
            Latest frame or None
        """
        with self.locks[cam_id]:
            if len(self.frame_buffers[cam_id]) > 0:
                return self.frame_buffers[cam_id][-1]['frame']
        return None

    def detect_objects(self, frame: np.ndarray,
                       filter_hand: bool = True) -> List[Dict]:
        """
        Detect objects in frame using YOLO

        Args:
            frame: Input image
            filter_hand: If True, exclude hand detections

        Returns:
            List of detections: [{class_id, confidence, bbox}]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()

                # Filter hand class (ID 0)
                if filter_hand and class_id == 0:
                    continue

                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox
                })

        return detections

    def get_dominant_class(self, cam_id: int,
                          lookback_time: float = 2.0,
                          filter_hand: bool = True) -> Optional[int]:
        """
        Get dominant product class from recent frames (voting algorithm)

        Args:
            cam_id: Camera device ID
            lookback_time: Time window to analyze (seconds)
            filter_hand: Exclude hand detections

        Returns:
            Most frequently detected class ID or None
        """
        current_time = time.time()
        cutoff_time = current_time - lookback_time

        class_votes = {}

        with self.locks[cam_id]:
            for frame_data in self.frame_buffers[cam_id]:
                if frame_data['timestamp'] < cutoff_time:
                    continue

                frame = frame_data['frame']
                detections = self.detect_objects(frame, filter_hand=filter_hand)

                for det in detections:
                    class_id = det['class_id']
                    class_votes[class_id] = class_votes.get(class_id, 0) + 1

        if not class_votes:
            return None

        # Return class with most votes
        dominant_class = max(class_votes, key=class_votes.get)
        self.logger.debug("Dominant class: %d (votes: %s)", dominant_class, class_votes)

        return dominant_class

    def visualize_detections(self, frame: np.ndarray,
                           detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame

        Args:
            frame: Input image
            detections: List of detections

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            confidence = det['confidence']

            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0)  # Green

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Class {class_id}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_frame

    def release(self):
        """Release all camera resources"""
        self.stop_streaming()

        for cap in self.cameras.values():
            cap.release()

        self.cameras.clear()
        self.frame_buffers.clear()
        self.logger.info("All cameras released")
