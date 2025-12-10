"""
Load Cell Driver with Kalman Filtering

Integrates legacy python_loadcell module with new Kalman filter
"""

import sys
import os
import logging
import time
import threading
from typing import Callable, Optional, Dict

# Add python_loadcell to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
loadcell_path = os.path.join(project_root, 'python_loadcell')
if loadcell_path not in sys.path:
    sys.path.insert(0, loadcell_path)

# Add src to path for absolute imports
src_path = os.path.dirname(current_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import legacy modules
try:
    from loadcell_serial import LoadCellSerial
    from loadcell_protocol import LoadCellProtocol
except ImportError as e:
    logging.error("Failed to import loadcell modules: %s", e)
    logging.error("Make sure python_loadcell directory exists")
    raise

# Import Kalman filter
from filters.kalman_filter import MultiChannelKalmanFilter


class LoadCellDriver:
    """
    Load Cell Driver with Kalman Filtering

    Features:
    - Wraps legacy LoadCellSerial interface
    - Applies Kalman filter for noise reduction
    - Multi-channel support (left/right sensors)
    - Event-driven weight change detection
    """

    def __init__(self, port: str, baudrate: int = 115200,
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0,
                 change_threshold: float = 5.0):
        """
        Initialize Load Cell Driver

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            baudrate: Communication speed
            process_noise: Kalman filter Q parameter
            measurement_noise: Kalman filter R parameter
            change_threshold: Minimum weight change to trigger event (grams)
        """
        self.logger = logging.getLogger(__name__)

        # Serial communication
        self.port = port
        self.baudrate = baudrate
        self.serial = LoadCellSerial()

        # Kalman filters (one per layer)
        self.filters = {}  # layer_id -> MultiChannelKalmanFilter

        # Configuration
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.change_threshold = change_threshold

        # State tracking
        self.previous_weights = {}  # layer_id -> [left, right]
        self.stable_weights = {}  # layer_id -> [left, right]

        # Threading
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Callbacks
        self.on_weight_change = None  # callback(layer, zone, delta_weight)

        self.logger.info("LoadCellDriver initialized: %s @ %d baud", port, baudrate)

    def connect(self) -> bool:
        """
        Connect to serial port

        Returns:
            True if connection successful
        """
        try:
            success = self.serial.connect(self.port, self.baudrate)
            if success:
                self.logger.info("Connected to load cell on %s", self.port)
            else:
                self.logger.error("Failed to connect to %s", self.port)
            return success
        except Exception as e:
            self.logger.error("Connection error: %s", e)
            return False

    def register_sensor(self, layer_id: int, left_addr: int, right_addr: int):
        """
        Register load cell sensors for a layer

        Args:
            layer_id: Layer identifier
            left_addr: I2C address of left load cell
            right_addr: I2C address of right load cell
        """
        # Create Kalman filter for this layer
        self.filters[layer_id] = MultiChannelKalmanFilter(
            num_channels=2,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise
        )

        # Initialize state
        self.previous_weights[layer_id] = [0.0, 0.0]
        self.stable_weights[layer_id] = [0.0, 0.0]

        self.logger.info("Registered layer %d: Left=%d, Right=%d",
                        layer_id, left_addr, right_addr)

    def read_raw_weights(self, left_addr: int, right_addr: int) -> Optional[tuple]:
        """
        Read raw weights from sensors

        Args:
            left_addr: Left sensor address
            right_addr: Right sensor address

        Returns:
            Tuple of (left_weight, right_weight) or None
        """
        try:
            # Read left sensor
            self.serial.read_weight()
            time.sleep(0.1)  # Wait for response

            rx_data = self.serial.get_rx_buffer()
            if len(rx_data) >= 8:
                left_result = LoadCellProtocol.parse_weight_response(rx_data)
                left_weight = left_result['weight'] if left_result else 0.0
            else:
                left_weight = 0.0

            # Read right sensor
            self.serial.read_weight()
            time.sleep(0.1)  # Wait for response

            rx_data = self.serial.get_rx_buffer()
            if len(rx_data) >= 8:
                right_result = LoadCellProtocol.parse_weight_response(rx_data)
                right_weight = right_result['weight'] if right_result else 0.0
            else:
                right_weight = 0.0

            return (left_weight, right_weight)

        except Exception as e:
            self.logger.error("Failed to read weights: %s", e)

        return None

    def read_filtered_weights(self, layer_id: int,
                            left_addr: int, right_addr: int) -> Optional[tuple]:
        """
        Read and filter weights using Kalman filter

        Args:
            layer_id: Layer identifier
            left_addr: Left sensor address
            right_addr: Right sensor address

        Returns:
            Tuple of (filtered_left, filtered_right) or None
        """
        raw_weights = self.read_raw_weights(left_addr, right_addr)
        if raw_weights is None:
            return None

        # Apply Kalman filter
        if layer_id in self.filters:
            filtered = self.filters[layer_id].filter(list(raw_weights))
            return tuple(filtered)

        return raw_weights

    def detect_weight_change(self, layer_id: int,
                           current_weights: tuple) -> Optional[Dict]:
        """
        Detect significant weight change

        Args:
            layer_id: Layer identifier
            current_weights: (left, right) filtered weights

        Returns:
            Dict with change info or None
        """
        if layer_id not in self.previous_weights:
            return None

        prev = self.previous_weights[layer_id]
        curr = list(current_weights)

        # Calculate deltas
        delta_left = curr[0] - prev[0]
        delta_right = curr[1] - prev[1]

        # Determine which zone changed
        zone = None
        delta = 0.0

        if abs(delta_left) > self.change_threshold:
            zone = 'left'
            delta = delta_left
        elif abs(delta_right) > self.change_threshold:
            zone = 'right'
            delta = delta_right

        if zone:
            return {
                'layer': layer_id,
                'zone': zone,
                'delta': delta,
                'current': curr,
                'previous': prev
            }

        return None

    def start_monitoring(self, sensors: Dict[int, tuple]):
        """
        Start continuous monitoring thread

        Args:
            sensors: Dict mapping layer_id -> (left_addr, right_addr)
        """
        if self.running:
            self.logger.warning("Monitoring already running")
            return

        self.running = True
        self.sensors = sensors

        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.thread.start()

        self.logger.info("Load cell monitoring started")

    def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            for layer_id, (left_addr, right_addr) in self.sensors.items():
                # Read filtered weights
                weights = self.read_filtered_weights(layer_id, left_addr, right_addr)

                if weights is None:
                    continue

                # Detect changes
                change = self.detect_weight_change(layer_id, weights)

                if change and self.on_weight_change:
                    # Trigger callback
                    self.on_weight_change(
                        layer=change['layer'],
                        zone=change['zone'],
                        delta=change['delta']
                    )

                    self.logger.info(
                        "Weight change: Layer %d, Zone %s, Δ=%.1fg",
                        change['layer'], change['zone'], change['delta']
                    )

                # Update state
                with self.lock:
                    self.previous_weights[layer_id] = list(weights)

            time.sleep(0.1)  # 10Hz polling

    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.running = False

        if self.thread:
            self.thread.join(timeout=2.0)

        self.logger.info("Load cell monitoring stopped")

    def get_variance(self, layer_id: int) -> float:
        """
        Get weight variance for stability check

        Args:
            layer_id: Layer identifier

        Returns:
            Total variance across both channels
        """
        if layer_id in self.filters:
            states = self.filters[layer_id].get_states()
            return sum(s['variance'] for s in states)
        return 0.0

    def is_stable(self, layer_id: int, threshold: float = 0.5) -> bool:
        """
        Check if weights are stable

        Args:
            layer_id: Layer identifier
            threshold: Variance threshold

        Returns:
            True if stable
        """
        return self.get_variance(layer_id) < threshold

    def disconnect(self):
        """Disconnect from serial port"""
        self.stop_monitoring()
        self.serial.disconnect()
        self.logger.info("Load cell driver disconnected")
