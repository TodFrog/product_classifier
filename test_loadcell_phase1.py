#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 Test: Load Cell + Kalman Filter
로드셀과 Kalman 필터만 단독으로 테스트하는 스크립트
"""

import sys
import os
import time
import yaml
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python_loadcell'))

from loadcell_serial import LoadCellSerial
from loadcell_protocol import LoadCellProtocol
from filters.kalman_filter import LoadCellKalmanFilter


class LoadCellTester:
    """
    Phase 1 로드셀 테스트

    기능:
    - Raw weight 읽기
    - Kalman 필터 적용
    - 무게 변화 감지
    - 안정성 검증
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize Load Cell Tester

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 60)
        print("Phase 1 Test: Load Cell + Kalman Filter")
        print("=" * 60)

        # Serial connection
        self.serial = LoadCellSerial()
        self.is_connected = False

        # Kalman filter configuration
        kalman_config = self.config['kalman']
        self.filter = LoadCellKalmanFilter(
            process_noise=kalman_config['process_noise'],
            measurement_noise=kalman_config['measurement_noise'],
            dead_zone=kalman_config.get('dead_zone', 3.0)
        )

        # Event detection configuration
        event_config = self.config['event_detection']
        self.weight_change_threshold = event_config['weight_change_threshold']
        self.settling_time = event_config['settling_time']
        self.variance_threshold = event_config['variance_threshold']

        # Weight tracking
        self.raw_weight = 0.0
        self.filtered_weight = 0.0
        self.zero_offset = 0.0

        # Stability tracking
        self.weight_history = []
        self.is_stable = False
        self.stable_weight = 0.0
        self.last_stable_weight = 0.0

        # Logging
        self.log_file = "phase1_test_log.txt"
        self.clear_log()

        print("\n[Configuration]")
        print(f"Process Noise (Q): {kalman_config['process_noise']}")
        print(f"Measurement Noise (R): {kalman_config['measurement_noise']}")
        print(f"Dead Zone: {kalman_config.get('dead_zone', 3.0)}g")
        print(f"Weight Change Threshold: {self.weight_change_threshold}g")
        print(f"Settling Time: {self.settling_time}s")
        print(f"Variance Threshold: {self.variance_threshold}")
        print("=" * 60)

    def connect(self, port: str = None, baudrate: int = 115200):
        """
        Connect to load cell

        Args:
            port: Serial port (auto-detect if None)
            baudrate: Baud rate

        Returns:
            True if connected successfully
        """
        # List available ports
        ports = LoadCellSerial.list_ports()

        if not ports:
            print("\n[ERROR] No serial ports found")
            return False

        print("\n[Available Ports]")
        for i, (port_name, desc) in enumerate(ports):
            print(f"{i + 1}. {port_name} - {desc}")

        # Auto-select or manual selection
        if port is None:
            # Try default port from config
            hw_config = self.config['hardware']
            port = hw_config.get('loadcell_port', '/dev/ttyUSB0')

            # Check if default port exists
            port_names = [p[0] for p in ports]
            if port not in port_names:
                # Use first available port
                port = ports[0][0]

        print(f"\n[Connecting] Port: {port}, Baudrate: {baudrate}")

        if self.serial.connect(port, baudrate):
            self.is_connected = True
            print("[SUCCESS] Connected to load cell")
            return True
        else:
            print("[ERROR] Failed to connect")
            return False

    def disconnect(self):
        """Disconnect from load cell"""
        if self.is_connected:
            self.serial.disconnect()
            self.is_connected = False
            print("\n[Disconnected]")

    def read_weight(self, sensor_address: int = 1) -> float:
        """
        Read weight from sensor

        Args:
            sensor_address: Sensor I2C address (not used, kept for API compatibility)

        Returns:
            Raw weight in grams
        """
        if not self.is_connected:
            return 0.0

        # Clear buffer
        self.serial.clear_rx_buffer()
        time.sleep(0.001)

        # Send read command (broadcast address)
        cmd = LoadCellProtocol.create_weight_read_command()
        self.serial.send_command(cmd)

        # Wait for response
        time.sleep(0.05)

        # Parse response
        rx_buffer = self.serial.get_rx_buffer()

        if len(rx_buffer) >= 8:
            weight_data = LoadCellProtocol.parse_weight_response(rx_buffer)
            if weight_data:
                return weight_data['weight']

        return 0.0

    def calibrate_zero(self, sensor_address: int = 1):
        """
        Calibrate zero point

        Args:
            sensor_address: Sensor I2C address
        """
        print("\n[Zero Calibration]")
        print("Make sure the sensor is empty...")
        time.sleep(2)

        # Read current weight
        raw = self.read_weight(sensor_address)
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
        self.is_stable = False
        self.stable_weight = 0.0
        self.last_stable_weight = 0.0

        print(f"Zero offset set: {self.zero_offset:.1f}g")
        self.log_event("ZERO_CALIBRATION", self.zero_offset)

    def update(self, sensor_address: int = 1):
        """
        Update weight reading and apply Kalman filter

        Args:
            sensor_address: Sensor I2C address
        """
        # Read raw weight
        self.raw_weight = self.read_weight(sensor_address)

        # Apply zero offset
        zeroed_weight = self.raw_weight - self.zero_offset

        # Apply Kalman filter
        self.filtered_weight = self.filter.filter(zeroed_weight)

        # Add to history for stability check
        self.weight_history.append(self.filtered_weight)

        # Keep only recent readings (last 2 seconds at 20Hz = 40 samples)
        if len(self.weight_history) > 40:
            self.weight_history = self.weight_history[-40:]

    def get_variance(self) -> float:
        """
        Calculate variance of recent weight readings

        Returns:
            Variance
        """
        if len(self.weight_history) < 2:
            return float('inf')

        recent = self.weight_history[-10:]  # Last 0.5 seconds at 20Hz
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)

        return variance

    def check_stability(self) -> bool:
        """
        Check if weight is stable

        Returns:
            True if stable
        """
        if len(self.weight_history) < 10:
            return False

        variance = self.get_variance()

        is_stable = variance < self.variance_threshold

        return is_stable

    def check_weight_change(self) -> float:
        """
        Check for significant weight change

        Returns:
            Weight change (0 if no significant change)
        """
        if not self.is_stable:
            return 0.0

        delta_weight = self.filtered_weight - self.last_stable_weight

        if abs(delta_weight) > self.weight_change_threshold:
            return delta_weight

        return 0.0

    def log_event(self, event_type: str, value: float):
        """
        Log event to file

        Args:
            event_type: Type of event
            value: Value to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {event_type}: {value:.2f}g\n")

    def clear_log(self):
        """Clear log file"""
        with open(self.log_file, 'w') as f:
            f.write("")

    def run_monitoring(self, sensor_address: int = 1, duration: int = None):
        """
        Run continuous monitoring

        Args:
            sensor_address: Sensor I2C address
            duration: Duration in seconds (None for infinite)
        """
        print("\n[Monitoring Started]")
        print("Press Ctrl+C to stop")
        print("-" * 60)

        # Perform zero calibration
        self.calibrate_zero(sensor_address)

        start_time = time.time()
        last_print_time = 0
        last_stability_state = False

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Update weight
                self.update(sensor_address)

                # Check stability
                current_stability = self.check_stability()
                variance = self.get_variance()

                # Detect stability state change
                if current_stability and not last_stability_state:
                    # Became stable
                    self.is_stable = True
                    self.stable_weight = self.filtered_weight

                    # Check for weight change
                    delta_weight = self.check_weight_change()

                    if delta_weight != 0:
                        # Significant change detected
                        action = "ADDED" if delta_weight > 0 else "REMOVED"
                        print(f"\n>>> [{action}] ΔW = {delta_weight:+.1f}g")
                        print(f"    Previous: {self.last_stable_weight:.1f}g → Current: {self.stable_weight:.1f}g")

                        # Log event
                        self.log_event(action, delta_weight)

                        # Update last stable weight
                        self.last_stable_weight = self.stable_weight

                elif not current_stability and last_stability_state:
                    # Became unstable
                    self.is_stable = False
                    print("\n[Weight changing...]")

                last_stability_state = current_stability

                # Print status periodically (every 0.5 seconds)
                current_time = time.time()
                if current_time - last_print_time > 0.5:
                    status = "STABLE" if current_stability else "UNSTABLE"
                    print(f"\rRaw: {self.raw_weight:7.1f}g | "
                          f"Filtered: {self.filtered_weight:7.1f}g | "
                          f"Variance: {variance:6.2f} | "
                          f"Status: {status:8s}",
                          end='', flush=True)

                    last_print_time = current_time

                # Sleep to maintain ~20Hz
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\n[Monitoring Stopped]")

    def run_interactive(self):
        """Run interactive mode"""
        sensor_address = 1  # Default sensor address

        while True:
            print("\n" + "=" * 60)
            print("Phase 1 Interactive Test")
            print("=" * 60)
            print("1. Read current weight")
            print("2. Calibrate zero")
            print("3. Start monitoring")
            print("4. Change sensor address")
            print("5. Tune Kalman parameters")
            print("0. Exit")
            print("-" * 60)

            choice = input("Select option: ").strip()

            if choice == '1':
                # Read weight
                raw = self.read_weight(sensor_address)
                zeroed = raw - self.zero_offset
                filtered = self.filter.filter(zeroed)

                print(f"\nRaw: {raw:.1f}g")
                print(f"Zeroed: {zeroed:.1f}g")
                print(f"Filtered: {filtered:.1f}g")

            elif choice == '2':
                # Calibrate zero
                self.calibrate_zero(sensor_address)

            elif choice == '3':
                # Start monitoring
                self.run_monitoring(sensor_address)

            elif choice == '4':
                # Change sensor address
                addr = input(f"Enter sensor address (current: {sensor_address}): ").strip()
                try:
                    sensor_address = int(addr)
                    print(f"Sensor address set to: {sensor_address}")
                except ValueError:
                    print("Invalid address")

            elif choice == '5':
                # Tune Kalman parameters
                print(f"\nCurrent Q (process noise): {self.filter.Q}")
                print(f"Current R (measurement noise): {self.filter.R}")

                q_str = input("Enter new Q (or press Enter to keep): ").strip()
                r_str = input("Enter new R (or press Enter to keep): ").strip()

                try:
                    if q_str:
                        self.filter.Q = float(q_str)
                        print(f"Q set to: {self.filter.Q}")

                    if r_str:
                        self.filter.R = float(r_str)
                        print(f"R set to: {self.filter.R}")
                except ValueError:
                    print("Invalid value")

            elif choice == '0':
                # Exit
                break

            else:
                print("Invalid option")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 1 Load Cell Test')
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--port',
        help='Serial port (auto-detect if not specified)'
    )
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Baud rate (default: 115200)'
    )
    parser.add_argument(
        '--address',
        type=int,
        default=1,
        help='Sensor I2C address (default: 1)'
    )
    parser.add_argument(
        '--monitor',
        action='store_true',
        help='Start monitoring immediately'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Monitoring duration in seconds (default: infinite)'
    )

    args = parser.parse_args()

    # Check config file
    if not os.path.exists(args.config):
        print(f"Error: {args.config} not found")
        sys.exit(1)

    # Initialize tester
    tester = LoadCellTester(config_path=args.config)

    # Connect
    if not tester.connect(port=args.port, baudrate=args.baudrate):
        print("Failed to connect to load cell")
        sys.exit(1)

    try:
        if args.monitor:
            # Start monitoring directly
            tester.run_monitoring(
                sensor_address=args.address,
                duration=args.duration
            )
        else:
            # Interactive mode
            tester.run_interactive()

    finally:
        tester.disconnect()

    print("\n" + "=" * 60)
    print("Test Complete")
    print(f"Log saved to: {tester.log_file}")
    print("=" * 60)


if __name__ == '__main__':
    main()
