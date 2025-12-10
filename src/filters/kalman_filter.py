"""
1D Kalman Filter for Load Cell Signal Processing
Reduces noise and estimates stable weight state
"""

import numpy as np


class LoadCellKalmanFilter:
    """
    1D Kalman Filter for single load cell channel

    State Model: x_k = x_k-1 (Constant weight assumption)
    Measurement Model: z_k = x_k + v_k (Observation with noise)

    Parameters:
        process_noise (Q): Model uncertainty (keep low for steady states)
        measurement_noise (R): Sensor noise covariance (tune based on hardware)
    """

    def __init__(self, process_noise=0.1, measurement_noise=1.0, initial_state=0.0, dead_zone=3.0):
        """
        Initialize Kalman Filter

        Args:
            process_noise: Process noise covariance Q
            measurement_noise: Measurement noise covariance R
            initial_state: Initial weight estimate
            dead_zone: Threshold below which output is snapped to zero (grams)
        """
        # State
        self.x = initial_state  # Estimated weight
        self.P = 1.0  # Error covariance

        # Model parameters
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        self.dead_zone = dead_zone  # Dead zone threshold

        # Constants
        self.A = 1.0  # State transition (constant)
        self.H = 1.0  # Measurement matrix (direct observation)

        # Statistics
        self.variance = measurement_noise
        self.innovation = 0.0  # Measurement residual

    def predict(self):
        """
        Prediction step: x_k|k-1 = A * x_k-1
        """
        # State prediction (no change expected)
        self.x = self.A * self.x

        # Error covariance prediction
        self.P = self.A * self.P * self.A + self.Q

    def update(self, measurement):
        """
        Update step with new measurement

        Args:
            measurement: Raw weight reading from load cell

        Returns:
            Filtered weight estimate
        """
        # Innovation (measurement residual)
        self.innovation = measurement - self.H * self.x

        # Innovation covariance
        S = self.H * self.P * self.H + self.R

        # Kalman gain
        K = self.P * self.H / S

        # State update
        self.x = self.x + K * self.innovation

        # Error covariance update
        self.P = (1 - K * self.H) * self.P

        # Update variance estimate
        self.variance = self.P

        return self.x

    def filter(self, measurement):
        """
        Complete filter cycle: predict + update

        Args:
            measurement: Raw weight reading

        Returns:
            Filtered weight estimate
        """
        self.predict()
        filtered_value = self.update(measurement)

        # Apply dead zone: snap to zero if below threshold
        if abs(filtered_value) < self.dead_zone:
            return 0.0

        return filtered_value

    def reset(self, state=0.0):
        """Reset filter to initial state"""
        self.x = state
        self.P = 1.0
        self.innovation = 0.0

    def is_stable(self, threshold=0.5):
        """
        Check if weight estimate is stable

        Args:
            threshold: Maximum variance for stability

        Returns:
            True if variance < threshold
        """
        return self.variance < threshold

    def get_state(self):
        """
        Get current filter state

        Returns:
            Dictionary with estimate, variance, innovation
        """
        return {
            'estimate': self.x,
            'variance': self.variance,
            'innovation': self.innovation,
            'stable': self.is_stable()
        }


class MultiChannelKalmanFilter:
    """
    Multi-channel Kalman Filter for dual load cell system
    Maintains independent filters for left and right sensors
    """

    def __init__(self, num_channels=2, process_noise=0.1, measurement_noise=1.0):
        """
        Initialize multi-channel filter

        Args:
            num_channels: Number of load cell channels (default: 2 for left/right)
            process_noise: Process noise for all channels
            measurement_noise: Measurement noise for all channels
        """
        self.filters = [
            LoadCellKalmanFilter(process_noise, measurement_noise)
            for _ in range(num_channels)
        ]
        self.num_channels = num_channels

    def filter(self, measurements):
        """
        Filter all channels simultaneously

        Args:
            measurements: List of raw weight readings [left, right]

        Returns:
            List of filtered weight estimates
        """
        if len(measurements) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} measurements, got {len(measurements)}")

        return [
            filter.filter(measurement)
            for filter, measurement in zip(self.filters, measurements)
        ]

    def get_total_weight(self, measurements):
        """
        Get total filtered weight across all channels

        Args:
            measurements: List of raw weight readings

        Returns:
            Total filtered weight
        """
        filtered = self.filter(measurements)
        return sum(filtered)

    def are_stable(self, threshold=0.5):
        """
        Check if all channels are stable

        Returns:
            True if all channels have variance < threshold
        """
        return all(f.is_stable(threshold) for f in self.filters)

    def get_states(self):
        """
        Get states of all channels

        Returns:
            List of state dictionaries
        """
        return [f.get_state() for f in self.filters]

    def reset(self):
        """Reset all filters"""
        for f in self.filters:
            f.reset()
