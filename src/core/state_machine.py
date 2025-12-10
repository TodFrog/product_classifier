"""
State Machine for Vending Machine Control

States: IDLE → EVENT_TRIGGER → INTERACTION → VERIFICATION → IDLE
"""

import logging
from enum import Enum
from typing import Optional, Dict, Callable
import time


class VendingState(Enum):
    """System states"""
    IDLE = "idle"
    EVENT_TRIGGER = "event_trigger"
    INTERACTION = "interaction"
    VERIFICATION = "verification"


class OperationMode(Enum):
    """Operation modes"""
    SETUP = "setup"        # Initial setup: adding items to inventory
    OPERATION = "operation"  # Normal operation: detecting item removal


class StateMachine:
    """
    Event-Driven State Machine for Vending System

    State Flow:
    1. IDLE: Monitor load cell, wait for ΔW > threshold
    2. EVENT_TRIGGER: Weight change detected, capture state
    3. INTERACTION: Analyze vision data, wait for weight settling
    4. VERIFICATION: Validate fusion result, update inventory
    """

    def __init__(self, settling_time: float = 2.0,
                 variance_threshold: float = 0.5,
                 initial_mode: OperationMode = OperationMode.SETUP):
        """
        Initialize State Machine

        Args:
            settling_time: Time to wait for weight stabilization (seconds)
            variance_threshold: Maximum variance for stability check
            initial_mode: Initial operation mode (SETUP or OPERATION)
        """
        self.state = VendingState.IDLE
        self.mode = initial_mode
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.settling_time = settling_time
        self.variance_threshold = variance_threshold

        # State data
        self.event_data = {}
        self.event_start_time = None

        # Callbacks
        self.callbacks = {
            VendingState.IDLE: [],
            VendingState.EVENT_TRIGGER: [],
            VendingState.INTERACTION: [],
            VendingState.VERIFICATION: []
        }

        self.logger.info("State machine initialized in %s mode", self.mode.value.upper())

    def register_callback(self, state: VendingState, callback: Callable):
        """
        Register callback for state entry

        Args:
            state: State to attach callback to
            callback: Function to call on state entry
        """
        self.callbacks[state].append(callback)
        self.logger.debug("Registered callback for state: %s", state.value)

    def transition_to(self, new_state: VendingState, data: Optional[Dict] = None):
        """
        Transition to new state

        Args:
            new_state: Target state
            data: Optional data to pass to callbacks
        """
        old_state = self.state
        self.state = new_state

        self.logger.info("State transition: %s → %s", old_state.value, new_state.value)

        # Execute callbacks
        for callback in self.callbacks[new_state]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error("Callback error in %s: %s", new_state.value, e)

    def on_weight_change(self, delta_weight: float, layer: int, zone: str):
        """
        Handle weight change event (IDLE → EVENT_TRIGGER)

        Args:
            delta_weight: Measured weight change (positive = added, negative = removed)
            layer: Layer ID where change occurred
            zone: Zone ('left' or 'right')
        """
        if self.state != VendingState.IDLE:
            self.logger.warning("Weight event ignored: not in IDLE state")
            return

        # Determine if item was added or removed
        is_addition = delta_weight > 0
        is_removal = delta_weight < 0

        # In SETUP mode, only process additions
        if self.mode == OperationMode.SETUP and not is_addition:
            self.logger.debug("SETUP mode: ignoring removal event (ΔW=%.1fg)", delta_weight)
            return

        # In OPERATION mode, only process removals
        if self.mode == OperationMode.OPERATION and not is_removal:
            self.logger.debug("OPERATION mode: ignoring addition event (ΔW=%.1fg)", delta_weight)
            return

        self.event_data = {
            'delta_weight': delta_weight,
            'is_addition': is_addition,
            'is_removal': is_removal,
            'layer': layer,
            'zone': zone,
            'trigger_time': time.time()
        }

        self.event_start_time = time.time()

        action = "ITEM ADDED" if is_addition else "ITEM REMOVED"
        self.logger.info("[%s] %s: Layer %d, Zone %s, ΔW=%.1fg",
                        self.mode.value.upper(), action, layer, zone, delta_weight)

        self.transition_to(VendingState.EVENT_TRIGGER, self.event_data)

        # Immediately proceed to interaction
        self.transition_to(VendingState.INTERACTION, self.event_data)

    def check_settling(self, variance: float) -> bool:
        """
        Check if weight has settled

        Args:
            variance: Current weight variance

        Returns:
            True if settled (time elapsed and variance < threshold)
        """
        if self.event_start_time is None:
            return False

        time_elapsed = time.time() - self.event_start_time
        variance_ok = variance < self.variance_threshold

        settled = time_elapsed >= self.settling_time and variance_ok

        if settled:
            self.logger.debug("Weight settled: time=%.1fs, variance=%.2f",
                            time_elapsed, variance)

        return settled

    def on_weight_settled(self, detected_class: Optional[int]):
        """
        Handle weight settling (INTERACTION → VERIFICATION)

        Args:
            detected_class: Detected product class from vision
        """
        if self.state != VendingState.INTERACTION:
            self.logger.warning("Settlement event ignored: not in INTERACTION state")
            return

        self.event_data['detected_class'] = detected_class
        self.event_data['settle_time'] = time.time()

        self.transition_to(VendingState.VERIFICATION, self.event_data)

    def on_verification_complete(self, fusion_result: Optional[tuple]):
        """
        Handle verification completion (VERIFICATION → IDLE)

        Args:
            fusion_result: Tuple of (product_id, count, validated) or None
        """
        if self.state != VendingState.VERIFICATION:
            self.logger.warning("Verification event ignored: not in VERIFICATION state")
            return

        self.event_data['fusion_result'] = fusion_result
        self.event_data['complete_time'] = time.time()

        # Calculate total event duration
        if 'trigger_time' in self.event_data:
            duration = self.event_data['complete_time'] - self.event_data['trigger_time']
            self.logger.info("Event completed in %.2f seconds", duration)

        # Return to IDLE
        self.transition_to(VendingState.IDLE, self.event_data)

        # Clear event data
        self.event_data = {}
        self.event_start_time = None

    def get_state(self) -> VendingState:
        """Get current state"""
        return self.state

    def get_event_data(self) -> Dict:
        """Get current event data"""
        return self.event_data.copy()

    def set_mode(self, mode: OperationMode):
        """
        Change operation mode

        Args:
            mode: New operation mode (SETUP or OPERATION)
        """
        old_mode = self.mode
        self.mode = mode
        self.logger.info("Mode changed: %s → %s", old_mode.value.upper(), mode.value.upper())

    def get_mode(self) -> OperationMode:
        """Get current operation mode"""
        return self.mode

    def is_setup_mode(self) -> bool:
        """Check if in SETUP mode"""
        return self.mode == OperationMode.SETUP

    def is_operation_mode(self) -> bool:
        """Check if in OPERATION mode"""
        return self.mode == OperationMode.OPERATION

    def reset(self):
        """Reset to IDLE state"""
        self.state = VendingState.IDLE
        self.event_data = {}
        self.event_start_time = None
        self.logger.info("State machine reset to IDLE")
