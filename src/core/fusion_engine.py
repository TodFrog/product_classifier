"""
Fusion Engine: Combines Vision (Class) + Weight (Count)

Philosophy: "Vision identifies the CLASS, Weight determines the COUNT."
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np


class FusionEngine:
    """
    Multi-modal Sensor Fusion Engine

    Combines:
    - Vision: YOLOv8 detection for product class identification
    - Weight: Kalman-filtered load cell data for quantity estimation
    """

    def __init__(self, product_database: Dict, tolerance_pct=0.10):
        """
        Initialize Fusion Engine

        Args:
            product_database: Dictionary mapping product_id -> {name, weight, tolerance}
            tolerance_pct: Allowed weight variance percentage (default: 10%)
        """
        self.product_db = product_database
        self.tolerance_pct = tolerance_pct
        self.logger = logging.getLogger(__name__)

    def estimate_count(self, delta_weight: float, unit_weight: float) -> int:
        """
        Estimate quantity from weight change

        Formula: Count = Round(ΔW / Unit_Weight)

        Args:
            delta_weight: Measured weight change (grams)
            unit_weight: Reference weight of single item (grams)

        Returns:
            Estimated item count
        """
        if unit_weight <= 0:
            self.logger.warning("Invalid unit weight: %f", unit_weight)
            return 0

        if abs(delta_weight) < unit_weight * 0.3:
            # Too small to be counted
            return 0

        count = round(abs(delta_weight) / unit_weight)
        return max(0, count)  # Ensure non-negative

    def validate_weight(self, delta_weight: float, count: int, unit_weight: float) -> bool:
        """
        Validate estimated count against weight measurement

        Validation Logic:
            |ΔW - (Count × Unit_Weight)| < (Total_Weight × tolerance_pct)

        Args:
            delta_weight: Measured weight change
            count: Estimated count
            unit_weight: Reference weight per item

        Returns:
            True if weight matches expected value within tolerance
        """
        expected_weight = count * unit_weight
        residual = abs(abs(delta_weight) - expected_weight)
        max_error = expected_weight * self.tolerance_pct

        is_valid = residual < max_error

        if not is_valid:
            self.logger.warning(
                "Weight validation failed: ΔW=%.1fg, Count=%d, Expected=%.1fg, Residual=%.1fg > %.1fg",
                delta_weight, count, expected_weight, residual, max_error
            )

        return is_valid

    def fuse(self, detected_class: int, delta_weight: float,
             is_removal: bool = True) -> Optional[Tuple[int, int, bool]]:
        """
        Main Fusion Logic: Combine class detection + weight measurement

        Args:
            detected_class: YOLO detected class ID
            delta_weight: Filtered weight change (positive or negative)
            is_removal: True if item removed, False if added

        Returns:
            Tuple of (product_id, count, validated) or None if fusion fails
            - product_id: Detected product class
            - count: Estimated quantity
            - validated: Whether weight matches expected value
        """
        # Filter out non-product classes (e.g., hand)
        if detected_class not in self.product_db:
            self.logger.warning("Unknown class ID: %d", detected_class)
            return None

        product = self.product_db[detected_class]

        # Skip non-product detections
        if not product.get('is_product', True):
            self.logger.debug("Skipping non-product class: %s", product['name'])
            return None

        unit_weight = product['weight']

        # Ensure delta_weight is positive for counting
        abs_delta = abs(delta_weight)

        # Estimate count
        count = self.estimate_count(abs_delta, unit_weight)

        if count == 0:
            self.logger.info("Weight change too small to count: %.1fg", abs_delta)
            return None

        # Validate weight
        validated = self.validate_weight(abs_delta, count, unit_weight)

        self.logger.info(
            "Fusion Result: Class=%d (%s), Count=%d, ΔW=%.1fg, Validated=%s",
            detected_class, product['name'], count, delta_weight, validated
        )

        return (detected_class, count, validated)

    def get_product_name(self, product_id: int) -> str:
        """Get product name from ID"""
        if product_id in self.product_db:
            return self.product_db[product_id]['name']
        return "Unknown"

    def get_unit_weight(self, product_id: int) -> float:
        """Get unit weight for product"""
        if product_id in self.product_db:
            return self.product_db[product_id]['weight']
        return 0.0
