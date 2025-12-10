"""
Inventory Manager: Tracks products in vending machine

Supports two modes:
- SETUP: Build initial inventory by adding products
- OPERATION: Track product removals during sales
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class InventoryManager:
    """
    Manages product inventory

    Features:
    - Track products by layer and zone
    - Support SETUP mode (adding items) and OPERATION mode (removing items)
    - Persist inventory to JSON file
    - Track inventory history
    """

    def __init__(self, inventory_file: str = 'inventory.json'):
        """
        Initialize Inventory Manager

        Args:
            inventory_file: Path to inventory JSON file
        """
        self.logger = logging.getLogger(__name__)
        self.inventory_file = inventory_file

        # Inventory structure: {layer: {zone: [(product_id, count, timestamp)]}}
        self.inventory = defaultdict(lambda: defaultdict(list))

        # History of all transactions
        self.history = []

        # Load existing inventory if available
        self.load()

    def add_item(self, product_id: int, count: int, layer: int, zone: str,
                 validated: bool = True) -> bool:
        """
        Add item(s) to inventory (SETUP mode)

        Args:
            product_id: Product class ID
            count: Number of items
            layer: Layer ID
            zone: Zone ('left' or 'right')
            validated: Whether weight was validated

        Returns:
            True if successful
        """
        timestamp = time.time()

        # Add to inventory
        self.inventory[layer][zone].append({
            'product_id': product_id,
            'count': count,
            'timestamp': timestamp,
            'validated': validated
        })

        # Record in history
        self.history.append({
            'action': 'add',
            'product_id': product_id,
            'count': count,
            'layer': layer,
            'zone': zone,
            'validated': validated,
            'timestamp': timestamp
        })

        self.logger.info(
            "✓ Inventory ADD: Product %d × %d → Layer %d, Zone %s [%s]",
            product_id, count, layer, zone,
            "VALIDATED" if validated else "UNVALIDATED"
        )

        # Save to file
        self.save()

        return True

    def remove_item(self, product_id: int, count: int, layer: int, zone: str,
                   validated: bool = True) -> bool:
        """
        Remove item(s) from inventory (OPERATION mode)

        Args:
            product_id: Product class ID
            count: Number of items
            layer: Layer ID
            zone: Zone ('left' or 'right')
            validated: Whether weight was validated

        Returns:
            True if successful
        """
        timestamp = time.time()

        # Check if item exists in inventory
        zone_items = self.inventory[layer][zone]

        # Find matching product
        found = False
        for item in zone_items:
            if item['product_id'] == product_id:
                # Reduce count
                if item['count'] >= count:
                    item['count'] -= count
                    found = True

                    # Remove entry if count reaches 0
                    if item['count'] == 0:
                        zone_items.remove(item)

                    break
                else:
                    self.logger.warning(
                        "Not enough inventory: Product %d has %d, requested %d",
                        product_id, item['count'], count
                    )

        if not found:
            self.logger.warning(
                "Product %d not found in inventory (Layer %d, Zone %s)",
                product_id, layer, zone
            )

        # Record in history regardless
        self.history.append({
            'action': 'remove',
            'product_id': product_id,
            'count': count,
            'layer': layer,
            'zone': zone,
            'validated': validated,
            'found_in_inventory': found,
            'timestamp': timestamp
        })

        self.logger.info(
            "✓ Inventory REMOVE: Product %d × %d ← Layer %d, Zone %s [%s]",
            product_id, count, layer, zone,
            "VALIDATED" if validated else "UNVALIDATED"
        )

        # Save to file
        self.save()

        return found

    def get_inventory(self, layer: Optional[int] = None,
                     zone: Optional[str] = None) -> Dict:
        """
        Get current inventory

        Args:
            layer: Optional layer filter
            zone: Optional zone filter

        Returns:
            Inventory dictionary
        """
        if layer is not None and zone is not None:
            return dict(self.inventory[layer][zone])
        elif layer is not None:
            return dict(self.inventory[layer])
        else:
            return dict(self.inventory)

    def get_total_count(self, product_id: Optional[int] = None) -> int:
        """
        Get total count of products in inventory

        Args:
            product_id: Optional product filter

        Returns:
            Total count
        """
        total = 0
        for layer_inv in self.inventory.values():
            for zone_items in layer_inv.values():
                for item in zone_items:
                    if product_id is None or item['product_id'] == product_id:
                        total += item['count']
        return total

    def get_product_summary(self) -> Dict[int, int]:
        """
        Get summary of all products

        Returns:
            Dictionary mapping product_id -> total_count
        """
        summary = defaultdict(int)
        for layer_inv in self.inventory.values():
            for zone_items in layer_inv.values():
                for item in zone_items:
                    summary[item['product_id']] += item['count']
        return dict(summary)

    def save(self):
        """Save inventory to JSON file"""
        try:
            data = {
                'inventory': self._serialize_inventory(),
                'history': self.history,
                'last_updated': time.time()
            }

            with open(self.inventory_file, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.debug("Inventory saved to %s", self.inventory_file)

        except Exception as e:
            self.logger.error("Failed to save inventory: %s", e)

    def load(self):
        """Load inventory from JSON file"""
        try:
            with open(self.inventory_file, 'r') as f:
                data = json.load(f)

            self._deserialize_inventory(data.get('inventory', {}))
            self.history = data.get('history', [])

            self.logger.info("Inventory loaded from %s", self.inventory_file)

        except FileNotFoundError:
            self.logger.info("No existing inventory file, starting fresh")
        except Exception as e:
            self.logger.error("Failed to load inventory: %s", e)

    def _serialize_inventory(self) -> Dict:
        """Convert inventory to JSON-serializable format"""
        serialized = {}
        for layer, zones in self.inventory.items():
            serialized[str(layer)] = {}
            for zone, items in zones.items():
                serialized[str(layer)][zone] = items
        return serialized

    def _deserialize_inventory(self, data: Dict):
        """Load inventory from JSON format"""
        self.inventory.clear()
        for layer_str, zones in data.items():
            layer = int(layer_str)
            for zone, items in zones.items():
                self.inventory[layer][zone] = items

    def print_summary(self):
        """Print inventory summary"""
        self.logger.info("=" * 60)
        self.logger.info("INVENTORY SUMMARY")
        self.logger.info("=" * 60)

        summary = self.get_product_summary()
        if not summary:
            self.logger.info("Inventory is empty")
        else:
            for product_id, count in sorted(summary.items()):
                self.logger.info("Product %d: %d items", product_id, count)

        self.logger.info("=" * 60)

    def clear(self):
        """Clear all inventory (use with caution!)"""
        self.inventory.clear()
        self.history.clear()
        self.save()
        self.logger.warning("Inventory cleared!")
