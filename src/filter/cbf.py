# src/filter/cbf.py

from __future__ import annotations
import unittest
import argparse
import sys
from typing import Dict, Any

from .base import FilterItem, StandardFilter, FilterSymbol, FilterBase
from .utils import InputType, HashMapping

# --- CBF-specific Types ---
class CBFItem(FilterItem):
    """An item containing a simple key for a Counting Bloom Filter."""
    __slots__ = ('key',)
    def __init__(self, key: InputType): self.key = key
    def get_key(self) -> InputType: return self.key
    def __repr__(self) -> str: return f"CBFItem(key={self.key!r})"

class CBFSymbol(FilterSymbol):
    """A Counting Bloom Filter cell, representing an integer counter."""
    __slots__ = ('count',)
    def __init__(self, count: int = 0): self.count = count
    def __iadd__(self, other: CBFSymbol) -> CBFSymbol:
        self.count += other.count
        return self
    def __isub__(self, other: CBFSymbol) -> CBFSymbol:
        self.count -= other.count
        return self
    def is_empty(self) -> bool: return self.count == 0
    def __getstate__(self) -> Dict[str, Any]: return {'count': self.count}
    def __str__(self) -> str: return f"count={self.count}"
    
    @classmethod
    def _get_key(cls, item: CBFItem) -> InputType: return item.get_key()
    @classmethod
    def from_item(cls, item: CBFItem, **kwargs) -> CBFSymbol: return cls(count=1)

# --- CBF Implementation ---
class CBF(StandardFilter[CBFSymbol, CBFItem]):
    """A Counting Bloom Filter (CBF) supporting additions and removals."""
    symbol_type = CBFSymbol

    def query(self, item: CBFItem) -> bool:
        """Checks for item presence (allows for false positives)."""
        if not self.m: return False
        key = self.__class__.symbol_type._get_key(item)
        return all(self.cells[i].count > 0 for i in self._get_indices(key))

    def estimate_count(self, item: CBFItem) -> int:
        """Estimates the count of an item (upper-bounded due to collisions)."""
        if not self.m: return 0
        key = self.__class__.symbol_type._get_key(item)
        return min(self.cells[i].count for i in self._get_indices(key))

# --- In-module Tests ---
cli_args = None

class VerboseTestCase(unittest.TestCase):
    def _print_filter_state(self, filter_instance: FilterBase, stage: str):
        if cli_args and cli_args.verbose_filters:
            print(f"\n--- [State after: {stage}] ---\n{filter_instance.to_string(False)}")

class TestCountingBloomFilter(VerboseTestCase):
    def setUp(self):
        self.hash_map = HashMapping.from_seeds(['S1', 'S2', 'S3'], 100)
        self.cbf = CBF(self.hash_map)
        self.apple = CBFItem("apple")
        self.banana = CBFItem("banana")

    def test_push_and_removal(self):
        self.cbf.push(self.apple)
        self.cbf.push(self.apple)
        self.cbf.push(self.banana)
        self._print_filter_state(self.cbf, "pushing 2 apples, 1 banana")
        
        self.assertTrue(self.cbf.query(self.apple))
        self.assertEqual(self.cbf.estimate_count(self.apple), 2)
        self.assertTrue(self.cbf.query(self.banana))
        
        self.cbf.remove(self.apple)
        self._print_filter_state(self.cbf, "removing 1 apple")
        self.assertTrue(self.cbf.query(self.apple)) # Still one left
        self.assertEqual(self.cbf.estimate_count(self.apple), 1)

        self.cbf.remove(self.apple)
        self._print_filter_state(self.cbf, "removing 2nd apple")
        self.assertFalse(self.cbf.query(self.apple)) # Now gone

        self.cbf.remove(self.banana)
        self.assertTrue(all(c.is_empty() for c in self.cbf.cells))

    def test_difference(self):
        cbf_a = CBF(self.hash_map)
        cbf_a.push(self.apple)
        cbf_a.push(self.banana)

        cbf_b = CBF(self.hash_map)
        cbf_b.push(self.apple)

        diff = cbf_a - cbf_b
        self.assertFalse(diff.query(self.apple))
        self.assertTrue(diff.query(self.banana))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Counting Bloom Filter tests.")
    parser.add_argument('-vf', '--verbose-filters', action='store_true', help="Print filter state during tests.")
    cli_args, unknown = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0]] + unknown, verbosity=2, exit=False)