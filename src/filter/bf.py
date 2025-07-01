# src/filter/bf.py

from __future__ import annotations
import unittest
import argparse
import sys
from typing import Dict, Any

from .base import FilterItem, StandardFilter, FilterSymbol, FilterBase
from .utils import InputType, HashMapping

# --- BF-specific Types ---
class BFItem(FilterItem):
    """An item containing a simple key for a Bloom Filter."""
    __slots__ = ('key',)
    def __init__(self, key: InputType): self.key = key
    def get_key(self) -> InputType: return self.key
    def __repr__(self) -> str: return f"BFItem(key={self.key!r})"

class BFSymbol(FilterSymbol):
    """A Bloom Filter cell, representing a single bit."""
    __slots__ = ('set',)
    def __init__(self, set: bool = False): self.set = set
    def __iadd__(self, other: BFSymbol) -> BFSymbol:
        if other.set: self.set = True
        return self
    def __isub__(self, other: BFSymbol) -> BFSymbol:
        raise NotImplementedError("Bloom Filters do not support removal.")
    def is_empty(self) -> bool: return not self.set
    def __getstate__(self) -> Dict[str, Any]: return {'set': self.set}
    def __str__(self) -> str: return "SET" if self.set else "EMPTY"

    @classmethod
    def _get_key(cls, item: BFItem) -> InputType: return item.get_key()
    @classmethod
    def from_item(cls, item: BFItem, **kwargs) -> BFSymbol: return cls(set=True)

# --- BF Implementation ---
class BF(StandardFilter[BFSymbol, BFItem]):
    """A classic Bloom Filter (BF)."""
    symbol_type = BFSymbol

    def __contains__(self, item: BFItem) -> bool:
        """Checks for item presence (allows for false positives)."""
        if not self.m: return False
        key = self.__class__.symbol_type._get_key(item)
        return all(self.cells[i].set for i in self._get_indices(key))

# --- In-module Tests ---
cli_args = None

class VerboseTestCase(unittest.TestCase):
    def _print_filter_state(self, filter_instance: FilterBase, stage: str):
        if cli_args and cli_args.verbose_filters:
            print(f"\n--- [State after: {stage}] ---\n{filter_instance.to_string(False)}")

class TestBloomFilter(VerboseTestCase):
    def setUp(self):
        self.hash_map = HashMapping.from_seeds(['S1', 'S2', 'S3'], 100)
        self.bf = BF(self.hash_map)
        self.items_present = [BFItem("apple"), BFItem(123)]
        self.items_absent = [BFItem("cherry"), BFItem(456)]
    
    def test_membership_check(self):
        for item in self.items_present: self.bf.push(item)
        self._print_filter_state(self.bf, "pushing initial items")

        for item in self.items_present: self.assertIn(item, self.bf)
        for item in self.items_absent: self.assertNotIn(item, self.bf)

    def test_removal_not_supported(self):
        with self.assertRaises(NotImplementedError):
            self.bf.remove(self.items_present[0])
    
    def test_copy_and_merge(self):
        self.bf.push(self.items_present[0])
        bf_copy = self.bf.copy()
        self.bf.push(self.items_present[1])
        
        bf_copy += self.bf
        self.assertIn(self.items_present[0], bf_copy)
        self.assertIn(self.items_present[1], bf_copy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bloom Filter tests.")
    parser.add_argument('-vf', '--verbose-filters', action='store_true', help="Print filter state during tests.")
    cli_args, unknown = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0]] + unknown, verbosity=2, exit=False)