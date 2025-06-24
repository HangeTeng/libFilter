from __future__ import annotations
import unittest
import argparse
import sys
from .base import FilterItem, StandardFilter, FilterSymbol, FilterBase
from .utils import InputType, HashMapping

# --- BF-specific Types ---
class BFItem(FilterItem):
    """An item containing a simple key for a Bloom Filter."""
    __slots__ = ('key',)

    def __init__(self, key: InputType):
        self.key = key

    def get_key(self) -> InputType:
        return self.key
    
    def __repr__(self) -> str:
        return f"BFItem(key={self.key!r})"

class BFSymbol(FilterSymbol):
    """A Bloom Filter cell, representing a single bit."""
    __slots__ = ('set',)

    def __init__(self):
        self.set: bool = False

    def __iadd__(self, other: BFSymbol) -> BFSymbol:
        if other.set:
            self.set = True
        return self

    def __isub__(self, other: BFSymbol) -> BFSymbol:
        raise NotImplementedError("Bloom Filters do not support removal.")

    def is_empty(self) -> bool:
        return not self.set

    def __str__(self) -> str:
        return "SET" if self.set else "EMPTY"

    @classmethod
    def _get_key(cls, item: BFItem) -> InputType:
        return item.get_key()
    
    @classmethod
    def from_item(cls, item: BFItem) -> BFSymbol:
        source = cls()
        source.set = True
        return source

# --- BF Implementation ---
class BF(StandardFilter[BFSymbol, BFItem]):
    """A classic Bloom Filter (BF)."""
    symbol_type = BFSymbol

    def __contains__(self, item: BFItem) -> bool:
        """Checks for the presence of an item (allows for false positives)."""
        if not self.m:
            return False
        key = self.__class__.symbol_type._get_key(item)
        return all(self.cells[i].set for i in self._get_indices(key))

# --- In-module Tests ---
cli_args = None

class VerboseTestCase(unittest.TestCase):
    """A test case that can print filter states based on a CLI flag."""
    def _print_filter_state(self, filter_instance: FilterBase, stage: str):
        if not (cli_args and cli_args.verbose_filters):
            return
        
        print("\n" + "=" * 70)
        print(f"  [VERBOSE] Filter State Snapshot\n  Test: {self.id()}\n  Stage: {stage}")
        print("-" * 70)
        print(filter_instance.to_string(verbose=False))
        print("=" * 70 + "\n")

def _get_occupied_indices(bf: BF, items: list[BFItem]) -> set[int]:
    """Helper to compute all indices that should be set for a list of items."""
    return {index for item in items for index in bf._get_indices(bf.symbol_type._get_key(item))}

class TestBloomFilter(VerboseTestCase):
    def setUp(self):
        # Use meaningful string seeds for different hash functions
        seeds = ['HH', 'MM', 'BH']
        self.m, self.k = 100, len(seeds)
        self.hash_map = HashMapping.from_seeds(seeds, self.m)
        self.bf = BF(self.hash_map)
        self.items_present = [BFItem("apple"), BFItem(123)]
        self.items_absent = [BFItem("cherry"), BFItem(456)]
    
    def test_membership_check(self):
        """Verifies basic 'in' operator functionality."""
        for item in self.items_present:
            self.bf.push(item)
        self._print_filter_state(self.bf, "After pushing initial items")

        for item in self.items_present:
            self.assertIn(item, self.bf)
        for item in self.items_absent:
            self.assertNotIn(item, self.bf)

    def test_internal_state_after_push(self):
        """Ensures filter's internal cells are correctly modified."""
        self.bf.push(self.items_present[0])
        self._print_filter_state(self.bf, f"After pushing '{self.items_present[0].key}'")
        
        self.bf.push(self.items_present[1])
        self._print_filter_state(self.bf, f"After pushing '{self.items_present[1].key}'")
        
        expected_indices = _get_occupied_indices(self.bf, self.items_present)
        for i, cell in enumerate(self.bf.cells):
            self.assertEqual(cell.set, i in expected_indices)

    def test_removal_not_supported(self):
        """Confirms that remove() raises the correct exception."""
        self.bf.push(self.items_present[0])
        self._print_filter_state(self.bf, "Before attempting removal")
        with self.assertRaises(NotImplementedError):
            self.bf.remove(self.items_present[0])
        self._print_filter_state(self.bf, "After attempting removal (state unchanged)")

    def test_string_representation(self):
        """Checks the output format of __str__ and to_string."""
        for item in self.items_present:
            self.bf.push(item)
        
        str_output = str(self.bf)
        expected_indices = _get_occupied_indices(self.bf, self.items_present)
        self.assertIn(f"Summary: {len(expected_indices)}/{self.m}", str_output)
        self.assertIn("- Index", str_output)
        self.assertNotIn("EMPTY", str_output)
        
        verbose_output = self.bf.to_string(verbose=True)
        self.assertIn("SET", verbose_output)
        self.assertIn("EMPTY", verbose_output)
    
    def test_repr_representation(self):
        """Validates the __repr__ output for clarity and correctness."""
        repr_output = repr(self.bf)
        self.assertIn("BF(hash_mapping=HashMapping", repr_output)
        self.assertIn("seed='HH'", repr_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the Bloom Filter.")
    parser.add_argument(
        '-vf', '--verbose-filters', action='store_true',
        help="Print filter state after each modification during tests."
    )
    
    args, unknown = parser.parse_known_args()
    cli_args = args
    main_args = [sys.argv[0]] + unknown
    
    print("\n" + "#" * 70)
    print("###" + " " * 21 + "RUNNING BLOOM FILTER TESTS" + " " * 21 + "###")
    print("#" * 70)
    if cli_args.verbose_filters:
        print("### Verbose filter state printing: ENABLED")
        print("#" * 70)
    
    unittest.main(argv=main_args, verbosity=2, exit=False)