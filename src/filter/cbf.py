from __future__ import annotations
import unittest
import argparse
import sys
from .base import FilterItem, StandardFilter, FilterSymbol, FilterBase
from .utils import InputType, HashMapping

# --- CBF-specific Types ---
class CBFItem(FilterItem):
    """An item containing a simple key for a Counting Bloom Filter."""
    __slots__ = ('key',)

    def __init__(self, key: InputType):
        self.key = key

    def get_key(self) -> InputType:
        return self.key
    
    def __repr__(self) -> str:
        return f"CBFItem(key={self.key!r})"

class CBFSymbol(FilterSymbol):
    """A Counting Bloom Filter cell, representing an integer counter."""
    __slots__ = ('count',)

    def __init__(self, count: int = 0):
        self.count: int = count

    def __iadd__(self, other: CBFSymbol) -> CBFSymbol:
        self.count += other.count
        return self

    def __isub__(self, other: CBFSymbol) -> CBFSymbol:
        self.count -= other.count
        return self

    def is_empty(self) -> bool:
        return self.count == 0

    def __str__(self) -> str:
        return f"count={self.count}"

    @classmethod
    def _get_key(cls, item: CBFItem) -> InputType:
        return item.get_key()
    
    @classmethod
    def from_item(cls, item: CBFItem) -> CBFSymbol:
        return cls(count=1)

# --- CBF Implementation ---
class CBF(StandardFilter[CBFSymbol, CBFItem]):
    """A classic Counting Bloom Filter (CBF)."""
    symbol_type = CBFSymbol

    def __contains__(self, item: CBFItem) -> bool:
        """Checks for the presence of an item (allows for false positives)."""
        if not self.m:
            return False
        key = self.__class__.symbol_type._get_key(item)
        return all(self.cells[i].count > 0 for i in self._get_indices(key))

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

class TestCountingBloomFilter(VerboseTestCase):
    def setUp(self):
        # Use meaningful string seeds for different hash functions
        seeds = ['HH', 'MM', 'BH'] 
        self.m, self.k = 100, len(seeds)
        self.hash_map = HashMapping.from_seeds(seeds, self.m)
        self.cbf = CBF(self.hash_map)
        self.items = [CBFItem("apple"), CBFItem("banana"), CBFItem("apple")]
    
    def test_push_and_membership(self):
        """Verifies item addition and membership checking."""
        apple, banana, _ = self.items
        
        self.cbf.push(apple)
        self._print_filter_state(self.cbf, "After pushing 'apple' once")
        self.assertIn(apple, self.cbf)
        
        self.cbf.push(banana)
        self._print_filter_state(self.cbf, "After pushing 'banana'")
        self.assertIn(banana, self.cbf)
        
        self.cbf.push(apple)
        self._print_filter_state(self.cbf, "After pushing 'apple' a second time")
        self.assertIn(apple, self.cbf)

    def test_removal(self):
        """Verifies that item removal correctly decrements counters."""
        apple, banana, _ = self.items

        self.cbf.push(apple)
        self.cbf.push(apple)
        self.cbf.push(banana)
        self._print_filter_state(self.cbf, "Initial state: 2 apples, 1 banana")
        self.assertTrue(apple in self.cbf and banana in self.cbf)

        self.cbf.remove(apple)
        self._print_filter_state(self.cbf, "After removing 'apple' once")
        self.assertIn(apple, self.cbf)
        
        self.cbf.remove(apple)
        self._print_filter_state(self.cbf, "After removing 'apple' twice")
        self.assertNotIn(apple, self.cbf)
        self.assertIn(banana, self.cbf)
        
        self.cbf.remove(banana)
        self._print_filter_state(self.cbf, "After removing 'banana'")
        self.assertNotIn(banana, self.cbf)
        self.assertTrue(all(c.is_empty() for c in self.cbf.cells))

    def test_internal_counts(self):
        """Directly checks the counter values in cells."""
        apple, banana, _ = self.items
        self.cbf.push(apple)
        self.cbf.push(apple)
        self.cbf.push(banana)
        self._print_filter_state(self.cbf, "State for count verification")

        apple_indices = list(self.cbf._get_indices(apple.key))
        banana_indices = list(self.cbf._get_indices(banana.key))
        
        expected_counts = {}
        for i in apple_indices:
            expected_counts[i] = expected_counts.get(i, 0) + 2
        for i in banana_indices:
            expected_counts[i] = expected_counts.get(i, 0) + 1

        for i, cell in enumerate(self.cbf.cells):
            self.assertEqual(cell.count, expected_counts.get(i, 0))

    def test_string_representation(self):
        """Checks the output format of __str__ and to_string."""
        self.cbf.push(self.items[0])
        str_output = str(self.cbf)
        self.assertIn("Summary:", str_output)
        self.assertIn("- Index", str_output)
        self.assertIn("count=1", str_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the Counting Bloom Filter.")
    parser.add_argument(
        '-vf', '--verbose-filters', action='store_true',
        help="Print filter state after each modification during tests."
    )

    args, unknown = parser.parse_known_args()
    cli_args = args
    main_args = [sys.argv[0]] + unknown
    
    print("\n" + "#" * 70)
    print("###" + " " * 19 + "RUNNING COUNTING BLOOM FILTER TESTS" + " " * 18 + "###")
    print("#" * 70)
    if cli_args.verbose_filters:
        print("### Verbose filter state printing: ENABLED")
        print("#" * 70)
    
    unittest.main(argv=main_args, verbosity=2, exit=False)