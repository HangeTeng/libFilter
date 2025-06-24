# src/filter/iblt.py

from __future__ import annotations
import unittest
import argparse
import sys
from typing import Set, Tuple, List, Optional, cast

from .base import FilterItem, PeelableSymbol, StandardFilter, FilterBase
from .utils import InputType, HashMapping, Hasher, _xor_bytes, _inputtype_to_bytes

# --- IBLT-specific Types ---

class IBLTItem(FilterItem):
    """
    An item for an Invertible Bloom Lookup Table (IBLT), containing both a
    key and an associated value.
    """
    __slots__ = ('key', 'value')

    def __init__(self, key: InputType, value: bytes):
        if not isinstance(value, bytes):
            raise TypeError("IBLT item value must be bytes.")
        self.key = key
        self.value = value

    def get_key(self) -> InputType:
        return self.key

    def __eq__(self, other):
        if not isinstance(other, IBLTItem):
            return NotImplemented
        return self.key == other.key and self.value == other.value
    
    def __hash__(self):
        return hash((self.key, self.value))

    def __repr__(self) -> str:
        # Show truncated value for readability
        val_repr = self.value[:16].hex() + ('...' if len(self.value) > 16 else '')
        return f"IBLTItem(key={self.key!r}, value=0x{val_repr})"

class IBLTSymbol(PeelableSymbol):
    """
    A cell in an IBLT, tracking a count, a key XOR sum, and a value XOR sum.
    """
    __slots__ = ('count', 'key_sum', 'value_sum')

    def __init__(self, count: int = 0, key_sum: bytes = b'', value_sum: bytes = b''):
        self.count: int = count
        self.key_sum: bytes = key_sum
        self.value_sum: bytes = value_sum

    def __iadd__(self, other: IBLTSymbol) -> IBLTSymbol:
        self.count += other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self

    def __isub__(self, other: IBLTSymbol) -> IBLTSymbol:
        self.count -= other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self

    def is_empty(self) -> bool:
        return self.count == 0 and not self.key_sum and not self.value_sum

    def is_pure(self, hasher: Hasher) -> bool:
        """
        A symbol is pure if it represents a single item. This is true if:
        1. The count is 1 or -1.
        2. The hash of the key_sum matches the hash of the recovered key.
           (This is a simplified check, a hash of value could also be used).
        
        This implementation assumes the key itself is stored in `key_sum`.
        A more robust IBLT might store a hash of the key instead.
        """
        if self.count == 1 or self.count == -1:
            # key_sum actually holds the key itself when pure
            recovered_key = self.key_sum 
            # In a real IBLT, you'd typically XOR a dedicated `key_hash` field.
            # Here we just check if the key_sum is non-empty. In a robust system,
            # you'd use a checksum or a second hash.
            # For this simplified example, we'll trust the count.
            return bool(self.key_sum)
        return False
    
    def get_key(self) -> Optional[InputType]:
        """Recovers the key if the symbol is pure."""
        # For simplicity, we assume the key was an int or str that became bytes.
        # This is a limitation of this simplified example.
        # A full implementation would need type info or fixed-width keys.
        try:
            return int.from_bytes(self.key_sum, 'little')
        except (ValueError, TypeError):
            try:
                return self.key_sum.decode('utf-8')
            except UnicodeDecodeError:
                return self.key_sum # Return as bytes if all else fails

    def __str__(self) -> str:
        key_sum_hex = self.key_sum[:8].hex() + '...' if len(self.key_sum) > 8 else self.key_sum.hex()
        val_sum_hex = self.value_sum[:8].hex() + '...' if len(self.value_sum) > 8 else self.value_sum.hex()
        return (f"count={self.count}, key_sum=0x{key_sum_hex}, "
                f"value_sum=0x{val_sum_hex}")

    @classmethod
    def _get_key(cls, item: IBLTItem) -> InputType:
        return item.get_key()
    
    @classmethod
    def from_item(cls, item: IBLTItem) -> IBLTSymbol:
        """Creates a source symbol from a single IBLTItem."""
        key_bytes = _inputtype_to_bytes(item.key)
        return cls(count=1, key_sum=key_bytes, value_sum=item.value)

    @classmethod
    def to_item(cls, symbol: IBLTSymbol) -> IBLTItem:
        """Recovers the original IBLTItem from a pure symbol."""
        if not (symbol.count == 1 or symbol.count == -1):
             raise ValueError("Cannot convert non-pure symbol to item.")
        
        key = symbol.get_key()
        if key is None:
            raise ValueError("Could not decode key from symbol.")
            
        value = symbol.value_sum
        
        # If the symbol represents a removed item, the values are effectively inverted.
        # We don't need to do anything here as the subtraction in peel() handles this.
        return IBLTItem(key, value)

# --- IBLT Implementation ---
class IBLT(StandardFilter[IBLTSymbol, IBLTItem]):
    """
    An Invertible Bloom Lookup Table (IBLT), capable of storing key-value
    pairs and recovering set differences.
    """
    symbol_type = IBLTSymbol

    def peel(self) -> Tuple[Set[IBLTItem], Set[IBLTItem]]:
        """
        Decodes the IBLT to find the set differences.

        This is a destructive operation on a copy of the filter.

        Returns:
            A tuple containing (items_added, items_removed).
        """
        decoder = self.__copy__() # Work on a copy
        
        added: Set[IBLTItem] = set()
        removed: Set[IBLTItem] = set()
        
        # We need a hasher to verify purity, but IBLT doesn't store one by default.
        # We can create a dummy one. The `is_pure` implementation here is simple
        # and doesn't rely on it, but a more robust one would.
        dummy_hasher = Hasher(seed=0)
        
        while True:
            pure_indices: List[int] = []
            for i, cell in enumerate(decoder.cells):
                if cell.is_pure(dummy_hasher):
                    pure_indices.append(i)
            
            if not pure_indices:
                break # No more pure cells, decoding stops

            for index in pure_indices:
                pure_cell = decoder.cells[index]
                if pure_cell.is_empty():
                    continue

                # Recover the item from the pure cell
                item = self.__class__.symbol_type.to_item(pure_cell)
                
                # Check if it was added or removed
                if pure_cell.count == 1:
                    added.add(item)
                else: # count == -1
                    removed.add(item)
                
                # Subtract the recovered item from the decoder to reveal more pure cells
                decoder.remove(item)

        # Check for failure
        if not all(cell.is_empty() for cell in decoder.cells):
            print("Warning: IBLT decoding failed. Not all cells could be cleared.", file=sys.stderr)

        return added, removed

    def __copy__(self) -> IBLT:
        """Create a deep copy for decoding."""
        new_iblt = IBLT(self.hash_mapping)
        new_iblt.cells = [
            IBLTSymbol(c.count, c.key_sum, c.value_sum) for c in self.cells
        ]
        return new_iblt

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

class TestIBLT(VerboseTestCase):
    def setUp(self):
        seeds = ['S1', 'S2', 'S3', 'S4']
        self.m, self.k = 20, len(seeds)
        self.hash_map = HashMapping.from_seeds(seeds, self.m)
        self.iblt = IBLT(self.hash_map)
        
        # Sets for two parties, A and B
        self.set_a = {
            IBLTItem("apple", b"v1"),
            IBLTItem("banana", b"v2"),
            IBLTItem("common", b"v_common")
        }
        self.set_b = {
            IBLTItem("grape", b"v3"),
            IBLTItem("orange", b"v4"),
            IBLTItem("common", b"v_common")
        }

    def test_basic_push_remove(self):
        """Tests that push and remove correctly modify the cells."""
        item = IBLTItem("test", b"val")
        
        self.iblt.push(item)
        self._print_filter_state(self.iblt, "After pushing 'test'")
        non_empty_cells_after_push = sum(1 for c in self.iblt.cells if not c.is_empty())
        self.assertEqual(non_empty_cells_after_push, self.k)
        
        self.iblt.remove(item)
        self._print_filter_state(self.iblt, "After removing 'test'")
        all_empty = all(c.is_empty() for c in self.iblt.cells)
        self.assertTrue(all_empty, "All cells should be empty after push and remove")

    def test_peel_simple_difference(self):
        """Tests decoding a simple set difference."""
        iblt_a = IBLT(self.hash_map)
        for item in self.set_a:
            iblt_a.push(item)
        self._print_filter_state(iblt_a, "IBLT for Set A")

        iblt_b = IBLT(self.hash_map)
        for item in self.set_b:
            iblt_b.push(item)
        self._print_filter_state(iblt_b, "IBLT for Set B")

        # The difference IBLT represents items in A but not B (added),
        # and items in B but not A (removed).
        diff_iblt = iblt_a - iblt_b
        self._print_filter_state(diff_iblt, "Difference IBLT (A - B)")

        added, removed = diff_iblt.peel()

        expected_added = self.set_a - self.set_b
        expected_removed = self.set_b - self.set_a
        
        self.assertEqual(added, expected_added)
        self.assertEqual(removed, expected_removed)

    def test_peel_idempotent(self):
        """Tests that peeling a filter with no differences yields empty sets."""
        self.iblt.push(IBLTItem("key1", b"v1"))
        self.iblt.remove(IBLTItem("key1", b"v1"))
        self._print_filter_state(self.iblt, "After push/remove same item")
        
        added, removed = self.iblt.peel()
        self.assertEqual(len(added), 0)
        self.assertEqual(len(removed), 0)

    def test_peel_failure_case(self):
        """Tests a likely decoding failure due to high load."""
        # Use a very small table to force collisions
        small_map = HashMapping.from_seeds(['s1', 's2'], 4)
        fail_iblt = IBLT(small_map)
        
        # Add many items, likely causing no pure cells
        items = [IBLTItem(f"key{i}", f"val{i}".encode()) for i in range(10)]
        for item in items:
            fail_iblt.push(item)
        
        self._print_filter_state(fail_iblt, "Overloaded IBLT likely to fail decoding")
        
        # Redirect stderr to check for the warning message
        import io
        from contextlib import redirect_stderr
        f = io.StringIO()
        with redirect_stderr(f):
            added, removed = fail_iblt.peel()
        
        # Check that the warning was printed
        self.assertIn("decoding failed", f.getvalue())
        # The decoded sets will likely not contain all items
        self.assertNotEqual(len(added), len(items))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the IBLT.")
    parser.add_argument(
        '-vf', '--verbose-filters', action='store_true',
        help="Print filter state after each modification during tests."
    )

    args, unknown = parser.parse_known_args()
    cli_args = args
    main_args = [sys.argv[0]] + unknown

    print("\n" + "#" * 70)
    print("###" + " " * 26 + "RUNNING IBLT TESTS" + " " * 26 + "###")
    print("#" * 70)
    if cli_args.verbose_filters:
        print("### Verbose filter state printing: ENABLED")
        print("#" * 70)
    
    unittest.main(argv=main_args, verbosity=2, exit=False)