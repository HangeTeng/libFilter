# src/filter/iblt.py

from __future__ import annotations
import unittest
import argparse
import sys
from typing import Set, Tuple, List, Dict, Any

from .base import FilterItem, PeelableSymbol, StandardFilter, FilterBase
from .utils import InputType, Hasher, _xor_bytes, _inputtype_to_bytes, HashMapping

# --- Type Serialization Helpers ---
class _DataType: BYTES, STR, INT = 0, 1, 2
class _TypedValue:
    __slots__ = ('value', 'type_id')
    def __init__(self, raw_value: InputType):
        if isinstance(raw_value, str): self.value, self.type_id = raw_value.encode('utf-8'), _DataType.STR
        elif isinstance(raw_value, int):
            if raw_value < 0: raise ValueError("Integers cannot be negative.")
            byte_len = (raw_value.bit_length() + 7) // 8 or 1
            self.value, self.type_id = raw_value.to_bytes(byte_len, 'little'), _DataType.INT
        elif isinstance(raw_value, bytes): self.value, self.type_id = raw_value, _DataType.BYTES
        else: raise TypeError(f"Unsupported type: {type(raw_value)}")
    def serialize(self) -> bytes: return self.type_id.to_bytes(1, 'little') + self.value
    @staticmethod
    def deserialize(data: bytes) -> InputType:
        if not data: return b''
        type_id, value_bytes = data[0], data[1:]
        if type_id == _DataType.STR: return value_bytes.decode('utf-8')
        if type_id == _DataType.INT: return int.from_bytes(value_bytes, 'little')
        if type_id == _DataType.BYTES: return value_bytes
        raise ValueError(f"Unknown type_id: {type_id}")

# --- IBLT-specific Types ---
class IBLTItem(FilterItem):
    __slots__ = ('key', 'value')
    def __init__(self, key: InputType, value: InputType): self.key, self.value = key, value
    def get_key(self) -> InputType: return self.key
    def __eq__(self, other):
        if not isinstance(other, IBLTItem): return NotImplemented
        return self.key == other.key and self.value == other.value
    def __hash__(self):
        return hash((_inputtype_to_bytes(self.key), _inputtype_to_bytes(self.value)))
    def __repr__(self) -> str: return f"IBLTItem(key={self.key!r}, value={self.value!r})"

class IBLTSymbol(PeelableSymbol):
    __slots__ = ('count', 'key_sum', 'value_sum')
    def __init__(self, count: int = 0, key_sum: bytes = b'', value_sum: bytes = b''):
        self.count, self.key_sum, self.value_sum = count, key_sum, value_sum
    def __iadd__(self, other: IBLTSymbol) -> IBLTSymbol:
        self.count += other.count; self.key_sum = _xor_bytes(self.key_sum, other.key_sum); self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self
    def __isub__(self, other: IBLTSymbol) -> IBLTSymbol:
        self.count -= other.count; self.key_sum = _xor_bytes(self.key_sum, other.key_sum); self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self
    def is_empty(self) -> bool:
        return self.count == 0 and all(b==0 for b in self.key_sum) and all(b==0 for b in self.value_sum)
    def is_pure(self) -> bool: return self.count == 1 or self.count == -1
    def __getstate__(self) -> Dict[str, Any]:
        return {'count': self.count, 'key_sum': self.key_sum, 'value_sum': self.value_sum}
    def __str__(self) -> str:
        k_hex, v_hex = self.key_sum[:4].hex(), self.value_sum[:4].hex()
        return f"cnt={self.count}, k_sum=0x{k_hex}..., v_sum=0x{v_hex}..."
    @classmethod
    def _get_key(cls, item: IBLTItem) -> InputType: return item.get_key()
    @classmethod
    def from_item(cls, item: IBLTItem, **kwargs) -> IBLTSymbol:
        return cls(1, _TypedValue(item.key).serialize(), _TypedValue(item.value).serialize())
    @classmethod
    def to_item(cls, symbol: "IBLTSymbol") -> IBLTItem:
        if not (symbol.count == 1 or symbol.count == -1): raise ValueError("Non-pure")
        return IBLTItem(_TypedValue.deserialize(symbol.key_sum), _TypedValue.deserialize(symbol.value_sum))

# --- IBLT Implementation ---
class IBLT(StandardFilter[IBLTSymbol, IBLTItem]):
    symbol_type = IBLTSymbol
    def peel(self, destructive: bool = False) -> Tuple[Set[IBLTItem], Set[IBLTItem]]:
        decoder = self if destructive else self.copy()
        added, removed = set(), set()
        pure_queue = [i for i, c in enumerate(decoder.cells) if c.is_pure()]
        processed = set(pure_queue)
        while pure_queue:
            cell = decoder.cells[pure_queue.pop(0)]
            if not cell.is_pure(): continue
            item = decoder.symbol_type.to_item(cell)
            (added if cell.count == 1 else removed).add(item)
            decoder.remove(item)
            for idx in decoder._get_indices(item.get_key()):
                if idx not in processed and decoder.cells[idx].is_pure():
                    pure_queue.append(idx); processed.add(idx)
        if not all(c.is_empty() for c in decoder.cells):
            print("Warning: IBLT decoding may be incomplete due to unresolvable collisions.")
        return added, removed

# --- In-module Tests ---
cli_args = None
class VerboseTestCase(unittest.TestCase):
    def _print_filter_state(self, filter_instance: FilterBase, stage: str):
        if cli_args and cli_args.verbose_filters:
            print(f"\n--- [State after: {stage}] ---\n{filter_instance.to_string(False)}")

class TestIBLT(VerboseTestCase):
    def setUp(self):
        self.hash_map = HashMapping.from_seeds(['S1', 'S2', 'S3'], 40)
        self.set_a = {IBLTItem("apple", "red"), IBLTItem(123, 456)}
        self.set_b = {IBLTItem("grape", "purple"), IBLTItem("apple", "red")}
        # A set with the same key as set_a but a different value
        self.set_c_diff_val = {IBLTItem("apple", "green"), IBLTItem(123, 456)}

    def test_copy_method(self):
        iblt = IBLT(self.hash_map)
        iblt.push(IBLTItem("key", "val"))
        iblt_copy = iblt.copy()
        self.assertNotEqual(id(iblt), id(iblt_copy))
        self.assertEqual(iblt.to_dict(), iblt_copy.to_dict())

    def test_peel_logic(self):
        iblt = IBLT(self.hash_map)
        for item in self.set_a: iblt.push(item)
        original_state = iblt.to_dict()
        added, _ = iblt.peel(destructive=False)
        self.assertEqual(added, self.set_a)
        self.assertEqual(iblt.to_dict(), original_state)
        added_d, _ = iblt.peel(destructive=True)
        self.assertEqual(added_d, self.set_a)
        self.assertTrue(all(c.is_empty() for c in iblt.cells))

    def test_set_difference(self):
        iblt_a = IBLT(self.hash_map); [iblt_a.push(item) for item in self.set_a]
        iblt_b = IBLT(self.hash_map); [iblt_b.push(item) for item in self.set_b]
        diff_iblt = iblt_a - iblt_b
        added, removed = diff_iblt.peel()
        self.assertEqual(added, self.set_a - self.set_b)
        self.assertEqual(removed, self.set_b - self.set_a)

    def test_value_difference_causes_cancellation(self):
        """
        Tests the IBLT's behavior with same keys but different values.
        
        This test demonstrates a key property of this IBLT implementation:
        it assumes a one-to-one mapping between a key and its value. When
        two items with the same key but different values are differenced,
        their `count` and `key_sum` cancel out (1-1=0, k^k=0), leaving a
        non-zero `value_sum`. This results in a non-pure cell (`count=0`)
        that cannot be decoded by the peel algorithm.
        """
        iblt_a = IBLT(self.hash_map)
        for item in self.set_a: iblt_a.push(item)

        iblt_c = IBLT(self.hash_map)
        for item in self.set_c_diff_val: iblt_c.push(item)
        
        diff_iblt = iblt_a - iblt_c
        self._print_filter_state(diff_iblt, "A - C (value difference)")

        added, removed = diff_iblt.peel()

        # The item 'apple'/'red' from set_a and 'apple'/'green' from set_c
        # have cancelled each other out, so they do NOT appear in the result.
        # The only item that doesn't have a matching key is 123/456, but since
        # its counterpart in set_c has the same value, it also cancels out.
        # In this specific case, nothing can be decoded.
        
        # We expect both sets to be empty because no pure cells can be found.
        self.assertEqual(added, set())
        self.assertEqual(removed, set())

        # Let's create a more illustrative case
        set_x = {IBLTItem("unique_x", 1)}
        set_y = {IBLTItem("apple", "red")}
        set_z = {IBLTItem("apple", "green"), IBLTItem("unique_x", 1)}
        
        ix = IBLT(self.hash_map); [ix.push(i) for i in set_x]
        iyz = IBLT(self.hash_map); [iyz.push(i) for i in set_y | set_z]

        # The difference should only contain the items whose keys are unique
        diff = iyz - ix
        added, removed = diff.peel()

        # 'unique_x' is cancelled out. 'apple'/'red' and 'apple'/'green'
        # cannot be resolved. The result is empty.
        self.assertEqual(added, set())
        self.assertEqual(removed, set())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IBLT tests.")
    parser.add_argument('-vf', '--verbose-filters', action='store_true', help="Print filter state during tests.")
    cli_args, unknown = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0]] + unknown, verbosity=2, exit=False)