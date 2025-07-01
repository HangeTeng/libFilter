# src/filter/iblt4nn.py

from __future__ import annotations
import unittest
import argparse
import sys
from typing import Set, Tuple, List, Dict, Any

from .prv import PRV
from .base import FilterItem, PeelableSymbol, StandardFilter, FilterBase
from .utils import Hasher, HashMapping

# --- IBLT for Neural Networks, using PRV for peeling ---

class NNItem(FilterItem):
    """Represents an item for the IBLT: an integer index and a float value."""
    __slots__ = ('idx', 'value')
    def __init__(self, idx: int, value: float): self.idx, self.value = idx, value
    def get_key(self) -> int: return self.idx
    def __eq__(self, other):
        if not isinstance(other, NNItem): return NotImplemented
        return self.idx == other.idx and abs(self.value - other.value) < 1e-9
    def __hash__(self) -> int: return hash((self.idx, round(self.value, 9)))
    def __repr__(self) -> str: return f"NNItem(idx={self.idx}, value={self.value:.4f})"

class NNSymbol(PeelableSymbol):
    """A cell for an IBLT that uses a PRV to encode indices for verifiable peeling."""
    __slots__ = ('count', 'idx_code_sum', 'value_sum')

    def __init__(self, count: int = 0, idx_code_sum: int = 0, value_sum: float = 0.0):
        self.count, self.idx_code_sum, self.value_sum = count, idx_code_sum, value_sum

    def __iadd__(self, other: "NNSymbol") -> "NNSymbol":
        self.count += other.count; self.idx_code_sum += other.idx_code_sum; self.value_sum += other.value_sum
        return self
    def __isub__(self, other: "NNSymbol") -> "NNSymbol":
        self.count -= other.count; self.idx_code_sum -= other.idx_code_sum; self.value_sum -= other.value_sum
        return self

    def is_empty(self) -> bool:
        return self.count == 0 and self.idx_code_sum == 0 and abs(self.value_sum) < 1e-9
    def is_pure(self, prv: PRV) -> bool:
        if self.count == 0: return False
        avg_code = self.idx_code_sum / self.count
        if avg_code != int(avg_code): return False
        return prv.index(int(avg_code)) is not None

    def __getstate__(self) -> Dict[str, Any]:
        return {'count': self.count, 'idx_code_sum': self.idx_code_sum, 'value_sum': self.value_sum}
    def __str__(self) -> str:
        return f"cnt={self.count}, code_sum={self.idx_code_sum}, val_sum={self.value_sum:.4f}"

    @classmethod
    def _get_key(cls, item: NNItem) -> int: return item.get_key()
    @classmethod
    def from_item(cls, item: NNItem, prv: PRV) -> "NNSymbol":
        return cls(1, int(prv.entry(item.idx)), item.value)
    @classmethod
    def to_item(cls, symbol: "NNSymbol", prv: PRV) -> NNItem:
        if not symbol.is_pure(prv): raise ValueError("Cannot convert non-pure symbol.")
        avg_code = int(symbol.idx_code_sum / symbol.count)
        original_idx = prv.index(avg_code)
        if original_idx is None: raise ValueError("Failed to decode index.")
        return NNItem(original_idx, symbol.value_sum)

class IBLT4NN(StandardFilter[NNSymbol, NNItem]):
    """
    An aggregation-only IBLT for neural networks, using a PRV for verifiable peeling.
    It supports merging (`+`, `+=`) but not direct item removal or subtraction.
    """
    symbol_type = NNSymbol

    def __init__(self, hash_mapping: HashMapping, prv: PRV):
        super().__init__(hash_mapping)
        self.prv = prv

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IBLT4NN":
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        prv_config = data['prv']
        prv = PRV(n=prv_config['n'], prp_type=prv_config['prp_type'], key=prv_config.get('key'))
        instance = cls(hash_mapping, prv)
        instance.cells = [cls.symbol_type(**s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict['prv'] = {'n': self.prv.n, 'prp_type': self.prv.prp_type, 'key': self.prv.key}
        return base_dict
    
    def __sub__(self, other: FilterBase): raise NotImplementedError("IBLT4NN does not support subtraction.")
    def __isub__(self, other: FilterBase): raise NotImplementedError("IBLT4NN does not support subtraction.")
    def remove(self, item: NNItem): raise NotImplementedError("IBLT4NN is an aggregation-only filter.")

    def push(self, item: NNItem) -> None:
        source_symbol = self.symbol_type.from_item(item, prv=self.prv)
        for index in self._get_indices(item.get_key()): self.cells[index] += source_symbol
            
    def _peel_remove_symbol(self, symbol_to_peel: NNSymbol, original_key: int):
        """Internal helper to subtract a pure symbol's influence during peeling."""
        for h_idx in self._get_indices(original_key):
            self.cells[h_idx] -= symbol_to_peel
            
    def peel(self, destructive: bool = False) -> Dict[int, float]:
        decoder = self if destructive else self.copy()
        decoded_items: Dict[int, float] = {}
        
        peeled_in_round = True
        while peeled_in_round:
            peeled_in_round = False
            pure_indices = [i for i, c in enumerate(decoder.cells) if c.is_pure(decoder.prv)]
            
            for idx in pure_indices:
                cell = decoder.cells[idx]
                if not cell.is_pure(decoder.prv): continue
                
                item = decoder.symbol_type.to_item(cell, prv=decoder.prv)
                if item.idx in decoded_items: continue
                
                decoded_items[item.idx] = item.value
                
                symbol_to_peel = self.symbol_type(cell.count, cell.idx_code_sum, cell.value_sum)
                decoder._peel_remove_symbol(symbol_to_peel, item.get_key())
                
                peeled_in_round = True
        
        if not all(c.is_empty() for c in decoder.cells):
             print("Warning: IBLT decoding may be incomplete.")
        return decoded_items

# --- In-module Tests ---
cli_args = None
class VerboseTestCase(unittest.TestCase):
    def _print_filter_state(self, filter_instance: FilterBase, stage: str):
        if cli_args and cli_args.verbose_filters:
            print(f"\n--- [State after: {stage}] ---\n{filter_instance.to_string(False)}")

class TestIBLT4NN(VerboseTestCase):
    def setUp(self):
        self.n_indices = 2048
        self.hash_map = HashMapping.from_seeds(['NNS1', 'NNS2', 'NNS3'], 50)
        self.prv = PRV(n=self.n_indices, prp_type='aes128')
        self.client1 = [NNItem(10, 0.5), NNItem(1024, -0.1)]
        self.client2 = [NNItem(10, 0.2), NNItem(100, 0.3)]
        
    def test_aggregation_and_peel(self):
        """Tests aggregation and peeling."""
        agg_iblt = IBLT4NN(self.hash_map, self.prv)
        for item in self.client1: agg_iblt.push(item)
        for item in self.client2: agg_iblt.push(item)
        
        decoded = agg_iblt.peel()
        expected = { 10: 0.5 + 0.2, 1024: -0.1, 100: 0.3 }
        
        self.assertEqual(len(decoded), len(expected))
        for idx, val in expected.items():
            self.assertIn(idx, decoded)
            self.assertAlmostEqual(decoded[idx], val, places=9)

    # --- Re-added test cases ---
    def test_merge_with_add(self):
        """Tests merging two IBLTs using + and +=."""
        iblt1 = IBLT4NN(self.hash_map, self.prv)
        for item in self.client1: iblt1.push(item)
        iblt1_copy = iblt1.copy()

        iblt2 = IBLT4NN(self.hash_map, self.prv)
        for item in self.client2: iblt2.push(item)
        
        # Test `+` operator (non-destructive)
        merged_iblt = iblt1 + iblt2
        self.assertEqual(iblt1.to_dict(), iblt1_copy.to_dict(), "`+` should be non-destructive.")
        
        # Test `+=` operator (destructive)
        iblt1 += iblt2
        self.assertEqual(iblt1.to_dict(), merged_iblt.to_dict(), "`+=` should yield same result as `+`.")

    def test_serialization(self):
        """Tests that the IBLT with PRV can be serialized and deserialized."""
        iblt = IBLT4NN(self.hash_map, self.prv)
        for item in self.client1: iblt.push(item)
        config = iblt.to_dict()
        rebuilt_iblt = IBLT4NN.from_dict(config)
        self.assertEqual(iblt.to_dict(), rebuilt_iblt.to_dict())
    # --- End of re-added test cases ---

    def test_unsupported_operations(self):
        """Ensures that disabled operations raise errors."""
        iblt = IBLT4NN(self.hash_map, self.prv)
        with self.assertRaises(NotImplementedError): _ = iblt - iblt
        with self.assertRaises(NotImplementedError): iblt -= iblt
        with self.assertRaises(NotImplementedError): iblt.remove(self.client1[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IBLT4NN tests.")
    parser.add_argument('-vf', '--verbose-filters', action='store_true', help="Print filter state.")
    cli_args, unknown = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0]] + unknown, verbosity=2, exit=False)