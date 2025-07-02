# src/filter/iblt4nn.py

from __future__ import annotations
import unittest
import argparse
import sys
from typing import Set, Tuple, List, Dict, Any
import galois

from ..core.prv import PRV
from ..core.base import FilterItem, PeelableSymbol, StandardFilter, FilterBase
from .utils import Hasher, HashMapping

# ... (NNItem, NNSymbol, IBLT4NN 类的定义保持不变) ...
class NNItem(FilterItem):
    __slots__ = ('idx', 'value')
    def __init__(self, idx: int, value: float): self.idx, self.value = idx, value
    def get_key(self) -> int: return self.idx
    def __eq__(self, other):
        if not isinstance(other, NNItem): return NotImplemented
        return self.idx == other.idx and abs(self.value - other.value) < 1e-9
    def __hash__(self) -> int: return hash((self.idx, round(self.value, 9)))
    def __repr__(self) -> str: return f"NNItem(idx={self.idx}, value={self.value:.4f})"
class NNSymbol(PeelableSymbol):
    __slots__ = ('GF', 'mask_sum', 'idx_code_sum', 'value_sum')
    def __init__(self, GF: galois.FieldClass, mask_sum=0, idx_code_sum=0, value_sum: float = 0.0):
        self.GF, self.mask_sum, self.idx_code_sum, self.value_sum = GF, GF(mask_sum), GF(idx_code_sum), value_sum
    def __iadd__(self, other: "NNSymbol") -> "NNSymbol":
        self.mask_sum += other.mask_sum; self.idx_code_sum += other.idx_code_sum; self.value_sum += other.value_sum
        return self
    def __isub__(self, other: "NNSymbol") -> "NNSymbol":
        self.mask_sum -= other.mask_sum; self.idx_code_sum -= other.idx_code_sum; self.value_sum -= other.value_sum
        return self
    def is_empty(self) -> bool:
        return self.mask_sum == 0 and self.idx_code_sum == 0 and abs(self.value_sum) < 1e-9
    def is_pure(self, prv: PRV) -> bool:
        if self.mask_sum == 0: return False
        potential_code = self.idx_code_sum / self.mask_sum
        decoded_idx = prv.index(int(potential_code))
        if decoded_idx is None: return False
        return potential_code == prv.entry(decoded_idx)
    def __getstate__(self) -> Dict[str, Any]:
        return {'mask_sum': int(self.mask_sum), 'idx_code_sum': int(self.idx_code_sum), 'value_sum': self.value_sum}
    def __str__(self) -> str:
        return f"mask={int(self.mask_sum)}, code={int(self.idx_code_sum)}, val={self.value_sum:.4f}"
    @classmethod
    def _get_key(cls, item: NNItem) -> int: return item.get_key()
    @classmethod
    def from_item(cls, item: NNItem, prv: PRV, r: galois.FieldArray) -> "NNSymbol":
        return cls(prv.GF, r, r * prv.entry(item.idx), item.value)
    @classmethod
    def to_item(cls, symbol: "NNSymbol", prv: PRV) -> NNItem:
        if not symbol.is_pure(prv): raise ValueError("Cannot convert non-pure symbol.")
        potential_code = symbol.idx_code_sum / symbol.mask_sum
        original_idx = prv.index(int(potential_code))
        if original_idx is None: raise ValueError("Failed to decode index.")
        return NNItem(original_idx, symbol.value_sum)
class IBLT4NN(StandardFilter[NNSymbol, NNItem]):
    symbol_type = NNSymbol
    def __init__(self, hash_mapping: HashMapping, prv: PRV, mask_seed: Any = "mask_seed"):
        self.cells: List[NNSymbol] = []
        self.hash_mapping, self.m, self.k, self.prv = hash_mapping, hash_mapping.table_size, len(hash_mapping.hashers), prv
        self._mask_hasher = Hasher(seed=mask_seed)
        self.cells = [self.symbol_type(prv.GF) for _ in range(self.m)]
    def _get_mask_for_idx(self, idx: int) -> galois.FieldArray:
        r_int = 0
        while r_int == 0: r_int = self._mask_hasher.digest_int(idx, nbytes=16)
        return self.prv.GF(r_int)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IBLT4NN":
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        prv_config = data['prv']
        prv = PRV(n=prv_config['n'], prp_type=prv_config['prp_type'], key=prv_config.get('key'))
        instance = cls(hash_mapping, prv, data['mask_seed'])
        GF = prv.GF
        instance.cells = [cls.symbol_type(GF, **s_data) for s_data in data['cells']]
        return instance
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict['prv'] = {'n': self.prv.n, 'prp_type': self.prv.prp_type, 'key': self.prv.key}
        base_dict['mask_seed'] = self._mask_hasher.seed
        return base_dict
    def push(self, item: NNItem) -> None:
        r = self._get_mask_for_idx(item.idx)
        source_symbol = self.symbol_type.from_item(item, prv=self.prv, r=r)
        for index in self._get_indices(item.get_key()): self.cells[index] += source_symbol
    def _peel_remove_symbol(self, symbol_to_peel: NNSymbol, original_key: int):
        for h_idx in self._get_indices(original_key): self.cells[h_idx] -= symbol_to_peel
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
                symbol_to_peel = self.symbol_type(decoder.prv.GF, cell.mask_sum, cell.idx_code_sum, cell.value_sum)
                decoder._peel_remove_symbol(symbol_to_peel, item.get_key())
                peeled_in_round = True
        if not all(c.is_empty() for c in decoder.cells): print("Warning: IBLT decoding may be incomplete.")
        return decoded_items
    def __sub__(self, other: FilterBase): raise NotImplementedError("IBLT4NN does not support subtraction.")
    def __isub__(self, other: FilterBase): raise NotImplementedError("IBLT4NN does not support subtraction.")
    def remove(self, item: NNItem): raise NotImplementedError("IBLT4NN is an aggregation-only filter.")

# --- In-module Tests ---
cli_args = None
class VerboseTestCase(unittest.TestCase):
    def _print_filter_state(self, filter_instance: FilterBase, stage: str):
        if cli_args and cli_args.verbose_filters:
            print(f"\n--- [State after: {stage}] ---\n{filter_instance.to_string(False)}")

# --- FIX: Inherit from VerboseTestCase ---
class TestIBLT4NN(VerboseTestCase):
    def setUp(self):
        self.n_indices = 2048
        self.hash_map = HashMapping.from_seeds(['NNS1', 'NNS2', 'NNS3'], 50)
        self.prv = PRV(n=self.n_indices, prp_type='aes128')
        self.client1 = [NNItem(10, 0.5), NNItem(1024, -0.1)]
        self.client2 = [NNItem(10, 0.2), NNItem(100, 0.3)]
        
    def test_aggregation_and_peel(self):
        """Tests aggregation and peeling with randomized masks."""
        agg_iblt = IBLT4NN(self.hash_map, self.prv)
        for item in self.client1: agg_iblt.push(item)
        for item in self.client2: agg_iblt.push(item)
        
        self._print_filter_state(agg_iblt, "After aggregation")
        
        decoded = agg_iblt.peel()
        expected = { 10: 0.5 + 0.2, 1024: -0.1, 100: 0.3 }
        
        self.assertEqual(len(decoded), len(expected))
        for idx, val in expected.items():
            self.assertIn(idx, decoded)
            self.assertAlmostEqual(decoded[idx], val, places=9)

    def test_merge_with_add(self):
        """Tests merging two IBLTs using + and += operators."""
        iblt1 = IBLT4NN(self.hash_map, self.prv)
        for item in self.client1: iblt1.push(item)
        iblt1_copy = iblt1.copy()

        iblt2 = IBLT4NN(self.hash_map, self.prv)
        for item in self.client2: iblt2.push(item)
        
        self._print_filter_state(iblt1, "IBLT 1 (before merge)")
        self._print_filter_state(iblt2, "IBLT 2 (before merge)")

        merged_iblt = iblt1 + iblt2
        self._print_filter_state(merged_iblt, "Merged IBLT (from `+`)")

        self.assertEqual(iblt1.to_dict(), iblt1_copy.to_dict(), "`+` should be non-destructive.")
        
        iblt1 += iblt2
        self.assertEqual(iblt1.to_dict(), merged_iblt.to_dict(), "`+=` should yield same result as `+`.")

    def test_serialization(self):
        """Tests that the IBLT with PRV can be serialized and deserialized."""
        iblt = IBLT4NN(self.hash_map, self.prv)
        for item in self.client1: iblt.push(item)
        config = iblt.to_dict()
        rebuilt_iblt = IBLT4NN.from_dict(config)
        self.assertEqual(iblt.to_dict(), rebuilt_iblt.to_dict())
        self.assertEqual(rebuilt_iblt.peel(), iblt.peel())

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