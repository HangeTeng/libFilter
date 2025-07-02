# libFilter/filters/iblt4nn.py

"""
Implementation of an IBLT for Neural Network (NN) gradient aggregation.

This specialized IBLT is designed for scenarios like federated learning,
where multiple clients contribute numerical updates (e.g., gradients) for
various indices (e.g., weight identifiers). It securely aggregates these
updates without revealing individual contributions.

It relies on a Pseudo-Random Vector (PRV) and finite field arithmetic to
encode and decode the index-value pairs.
"""

from __future__ import annotations
from typing import Set, Tuple, List, Dict, Any, Type
import math

import galois

from ..core.prv import PRV
from ..core.base import FilterItem, PeelableSymbol, StandardFilter, FilterBase
from ..core.utils import Hasher, HashMapping

# --- IBLT4NN-specific Components ---

class NNItem(FilterItem):
    """An item representing a numerical update for a specific index."""
    __slots__ = ('idx', 'value')

    def __init__(self, idx: int, value: float):
        self.idx = idx
        self.value = value

    def get_key(self) -> int:
        """The item's index is used as its key for hashing."""
        return self.idx

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NNItem):
            return NotImplemented
        # Use math.isclose for robust float comparison.
        return self.idx == other.idx and math.isclose(self.value, other.value)

    def __hash__(self) -> int:
        return hash((self.idx, self.value))

    def __repr__(self) -> str:
        return f"NNItem(idx={self.idx}, value={self.value:.4f})"


class NNSymbol(PeelableSymbol):
    """A cell in an IBLT4NN, using finite field arithmetic for sums."""
    __slots__ = ('GF', 'mask_sum', 'idx_code_sum', 'value_sum')

    def __init__(
        self,
        GF: galois.FieldClass,
        mask_sum: Any = 0,
        idx_code_sum: Any = 0,
        value_sum: float = 0.0
    ):
        super().__init__(GF=GF, mask_sum=mask_sum, idx_code_sum=idx_code_sum, value_sum=value_sum)
        self.GF = GF
        self.mask_sum = GF(mask_sum)
        self.idx_code_sum = GF(idx_code_sum)
        self.value_sum = value_sum

    def __iadd__(self, other: "NNSymbol") -> "NNSymbol":
        self.mask_sum += other.mask_sum
        self.idx_code_sum += other.idx_code_sum
        self.value_sum += other.value_sum
        return self

    def __isub__(self, other: "NNSymbol") -> "NNSymbol":
        self.mask_sum -= other.mask_sum
        self.idx_code_sum -= other.idx_code_sum
        self.value_sum -= other.value_sum
        return self

    def is_empty(self) -> bool:
        """A symbol is empty if all its components are zero."""
        return self.mask_sum == 0 and self.idx_code_sum == 0 and math.isclose(self.value_sum, 0.0)

    def is_pure(self, prv: PRV) -> bool:
        """
        A symbol is pure if it can be decoded back to a single valid index.
        This happens when `mask_sum` corresponds to a single item's mask,
        allowing `idx_code` to be recovered and verified against the PRV.
        """
        if self.mask_sum == 0:
            return False
        
        # Recover the potential index code by dividing by the mask sum.
        potential_code = self.idx_code_sum / self.mask_sum
        
        # Check if this code maps back to a valid index in the PRV.
        decoded_idx = prv.index(int(potential_code))
        if decoded_idx is None:
            return False
        
        # Verify that the PRV entry for the decoded index matches the code.
        return potential_code == prv.entry(decoded_idx)

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the symbol."""
        # Note: GF class itself is not serialized; it's restored from PRV.
        return {
            'mask_sum': int(self.mask_sum),
            'idx_code_sum': int(self.idx_code_sum),
            'value_sum': self.value_sum
        }

    @classmethod
    def from_item(
        cls: Type[NNSymbol],
        item: NNItem,
        *,
        prv: PRV,
        r: galois.FieldArray
    ) -> NNSymbol:
        """Creates a symbol from an item, its PRV code, and a random mask `r`."""
        return cls(
            GF=prv.GF,
            mask_sum=r,
            idx_code_sum=r * prv.entry(item.idx),
            value_sum=item.value
        )

    @classmethod
    def to_item(cls, symbol: "NNSymbol", prv: PRV) -> NNItem:
        """Decodes a pure symbol back into an NNItem."""
        if not symbol.is_pure(prv):
            raise ValueError("Cannot convert a non-pure symbol to an item.")
            
        potential_code = symbol.idx_code_sum / symbol.mask_sum
        original_idx = prv.index(int(potential_code))
        
        # This check is somewhat redundant with is_pure, but serves as a safeguard.
        if original_idx is None:
            raise ValueError("Failed to decode a valid index from the symbol.")
            
        return NNItem(original_idx, symbol.value_sum)


class IBLT4NN(StandardFilter[NNSymbol, NNItem]):
    """
    An IBLT for aggregating numerical updates.
    This filter is aggregation-only and does not support item removal or
    standard set difference operations.
    """
    symbol_type = NNSymbol

    def __init__(self, hash_mapping: HashMapping, prv: PRV, mask_seed: Any = "default_mask_seed"):
        # We cannot call super().__init__ because NNSymbol requires a `GF` argument.
        self.hash_mapping = hash_mapping
        self.m = hash_mapping.table_size
        self.k = len(hash_mapping.hashers)
        self.prv = prv
        self._mask_hasher = Hasher(seed=mask_seed)
        
        # Initialize cells with the correct Galois Field class.
        self.cells = [self.symbol_type(prv.GF) for _ in range(self.m)]

    def _get_mask_for_idx(self, idx: int) -> galois.FieldArray:
        """Generates a non-zero, deterministic random mask for an index."""
        r_int = 0
        while r_int == 0:
            # Hash the item's own index to get a deterministic mask.
            r_int = self._mask_hasher.digest_int(idx, nbytes=16)
        return self.prv.GF(r_int)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IBLT4NN":
        """Reconstructs an IBLT4NN from its serialized dictionary state."""
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        prv_config = data['prv']
        
        # The key might be bytes, so handle it correctly.
        key_bytes = prv_config.get('key')
        if isinstance(key_bytes, list): # Simple JSON-to-list conversion
             key_bytes = bytes(key_bytes)
             
        prv = PRV(n=prv_config['n'], prp_type=prv_config['prp_type'], key=key_bytes)
        
        instance = cls(hash_mapping, prv, data.get('mask_seed'))
        
        # Recreate each symbol from its state, providing the GF from the new PRV.
        GF = prv.GF
        instance.cells = [cls.symbol_type(GF, **s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the filter's state into a dictionary."""
        # Convert key to list of ints for JSON compatibility.
        key_list = list(self.prv.key) if self.prv.key else None
        
        return {
            'hash_mapping': self.hash_mapping.get_config(),
            'cells': [s.get_state() for s in self.cells],
            'prv': {'n': self.prv.n, 'prp_type': self.prv.prp_type, 'key': key_list},
            'mask_seed': self._mask_hasher.seed
        }

    def push(self, item: NNItem) -> None:
        """Adds an item's contribution to the filter."""
        # Each push uses a deterministic mask `r` derived from the item's index.
        r = self._get_mask_for_idx(item.idx)
        source_symbol = self.symbol_type.from_item(item, prv=self.prv, r=r)
        
        for index in self._get_indices(item.get_key()):
            self.cells[index] += source_symbol

    def _peel_remove_item(self, item: NNItem):
        """Helper to subtract a fully decoded item's contribution from the IBLT."""
        r = self._get_mask_for_idx(item.idx)
        symbol_to_peel = self.symbol_type.from_item(item, prv=self.prv, r=r)
        
        for h_idx in self._get_indices(item.get_key()):
            self.cells[h_idx] -= symbol_to_peel

    def peel(self, destructive: bool = False) -> Dict[int, float]:
        """
        Iteratively decodes the IBLT to recover aggregated values.

        Args:
            destructive: If True, performs peeling in-place. If False,
                         works on a copy.

        Returns:
            A dictionary mapping decoded indices to their aggregated float values.
        """
        decoder = self if destructive else self.copy()
        decoded_items: Dict[int, float] = {}
        
        peeled_in_round = True
        while peeled_in_round:
            peeled_in_round = False
            
            # Find all currently pure cells.
            pure_indices = [i for i, c in enumerate(decoder.cells) if c.is_pure(decoder.prv)]
            
            for idx in pure_indices:
                cell = decoder.cells[idx]
                
                if not cell.is_pure(decoder.prv):
                    continue
                
                item = decoder.symbol_type.to_item(cell, prv=decoder.prv)
                
                # If we've already decoded this index, skip.
                if item.idx in decoded_items:
                    continue
                
                decoded_items[item.idx] = item.value
                
                # Subtract the contribution of the decoded item to find more pure cells.
                decoder._peel_remove_item(item)
                
                peeled_in_round = True
        
        if not all(c.is_empty() for c in decoder.cells):
            print("Warning: IBLT4NN decoding may be incomplete.")
            
        return decoded_items

    # --- Disable unsupported operations ---
    
    def remove(self, item: NNItem):
        """Removal of single items is not supported."""
        raise NotImplementedError("IBLT4NN is an aggregation-only filter and does not support `remove`.")

    def __sub__(self, other: FilterBase):
        """Standard filter subtraction is not meaningful for IBLT4NN."""
        raise NotImplementedError("IBLT4NN does not support the `-` operation.")

    def __isub__(self, other: FilterBase):
        """In-place subtraction is not meaningful for IBLT4NN."""
        raise NotImplementedError("IBLT4NN does not support the `-=` operation.")