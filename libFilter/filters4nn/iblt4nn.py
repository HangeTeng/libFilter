# libFilter/filters4nn/iblt4nn.py

"""
Implementation of an IBLT for Neural Network (NN) gradient aggregation.
"""

from __future__ import annotations
from typing import Dict, Any, Type

from ..core.prv import PRV
from ..core.base import StandardFilter, FilterBase
from ..core.utils import HashMapping
from .nn_utils import NNItem, NNSymbol

class IBLT4NN(StandardFilter[NNSymbol, NNItem]):
    """
    An IBLT for aggregating numerical updates (weights).
    This filter is aggregation-only and does not support item removal or
    standard set difference operations.
    """
    symbol_type = NNSymbol

    def __init__(
        self,
        hash_mapping: HashMapping,
        prv: PRV,
        mask_seed: Any = "default_mask_seed",
        ndigits: int = 6,
    ):
        self.prv = prv
        # Keep mask_seed in signature for backward compatibility with old callsites.
        self._ndigits = int(ndigits)
        self.hash_mapping = hash_mapping
        self.m = hash_mapping.table_size
        self.k = len(hash_mapping.hashers)
        self.cells = [self.symbol_type(self.prv.GF, ndigits=self._ndigits) for _ in range(self.m)]

    @classmethod
    def from_dict(cls: Type["IBLT4NN"], data: Dict[str, Any]) -> "IBLT4NN":
        """Reconstructs an IBLT4NN from its serialized dictionary state."""
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        prv_config = data['prv']
        
        key_bytes = prv_config.get('key')
        if isinstance(key_bytes, list):
             key_bytes = bytes(key_bytes)
             
        prv = PRV(n=prv_config['n'], prp_type=prv_config['prp_type'], key=key_bytes)
        
        instance = cls(
            hash_mapping,
            prv,
            data.get('mask_seed'),
            ndigits=data.get('ndigits', 6),
        )
        
        GF = prv.GF
        instance.cells = [cls.symbol_type(GF, **s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the filter's state into a dictionary."""
        key_list = list(self.prv.key) if self.prv.key else None
        
        return {
            'hash_mapping': self.hash_mapping.get_config(),
            'cells': [s.get_state() for s in self.cells],
            'prv': {'n': self.prv.n, 'prp_type': self.prv.prp_type, 'key': key_list},
            'ndigits': self._ndigits,
        }

    def push(self, item: NNItem) -> None:
        """Adds an item's contribution to the filter."""
        source_symbol = self.symbol_type.from_item(
            item,
            prv=self.prv,
            ndigits=self._ndigits,
        )
        
        for index in self._get_indices(item.get_key()):
            self.cells[index] += source_symbol

    def peel(self, destructive: bool = False) -> Dict[int, float]:
        """
        Iteratively decodes the IBLT to recover aggregated weights.

        Args:
            destructive: If True, performs peeling in-place.

        Returns:
            A dictionary mapping decoded indices to their aggregated float weights.
        """
        decoder = self if destructive else self.copy()
        decoded_weights: Dict[int, float] = {}
        
        pure_indices = [i for i, c in enumerate(decoder.cells) if c.is_pure(decoder.prv)]
        
        while pure_indices:
            idx = pure_indices.pop()
            cell = decoder.cells[idx]
            
            if not cell.is_pure(decoder.prv):
                continue

            # Create a safe, independent copy of the symbol for peeling.
            symbol_to_peel = cell.copy()
            item = symbol_to_peel.to_item(prv=decoder.prv)

            # if item.idx in [3567,3414,6532,2975,4945]:
            #     print("item.idx",item.idx)
            #     print("idx",idx)
            #     print("affected_indices",list(decoder._get_indices(item.get_key())))

            if item.idx in decoded_weights:
                continue
            
            affected_indices = list(decoder._get_indices(item.get_key()))
            scalar = affected_indices.count(idx)
            symbol_to_peel.div_scalar(scalar)

            decoded_weights[item.idx] = item.weight / scalar
            
            for affected_idx in affected_indices:
                affected_cell = decoder.cells[affected_idx]
                affected_cell -= symbol_to_peel
                if affected_cell.is_pure(decoder.prv):
                    pure_indices.append(affected_idx)
                    
        if not all(c.is_empty() for c in decoder.cells):
            print("Warning: IBLT4NN decoding may be incomplete.")
            # # Keep only the first/last 5 unpeeled cell indices, and include the total count.
            # unpeeled_indices = [i for i, c in enumerate(decoder.cells) if not c.is_empty()]
            # n = len(unpeeled_indices)
            # if n > 10:
            #     display_indices = unpeeled_indices[:5] + ["..."] + unpeeled_indices[-5:]
            # else:
            #     display_indices = unpeeled_indices
            # print(f"Unpeeled cell (total {n}):", display_indices)
        # else:
        #     print("IBLT4NN is fully decoded")

        return decoded_weights

    def is_fully_decoded(self) -> bool:
        """Checks if all cells are empty, indicating complete decoding."""
        return all(c.is_empty() for c in self.cells)
    
    def scalar_mul_weight(self, scalar: float):
        """Multiplies all cells by a scalar."""
        for cell in self.cells:
            cell.mul_scalar_weight(scalar)
    
    def remove(self, item: NNItem):
        """User-facing removal of single items is not supported."""
        raise NotImplementedError("IBLT4NN is an aggregation-only filter and does not support `remove`.")

    def __sub__(self, other: FilterBase):
        """Filter subtraction is not meaningful for IBLT4NN."""
        raise NotImplementedError("IBLT4NN does not support the `-` operation.")

    def __isub__(self, other: FilterBase):
        """In-place filter subtraction is not meaningful for IBLT4NN."""
        raise NotImplementedError("IBLT4NN does not support the `-=` operation.")