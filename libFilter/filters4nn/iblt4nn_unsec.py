# libFilter/filters4nn/iblt4nn_unsec.py

"""
Implementation of an IBLT for Neural Network (NN) gradient aggregation (unsecured version).
This version does not require a PRV and uses a count-based symbol (NNSymbol_unsec).
"""

from __future__ import annotations
from typing import Dict, Any, Type

from ..core.base import StandardFilter, FilterBase
from ..core.utils import HashMapping
from .nn_utils import NNItem, NNSymbol_unsec

class IBLT4NN_unsec(StandardFilter[NNSymbol_unsec, NNItem]):
    """
    An IBLT for aggregating numerical updates (weights) using count-based logic.
    This filter supports addition and subtraction of items, and can decode
    the aggregated weights for each index. No PRV or cryptographic masking is used.
    """
    symbol_type = NNSymbol_unsec

    def __init__(self, hash_mapping: HashMapping):
        self.hash_mapping = hash_mapping
        self.m = hash_mapping.table_size
        self.k = len(hash_mapping.hashers)
        self.cells = [self.symbol_type() for _ in range(self.m)]

    @classmethod
    def from_dict(cls: Type["IBLT4NN_unsec"], data: Dict[str, Any]) -> "IBLT4NN_unsec":
        """Reconstructs an IBLT4NN_unsec from its serialized dictionary state."""
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        instance = cls(hash_mapping)
        instance.cells = [cls.symbol_type(**s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the filter's state into a dictionary."""
        return {
            'hash_mapping': self.hash_mapping.get_config(),
            'cells': [s.get_state() for s in self.cells],
        }

    def push(self, item: NNItem) -> None:
        """Adds an item's contribution to the filter."""
        source_symbol = self.symbol_type.from_item(item)
        for index in self._get_indices(item.get_key()):
            self.cells[index] += source_symbol

    def remove(self, item: NNItem) -> None:
        """Removes an item's contribution from the filter."""
        source_symbol = self.symbol_type.from_item(item)
        for index in self._get_indices(item.get_key()):
            self.cells[index] -= source_symbol

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

        pure_indices = [i for i, c in enumerate(decoder.cells) if c.is_pure()]

        while pure_indices:
            idx = pure_indices.pop()
            cell = decoder.cells[idx]

            if not cell.is_pure():
                continue

            # Make a copy to avoid mutating the cell during subtraction
            symbol_to_peel = NNSymbol_unsec(cell.count, cell.idx_sum, cell.weight_sum)
            item = symbol_to_peel.to_item()

            if item.idx == 3567:
                affected_indices = list(decoder._get_indices(item.get_key()))
                print("affected_indices",affected_indices)


            if item.idx in decoded_weights:
                continue

            decoded_weights[item.idx] = item.weight

            affected_indices = list(decoder._get_indices(item.get_key()))
            for affected_idx in affected_indices:
                affected_cell = decoder.cells[affected_idx]
                affected_cell -= symbol_to_peel
                if affected_cell.is_pure():
                    pure_indices.append(affected_idx)

        if not all(c.is_empty() for c in decoder.cells):
            print("Warning: IBLT4NN_unsec decoding may be incomplete.")
            print("Unpeeled cells:", [c.get_state() for c in decoder.cells if not c.is_empty()])

        return decoded_weights

    def is_fully_decoded(self) -> bool:
        """Checks if all cells are empty, indicating complete decoding."""
        return all(c.is_empty() for c in self.cells)

    def __sub__(self, other: FilterBase):
        """Subtracts another IBLT4NN_unsec filter."""
        if not isinstance(other, IBLT4NN_unsec):
            raise TypeError("Can only subtract another IBLT4NN_unsec.")
        if self.m != other.m or self.k != other.k:
            raise ValueError("IBLT4NN_unsec filters must have the same shape for subtraction.")
        result = self.copy()
        for i in range(self.m):
            result.cells[i] -= other.cells[i]
        return result

    def __isub__(self, other: FilterBase):
        """In-place subtraction of another IBLT4NN_unsec filter."""
        if not isinstance(other, IBLT4NN_unsec):
            raise TypeError("Can only subtract another IBLT4NN_unsec.")
        if self.m != other.m or self.k != other.k:
            raise ValueError("IBLT4NN_unsec filters must have the same shape for subtraction.")
        for i in range(self.m):
            self.cells[i] -= other.cells[i]
        return self