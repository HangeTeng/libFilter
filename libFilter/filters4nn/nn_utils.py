# libFilter/filters4nn/nn_utils.py

"""
Shared components for Neural Network-oriented filters (IBLT4NN, RIBLT4NN).

This module defines the common data structures and interfaces used for
secure numerical aggregation, including the item representation (NNItem)
and the finite-field-based cell symbol (NNSymbol). It includes robust
handling of floating-point arithmetic.
"""

from __future__ import annotations
import math
from typing import Any, Dict, Type

import galois

from ..core.base import FilterItem, PeelableSymbol
from ..core.prv import PRV

# A small tolerance for floating point comparisons, used for proactive zeroing.
# The default relative tolerance of math.isclose is 1e-9, which is a good choice.
ZERO_TOLERANCE = 1e-9

class NNItem(FilterItem):
    """An item representing a numerical update (weight) for a specific index."""
    __slots__ = ('idx', 'weight')

    def __init__(self, idx: int, weight: float):
        self.idx = idx
        self.weight = weight

    def get_key(self) -> int:
        """The item's index is used as its key for hashing."""
        return self.idx

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NNItem):
            return NotImplemented
        return self.idx == other.idx and math.isclose(self.weight, other.weight)

    def __hash__(self) -> int:
        return hash((self.idx, self.weight))

    def __repr__(self) -> str:
        return f"NNItem(idx={self.idx}, weight={self.weight:.4f})"

    def copy(self) -> "NNItem":
        """Returns a new NNItem instance with the same state."""
        return NNItem(self.idx, self.weight)


class NNSymbol(PeelableSymbol):
    """
    A cell for NN-filters, using finite field arithmetic for secure sums.
    Includes proactive zeroing for floating-point stability.
    """
    __slots__ = ('GF', 'mask_sum', 'idx_code_sum', 'weight_sum')

    def __init__(
        self,
        GF: galois.FieldClass,
        mask_sum: Any = 0,
        idx_code_sum: Any = 0,
        weight_sum: float = 0.0
    ):
        super().__init__(GF=GF, mask_sum=mask_sum, idx_code_sum=idx_code_sum, weight_sum=weight_sum)
        self.GF = GF
        self.mask_sum = GF(mask_sum)
        self.idx_code_sum = GF(idx_code_sum)
        self.weight_sum = weight_sum

    def _zero_if_close(self):
        """If weight_sum is very close to zero, set it to exactly 0.0."""
        if math.isclose(self.weight_sum, 0.0, abs_tol=ZERO_TOLERANCE):
            self.weight_sum = 0.0

    def __iadd__(self, other: "NNSymbol") -> "NNSymbol":
        self.mask_sum += other.mask_sum
        self.idx_code_sum += other.idx_code_sum
        self.weight_sum += other.weight_sum
        self._zero_if_close()
        return self

    def __isub__(self, other: "NNSymbol") -> "NNSymbol":
        self.mask_sum -= other.mask_sum
        self.idx_code_sum -= other.idx_code_sum
        self.weight_sum -= other.weight_sum
        self._zero_if_close()
        return self

    def is_empty(self) -> bool:
        """
        A symbol is empty if all its components are zero.
        With proactive zeroing, direct comparison is safe.
        """
        return self.mask_sum == 0 and self.idx_code_sum == 0 and self.weight_sum == 0.0

    def is_pure(self, prv: PRV) -> bool:
        """
        Checks if the symbol is pure by verifying its contents against the PRV.
        """
        if self.mask_sum == 0:
            return False
        
        potential_code = self.idx_code_sum / self.mask_sum
        
        decoded_idx = prv.index(int(potential_code))
        if decoded_idx is None:
            return False
        
        return potential_code == prv.entry(decoded_idx)

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the symbol."""
        return {
            'mask_sum': int(self.mask_sum),
            'idx_code_sum': int(self.idx_code_sum),
            'weight_sum': self.weight_sum
        }

    def copy(self) -> "NNSymbol":
        """Returns a new NNSymbol instance with the same state."""
        return NNSymbol(
            GF=self.GF,
            mask_sum=self.mask_sum,
            idx_code_sum=self.idx_code_sum,
            weight_sum=self.weight_sum
        )
    
    @classmethod
    def from_item(
        cls: Type["NNSymbol"],
        item: NNItem,
        *,
        prv: PRV,
        r: galois.FieldArray
    ) -> "NNSymbol":
        """Creates a symbol from an item, its PRV code, and a random mask `r`."""
        return cls(
            GF=prv.GF,
            mask_sum=r,
            idx_code_sum=r * prv.entry(item.idx),
            weight_sum=item.weight
        )

    @classmethod
    def to_item(cls, symbol: "NNSymbol", prv: PRV) -> NNItem:
        """Decodes a pure symbol back into an NNItem."""
        if not symbol.is_pure(prv):
            raise ValueError("Cannot convert a non-pure symbol to an item.")
            
        potential_code = symbol.idx_code_sum / symbol.mask_sum
        original_idx = prv.index(int(potential_code))
        
        if original_idx is None:
            raise ValueError("Failed to decode a valid index from the symbol.")
            
        return NNItem(original_idx, symbol.weight_sum)