# libFilter/filters4nn/nn_utils.py

"""
Shared components for Neural Network-oriented filters (IBLT4NN, RIBLT4NN).

This module defines the common data structures and interfaces used for
secure numerical aggregation with a fixed-point design for weights.
"""

from __future__ import annotations
import math
from typing import Any, Dict, Type

import galois

from libFilter.core.base import FilterItem, PeelableSymbol
from libFilter.core.prv import PRV

ZERO_TOLERANCE = 1e-9

class NNItem(FilterItem):
    """An item representing a numerical update (weight) for a specific index."""
    __slots__ = ('idx', 'weight')

    def __init__(self, idx: int, weight: float):
        self.idx = idx
        self.weight = weight

    def get_key(self) -> int:
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
        return NNItem(self.idx, self.weight)

class NNSymbol(PeelableSymbol):
    """
    A cell for NN-filters using fixed-point arithmetic for the weights.
    Internally, 'weight_sum' is an integer that encodes weight*scale.
    """
    __slots__ = ('GF', 'mask_sum', 'idx_code_sum', 'weight_sum', 'ndigits', '_scale')

    def __init__(
        self,
        GF: galois.FieldClass,
        mask_sum: Any = 0,
        idx_code_sum: Any = 0,
        weight_sum: int = 0,
        ndigits: int = 6
    ):
        super().__init__(GF=GF, mask_sum=mask_sum, idx_code_sum=idx_code_sum, weight_sum=weight_sum, ndigits=ndigits)
        self.GF = GF
        self.mask_sum = GF(mask_sum)
        self.idx_code_sum = GF(idx_code_sum)
        self.weight_sum = int(weight_sum)
        self.ndigits = int(ndigits)
        if self.ndigits < 0:
            raise ValueError("ndigits must be non-negative.")
        self._scale = 10 ** self.ndigits

    def _zero_if_close(self):
        """If weight_sum as float is very close to zero, set it to exactly 0."""
        if abs(self.weight_sum) <= int(ZERO_TOLERANCE * self._scale):
            self.weight_sum = 0

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

    def div_scalar(self, scalar: int) -> "NNSymbol":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero in the field.")
        if scalar == 1:
            return self
        self.mask_sum /= self.GF(scalar)
        self.idx_code_sum /= self.GF(scalar)
        # integer division for fixed-point
        self.weight_sum //= scalar
        self._zero_if_close()
        return self

    def mul_scalar_weight(self, scalar: int) -> "NNSymbol":
        self.weight_sum *= scalar
        self._zero_if_close()
        return self

    def is_empty(self) -> bool:
        return self.mask_sum == 0 and self.idx_code_sum == 0 and self.weight_sum == 0

    def is_pure(self, prv: PRV) -> bool:
        if self.mask_sum == 0:
            return False
        potential_code = self.idx_code_sum / self.mask_sum
        decoded_idx = prv.index(int(potential_code))
        if decoded_idx is None:
            return False
        return potential_code == prv.entry(decoded_idx)

    def get_state(self) -> Dict[str, Any]:
        return {
            'mask_sum': int(self.mask_sum),
            'idx_code_sum': int(self.idx_code_sum),
            'weight_sum': int(self.weight_sum),
            'ndigits': self.ndigits
        }

    def copy(self) -> "NNSymbol":
        return NNSymbol(
            GF=self.GF,
            mask_sum=self.mask_sum,
            idx_code_sum=self.idx_code_sum,
            weight_sum=self.weight_sum,
            ndigits=self.ndigits
        )

    def negated(self) -> "NNSymbol":
        return NNSymbol(
            self.GF,
            mask_sum=-self.mask_sum,
            idx_code_sum=-self.idx_code_sum,
            weight_sum=-self.weight_sum,
            ndigits=self.ndigits
        )

    @classmethod
    def from_item(
        cls: Type["NNSymbol"],
        item: NNItem,
        *,
        prv: PRV,
        r: galois.FieldArray,
        ndigits: int = 6
    ) -> "NNSymbol":
        scale = 10 ** ndigits
        weight_sum = int(round(item.weight * scale))
        return cls(
            GF=prv.GF,
            mask_sum=r,
            idx_code_sum=r * prv.entry(item.idx),
            weight_sum=weight_sum,
            ndigits=ndigits
        )

    def to_item(self, prv: PRV) -> NNItem:
        if not self.is_pure(prv):
            raise ValueError("Cannot convert a non-pure symbol to an item.")
        potential_code = self.idx_code_sum / self.mask_sum
        k_int = int(potential_code)
        original_idx = prv.index(k_int)
        if original_idx is None:
            raise ValueError("Failed to decode a valid index from the symbol.")
        weight = round(self.weight_sum / self._scale, self.ndigits)
        return NNItem(original_idx, weight)

class NNSymbol_unsec:
    """
    A simplified, 'unsecured' NNSymbol variant using fixed-point encoding for weights.
    """
    __slots__ = ("count", "idx_sum", "weight_sum", "ndigits", "_scale")

    def __init__(self, count: int = 0, idx_sum: int = 0, weight_sum: int = 0, ndigits: int = 6):
        self.count = count
        self.idx_sum = idx_sum
        self.weight_sum = int(weight_sum)
        self.ndigits = int(ndigits)
        self._scale = 10 ** self.ndigits

    def __iadd__(self, other: "NNSymbol_unsec") -> "NNSymbol_unsec":
        self.count += other.count
        self.idx_sum += other.idx_sum
        self.weight_sum += other.weight_sum
        return self

    def __isub__(self, other: "NNSymbol_unsec") -> "NNSymbol_unsec":
        self.count -= other.count
        self.idx_sum -= other.idx_sum
        self.weight_sum -= other.weight_sum
        return self

    def is_empty(self) -> bool:
        return self.count == 0 and self.idx_sum == 0 and self.weight_sum == 0

    def is_pure(self) -> bool:
        return self.count == 1 or self.count == -1

    def get_state(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "idx_sum": self.idx_sum,
            "weight_sum": int(self.weight_sum),
            "ndigits": self.ndigits
        }

    @classmethod
    def from_item(cls, item: "NNItem", ndigits: int = 6) -> "NNSymbol_unsec":
        scale = 10 ** ndigits
        weight_sum = int(round(item.weight * scale))
        return cls(count=1, idx_sum=item.idx, weight_sum=weight_sum, ndigits=ndigits)

    def to_item(self) -> "NNItem":
        if not self.is_pure():
            raise ValueError("Cannot decode item from a non-pure symbol (count is {}).".format(self.count))
        idx = self.idx_sum // self.count
        weight = round(self.weight_sum / self._scale / self.count, self.ndigits)
        return NNItem(idx, weight)

    def copy(self) -> "NNSymbol_unsec":
        return NNSymbol_unsec(self.count, self.idx_sum, self.weight_sum, self.ndigits)