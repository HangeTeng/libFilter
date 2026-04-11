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
from collections import OrderedDict
from typing import Any, Dict, Optional, Type

import galois

from ..core.base import FilterItem, PeelableSymbol
from ..core.prv import PRV

# Bounded LRU caches attached to PRV instances to avoid repeated AES decrypt/encrypt
# during peel (hot path: many cells × is_pure / to_item).
_INDEX_CACHE_MAX = 8192
_ENTRY_CACHE_MAX = 4096

_CACHE_MISS = object()


def _lru_get(cache: OrderedDict, key: int) -> Any:
    if key not in cache:
        return _CACHE_MISS
    cache.move_to_end(key)
    return cache[key]


def _lru_set(cache: OrderedDict, key: int, value: Any, max_size: int) -> None:
    if key in cache:
        del cache[key]
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)


def _prv_index_cached(prv: PRV, k_int: int) -> Optional[int]:
    caches = getattr(prv, "_nn_utils_caches", None)
    if caches is None:
        caches = {"index": OrderedDict(), "entry": OrderedDict()}
        prv._nn_utils_caches = caches
    idx_cache: OrderedDict = caches["index"]
    hit = _lru_get(idx_cache, k_int)
    if hit is not _CACHE_MISS:
        return hit
    res = prv.index(k_int)
    _lru_set(idx_cache, k_int, res, _INDEX_CACHE_MAX)
    return res


def _prv_entry_cached(prv: PRV, i: int) -> "galois.FieldArray":
    caches = getattr(prv, "_nn_utils_caches", None)
    if caches is None:
        caches = {"index": OrderedDict(), "entry": OrderedDict()}
        prv._nn_utils_caches = caches
    ent_cache: OrderedDict = caches["entry"]
    hit = _lru_get(ent_cache, i)
    if hit is not _CACHE_MISS:
        return hit
    e = prv.entry(i)
    _lru_set(ent_cache, i, e, _ENTRY_CACHE_MAX)
    return e

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
    __slots__ = ('GF', 'idx_code_sum', 'weight_sum', 'ndigits', '_scale')

    def __init__(
        self,
        GF: galois.FieldClass,
        idx_code_sum: Any = 0,
        weight_sum: int = 0,
        ndigits: int = 6
    ):
        super().__init__(GF=GF, idx_code_sum=idx_code_sum, weight_sum=weight_sum, ndigits=ndigits)
        self.GF = GF
        self.idx_code_sum = GF(idx_code_sum)
        self.weight_sum = int(weight_sum)
        self.ndigits = int(ndigits)
        if self.ndigits < 0:
            raise ValueError("ndigits must be non-negative.")
        self._scale = 10 ** self.ndigits

    def _zero_if_close(self):
        """Fixed-point mode keeps weight_sum as integer, no-op."""
        return

    def _weight_as_gf(self) -> "galois.FieldArray":
        """
        Convert signed fixed-point integer weight into a valid field element.
        """
        return self.GF(self.weight_sum % self.GF.order)

    def __iadd__(self, other: "NNSymbol") -> "NNSymbol":
        if self.ndigits != other.ndigits:
            raise ValueError("Cannot add symbols with different ndigits.")
        self.idx_code_sum += other.idx_code_sum
        self.weight_sum += other.weight_sum
        return self

    def __isub__(self, other: "NNSymbol") -> "NNSymbol":
        if self.ndigits != other.ndigits:
            raise ValueError("Cannot subtract symbols with different ndigits.")
        self.idx_code_sum -= other.idx_code_sum
        self.weight_sum -= other.weight_sum
        return self

    def div_scalar(self, scalar: int) -> "NNSymbol":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero in the field.")
        if scalar == 1:
            return self

        self.idx_code_sum /= self.GF(scalar)
        if self.weight_sum % scalar != 0:
            raise ValueError("Fixed-point weight_sum cannot be divided evenly by scalar.")
        self.weight_sum //= scalar
        return self

    def mul_scalar_weight(self, scalar: int) -> "NNSymbol":
        self.weight_sum = int(round(self.weight_sum * scalar))
        return self

    def is_empty(self) -> bool:
        """
        A symbol is empty if all its components are zero.
        With proactive zeroing, direct comparison is safe.
        """
        return self.idx_code_sum == 0 and self.weight_sum == 0

    def is_pure(self, prv: PRV) -> bool:
        """
        Checks if the symbol is pure by verifying its contents against the PRV.
        """
        if self.weight_sum == 0:
            return False
        # Empty XOR bucket: no code contribution.
        if self.idx_code_sum == 0:
            return False

        potential_code = self.idx_code_sum / self._weight_as_gf()
        k_int = int(potential_code)
        # PRP domain check before decrypt (same as PRV.index, but cheaper than full index()).
        if k_int < 0 or k_int >= prv.value_limit:
            return False

        decoded_idx = _prv_index_cached(prv, k_int)
        if decoded_idx is None:
            return False

        return potential_code == _prv_entry_cached(prv, decoded_idx)

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the symbol."""
        return {
            'idx_code_sum': int(self.idx_code_sum),
            'weight_sum': self.weight_sum,
            'ndigits': self.ndigits,
        }

    def copy(self) -> "NNSymbol":
        """Returns a new NNSymbol instance with the same state."""
        return NNSymbol(
            GF=self.GF,
            idx_code_sum=self.idx_code_sum,
            weight_sum=self.weight_sum,
            ndigits=self.ndigits,
        )

    def negated(self) -> "NNSymbol":
        return NNSymbol(
            GF=self.GF,
            idx_code_sum=-self.idx_code_sum,
            weight_sum=-self.weight_sum,
            ndigits=self.ndigits,
        )

    @classmethod
    def from_item(
        cls: Type["NNSymbol"],
        item: NNItem,
        *,
        prv: PRV,
        ndigits: int = 6,
    ) -> "NNSymbol":
        """Creates a fixed-point symbol from an item."""
        scale = 10 ** ndigits
        fixed_weight = int(round(item.weight * scale))
        if fixed_weight == 0:
            raise ValueError(
                f"Item weight {item.weight} becomes 0 after quantization; increase ndigits."
            )
        return cls(
            GF=prv.GF,
            idx_code_sum=prv.entry(item.idx) * prv.GF(fixed_weight % prv.GF.order),
            weight_sum=fixed_weight,
            ndigits=ndigits,
        )

    def to_item(self, prv: PRV) -> NNItem:
        """Decodes a pure symbol back into an NNItem."""
        if not self.is_pure(prv):
            raise ValueError("Cannot convert a non-pure symbol to an item.")

        potential_code = self.idx_code_sum / self._weight_as_gf()
        k_int = int(potential_code)
        original_idx = _prv_index_cached(prv, k_int)

        if original_idx is None:
            raise ValueError("Failed to decode a valid index from the symbol.")

        weight = round(self.weight_sum / self._scale, self.ndigits)
        return NNItem(original_idx, weight)

class NNSymbol_unsec:
    """
    A simplified, 'unsecured' NNSymbol variant that does not require a PRV for decoding.
    Instead, it stores a count, idx_sum, and weight_sum, similar to IBLTSymbol logic.
    Addition and subtraction are supported.
    """
    __slots__ = ("count", "idx_sum", "weight_sum")

    def __init__(self, count: int = 0, idx_sum: int = 0, weight_sum: float = 0.0):
        self.count = count
        self.idx_sum = idx_sum
        self.weight_sum = weight_sum

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
        return self.count == 0 and self.idx_sum == 0 and self.weight_sum == 0.0

    def is_pure(self) -> bool:
        return self.count == 1 or self.count == -1

    def get_state(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "idx_sum": self.idx_sum,
            "weight_sum": self.weight_sum
        }

    @classmethod
    def from_item(cls, item: "NNItem") -> "NNSymbol_unsec":
        return cls(count=1, idx_sum=item.idx, weight_sum=item.weight)

    def to_item(self) -> "NNItem":
        if not self.is_pure():
            raise ValueError("Cannot decode item from a non-pure symbol (count is {}).".format(self.count))
        idx = self.idx_sum // self.count
        weight = self.weight_sum / self.count
        return NNItem(idx, weight)

    def copy(self) -> "NNSymbol_unsec":
        return NNSymbol_unsec(self.count, self.idx_sum, self.weight_sum)
