# libFilter/filters/cbf.py

"""
Implementation of a Counting Bloom Filter (CBF).

This module provides the CBF class and its components, CBFItem and CBFSymbol.
A Counting Bloom Filter extends a standard Bloom Filter by replacing bit
arrays with an array of counters. This allows it to support item removal
and count estimation, though the estimates can be inflated due to hash
collisions.
"""

from __future__ import annotations
from typing import Any, Dict, Type

# Relative imports from within the library.
from ..core.base import FilterItem, StandardFilter, FilterSymbol
from ..core.utils import InputType

# --- CBF-specific Components ---

class CBFItem(FilterItem):
    """An item containing a simple key for a Counting Bloom Filter."""
    __slots__ = ('key',)

    def __init__(self, key: InputType):
        self.key = key

    def get_key(self) -> InputType:
        """Returns the item's key."""
        return self.key

    def __repr__(self) -> str:
        return f"CBFItem(key={self.key!r})"


class CBFSymbol(FilterSymbol):
    """
    A Counting Bloom Filter cell, representing an integer counter.
    
    The counter increments on item addition and decrements on removal.
    """
    __slots__ = ('count',)

    def __init__(self, count: int = 0):
        """Initializes the symbol with a given count."""
        super().__init__(count=count)
        self.count = count

    def __iadd__(self, other: CBFSymbol) -> CBFSymbol:
        """Performs in-place addition by summing the counts."""
        self.count += other.count
        return self

    def __isub__(self, other: CBFSymbol) -> CBFSymbol:
        """Performs in-place subtraction by differencing the counts."""
        self.count -= other.count
        return self

    def is_empty(self) -> bool:
        """Returns True if the counter is zero."""
        return self.count == 0

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the serializable state, conforming to the base interface.
        """
        return {'count': self.count}

    @classmethod
    def from_item(cls: Type[CBFSymbol], item: CBFItem, **kwargs: Any) -> CBFSymbol:
        """Creates a symbol with a count of 1, representing a single item addition."""
        return cls(count=1)


# --- CBF Implementation ---

class CBF(StandardFilter[CBFSymbol, CBFItem]):
    """
    A Counting Bloom Filter (CBF) that supports additions and removals.

    It provides a `__contains__` method for membership testing (with a
    possibility of false positives) and an `estimate_count` method to
    approximate the number of times an item has been added.
    """
    symbol_type = CBFSymbol

    def __contains__(self, item: CBFItem) -> bool:
        """
        Checks for an item's presence using the 'in' operator.
        
        An item is considered present if all its corresponding counters
        are greater than zero.

        Args:
            item: The CBFItem to check.

        Returns:
            True if the item is likely present (can be a false positive).
            False if the item is definitely not present.
        """
        if self.m == 0:
            return False
        key = self._get_item_key(item)
        return all(self.cells[i].count > 0 for i in self._get_indices(key))

    def estimate_count(self, item: CBFItem) -> int:
        """
        Estimates the count of an item.

        The estimated count is the minimum value among all counters mapped
        by the item's key. This is an upper-bounded estimate due to potential
        hash collisions with other items.

        Args:
            item: The CBFItem whose count is to be estimated.

        Returns:
            An integer representing the estimated count of the item.
        """
        if self.m == 0:
            return 0
        key = self._get_item_key(item)
        
        # The `min` function is necessary because some counters may be inflated
        # by hashes from other items. The true count is at most the minimum
        # of the associated counters.
        return min(self.cells[i].count for i in self._get_indices(key))