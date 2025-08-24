# libFilter/filters/bf.py

"""
Implementation of a classic Bloom Filter (BF).

This module provides the BF class and its necessary components, BFItem and
BFSymbol, which implement the core filter interfaces defined in `base.py`.
The Bloom Filter is a space-efficient probabilistic data structure used to
test whether an element is a member of a set. False positive matches are
possible, but false negatives are not.
"""

from __future__ import annotations
from typing import Any, Dict, Type

# Relative imports from within the library.
from ..core.base import FilterItem, StandardFilter, FilterSymbol
from ..core.utils import InputType

# --- BF-specific Components ---

class BFItem(FilterItem):
    """
    An item specifically for a Bloom Filter, containing a single key.
    
    This class is a concrete implementation of the FilterItem interface.
    """
    __slots__ = ('key',)

    def __init__(self, key: InputType):
        self.key = key

    def get_key(self) -> InputType:
        """Returns the item's key."""
        return self.key

    def __repr__(self) -> str:
        return f"BFItem(key={self.key!r})"


class BFSymbol(FilterSymbol):
    """
    A Bloom Filter cell, representing a single bit.
    
    This class implements the FilterSymbol interface. It's essentially a
    boolean flag. Addition (`+=`) is a logical OR operation. Subtraction
    is not supported, as it would break the filter's guarantees.
    """
    __slots__ = ('is_set',)

    def __init__(self, is_set: bool = False):
        """Initializes the symbol. `is_set` is True if the bit is 1."""
        super().__init__(is_set=is_set)
        self.is_set = is_set

    def __iadd__(self, other: BFSymbol) -> BFSymbol:
        """
        Performs in-place addition (logical OR). If the other symbol is
        set, this one becomes set.
        """
        if other.is_set:
            self.is_set = True
        return self

    def __isub__(self, other: BFSymbol) -> BFSymbol:
        """Subtraction is not supported for a standard Bloom Filter."""
        raise NotImplementedError("Bloom Filters do not support item removal.")

    def is_empty(self) -> bool:
        """Returns True if the bit is not set (is 0)."""
        return not self.is_set

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the symbol."""
        return {'is_set': self.is_set}
    
    @classmethod
    def from_item(cls: Type[BFSymbol], item: BFItem, **kwargs: Any) -> BFSymbol:
        """Creates a 'SET' symbol from any item, as items only turn bits on."""
        # The item's content doesn't matter; its presence is what counts.
        return cls(is_set=True)


# --- BF Implementation ---

class BF(StandardFilter[BFSymbol, BFItem]):
    """
    A classic Bloom Filter (BF).

    It supports efficient `push` (add) and `__contains__` (check) operations.
    Removal is not supported. It is prone to false positives but guarantees
    no false negatives.
    
    Attributes:
        symbol_type (Type[BFSymbol]): The class for the cells in the filter.
    """
    symbol_type = BFSymbol

    def __contains__(self, item: BFItem) -> bool:
        """
        Checks if an item is possibly in the filter.

        Args:
            item: The BFItem to check.

        Returns:
            True if the item is likely present (could be a false positive).
            False if the item is definitely not present.
        """
        if self.m == 0:  # Table size
            return False
            
        key = self._get_item_key(item)
        indices = self._get_indices(key)
        
        return all(self.cells[i].is_set for i in indices)