# libFilter/filters/iblt.py

"""
Implementation of an Invertible Bloom Lookup Table (IBLT).

This module provides the IBLT class and its components. An IBLT is a
probabilistic data structure that stores a set of key-value pairs and can
be used to efficiently compute the difference between two sets. Unlike
Bloom Filters, an IBLT can, with high probability, decode its contents
back into the original items, provided the table is not overloaded.
"""

from __future__ import annotations
from typing import Set, Tuple, List, Dict, Any, Type

# Relative imports from within the library.
from ..core.base import FilterItem, PeelableSymbol, StandardFilter
from ..core.utils import (
    InputType, _xor_bytes, HashMapping,
    serialize_typed_value, deserialize_typed_value
)

# --- IBLT-specific Components ---

class IBLTItem(FilterItem):
    """
    An item for an IBLT, containing a key-value pair.
    
    Both key and value must be of a type supported by the serialization
    helpers (int, str, or bytes).
    """
    __slots__ = ('key', 'value')

    def __init__(self, key: InputType, value: InputType):
        self.key = key
        self.value = value

    def get_key(self) -> InputType:
        """Returns the item's key, used for hashing."""
        return self.key

    def __eq__(self, other: object) -> bool:
        """Checks for equality based on both key and value."""
        if not isinstance(other, IBLTItem):
            return NotImplemented
        return self.key == other.key and self.value == other.value

    def __hash__(self) -> int:
        """Computes a hash based on the byte representations of key and value."""
        # Note: Direct hashing of key/value might be problematic for mutable types,
        # but InputType is constrained to immutable-behaving types.
        # Serializing before hashing ensures consistency.
        key_bytes = serialize_typed_value(self.key)
        value_bytes = serialize_typed_value(self.value)
        return hash((key_bytes, value_bytes))

    def __repr__(self) -> str:
        return f"IBLTItem(key={self.key!r}, value={self.value!r})"


class IBLTSymbol(PeelableSymbol):
    """
    A cell (symbol) in an IBLT.
    
    It stores a counter and XOR sums of the keys and values of items
    hashed to it.
    """
    __slots__ = ('count', 'key_sum', 'value_sum')

    def __init__(
        self,
        count: int = 0,
        key_sum: bytes = b'',
        value_sum: bytes = b''
    ):
        super().__init__(count=count, key_sum=key_sum, value_sum=value_sum)
        self.count = count
        self.key_sum = key_sum
        self.value_sum = value_sum

    def __iadd__(self, other: IBLTSymbol) -> IBLTSymbol:
        """Adds another symbol by summing counts and XORing sums."""
        self.count += other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self

    def __isub__(self, other: IBLTSymbol) -> IBLTSymbol:
        """Subtracts a symbol by decrementing count and XORing sums."""
        self.count -= other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self

    def is_empty(self) -> bool:
        """
        Returns True if the symbol is in a neutral (zero) state.
        
        A cell is empty if its count is zero AND its checksums are also
        effectively zero (either empty bytes or all-zero bytes).
        """
        # all(b == 0 for b in self.key_sum) is a concise way to check
        # if a byte string contains only null bytes.
        key_is_zero = not self.key_sum or all(b == 0 for b in self.key_sum)
        value_is_zero = not self.value_sum or all(b == 0 for b in self.value_sum)
        
        return self.count == 0 and key_is_zero and value_is_zero

    def is_pure(self) -> bool:
        """
        Returns True if the symbol represents a single, recoverable item.
        This is the case if its count is 1 or -1.
        """
        return self.count == 1 or self.count == -1

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the symbol."""
        return {
            'count': self.count,
            'key_sum': self.key_sum,
            'value_sum': self.value_sum
        }
    
    @classmethod
    def from_item(cls: Type[IBLTSymbol], item: IBLTItem, **kwargs: Any) -> IBLTSymbol:
        """Creates a symbol from an IBLTItem."""
        return cls(
            count=1,
            key_sum=serialize_typed_value(item.key),
            value_sum=serialize_typed_value(item.value)
        )

    @classmethod
    def to_item(cls, symbol: IBLTSymbol) -> IBLTItem:
        """Decodes a pure symbol back into an IBLTItem."""
        if not symbol.is_pure():
            raise ValueError(
                "Cannot decode item from a non-pure symbol "
                f"(count is {symbol.count})."
            )
        
        key = deserialize_typed_value(symbol.key_sum)
        value = deserialize_typed_value(symbol.value_sum)
        return IBLTItem(key, value)


# --- IBLT Implementation ---

class IBLT(StandardFilter[IBLTSymbol, IBLTItem]):
    """
    An Invertible Bloom Lookup Table (IBLT).
    """
    symbol_type = IBLTSymbol

    def peel(self, destructive: bool = False) -> Tuple[Set[IBLTItem], Set[IBLTItem]]:
        """
        Attempts to decode the IBLT to recover added and removed items.
        
        The peeling process iteratively finds "pure" cells (those containing
        a single item), decodes the item, and subtracts it from the table,
        potentially creating new pure cells.

        Args:
            destructive: If True, performs peeling on the IBLT in-place,
                         leaving it empty on success. If False (default),
                         works on a copy, leaving the original unchanged.

        Returns:
            A tuple containing two sets:
            - `added`: Items present in the IBLT (count > 0).
            - `removed`: Items that were subtracted from the IBLT (count < 0).
        
        Note:
            If the IBLT is over-saturated or contains too many hash
            collisions, decoding may be incomplete.
        """
        decoder = self if destructive else self.copy()
        added: Set[IBLTItem] = set()
        removed: Set[IBLTItem] = set()
        
        pure_indices = [i for i, cell in enumerate(decoder.cells) if cell.is_pure()]
        
        while pure_indices:
            idx = pure_indices.pop()
            cell = decoder.cells[idx]
            
            # The cell might have become non-pure due to other peeling operations
            if not cell.is_pure():
                continue
            
            item = self.symbol_type.to_item(cell)

            if cell.count == 1:
                added.add(item)
            else: # count == -1
                removed.add(item)
            
            # "Peel" the decoded item from the table
            decoder.remove(item)
            
            # Check if this peeling action created new pure cells
            for affected_idx in decoder._get_indices(item.get_key()):
                if decoder.cells[affected_idx].is_pure():
                    # We can add to list without checking for duplicates because
                    # an index will only become pure once.
                    pure_indices.append(affected_idx)
        
        # After peeling, if the decoder is not completely empty, it means
        # some items could not be resolved.
        if not all(c.is_empty() for c in decoder.cells):
            # In a real application, you might use a logger here.
            print("Warning: IBLT decoding may be incomplete due to unresolvable collisions.")
            
        return added, removed