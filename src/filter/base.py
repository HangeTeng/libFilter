from __future__ import annotations
import abc
import copy
import math
from typing import List, Iterator, TypeVar, Generic, Type
from .utils import InputType, HashMapping

# --- Generic Types ---
ItemType = TypeVar('ItemType', bound='FilterItem')
SymbolType = TypeVar('SymbolType', bound='FilterSymbol')

# --- Abstract Interfaces ---
class FilterItem(abc.ABC):
    """Interface for items that can be added to a filter."""
    __slots__ = ()
    
    @abc.abstractmethod
    def get_key(self) -> InputType:
        """Extracts a unique, hashable key from the item."""
        pass

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Provides an unambiguous representation of the item."""
        pass

class FilterSymbol(abc.ABC):
    """Interface for a single cell within a filter."""
    __slots__ = ()

    @abc.abstractmethod
    def __iadd__(self, other: FilterSymbol) -> FilterSymbol:
        """In-place adds the influence of another symbol."""
        pass

    @abc.abstractmethod
    def __isub__(self, other: FilterSymbol) -> FilterSymbol:
        """In-place removes the influence of another symbol."""
        pass

    @abc.abstractmethod
    def is_empty(self) -> bool:
        """Checks if the symbol is in its initial (zero) state."""
        pass
    
    @classmethod
    @abc.abstractmethod
    def from_item(cls, item: ItemType) -> FilterSymbol:
        """Creates a 'source' symbol from a filter item."""
        pass

    @classmethod
    @abc.abstractmethod
    def _get_key(cls, item: ItemType) -> InputType:
        """A helper to extract the key from an item, used by filters."""
        pass

    def __str__(self) -> str:
        return self.__class__.__name__
        
    def __repr__(self) -> str:
        return f"<{str(self)}>"

class PeelableSymbol(FilterSymbol):
    """Extended interface for symbols in decodable filters (e.g., IBLT)."""
    
    @abc.abstractmethod
    def is_pure(self, hasher: Hasher) -> bool:
        """Checks if the symbol contains a single, uncorrupted item."""
        pass
    
    @classmethod
    @abc.abstractmethod
    def to_item(cls, symbol: "PeelableSymbol") -> ItemType:
        """Recovers the original item from a pure symbol."""
        pass

# --- Filter Base Classes ---
class FilterBase(abc.ABC, Generic[SymbolType, ItemType]):
    """Abstract base class for all filters, defining common behaviors."""
    __slots__ = 'cells'

    def __init__(self):
        self.cells: List[SymbolType] = []
    
    @abc.abstractmethod
    def push(self, item: ItemType) -> None:
        """Adds an item to the filter."""
        pass

    @abc.abstractmethod
    def remove(self, item: ItemType) -> None:
        """Removes an item from the filter."""
        pass
    
    def __iadd__(self, other: FilterBase) -> FilterBase:
        """In-place merges another filter into this one."""
        if len(self.cells) != len(other.cells):
            raise ValueError("Filters must be the same size to merge.")
        for i in range(len(self.cells)):
            self.cells[i] += other.cells[i]
        return self

    def __isub__(self, other: FilterBase) -> FilterBase:
        """In-place subtracts another filter from this one."""
        if len(self.cells) != len(other.cells):
            raise ValueError("Filters must be the same size for subtraction.")
        for i in range(len(self.cells)):
            self.cells[i] -= other.cells[i]
        return self

    def __add__(self, other: FilterBase) -> FilterBase:
        """Returns a new filter containing the merge of two filters."""
        result = copy.deepcopy(self)
        result += other
        return result

    def __sub__(self, other: FilterBase) -> FilterBase:
        """Returns a new filter representing the difference of two filters."""
        result = copy.deepcopy(self)
        result -= other
        return result

    def __len__(self) -> int:
        """Returns the number of cells (m) in the filter."""
        return len(self.cells)

    def to_string(self, verbose: bool = False, display_limit: int = 32) -> str:
        """Generates a detailed string representation of the filter."""
        size = len(self.cells)
        k_val = getattr(self, 'k', 'N/A')
        if size == 0:
            return f"{self.__class__.__name__}(size=0, k={k_val})"

        header = f"{self.__class__.__name__}(size={size}, k={k_val})"
        lines = [header]
        
        non_empty_cells = [(i, cell) for i, cell in enumerate(self.cells) if not cell.is_empty()]
        num_non_empty = len(non_empty_cells)
        occupancy = num_non_empty / size
        
        lines.append(f"  Summary: {num_non_empty}/{size} cells occupied ({occupancy:.2%})")

        if not verbose:
            if num_non_empty > 0:
                for i, (index, cell) in enumerate(non_empty_cells):
                    if i >= display_limit:
                        lines.append(f"  ... and {num_non_empty - display_limit} more non-empty cells")
                        break
                    lines.append(f"  - Index [{index}]: {cell}")
            else:
                lines.append("  (All cells are empty)")
        else:
            limit = min(size, display_limit)
            for i in range(limit):
                cell = self.cells[i]
                mark = "*" if not cell.is_empty() else " "
                lines.append(f"  {mark} [{i:>{len(str(size-1))}}]: {cell}")
            if size > limit:
                lines.append(f"  ... ({size - limit} more cells)")

        return "\n".join(lines)
    
    def __str__(self) -> str:
        """Returns a concise, human-readable string representation."""
        return self.to_string(verbose=False)

    @abc.abstractmethod
    def __repr__(self) -> str:
        """Provides an unambiguous, ideally reconstructible, representation."""
        pass

class StandardFilter(FilterBase[SymbolType, ItemType]):
    """A concrete base for filters using a k-hash mapping (BF, CBF, IBLT)."""
    __slots__ = 'hash_mapping', 'm', 'k'
    
    symbol_type: Type[SymbolType] # Must be defined by subclasses

    def __init__(self, hash_mapping: HashMapping):
        super().__init__()
        if not hasattr(self.__class__, 'symbol_type'):
            raise NotImplementedError(f"{self.__class__.__name__} must define 'symbol_type'.")
            
        self.hash_mapping = hash_mapping
        self.m = hash_mapping.table_size
        self.k = len(hash_mapping.hashers)
        self.cells = [self.__class__.symbol_type() for _ in range(self.m)]

    def _get_indices(self, key: InputType) -> Iterator[int]:
        """Gets the k indices for a given key."""
        return self.hash_mapping.indices(key)
    
    def push(self, item: ItemType) -> None:
        st = self.__class__.symbol_type
        source_symbol = st.from_item(item)
        for index in self._get_indices(st._get_key(item)):
            self.cells[index] += source_symbol

    def remove(self, item: ItemType) -> None:
        st = self.__class__.symbol_type
        source_symbol = st.from_item(item)
        for index in self._get_indices(st._get_key(item)):
            self.cells[index] -= source_symbol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(hash_mapping={self.hash_mapping!r})"