# libfilter/core/base.py (Corrected Version)

"""
Defines the abstract base classes and interfaces for the filter library.

This module establishes the core contracts that all filter implementations
and their components (items, symbols) must adhere to. It promotes a
consistent API and structure across different filter types.
"""

from __future__ import annotations
import abc
from typing import (
    List, Iterator, TypeVar, Generic, Type, Any, Dict
)

# Use a relative import to access sibling modules within the same package.
from .utils import InputType, HashMapping


# --- Generic Type Variables ---

ItemType = TypeVar('ItemType', bound='FilterItem')
SymbolType = TypeVar('SymbolType', bound='FilterSymbol')


# --- Abstract Interfaces ---

class FilterItem(abc.ABC):
    __slots__ = ()

    @abc.abstractmethod
    def get_key(self) -> InputType:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError


class FilterSymbol(abc.ABC):
    __slots__ = ()

    def __init__(self, **kwargs: Any):
        super().__init__()

    @abc.abstractmethod
    def __iadd__(self, other: FilterSymbol) -> FilterSymbol:
        raise NotImplementedError

    @abc.abstractmethod
    def __isub__(self, other: FilterSymbol) -> FilterSymbol:
        raise NotImplementedError

    @abc.abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_item(cls: Type[SymbolType], item: ItemType, **kwargs: Any) -> SymbolType:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        state = self.get_state()
        state_str = ", ".join(f"{k}={v!r}" for k, v in state.items())
        return f"{self.__class__.__name__}({state_str})"


class PeelableSymbol(FilterSymbol):
    __slots__ = ()

    @abc.abstractmethod
    def is_pure(self) -> bool:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def to_item(cls, symbol: PeelableSymbol) -> ItemType:
        raise NotImplementedError


# --- Filter Base Classes ---

class FilterBase(abc.ABC, Generic[SymbolType, ItemType]):
    __slots__ = ('cells',)

    def __init__(self) -> None:
        self.cells: List[SymbolType] = []

    @classmethod
    @abc.abstractmethod
    def from_dict(cls: Type[FilterBase], data: Dict[str, Any]) -> FilterBase:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def copy(self) -> FilterBase:
        return self.__class__.from_dict(self.to_dict())
    
    @abc.abstractmethod
    def push(self, item: ItemType) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def remove(self, item: ItemType) -> None:
        raise NotImplementedError

    def __iadd__(self, other: FilterBase) -> FilterBase:
        if len(self.cells) != len(other.cells):
            raise ValueError("Filters must have the same size for addition.")
        for i, cell in enumerate(other.cells):
            self.cells[i] += cell
        return self

    def __isub__(self, other: FilterBase) -> FilterBase:
        if len(self.cells) != len(other.cells):
            raise ValueError("Filters must have the same size for subtraction.")
        for i, cell in enumerate(other.cells):
            self.cells[i] -= cell
        return self

    def __add__(self, other: FilterBase) -> FilterBase:
        result = self.copy()
        result += other
        return result

    def __sub__(self, other: FilterBase) -> FilterBase:
        result = self.copy()
        result -= other
        return result

    def __len__(self) -> int:
        return len(self.cells)

    def __str__(self) -> str:
        return self.to_string(verbose=False)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} len={len(self)}>"

    def to_string(self, verbose: bool, display_limit: int = 32) -> str:
        size = len(self.cells)
        k_val = getattr(self, 'k', 'N/A')
        
        header = f"{self.__class__.__name__}(size={size}, k={k_val})"
        if size == 0:
            return header
        
        non_empty_cells = [(i, c) for i, c in enumerate(self.cells) if not c.is_empty()]
        occupation = len(non_empty_cells)
        ratio = occupation / size if size > 0 else 0
        summary = f"  Summary: {occupation}/{size} cells occupied ({ratio:.2%})"
        
        lines = [header, summary]
        
        items_to_show = self.cells if verbose else non_empty_cells
        limit = min(len(items_to_show), display_limit)
        
        if limit == 0:
            return "\n".join(lines)
            
        index_width = len(str(size - 1))

        for i in range(limit):
            if verbose:
                idx, cell = i, items_to_show[i]
            else:
                idx, cell = items_to_show[i]
            
            mark = "*" if not cell.is_empty() else " "
            lines.append(f"  {mark} [{idx:>{index_width}}]: {cell!r}")
        
        if len(items_to_show) > limit:
            lines.append(f"  ... and {len(items_to_show) - limit} more")
            
        return "\n".join(lines)


class StandardFilter(FilterBase[SymbolType, ItemType]):
    """
    A base class for standard filters using a k-hash mapping strategy.
    
    This class provides the common structure for filters like Bloom Filters,
    Counting Bloom Filters, and IBLTs. It manages the `HashMapping`, cell
    initialization, serialization, and the core `push`/`remove` logic based
    on mapping a key to k indices.
    """
    __slots__ = ('hash_mapping', 'm', 'k')
    
    symbol_type: Type[SymbolType]

    def __init__(self, hash_mapping: HashMapping):
        super().__init__()
        if not hasattr(self.__class__, 'symbol_type'):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define a 'symbol_type' class attribute."
            )
        
        self.hash_mapping = hash_mapping
        self.m = hash_mapping.table_size
        self.k = len(hash_mapping.hashers)
        
        self.cells = [self.symbol_type() for _ in range(self.m)]

    @classmethod
    def from_dict(cls: Type[StandardFilter], data: Dict[str, Any]) -> StandardFilter:
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        instance = cls(hash_mapping)
        instance.cells = [cls.symbol_type(**s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash_mapping': self.hash_mapping.get_config(),
            'cells': [s.get_state() for s in self.cells]
        }

    def _get_indices(self, key: InputType) -> Iterator[int]:
        """
        Retrieves the k indices for a given key using the hash mapping.
        
        This is a crucial helper method used by subclasses for operations
        like `push`, `remove`, and `__contains__`.
        """
        return self.hash_mapping.indices(key)
    
    def _get_item_key(self, item: ItemType) -> InputType:
        """
        A helper to consistently get the key from a filter item.
        
        This indirection simplifies the logic in `push`/`remove` and allows
        for future overrides if needed.
        """
        return item.get_key()

    def push(self, item: ItemType) -> None:
        """Adds an item by adding its symbol representation to k cells."""
        source_symbol = self.symbol_type.from_item(item)
        item_key = self._get_item_key(item)
        for index in self._get_indices(item_key):
            self.cells[index] += source_symbol

    def remove(self, item: ItemType) -> None:
        """Removes an item by subtracting its symbol representation from k cells."""
        source_symbol = self.symbol_type.from_item(item)
        item_key = self._get_item_key(item)
        for index in self._get_indices(item_key):
            self.cells[index] -= source_symbol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={self.m}, k={self.k})"