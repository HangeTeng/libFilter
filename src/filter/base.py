# src/filter/base.py

from __future__ import annotations
import abc
from typing import List, Iterator, TypeVar, Generic, Type, Any, Dict

from .utils import InputType, HashMapping, Hasher

# --- Generic Types ---
ItemType = TypeVar('ItemType', bound='FilterItem')
SymbolType = TypeVar('SymbolType', bound='FilterSymbol')

# --- Abstract Interfaces ---
class FilterItem(abc.ABC):
    """Interface for items that can be added to a filter."""
    __slots__ = ()
    @abc.abstractmethod
    def get_key(self) -> InputType: pass
    @abc.abstractmethod
    def __repr__(self) -> str: pass

class FilterSymbol(abc.ABC):
    """Interface for a single cell within a filter."""
    __slots__ = ()

    def __init__(self, **kwargs): super().__init__()
    @abc.abstractmethod
    def __iadd__(self, other: "FilterSymbol") -> "FilterSymbol": pass
    @abc.abstractmethod
    def __isub__(self, other: "FilterSymbol") -> "FilterSymbol": pass
    @abc.abstractmethod
    def is_empty(self) -> bool: pass
    @abc.abstractmethod
    def __getstate__(self) -> Dict[str, Any]:
        """Return state as a dict. Keys must match __init__ args."""
        pass
    @classmethod
    @abc.abstractmethod
    def from_item(cls: Type[SymbolType], item: ItemType, **kwargs) -> SymbolType: pass
    @classmethod
    @abc.abstractmethod
    def _get_key(cls, item: ItemType) -> InputType: pass
    def __str__(self) -> str: return self.__class__.__name__
    def __repr__(self) -> str: return f"<{str(self)}>"

class PeelableSymbol(FilterSymbol):
    """Extended interface for symbols in decodable filters (e.g., IBLT)."""
    __slots__ = ()
    @abc.abstractmethod
    def is_pure(self) -> bool: pass
    @classmethod
    @abc.abstractmethod
    def to_item(cls, symbol: "PeelableSymbol") -> ItemType: pass

# --- Filter Base Classes ---
class FilterBase(abc.ABC, Generic[SymbolType, ItemType]):
    """Abstract base class for all filters."""
    __slots__ = 'cells'

    def __init__(self): self.cells: List[SymbolType] = []
    @classmethod
    @abc.abstractmethod
    def from_dict(cls: Type["FilterBase"], data: Dict[str, Any]) -> "FilterBase": pass
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]: pass
    def copy(self) -> "FilterBase": return self.__class__.from_dict(self.to_dict())
    
    @abc.abstractmethod
    def push(self, item: ItemType) -> None: pass
    @abc.abstractmethod
    def remove(self, item: ItemType) -> None: pass

    def __iadd__(self, other: FilterBase) -> FilterBase:
        if len(self.cells) != len(other.cells): raise ValueError("Size mismatch")
        for i, cell in enumerate(other.cells): self.cells[i] += cell
        return self
    def __isub__(self, other: FilterBase) -> FilterBase:
        if len(self.cells) != len(other.cells): raise ValueError("Size mismatch")
        for i, cell in enumerate(other.cells): self.cells[i] -= cell
        return self
    def __add__(self, other: FilterBase) -> FilterBase:
        result = self.copy(); result += other; return result
    def __sub__(self, other: FilterBase) -> FilterBase:
        result = self.copy(); result -= other; return result

    def __len__(self) -> int: return len(self.cells)
    def __str__(self) -> str: return self.to_string(False)
    def __repr__(self) -> str: return f"<{self.__class__.__name__}>"

    def to_string(self, verbose: bool, display_limit: int = 32) -> str:
        """Generates a detailed string representation of the filter."""
        size, k_val = len(self.cells), getattr(self, 'k', 'N/A')
        header = f"{self.__class__.__name__}(size={size}, k={k_val})"
        if size == 0: return header
        
        non_empty = [(i, c) for i, c in enumerate(self.cells) if not c.is_empty()]
        summary = f"  Summary: {len(non_empty)}/{size} cells occupied ({len(non_empty)/size:.2%})"
        lines = [header, summary]
        
        cells_to_show = self.cells if verbose else [c for _, c in non_empty]
        limit = display_limit if not verbose else min(size, display_limit)

        for i, cell in enumerate(cells_to_show[:limit]):
            idx_str = non_empty[i][0] if not verbose else i
            mark = "*" if not cell.is_empty() else " "
            lines.append(f"  {mark} [{idx_str:>{len(str(size-1))}}]: {cell}")
        
        if len(cells_to_show) > limit:
            lines.append(f"  ... and {len(cells_to_show) - limit} more")
        return "\n".join(lines)

class StandardFilter(FilterBase[SymbolType, ItemType]):
    """A base for filters using a k-hash mapping (BF, CBF, IBLT)."""
    __slots__ = 'hash_mapping', 'm', 'k'
    symbol_type: Type[SymbolType]

    def __init__(self, hash_mapping: HashMapping):
        super().__init__()
        if not hasattr(self.__class__, 'symbol_type'):
            raise NotImplementedError(f"{self.__class__.__name__} must define 'symbol_type'.")
        self.hash_mapping = hash_mapping; self.m = hash_mapping.table_size; self.k = len(hash_mapping.hashers)
        self.cells = [self.__class__.symbol_type() for _ in range(self.m)]

    @classmethod
    def from_dict(cls: Type["StandardFilter"], data: Dict[str, Any]) -> "StandardFilter":
        hash_mapping = HashMapping.from_config(data['hash_mapping'])
        instance = cls(hash_mapping)
        instance.cells = [cls.symbol_type(**s_data) for s_data in data['cells']]
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hash_mapping': self.hash_mapping.get_config(),
            'cells': [s.__getstate__() for s in self.cells]
        }

    def _get_indices(self, key: InputType) -> Iterator[int]:
        return self.hash_mapping.indices(key)
    
    def push(self, item: ItemType) -> None:
        source_symbol = self.__class__.symbol_type.from_item(item)
        for index in self._get_indices(self.__class__.symbol_type._get_key(item)):
            self.cells[index] += source_symbol

    def remove(self, item: ItemType) -> None:
        source_symbol = self.__class__.symbol_type.from_item(item)
        for index in self._get_indices(self.__class__.symbol_type._get_key(item)):
            self.cells[index] -= source_symbol

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.hash_mapping!r})"