# libFilter/filters/riblt.py

"""
Implementation of a Robust Invertible Bloom Lookup Table (RIBLT).

This version incorporates a hash_sum field for robust, self-verifying
purity checks in cells, making it resilient to key/value collisions.
The index generation is based solely on the item's key. The implementation
is refactored for clarity, logical cohesion, and efficiency, and its
behavior aligns with standard IBLT principles for strict key-value pair
reconciliation.
"""

from __future__ import annotations
from typing import Set, Tuple, List, Dict, Any, Type, Optional
import heapq
import hashlib

# Relative imports from within the library.
from ..core.base import FilterItem, PeelableSymbol, FilterBase
from ..core.utils import (
    InputType, _xor_bytes, IndexGenerator,
    serialize_typed_value, deserialize_typed_value
)

# --- Helper for hash_sum ---
def _hash_key(key_bytes: bytes) -> bytes:
    """A consistent hash function for the key part of a symbol."""
    return hashlib.sha256(key_bytes).digest()


# --- RIBLT-specific Components ---

class RIBLTItem(FilterItem):
    """An item for a RIBLT, containing a key-value pair."""
    __slots__ = ('key', 'value')

    def __init__(self, key: InputType, value: InputType):
        self.key = key
        self.value = value

    def get_key(self) -> InputType:
        return self.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RIBLTItem): return NotImplemented
        return self.key == other.key and self.value == other.value

    def __hash__(self) -> int:
        key_bytes = serialize_typed_value(self.key)
        value_bytes = serialize_typed_value(self.value)
        return hash((key_bytes, value_bytes))

    def __repr__(self) -> str:
        return f"RIBLTItem(key={self.key!r}, value={self.value!r})"


class RIBLTSymbol(PeelableSymbol):
    """
    A cell in a RIBLT, with a hash_sum for self-verification.
    """
    __slots__ = ('count', 'key_sum', 'value_sum', 'hash_sum')

    def __init__(self, count: int = 0, key_sum: bytes = b'', value_sum: bytes = b'', hash_sum: bytes = b''):
        super().__init__(count=count, key_sum=key_sum, value_sum=value_sum, hash_sum=hash_sum)
        self.count = count
        self.key_sum = key_sum
        self.value_sum = value_sum
        self.hash_sum = hash_sum

    def __iadd__(self, other: RIBLTSymbol) -> RIBLTSymbol:
        self.count += other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        self.hash_sum = _xor_bytes(self.hash_sum, other.hash_sum)
        return self

    def __isub__(self, other: RIBLTSymbol) -> RIBLTSymbol:
        self.count -= other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        self.hash_sum = _xor_bytes(self.hash_sum, other.hash_sum)
        return self

    def is_empty(self) -> bool:
        """
        A cell is empty only if all its components are zero. This provides
        strict checking for full reconciliation, including values.
        """
        return (self.count == 0 and
                (not self.key_sum or all(b == 0 for b in self.key_sum)) and
                (not self.value_sum or all(b == 0 for b in self.value_sum)) and
                (not self.hash_sum or all(b == 0 for b in self.hash_sum)))

    def is_pure(self) -> bool:
        """
        A symbol is pure if it represents a single, verifiable item.
        """
        if not (self.count == 1 or self.count == -1):
            return False
        
        if not self.key_sum:
            return False
        expected_hash = _hash_key(self.key_sum)
        return self.hash_sum == expected_hash

    def get_state(self) -> Dict[str, Any]:
        return {
            'count': self.count, 
            'key_sum': self.key_sum, 
            'value_sum': self.value_sum, 
            'hash_sum': self.hash_sum
        }

    @classmethod
    def from_item(cls: Type[RIBLTSymbol], item: RIBLTItem, **kwargs: Any) -> RIBLTSymbol:
        key_b = serialize_typed_value(item.key)
        val_b = serialize_typed_value(item.value)
        hash_b = _hash_key(key_b)
        return cls(1, key_b, val_b, hash_b)

    @classmethod
    def to_item(cls, symbol: RIBLTSymbol) -> RIBLTItem:
        if not symbol.is_pure():
            raise ValueError("Cannot decode item from a non-pure symbol.")
        key = deserialize_typed_value(symbol.key_sum)
        value = deserialize_typed_value(symbol.value_sum)
        return RIBLTItem(key, value)


class SymbolQueue:
    """
    A self-managing queue that stores symbols and handles their diffusion
    into the RIBLT's cells as the table expands.
    """
    __slots__ = ('_heap', '_entries', '_next_id')

    def __init__(self):
        self._heap: List[Tuple[int, int]] = []
        self._entries: Dict[int, Dict[str, Any]] = {}
        self._next_id = 0

    def enqueue_and_diffuse(self, symbol: RIBLTSymbol, generator: IndexGenerator, cells: List[RIBLTSymbol]):
        while generator.curr < len(cells):
            cells[generator.curr] += symbol
            generator.jump()
        
        entry_id = self._next_id
        self._entries[entry_id] = {'symbol': symbol, 'generator': generator}
        heapq.heappush(self._heap, (generator.curr, entry_id))
        self._next_id += 1

    def expand_and_diffuse(self, cells: List[RIBLTSymbol]):
        limit_index = len(cells)
        while self._heap and self._heap[0][0] < limit_index:
            top_entry_id = self._heap[0][1]
            entry = self._entries[top_entry_id]
            symbol, generator = entry['symbol'], entry['generator']
            
            while generator.curr < limit_index:
                cells[generator.curr] += symbol
                generator.jump()
            
            heapq.heapreplace(self._heap, (generator.curr, top_entry_id))
    
    def clear(self):
        self._heap.clear()
        self._entries.clear()
        
    def __len__(self) -> int:
        return len(self._entries)


class RIBLT(FilterBase[RIBLTSymbol, RIBLTItem]):
    """
    A Robust Invertible Bloom Lookup Table (with hash_sum verification).
    """
    __slots__ = (
        'cells', '_symbol_queue', '_seed', '_added_items', '_removed_items',
        '_peeled_indices', 'done_expanding'
    )

    def __init__(self, seed: Any = "default_riblt_seed"):
        super().__init__()
        self._seed = seed
        self._symbol_queue = SymbolQueue()
        self.done_expanding = False
        self._added_items: Set[RIBLTItem] = set()
        self._removed_items: Set[RIBLTItem] = set()
        self._peeled_indices: Set[int] = set()

    def _get_generator_for_item(self, item_key: InputType) -> IndexGenerator:
        return IndexGenerator(key=item_key, seed=self._seed)

    def _process_item(self, item: RIBLTItem, count: int):
        if self.done_expanding:
            raise RuntimeError("Cannot modify RIBLT after calling `set_done_expanding`.")

        if count == 1 and item in self._removed_items:
            self._removed_items.remove(item)
            return
        if count == -1 and item in self._added_items:
            self._added_items.remove(item)
            return

        symbol = RIBLTSymbol.from_item(item)
        symbol.count = count
        generator = self._get_generator_for_item(item.get_key())
        self._symbol_queue.enqueue_and_diffuse(symbol, generator, self.cells)

    def push(self, item: RIBLTItem) -> None:
        self._process_item(item, 1)

    def remove(self, item: RIBLTItem) -> None:
        self._process_item(item, -1)

    def expand(self, n: int):
        if self.done_expanding:
            raise RuntimeError("Cannot expand after calling `set_done_expanding`.")
        if n <= 0: return

        self.cells.extend([RIBLTSymbol() for _ in range(n)])
        self._symbol_queue.expand_and_diffuse(self.cells)

    def set_done_expanding(self):
        self.done_expanding = True
        self._symbol_queue.clear()

    def peel(self) -> bool:
        items_peeled_this_round = 0
        pure_indices = [
            i for i in range(len(self.cells) - 1, -1, -1) 
            if self.cells[i].is_pure() and i not in self._peeled_indices
        ]

        while pure_indices:
            idx = pure_indices.pop()
            
            if not self.cells[idx].is_pure():
                continue
                
            self._peeled_indices.add(idx)
            
            cell_to_peel = self.cells[idx]
            item = RIBLTSymbol.to_item(cell_to_peel)
            
            items_peeled_this_round += 1
            if cell_to_peel.count == 1:
                self._added_items.add(item)
            else:
                self._removed_items.add(item)
            
            inverse_symbol = RIBLTSymbol(
                count=cell_to_peel.count,
                key_sum=cell_to_peel.key_sum,
                value_sum=cell_to_peel.value_sum,
                hash_sum=cell_to_peel.hash_sum
            )

            peel_generator = self._get_generator_for_item(item.get_key())
            
            while peel_generator.curr < len(self.cells):
                affected_idx = peel_generator.curr
                if affected_idx != idx and affected_idx not in self._peeled_indices:
                    self.cells[affected_idx] -= inverse_symbol
                    if self.cells[affected_idx].is_pure():
                        pure_indices.append(affected_idx)
                peel_generator.jump()
            
            self.cells[idx] = RIBLTSymbol()

        return items_peeled_this_round > 0

    @property
    def added_items(self) -> Set[RIBLTItem]:
        return self._added_items
    
    @property
    def removed_items(self) -> Set[RIBLTItem]:
        return self._removed_items

    def is_fully_decoded(self) -> bool:
        return all(c.is_empty() for c in self.cells)

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Full serialization for streaming RIBLT is non-trivial.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RIBLT":
        raise NotImplementedError("Full serialization for streaming RIBLT is non-trivial.")