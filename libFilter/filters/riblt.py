# libFilter/filters/riblt.py

"""
Implementation of a Robust Invertible Bloom Lookup Table (RIBLT).

The RIBLT is a probabilistic data structure for set reconciliation that,
unlike a standard IBLT, does not require a pre-set size. It can dynamically
grow and process a stream of items, making it "robust" to unknown set
difference sizes. Decoding can be performed incrementally as more data arrives.
"""

from __future__ import annotations
from typing import Set, Tuple, List, Dict, Any, Type, Optional, Iterator
import heapq

# Relative imports from within the library.
from ..core.base import FilterItem, PeelableSymbol, FilterBase
from ..core.utils import (
    InputType, _xor_bytes, IndexGenerator,
    serialize_typed_value, deserialize_typed_value
)

# --- RIBLT-specific Components (reusing IBLT's Item and Symbol) ---

# For set-reconciliation, RIBLT's item and symbol structure is identical to IBLT's.
# We can re-use them directly or alias them for clarity.
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
    """A cell (symbol) in a RIBLT, identical in structure to an IBLTSymbol."""
    __slots__ = ('count', 'key_sum', 'value_sum')

    def __init__(self, count: int = 0, key_sum: bytes = b'', value_sum: bytes = b''):
        super().__init__(count=count, key_sum=key_sum, value_sum=value_sum)
        self.count = count
        self.key_sum = key_sum
        self.value_sum = value_sum

    def __iadd__(self, other: RIBLTSymbol) -> RIBLTSymbol:
        self.count += other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self

    def __isub__(self, other: RIBLTSymbol) -> RIBLTSymbol:
        self.count -= other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        return self

    def is_empty(self) -> bool:
        key_is_zero = not self.key_sum or all(b == 0 for b in self.key_sum)
        value_is_zero = not self.value_sum or all(b == 0 for b in self.value_sum)
        return self.count == 0 and key_is_zero and value_is_zero

    def is_pure(self) -> bool:
        if not (self.count == 1 or self.count == -1):
            return False
        # To be truly pure, the key must also hash to the value (or a hash thereof).
        # This IBLT/RIBLT version relies on being a "dumb" container, where peeling
        # and checking happens externally or is assumed correct. For simplicity,
        # we only check the count here, matching the C++ `isPure` core logic.
        return True

    def get_state(self) -> Dict[str, Any]:
        return {'count': self.count, 'key_sum': self.key_sum, 'value_sum': self.value_sum}

    @classmethod
    def from_item(cls: Type[RIBLTSymbol], item: RIBLTItem, **kwargs: Any) -> RIBLTSymbol:
        return cls(1, serialize_typed_value(item.key), serialize_typed_value(item.value))

    @classmethod
    def to_item(cls, symbol: RIBLTSymbol) -> RIBLTItem:
        if not symbol.is_pure():
            raise ValueError("Cannot decode item from a non-pure symbol.")
        key = deserialize_typed_value(symbol.key_sum)
        value = deserialize_typed_value(symbol.value_sum)
        return RIBLTItem(key, value)


class SymbolQueue:
    """
    A queue for managing symbols whose index sequences extend beyond the
    current size of the RIBLT. It uses a min-heap to efficiently find
    the next symbol to process when the RIBLT expands.
    """
    __slots__ = ('_heap', '_entries', '_next_id')

    def __init__(self):
        self._heap = []  # The min-heap: (next_index, entry_id, symbol_entry)
        self._entries = {} # Maps entry_id to symbol_entry for updates
        self._next_id = 0 # Unique ID to handle heap tie-breaking

    def enqueue(self, symbol: RIBLTSymbol, generator: IndexGenerator):
        """Adds a new symbol and its generator to the queue."""
        entry_id = self._next_id
        symbol_entry = {'symbol': symbol, 'generator': generator}
        self._entries[entry_id] = symbol_entry
        heap_item = (generator.curr, entry_id)
        heapq.heappush(self._heap, heap_item)
        self._next_id += 1

    def next_coded_index(self) -> int:
        """Returns the index of the next symbol to be processed, or infinity."""
        return self._heap[0][0] if self._heap else float('inf')

    def pop_and_reschedule(self) -> Tuple[RIBLTSymbol, IndexGenerator]:
        """
        Pops the symbol with the smallest next index, updates its generator,
        and reschedules it in the queue.
        """
        next_idx, entry_id = heapq.heappop(self._heap)
        
        entry = self._entries.pop(entry_id)
        symbol = entry['symbol']
        generator = entry['generator']

        # The generator is advanced by the calling `diffuse` function.
        # We just need to re-enqueue it with its new `curr` index.
        self.enqueue(symbol, generator)
        
        # Return a copy of the generator state at the time of popping
        return symbol, generator

    def top_generator(self) -> IndexGenerator:
        """Returns the generator of the top item without removing it."""
        entry_id = self._heap[0][1]
        return self._entries[entry_id]['generator']
        
    def __len__(self) -> int:
        return len(self._entries)


class RIBLT(FilterBase[RIBLTSymbol, RIBLTItem]):
    """
    A Robust Invertible Bloom Lookup Table.
    """
    __slots__ = (
        'cells', '_symbol_queue', '_seed', '_peeled_items',
        '_next_peel_idx', 'done_expanding'
    )

    def __init__(self, seed: Any = "default_riblt_seed"):
        super().__init__()
        self._seed = seed
        self._symbol_queue = SymbolQueue()
        self.done_expanding = False
        
        # --- Attributes for Incremental Peeling ---
        self._peeled_items: Set[RIBLTItem] = set()
        self._next_peel_idx = 0

    def push(self, item: RIBLTItem) -> None:
        """Adds an item to the RIBLT."""
        self._diffuse(RIBLTSymbol.from_item(item), is_new_item=True)
        
    def remove(self, item: RIBLTItem) -> None:
        """Subtracts an item from the RIBLT."""
        symbol = RIBLTSymbol.from_item(item)
        # Negate the count for subtraction
        symbol.count = -1
        self._diffuse(symbol, is_new_item=True)

    def _diffuse(self, symbol: RIBLTSymbol, is_new_item: bool, generator: IndexGenerator = None):
        """
        Spreads a symbol's contribution across the filter's cells.
        If `is_new_item` is True, a new generator is created.
        """
        if self.done_expanding and is_new_item:
            raise RuntimeError("Cannot add new items after calling `set_done_expanding`.")

        if generator is None:
            # For new items, create a fresh generator.
            # We use the raw key/value sum as the seed source for the generator.
            # This ensures add/remove ops on the same item use the same index sequence.
            key_for_gen = symbol.key_sum + symbol.value_sum
            generator = IndexGenerator(key=key_for_gen, seed=self._seed)

        # Diffuse the symbol as far as possible within the current bounds.
        while generator.curr < len(self.cells):
            self.cells[generator.curr] += symbol
            generator.jump()

        # If it's a new item that will continue to evolve, enqueue it.
        if is_new_item and not self.done_expanding:
            self._symbol_queue.enqueue(symbol, generator)

    def expand(self, n: int):
        """Expands the RIBLT by `n` cells, processing queued symbols."""
        if self.done_expanding:
            raise RuntimeError("Cannot expand after calling `set_done_expanding`.")
        
        current_size = len(self.cells)
        new_size = current_size + n
        self.cells.extend([RIBLTSymbol() for _ in range(n)])

        # Process any queued symbols that now fall within the new bounds.
        while self._symbol_queue and self._symbol_queue.next_coded_index() < new_size:
            symbol, generator = self._symbol_queue.pop_and_reschedule()
            # This is not a new item, so `is_new_item` is False.
            self._diffuse(symbol, is_new_item=False, generator=generator)

    def set_done_expanding(self):
        """
        Signals that no more items will be added. This allows the symbol
        queue to be cleared to save memory.
        """
        self.done_expanding = True
        self._symbol_queue = SymbolQueue() # Free memory

    def peel(self) -> None:
        """
        Performs an incremental peel operation.
        
        This method attempts to decode items from the current state of the RIBLT.
        Successfully decoded items are stored in the `added_items` and
        `removed_items` properties and are subtracted from the RIBLT state.
        
        This method can be called multiple times. If the first call doesn't
        fully decode the filter, you can `push`/`remove` more items and then
        call `peel` again to continue the process.
        """
        pure_indices = []
        # Scan for new pure cells since the last peel operation.
        for i in range(self._next_peel_idx, len(self.cells)):
            if self.cells[i].is_pure():
                pure_indices.append(i)
        self._next_peel_idx = len(self.cells)

        while pure_indices:
            idx = pure_indices.pop()
            cell = self.cells[idx]
            
            if not cell.is_pure():
                continue

            item = RIBLTSymbol.to_item(cell)
            
            # If we've already peeled this item, it means it was part of a
            # more complex structure that has now resolved. We can ignore it.
            if item in self._peeled_items:
                continue
                
            self._peeled_items.add(item)
            
            # Subtract the peeled item's contribution from the RIBLT.
            # This is like `remove`, but uses the already negated count if needed.
            symbol_to_peel = RIBLTSymbol(
                count=-cell.count, # If count was 1, subtract 1. If -1, add 1.
                key_sum=cell.key_sum,
                value_sum=cell.value_sum
            )

            # We must use a new generator for the peeling diffusion.
            key_for_gen = symbol_to_peel.key_sum + symbol_to_peel.value_sum
            peel_generator = IndexGenerator(key=key_for_gen, seed=self._seed)
            
            # Diffuse the negated symbol.
            while peel_generator.curr < len(self.cells):
                affected_idx = peel_generator.curr
                self.cells[affected_idx] += symbol_to_peel
                
                # Check if this action created a new pure cell.
                if self.cells[affected_idx].is_pure():
                    pure_indices.append(affected_idx)
                
                peel_generator.jump()

    @property
    def added_items(self) -> Set[RIBLTItem]:
        """Returns the set of successfully peeled items with a positive count."""
        # This is a placeholder; a real implementation might track counts.
        # For simple diff, we assume peeled items are the 'added' set.
        return self._peeled_items
    
    @property
    def removed_items(self) -> Set[RIBLTItem]:
        """Returns the set of successfully peeled items with a negative count."""
        # This RIBLT doesn't distinguish between added/removed in the peeled set.
        # For a full diff, one would need to inspect the original cell's count.
        # For now, we return an empty set.
        return set()

    def is_fully_decoded(self) -> bool:
        """Checks if the RIBLT is empty, indicating successful peeling."""
        return all(c.is_empty() for c in self.cells)

    # --- Serialization and other base methods ---
    # For RIBLT, full serialization is complex due to the queue state.
    # We provide a minimal implementation.
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Full serialization for RIBLT is complex and not yet implemented.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RIBLT":
        raise NotImplementedError("Full serialization for RIBLT is complex and not yet implemented.")