# src/filter/riblt.py

import heapq
from typing import Any, Tuple, Iterator, Deque, List, Callable
from collections import deque
from .base import FilterBase
from .utils import InputType, Hasher, IndexGenerator, _inputtype_to_bytes
from .iblt import IBLTSymbol # RIBLT can reuse IBLTSymbol

class RIBLT(FilterBase[IBLTSymbol]):
    """可动态扩展的、受 Raptor 启发的 IBLT。"""
    
    class _SymbolQueue:
        def __init__(self):
            self.pq: List[Tuple[float, int, IndexGenerator, IBLTSymbol]] = []
            self.counter = 0

        def enqueue(self, gen: IndexGenerator, sym: IBLTSymbol):
            heapq.heappush(self.pq, (gen.curr, self.counter, gen, sym))
            self.counter += 1
        
        def next_coded_index(self) -> float:
            return self.pq[0][0] if self.pq else float('inf')

        def pop(self) -> Tuple[IndexGenerator, IBLTSymbol]:
            _, _, gen, sym = heapq.heappop(self.pq)
            return gen, sym

    def __init__(self, seed: InputType = "riblt-seed", hash_algo: str = "sha256"):
        super().__init__()
        self.seed = seed
        self.hasher = Hasher(seed=f"riblt-hasher-{seed}", algo=hash_algo)
        self.cells = [] # 初始为空
        self._queue = self._SymbolQueue()
        self.done_expanding = False

    def _get_indices(self, item: InputType) -> Iterator[int]:
        gen = IndexGenerator(item, self.seed)
        while gen.curr < len(self.cells):
            yield int(gen.curr)
            gen.jump()
    
    def _diffuse(self, symbol: IBLTSymbol, gen: IndexGenerator, on_diffuse: Callable[[int], None] = None):
        while gen.curr < len(self.cells):
            idx = int(gen.curr)
            self.cells[idx].add(symbol)
            if on_diffuse: on_diffuse(idx)
            gen.jump()
            
    def add(self, key: InputType, value: Any = b''):
        source_symbol = IBLTSymbol(1, _inputtype_to_bytes(key), _inputtype_to_bytes(value), 
                                   self.hasher.digest_int(key))
        gen = IndexGenerator(key, self.seed)
        self._diffuse(source_symbol, gen)
        if not self.done_expanding: self._queue.enqueue(gen, source_symbol)

    def expand(self, n: int):
        if self.done_expanding: raise RuntimeError("不能对已固化的 RIBLT 进行扩展。")
        self.cells.extend([IBLTSymbol() for _ in range(n)])
        
        processed = []
        while self._queue.next_coded_index() < len(self.cells):
            gen, sym = self._queue.pop()
            self._diffuse(sym, gen)
            processed.append((gen, sym))
        
        for gen, sym in processed: self._queue.enqueue(gen, sym)

    def set_done_expanding(self):
        self.done_expanding = True
        self._queue = self._SymbolQueue()
        
    def peel(self) -> Iterator[Tuple[bytes, Any, int]]:
        pure_indices: Deque[int] = deque(i for i, cell in enumerate(self.cells) if cell.is_pure(self.hasher))
        while pure_indices:
            index = pure_indices.popleft()
            cell = self.cells[index]
            if not cell.is_pure(self.hasher): continue
            
            key, value = cell.get_key_value()
            yield key, value, cell.count
            
            symbol_to_remove = IBLTSymbol(1, key, value, self.hasher.digest_int(key))
            if cell.count > 0: symbol_to_remove = symbol_to_remove.negate()
            
            gen_to_remove = IndexGenerator(key, self.seed)
            def on_peel(idx):
                if self.cells[idx].is_pure(self.hasher) and idx not in pure_indices:
                    pure_indices.append(idx)
            self._diffuse(symbol_to_remove, gen_to_remove, on_peel)