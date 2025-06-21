"""
大方案：Panther

Filter

架构：
SketchSymbol
有的方法好像不必要实现全部函数，比如bf不需要ispure，你看怎么处理让它继承的时候操作精简一些
用未实例化报错合适吗？


IndexMapping
hashfamily在这里用合适一些吧？毕竟riblt本质上只需要一个hash，不像其他要多个

Sketch
RIBLT中用于延迟扩散的 SymbolQueue


单拎出来：
基础类型定义
hash 不同类型的hash



我这里有一些需求，我有时候这个架构实现我自己的riblt4panther，sketchsymbol里的结构很不一样
（比如val的只可能很多，插入的方式不一样，算的hash不一样可能只用一个val，
合并的方式也不一样不是异或是，不同val用不同的，也可以不用hashsum），
我希望能有一个接口用于我在实例化一个特别的riblt的时候可以从外部设置不同的val或者传入一个函数来实现加法 合并 判纯

"""

import math
import random
import heapq
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Optional, Dict, Union

# --- 基础类型定义 ---
Scalar = Union[bool, bytes, float, int, str]
Value = Union[
    bool, bytes, float, int, str,
    List[bool], List[bytes], List[float], List[int], List[str]
]
ValueDict = Dict[str, Value]


# --- SketchSymbol 抽象类 ---
class SketchSymbol(ABC):
    def __init__(self, val: ValueDict):
        self.val = val

    @abstractmethod
    def add(self, other: 'SketchSymbol'):
        pass

    @abstractmethod
    def sub(self, other: 'SketchSymbol'):
        pass

    @abstractmethod
    def is_pure(self) -> bool:
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        pass


# --- IBLTSymbol 实现类 ---
class IBLTSymbol(SketchSymbol):
    def __init__(self, val: ValueDict = None):
        default = {
            "count": 0,
            "indexSum": 0,
            "valueSum": 0,
            "hashSum": 0,
        }
        merged_val = default if val is None else {**default, **val}
        super().__init__(merged_val)

    @staticmethod
    def _hash_int(x: int) -> int:
        import hashlib
        h = hashlib.sha256(str(x).encode()).digest()
        return int.from_bytes(h[:8], byteorder='big', signed=False)

    def add(self, other: 'IBLTSymbol'):
        self.val["count"] += other.val["count"]
        self.val["indexSum"] += other.val["indexSum"]
        self.val["valueSum"] += other.val["valueSum"]
        self.val["hashSum"] ^= other.val["hashSum"]

    def sub(self, other: 'IBLTSymbol'):
        self.val["count"] -= other.val["count"]
        self.val["indexSum"] -= other.val["indexSum"]
        self.val["valueSum"] -= other.val["valueSum"]
        self.val["hashSum"] ^= other.val["hashSum"]

    def is_zero(self) -> bool:
        return all(self.val[k] == 0 for k in self.val)

    def is_pure(self) -> bool:
        if abs(self.val["count"]) != 1:
            return False
        return self._hash_int(self.val["indexSum"]) == self.val["hashSum"]

    @classmethod
    def from_kv(cls, index: int, value: int) -> 'IBLTSymbol':
        return cls({
            "count": 1,
            "indexSum": index,
            "valueSum": value,
            "hashSum": cls._hash_int(index),
        })


class IndexGenerator(ABC):
    """
    索引生成器的抽象基类。
    用于在 Sketch 的内部 storage 中为符号生成一系列(随机)索引。
    """

    @abstractmethod
    def jump(self) -> None:
        """更新内部状态，跳到下一个索引。"""
        pass

    @property
    @abstractmethod
    def curr(self) -> int:
        """返回当前索引值。"""
        pass


class IBLTIndexGenerator(IndexGenerator):
    """
    演示版的 IBLT 索引生成器，以一个 int 作为随机种子，
    并使用 “Magic Formula” 计算跳跃。
    """
    def __init__(self, sym:IBLTSymbol, ):
        self._random = random.Random(seed)
        self._curr = 0

    def jump(self) -> None:
        # 生成一个随机数
        rand_val = self._random.randint(1, 2**31 - 1)
        # 这里与 C++ 的公式并不完全相同，仅做演示：
        new_offset = math.ceil((self._curr + 1.5) * ((1 << 32) / math.sqrt(rand_val + 1) - 1))
        self._curr += int(new_offset)

    @property
    def curr(self) -> int:
        return self._curr


class SketchQueue(ABC):
    """
    抽象基类：维护 (Symbol, IndexGenerator) 的优先队列
    """
    @abstractmethod
    def enqueue(self, symbol: SketchSymbol, generator: IndexGenerator) -> None:
        pass

    @abstractmethod
    def next_coded_index(self) -> int:
        pass

    @abstractmethod
    def top(self) -> (SketchSymbol, IndexGenerator):
        pass

    @abstractmethod
    def bubble_up(self) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

class IBLTQueue(SketchQueue):
    """
    具体的队列实现，使用 heapq 按照 “下一个写入索引” 升序组织。
    """
    
    def __init__(self):
        # priority queue 存储 (curr_index, symbol, generator)
        self._heap: List = []

    def enqueue(self, symbol: SketchSymbol, generator: IndexGenerator) -> None:
        heapq.heappush(self._heap, (generator.curr, symbol, generator))

    def next_coded_index(self) -> int:
        if not self._heap:
            return float('inf')
        return self._heap[0][0]

    def top(self) -> (SketchSymbol, IndexGenerator):
        if not self._heap:
            raise IndexError("Queue is empty.")
        _, symbol, generator = self._heap[0]
        return (symbol, generator)

    def bubble_up(self) -> None:
        """
        将堆顶元素弹出并重新放回，以更新其 curr 索引在堆中的顺序。
        """
        if not self._heap:
            raise IndexError("Queue is empty.")
        _, symbol, generator = heapq.heappop(self._heap)
        # generator.curr 可能已更新
        heapq.heappush(self._heap, (generator.curr, symbol, generator))

    def clear(self) -> None:
        self._heap.clear()    



# --- SketchQueue 抽象类 ---
class QueueEntry:
    def __init__(self, coded_index: int, symbol_index: int):
        self.coded_index = coded_index
        self.symbol_index = symbol_index

    def __lt__(self, other):
        return self.coded_index < other.coded_index


# --- Sketch 抽象类 ---
class Sketch(ABC):
    def __init__(self):
        self.symbols: List[SketchSymbol] = [] 需要插入的元素
        self.queue: SketchQueue = self.make_queue()
        self.done_expanding = False
        self.next_peel = 0

    @abstractmethod
    def make_generator(self, symbol: SketchSymbol) -> IndexGenerator:
        pass

    @abstractmethod
    def make_queue(self) -> SketchQueue:
        pass

    def expand(self, n: int):
        if self.done_expanding:
            raise RuntimeError("Can't expand fixed size sketch")
        self.symbols.extend([self.zero_symbol() for _ in range(n)])
        while self.queue.next_coded_index() < len(self.symbols):
            gen, sym = self.queue.top()
            self.diffuse_symbol(sym, gen)
            self.queue.bubble_up()

    def push(self, sym: SketchSymbol):
        self.expand(1)
        self.symbols[-1].add(sym)

    def set_done_expanding(self):
        self.done_expanding = True
        self.queue.clear()

    def add(self, sym: SketchSymbol, on_symbol: Callable[[int], None] = lambda _: None):
        gen = self.make_generator(sym)
        self.diffuse_symbol(sym, gen, on_symbol)
        if not self.done_expanding:
            self.queue.enqueue(sym, gen)

    def peel(self, on_pure: Callable[[SketchSymbol], None]):
        decodeable = []
        while self.next_peel < len(self.symbols):
            if not self.symbols[self.next_peel].is_pure():
                self.next_peel += 1
                continue
            decodeable.append(self.next_peel)
            self.next_peel += 1
            while decodeable:
                idx = decodeable.pop()
                if self.symbols[idx].is_zero():
                    continue
                on_pure(self.symbols[idx])
                neg = self.copy_and_negate(self.symbols[idx])
                self.add(neg, lambda new_idx: (
                    decodeable.append(new_idx) if self.symbols[new_idx].is_pure() else None
                ))

    def is_balanced(self):
        return len(self.symbols) > 0 and self.symbols[0].is_zero()

    def diffuse_symbol(self, sym: SketchSymbol, gen: IndexGenerator, cb: Callable[[int], None] = lambda _: None):
        while gen.curr < len(self.symbols):
            self.symbols[gen.curr].add(sym)
            cb(gen.curr)
            gen.jump()

    @abstractmethod
    def zero_symbol(self) -> SketchSymbol:
        pass

    @abstractmethod
    def copy_and_negate(self, sym: SketchSymbol) -> SketchSymbol:
        pass


# --- IBLT 实现类 ---
class IBLT(Sketch):
    def make_generator(self, symbol: IBLTSymbol) -> IndexGenerator:
        return IBLTIndexGenerator(symbol.val["indexSum"])

    def make_queue(self) -> SketchQueue:
        return IBLTQueue()

    def zero_symbol(self) -> SketchSymbol:
        return IBLTSymbol()

    def copy_and_negate(self, sym: IBLTSymbol) -> SketchSymbol:
        new_sym = IBLTSymbol(dict(sym.val))
        new_sym.val["count"] *= -1
        return new_sym


# --- 测试函数 ---
def test_iblt():
    iblt = IBLT()
    kvs = [(10, 100), (20, 200), (30, 300)]
    for index, value in kvs:
        sym = IBLTSymbol.from_kv(index, value)
        iblt.add(sym)

    iblt.set_done_expanding()

    print("Peeling:")
    iblt.peel(lambda s: print("Decoded:", s.val))


if __name__ == "__main__":
    test_iblt()
