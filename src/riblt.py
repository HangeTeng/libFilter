import random
import math
import heapq
from typing import Optional, Callable, List


class Hash:
    def __init__(self, value=None):
        if value:
            self.h = self.hash_function(value)
        else:
            self.h = b'\0' * 32  # 默认空哈希

    def __eq__(self, other):
        return self.h == other.h

    def __ne__(self, other):
        return self.h != other.h

    def hash_function(self, value):
        # 这里使用简单的 hash 函数模拟
        return value.encode('utf-8')[:32]  # 简化为截取前32字节

    def get_prng(self):
        # 使用哈希值作为种子生成 PRNG
        return random.Random(self.h)


class CodedSymbol:
    def __init__(self, val_raw: str, count: int = 1):
        self.hash = Hash(val_raw)
        self.count = count
        self.val = f"{len(val_raw):<4}" + val_raw  # 将值前面加上长度信息

    def add(self, other):
        self.update(other, other.count)

    def sub(self, other):
        self.update(other, -other.count)

    def negate(self):
        self.count *= -1

    def is_pure(self):
        if self.count == 1 or self.count == -1:
            decoded_val = self.decode_val()
            if decoded_val:
                return Hash(decoded_val) == self.hash
        return False

    def is_zero(self):
        return self.count == 0 and self.hash == Hash()

    def get_val(self):
        decoded_val = self.decode_val()
        if not decoded_val:
            raise ValueError("val not decodeable (not pure?)")
        return decoded_val

    def decode_val(self):
        # 解码值
        if len(self.val) < 4:
            raise ValueError("val unexpectedly small")
        size = int(self.val[:4])
        return self.val[4:4 + size]

    def update(self, other, inc):
        # 更新符号
        max_len = max(len(self.val), len(other.val))
        self.val = self.val.ljust(max_len, '\0')
        other_val = other.val.ljust(max_len, '\0')

        # XOR 值
        self.val = ''.join(chr(ord(self_val) ^ ord(other_val)) for self_val, other_val in zip(self.val, other_val))

        # XOR 哈希
        self.hash = Hash(''.join(chr(h ^ ord(other_hash)) for h, other_hash in zip(self.hash.h, other.hash.h)))

        self.count += inc


class IndexGenerator:
    def __init__(self, sym: CodedSymbol):
        self.prng = sym.hash.get_prng()
        self.curr = 0

    def jump(self):
        rand = self.prng.randint(0, 2**64 - 1)

        # 魔法公式，来自 RIBLT 论文
        self.curr += int(math.ceil((self.curr + 1.5) * ((2**32) / math.sqrt(rand + 1) - 1)))


class SymbolQueue:
    class QueueEntry:
        def __init__(self, coded_stream_index: int, symbol_index: int):
            self.coded_stream_index = coded_stream_index
            self.symbol_index = symbol_index

        def __lt__(self, other):
            return self.coded_stream_index < other.coded_stream_index

    def __init__(self):
        self.symbol_vec = []
        self.coded_queue = []

    def enqueue(self, sym: CodedSymbol, gen: IndexGenerator):
        entry = self.QueueEntry(gen.curr, len(self.symbol_vec))
        heapq.heappush(self.coded_queue, entry)
        self.symbol_vec.append(sym)

    def next_coded_index(self):
        if not self.symbol_vec:
            return float('inf')
        return self.coded_queue[0].coded_stream_index

    def top(self):
        if not self.symbol_vec:
            raise ValueError("SymbolQueue empty")
        return self.symbol_vec[self.coded_queue[0].symbol_index]

    def bubble_up(self):
        if not self.symbol_vec:
            raise ValueError("can't bubbleUp empty queue")
        entry = self.coded_queue.pop(0)
        entry.coded_stream_index = self.symbol_vec[entry.symbol_index].count  # 使用 count 作为更新的依据
        heapq.heappush(self.coded_queue, entry)

    def clear(self):
        self.symbol_vec.clear()
        self.coded_queue.clear()


class RIBLT:
    def __init__(self):
        self.coded_symbols = []
        self.queue = SymbolQueue()
        self.done_expanding = False
        self.next_peel = 0

    def expand(self, n):
        if self.done_expanding:
            raise ValueError("can't expand fixed size vector")
        self.coded_symbols.extend([None] * n)

        while self.queue.next_coded_index() < len(self.coded_symbols):
            top = self.queue.top()
            self.diffuse_symbol(top, top.count)
            self.queue.bubble_up()

    def push(self, sym: CodedSymbol):
        self.expand(1)
        self.coded_symbols[-1] = sym
        self.queue.enqueue(sym, IndexGenerator(sym))

    def set_done_expanding(self):
        self.done_expanding = True
        self.queue.clear()

    def add(self, sym: CodedSymbol, on_sym: Optional[Callable[[int], None]] = None):
        gen = IndexGenerator(sym)
        self.diffuse_symbol(sym, gen, on_sym)
        if not self.done_expanding:
            self.queue.enqueue(sym, gen)

    def peel(self, on_sym: Callable[[CodedSymbol], None]):
        decodeable = []
        for i in range(self.next_peel, len(self.coded_symbols)):
            if not self.coded_symbols[i].is_pure():
                continue

            decodeable.append(i)

            while decodeable:
                index = decodeable.pop()
                if self.coded_symbols[index].is_zero():
                    continue

                on_sym(self.coded_symbols[index])

                sym = self.coded_symbols[index]
                sym.negate()
                self.add(sym, lambda new_index: decodeable.append(new_index) if self.coded_symbols[new_index].is_pure() else None)

    def is_balanced(self):
        return len(self.coded_symbols) > 0 and self.coded_symbols[0].is_zero()

    def diffuse_symbol(self, sym: CodedSymbol, gen: IndexGenerator, cb: Optional[Callable[[int], None]] = None):
        while gen.curr < len(self.coded_symbols):
            self.coded_symbols[gen.curr] = sym
            if cb:
                cb(gen.curr)
            gen.jump()

def test_riblt():
    # 创建 RIBLT 对象
    riblt = RIBLT()

    # 定义回调函数来处理解码后的符号
    def on_symbol(symbol: CodedSymbol):
        print(f"Decoded Symbol: {symbol.get_val()}")

    # 添加几个符号
    print("Adding Symbol 1: 'hello'")
    sym1 = CodedSymbol('hello', 1)
    riblt.add(sym1)

    print("\nAdding Symbol 2: 'world'")
    sym2 = CodedSymbol('world', 1)
    riblt.add(sym2)

    print("\nAdding Symbol 3: 'riblt'")
    sym3 = CodedSymbol('riblt', 1)
    riblt.add(sym3)

    # 执行符号扩展
    print("\nExpanding RIBLT with 3 symbols...")
    riblt.expand(3)

    # 执行符号解码过程
    print("\nPeeling and decoding symbols:")
    riblt.peel(on_symbol)

    # 检查 RIBLT 是否平衡
    if riblt.is_balanced():
        print("\nRIBLT is balanced!")
    else:
        print("\nRIBLT is not balanced.")

# 运行测试函数
test_riblt()
