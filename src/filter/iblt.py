# src/filter/iblt.py

from typing import Any, Tuple, Iterator, Deque, List
from collections import deque
from .base import PeelableSymbol, StandardFilter
from .utils import InputType, Hasher, _xor_bytes, _inputtype_to_bytes

class IBLTSymbol(PeelableSymbol):
    def __init__(self, count: int = 0, key_sum: bytes = b'', 
                 value_sum: Any = b'', key_hash_sum: int = 0):
        self.count = count; self.key_sum = key_sum
        self.value_sum = value_sum; self.key_hash_sum = key_hash_sum

    def add(self, other: 'IBLTSymbol') -> None:
        self.count += other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        self.key_hash_sum ^= other.key_hash_sum

    def sub(self, other: 'IBLTSymbol') -> None:
        self.count -= other.count
        self.key_sum = _xor_bytes(self.key_sum, other.key_sum)
        self.value_sum = _xor_bytes(self.value_sum, other.value_sum)
        self.key_hash_sum ^= other.key_hash_sum

    def is_empty(self) -> bool:
        return self.count == 0 and self.key_hash_sum == 0 and not self.key_sum

    def is_pure(self, hasher: Hasher) -> bool:
        if abs(self.count) != 1: return False
        return hasher.digest_int(self.key_sum) == self.key_hash_sum

    def get_key_value(self) -> Tuple[bytes, Any]:
        return self.key_sum, self.value_sum
    
    def negate(self) -> 'IBLTSymbol':
        return IBLTSymbol(-self.count, self.key_sum, self.value_sum, self.key_hash_sum)
        
    def __str__(self) -> str:
        if self.is_empty(): return "Empty"
        key_hex = self.key_sum.hex()[:12]
        val_str = self.value_sum.hex()[:12] if isinstance(self.value_sum, bytes) else str(self.value_sum)
        return (f"count={self.count}, key_sum={key_hex}..., "
                f"value_sum={val_str}..., key_hash_sum={self.key_hash_sum}")

class IBLT(StandardFilter[IBLTSymbol]):
    """可逆布隆查找表 (Invertible Bloom Lookup Table)。"""
    def __init__(self, m: int, hashers: List[Hasher], validation_hasher: Hasher = None):
        super().__init__(m, IBLTSymbol, hashers)
        if validation_hasher is None:
            default_seed = f"iblt-validation-{self.k}-{hashers[0].seed}"
            self.validation_hasher = Hasher(seed=default_seed)
        else:
            self.validation_hasher = validation_hasher
    
    @classmethod
    def from_seeds(cls, m: int, hash_seeds: List[int], hash_algo: str = "sha256", 
                   validation_hasher: Hasher = None) -> 'IBLT':
        hashers = [Hasher(seed=s, algo=hash_algo) for s in hash_seeds]
        return cls(m, hashers, validation_hasher)

    def _create_source_symbol(self, key: InputType, value: Any) -> IBLTSymbol:
        key_bytes = _inputtype_to_bytes(key); value_bytes = _inputtype_to_bytes(value)
        key_hash = self.validation_hasher.digest_int(key_bytes)
        return IBLTSymbol(1, key_bytes, value_bytes, key_hash)

    def add(self, key: InputType, value: Any = b''):
        source_symbol = self._create_source_symbol(key, value)
        for index in self._get_indices(key):
            self.cells[index].add(source_symbol)

    def sub(self, key: InputType, value: Any = b''):
        source_symbol = self._create_source_symbol(key, value)
        for index in self._get_indices(key):
            self.cells[index].sub(source_symbol)

    def peel(self) -> Iterator[Tuple[bytes, Any, int]]:
        pure_indices: Deque[int] = deque(i for i, cell in enumerate(self.cells) if cell.is_pure(self.validation_hasher))
        while pure_indices:
            index = pure_indices.popleft()
            cell = self.cells[index]
            if not cell.is_pure(self.validation_hasher): continue
            key, value = cell.get_key_value()
            yield key, value, cell.count
            self.sub(key, value)
            for idx in self._get_indices(key):
                if self.cells[idx].is_pure(self.validation_hasher) and idx not in pure_indices:
                    pure_indices.append(idx)
                    
    def __sub__(self, other: 'IBLT') -> 'IBLT':
        if self.m != other.m or len(self.hashers) != len(other.hashers):
            raise ValueError("IBLTs must have same m and number of hashers to be subtracted.")
        diff = IBLT(self.m, self.hashers, self.validation_hasher)
        for i in range(self.m):
            diff.cells[i].add(self.cells[i])
            diff.cells[i].sub(other.cells[i])
        return diff

# --- 模块自测试代码 ---
if __name__ == '__main__':
    print("--- Running Self-Test for Invertible Bloom Lookup Table (IBLT) ---")
    set_A = {f"item-{i}" for i in range(10)}
    set_B = {f"item-{i}" for i in range(5, 15)}

    iblt_A = IBLT.from_seeds(m=100, hash_seeds=list(range(4)))
    for item in set_A: iblt_A.add(item, f"val_A_{item}")
    
    iblt_B = IBLT.from_seeds(m=100, hash_seeds=list(range(4)))
    for item in set_B: iblt_B.add(item, f"val_B_{item}")
    
    print("\n计算 A 和 B 的差集 IBLT...")
    diff_iblt = iblt_A - iblt_B
    # print(diff_iblt) # 取消注释以查看庞大的差集IBLT内部

    decoded_A = set()
    decoded_B = set()
    
    print("\n开始解码...")
    for key, value, count in diff_iblt.peel():
        if count == 1: decoded_A.add(key.decode())
        elif count == -1: decoded_B.add(key.decode())

    print(f"只在 A 中的 (预期): {sorted(list(set_A - set_B))}")
    print(f"只在 A 中的 (解码): {sorted(list(decoded_A))}")
    print(f"只在 B 中的 (预期): {sorted(list(set_B - set_A))}")
    print(f"只在 B 中的 (解码): {sorted(list(decoded_B))}")
    
    assert set_A - set_B == decoded_A
    assert set_B - set_A == decoded_B
    print("\n--- IBLT Self-Test Completed Successfully! ---")