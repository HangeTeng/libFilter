# src/filter/utils.py

import hashlib
import random
import math
from typing import Final, Union, List, Iterator

# --- 类型与常量定义 ---
ALLOWED_HASH_TYPES: Final = frozenset(["sha256", "blake2b", "sha1", "blake2s"])
InputType = Union[int, str, bytes]

# --- 工具函数 ---
def _inputtype_to_bytes(data: InputType) -> bytes:
    match data:
        case bytes(): return data
        case str(): return data.encode("utf-8")
        case int() if data >= 0:
            byte_length = (data.bit_length() + 7) // 8 or 1
            return data.to_bytes(byte_length, "little")
        case int(): raise ValueError("整数不能为负")
        case _: raise TypeError(f"不支持的类型: {type(data).__name__}")

def _xor_bytes(b1: bytes, b2: bytes) -> bytes:
    length = max(len(b1), len(b2))
    b1 = b1.ljust(length, b'\0')
    b2 = b2.ljust(length, b'\0')
    return bytes(x ^ y for x, y in zip(b1, b2))

class prng:
    """仅支持 random() 的精简 PRNG。"""
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

class Hasher:
    """一个可配置的哈希计算器，用于将输入数据映射到一个64位整数。"""
    def __init__(self, seed: InputType, algo: str = "sha256"):
        algo = algo.lower()
        if algo not in ALLOWED_HASH_TYPES:
            raise ValueError(f"不支持的哈希算法: {algo}")
        self._base_hasher = hashlib.new(algo)
        self._base_hasher.update(_inputtype_to_bytes(seed))

    def digest_int(self, data: InputType, nbytes: int = 8) -> int:
        h = self._base_hasher.copy()
        h.update(_inputtype_to_bytes(data))
        return int.from_bytes(h.digest()[:nbytes], "big")

    def prng(self, data: InputType) -> prng:
        return prng(self.digest_int(data))

class HashMapping:
    """BF / CBF / IBLT 等数据结构共用的 k 重哈希映射器。"""
    def __init__(self, hashers: List[Hasher], table_size: int):
        if not table_size > 0: raise ValueError("table_size 必须为正数")
        self.hashers = hashers
        self.table_size = table_size

    @classmethod
    def from_seeds(cls, seeds: List[int], table_size: int, algo: str = "sha256") -> "HashMapping":
        hashers = [Hasher(seed=s, algo=algo) for s in seeds]
        return cls(hashers, table_size)

    def indices(self, key: InputType) -> Iterator[int]:
        if self.table_size <= 0: return
        for hasher in self.hashers:
            yield hasher.digest_int(key) % self.table_size

class IndexGenerator:
    """RIBLT 使用的索引序列生成器。"""
    def __init__(self, key: InputType, seed: InputType):
        hasher = Hasher(seed)
        self._prng = hasher.prng(key)
        self.curr = 0

    def jump(self) -> None:
        """使用随机浮点数跳跃，公式与 C++ 版本对应。"""
        r = self._prng.random()
        self.curr += math.ceil((self.curr + 1.5) * (((1 << 32) / math.sqrt(r + 1e-9)) - 1.0))