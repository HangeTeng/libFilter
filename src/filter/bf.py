# src/filter/bf.py

from typing import List
from .base import FilterSymbol, StandardFilter
from .utils import InputType, Hasher, HashMapping

class BFSymbol(FilterSymbol):
    def __init__(self, value: bool = False):
        self.value = value
    
    def add(self, other: 'BFSymbol') -> None:
        self.value = self.value or other.value
    
    def is_empty(self) -> bool:
        return not self.value
    
    def __str__(self) -> str:
        return f"1" if self.value else "0"

class BF(StandardFilter[BFSymbol]):
    """标准布隆过滤器 (Bloom Filter)。"""
    def __init__(self, hash_mapping: HashMapping):
        # 调用父类构造函数，并传入自己的符号工厂和哈希映射器
        super().__init__(BFSymbol, hash_mapping)
    
    @classmethod
    def from_hashers(cls, m: int, hashers: List[Hasher]) -> 'BF':
        """通过 Hasher 列表创建 BF。"""
        if not m > 0: raise ValueError("过滤器大小 (m) 必须为正数")
        if not hashers: raise ValueError("hashers 列表不能为空")
        hash_mapping = HashMapping(hashers, m)
        return cls(hash_mapping)

    @classmethod
    def from_seeds(cls, m: int, hash_seeds: List[int], hash_algo: str = "sha256") -> 'BF':
        """通过种子列表创建 BF。"""
        if not m > 0: raise ValueError("过滤器大小 (m) 必须为正数")
        if not hash_seeds: raise ValueError("hash_seeds 列表不能为空")
        hash_mapping = HashMapping.from_seeds(hash_seeds, m, hash_algo)
        return cls(hash_mapping)

    def add(self, item: InputType) -> None:
        source_symbol = BFSymbol(True)
        for index in self._get_indices(item):
            self.cells[index].add(source_symbol)

    def query(self, item: InputType) -> bool:
        return all(self.cells[index].value for index in self._get_indices(item))
    
    def __contains__(self, item: InputType) -> bool:
        return self.query(item)

# --- 模块自测试代码 (现在绝对可以工作) ---
if __name__ == '__main__':
    print("--- Running Self-Test for Bloom Filter (BF) ---")
    m_test, k_test = 20, 3
    # 正确的调用方式
    bf = BF.from_seeds(m=m_test, hash_seeds=list(range(k_test)))
    print(f"创建了一个 m={bf.m}, k={bf.k} 的布隆过滤器。")
    
    items_to_add = ["apple", "banana", 123]
    print(f"\n添加元素: {items_to_add}")
    for item in items_to_add:
        bf.add(item)

    print("\n过滤器内部状态:")
    print(bf)

    print("\n基本查询验证:")
    for item in items_to_add:
        assert item in bf, f"测试失败: '{item}' 应该在过滤器中。"
        print(f"'{item}' in filter? -> {item in bf} (正确)")
    
    item_not_added = "cherry"
    print(f"'{item_not_added}' in filter? -> {item_not_added in bf} (可能误报)")
    
    print("\n--- BF Self-Test Completed Successfully! ---")