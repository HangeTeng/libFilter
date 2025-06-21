# src/filter/cbf.py

from typing import List
from .base import FilterSymbol, StandardFilter
from .utils import InputType, Hasher, HashMapping

class CBFSymbol(FilterSymbol):
    """计数布隆过滤器的单元格。"""
    def __init__(self, count: int = 0):
        self.count = count
    
    def add(self, other: 'CBFSymbol') -> None:
        self.count += other.count

    def is_empty(self) -> bool:
        return self.count == 0
    
    def __str__(self) -> str:
        return f"Count = {self.count}"

class CBF(StandardFilter[CBFSymbol]):
    """
    计数布隆过滤器 (Counting Bloom Filter)。
    它的构造函数与其父类一致，提供便利的工厂方法。
    """
    def __init__(self, hash_mapping: HashMapping):
        super().__init__(CBFSymbol, hash_mapping)

    @classmethod
    def from_hashers(cls, m: int, hashers: List[Hasher]) -> 'CBF':
        """通过 Hasher 列表创建 CBF。"""
        if not m > 0: raise ValueError("过滤器大小 (m) 必须为正数")
        if not hashers: raise ValueError("hashers 列表不能为空")
        hash_mapping = HashMapping(hashers, m)
        return cls(hash_mapping)

    @classmethod
    def from_seeds(cls, m: int, hash_seeds: List[int], hash_algo: str = "sha256") -> 'CBF':
        """通过种子列表创建 CBF。"""
        if not m > 0: raise ValueError("过滤器大小 (m) 必须为正数")
        if not hash_seeds: raise ValueError("hash_seeds 列表不能为空")
        hash_mapping = HashMapping.from_seeds(hash_seeds, m, hash_algo)
        return cls(hash_mapping)
    
    def add(self, item: InputType) -> None:
        """向过滤器中添加一个元素，对应计数器加一。"""
        source_symbol = CBFSymbol(1)
        for index in self._get_indices(item):
            self.cells[index].add(source_symbol)
    
    def sub(self, item: InputType) -> None:
        """
        从过滤器中移除一个元素，对应计数器减一。
        为防止下溢，只在元素可能存在时操作。
        """
        if self.query(item):
            source_symbol = CBFSymbol(-1)
            for index in self._get_indices(item):
                # 额外的保护，确保计数器不降到负数
                if self.cells[index].count > 0:
                    self.cells[index].add(source_symbol)
    
    def query(self, item: InputType) -> bool:
        """查询一个元素是否存在（所有对应计数器 > 0）。"""
        return all(self.cells[index].count > 0 for index in self._get_indices(item))
        
    def __contains__(self, item: InputType) -> bool:
        return self.query(item)

# --- 模块自测试代码 ---
if __name__ == '__main__':
    print("--- Running Self-Test for Counting Bloom Filter (CBF) ---")
    m_test, k_test = 15, 3
    cbf = CBF.from_seeds(m=m_test, hash_seeds=list(range(k_test)))
    print(f"创建了一个 m={cbf.m}, k={cbf.k} 的计数布隆过滤器。")

    print("\n添加 'apple' 两次, 'banana' 一次。")
    cbf.add("apple")
    cbf.add("apple")
    cbf.add("banana")
    
    print("\n过滤器内部状态:")
    print(cbf)
    
    print("\n--- 查询验证 ---")
    assert "apple" in cbf, "测试失败: 'apple' 应该在过滤器中"
    print("'apple' in cbf (count=2) -> True (正确)")
    
    assert "banana" in cbf, "测试失败: 'banana' 应该在过滤器中"
    print("'banana' in cbf (count=1) -> True (正确)")

    assert not ("cherry" in cbf), "测试失败(可能因哈希碰撞): 'cherry' 不应在过滤器中"
    print("'cherry' in cbf -> False (正确)")
    
    print("\n--- 删除验证 ---")
    print("\n移除 'apple' 一次。")
    cbf.sub("apple")
    assert "apple" in cbf, "测试失败: 'apple' 移除一次后仍应存在"
    print("查询 'apple' -> True (正确)")
    
    print("\n再次移除 'apple'。")
    cbf.sub("apple")
    assert not ("apple" in cbf), "测试失败: 'apple' 移除两次后不应存在"
    print("查询 'apple' -> False (正确)")
    
    print("\n尝试移除不存在的 'cherry'。")
    cbf.sub("cherry") # 不应该产生错误或改变状态
    print("操作完成，状态应无变化。")
    
    print("\n移除 'banana'。")
    cbf.sub("banana")
    assert not ("banana" in cbf), "测试失败: 'banana' 移除后不应存在"
    print("查询 'banana' -> False (正确)")
    
    print("\n最终过滤器内部状态:")
    print(cbf)
    
    print("\n--- CBF Self-Test Completed Successfully! ---")