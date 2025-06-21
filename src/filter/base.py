# src/filter/base.py (完整文件内容)

import abc
import math # 需要 math 库来计算宽度
from typing import List, Iterator, TypeVar, Generic, Callable, Any, Tuple
from .utils import InputType, Hasher, HashMapping

SymbolType = TypeVar('SymbolType', bound='FilterSymbol')

# ... (FilterSymbol, PeelableSymbol 的定义不变) ...
class FilterSymbol(abc.ABC):
    @abc.abstractmethod
    def add(self, other: 'FilterSymbol') -> None: pass
    @abc.abstractmethod
    def is_empty(self) -> bool: pass
    @abc.abstractmethod
    def __str__(self) -> str: pass

class PeelableSymbol(FilterSymbol):
    @abc.abstractmethod
    def sub(self, other: 'PeelableSymbol') -> None: pass
    @abc.abstractmethod
    def is_pure(self, hasher: Hasher) -> bool: pass
    @abc.abstractmethod
    def get_key_value(self) -> Tuple[bytes, Any]: pass
    @abc.abstractmethod
    def negate(self) -> 'PeelableSymbol': pass


class FilterBase(abc.ABC, Generic[SymbolType]):
    """所有过滤器的抽象基类。"""
    def __init__(self):
        self.cells: List[SymbolType] = []
    
    @abc.abstractmethod
    def _get_indices(self, item: InputType) -> Iterator[int]:
        """为给定项生成索引序列。"""
        pass

    def __len__(self) -> int:
        return len(self.cells)

    def __str__(self) -> str:
        """
        提供过滤器的字符串表示，打印所有单元格的状态，并对齐索引。
        """
        size = len(self.cells)
        if size == 0:
            return f"{self.__class__.__name__}(size=0):\n  (empty)"
            
        lines = [f"{self.__class__.__name__}(size={size}):"]
        
        # 计算索引列所需的最大宽度
        # 例如，如果 size=1000, m-1=999, len("999")=3
        # 如果 size=10, m-1=9, len("9")=1
        # 使用 math.log10 可以高效计算位数，但要处理 size=1 的情况
        max_index_width = math.floor(math.log10(size - 1)) + 1 if size > 1 else 1

        for i, cell in enumerate(self.cells):
            # 使用 f-string 进行格式化：
            # {i:<{width}} - 左对齐
            # {i:>{width}} - 右对齐
            # 我们使用右对齐，看起来更像数字列表
            lines.append(f"  [{i:>{max_index_width}}]: {cell}")
            
        return "\n".join(lines)

# --- 修改后的 StandardFilter ---

class StandardFilter(FilterBase[SymbolType]):
    """
    适用于 BF, CBF, IBLT 等固定大小过滤器的基类。
    它现在通过 HashMapping 对象来维护索引映射。
    """
    def __init__(self, symbol_factory: Callable[[], SymbolType], 
                 hash_mapping: HashMapping):
        """
        主构造函数：直接接收一个预先配置好的 HashMapping 对象。
        
        Args:
            symbol_factory (Callable): 用于创建空符号的工厂函数。
            hash_mapping (HashMapping): 一个配置好的哈希映射器实例。
        """
        super().__init__()
        self.hash_mapping = hash_mapping
        self.m = hash_mapping.table_size
        self.k = len(hash_mapping.hashers)
        self.cells = [symbol_factory() for _ in range(self.m)]

    def _get_indices(self, item: InputType) -> Iterator[int]:
        return self.hash_mapping.indices(item)