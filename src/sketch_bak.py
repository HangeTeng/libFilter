import hashlib
from typing import Dict, Union, List
from abc import ABC, abstractmethod

# Define the Scalar type, which can be boolean, bytes, float, integer, or string
Scalar = Union[bool, bytes, float, int, str]

# Define the Value type, which can be Scalar or a list of Scalar
Value = Union[
    bool, bytes, float, int, str,
    List[bool], List[bytes], List[float], List[int], List[str]
]

# Define SymbolValue as a dictionary where keys are strings and values are of Value type
ValueDict = Dict[str, Value]

class SketchSymbol(ABC):
    """Abstract base class representing a sketch symbol"""

    def __init__(self, val: ValueDict):
        self.val = val

    def __str__(self):
        return str(self.val)

    @abstractmethod
    def add(self, other: 'SketchSymbol'):
        """
        Perform addition operation on the symbol, adding the content of another symbol to the current one
        
        Args:
            other: Another SketchSymbol object to be added
        """
        pass

    @abstractmethod
    def sub(self, other: 'SketchSymbol'):
        """
        Perform subtraction operation on the symbol, subtracting the content of another symbol from the current one
        
        Args:
            other: Another SketchSymbol object to be subtracted
        """
        pass

    @abstractmethod
    def is_pure(self) -> bool:
        """
        Check if the symbol is pure, which means its value contains only one fixed non-zero element
        
        Returns:
            True if the symbol is pure, False otherwise
        """
        pass

    @abstractmethod
    def is_zero(self) -> bool:
        """
        Check if the symbol is zero, which means it contains no elements
        
        Returns:
            True if the symbol contains no elements, False otherwise
        """
        pass


class IBLTSymbol(SketchSymbol):
    """
    IBLT Symbol represents a single cell in an Invertible Bloom Lookup Table.
    
    Each symbol stores:
        - count:     The number of key-value pairs mapped to this cell
        - indexSum:  The XOR sum of all integer keys (used instead of keySum)
        - valueSum:  The XOR sum of all integer values
        - hashSum:   The XOR sum of all key hashes for collision checking
    """

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
        """
        Generate a fixed-size hash (as int) from an integer key using SHA-256

        Args:
            x: The integer key to be hashed

        Returns:
            A 64-bit integer derived from SHA-256 hash of the input
        """
        h = hashlib.sha256(str(x).encode()).digest()
        return int.from_bytes(h[:8], byteorder='big')  # use first 8 bytes

    def add(self, other: 'IBLTSymbol'):
        """
        Add another IBLTSymbol into the current one using XOR and integer sum rules

        Args:
            other: Another IBLTSymbol to add
        """
        self.val["count"] += other.val["count"]
        self.val["indexSum"] += other.val["indexSum"]
        self.val["valueSum"] += other.val["valueSum"]
        self.val["hashSum"] ^= other.val["hashSum"]

    def sub(self, other: 'IBLTSymbol'):
        """
        Subtract another IBLTSymbol from the current one using XOR and integer subtraction

        Args:
            other: Another IBLTSymbol to subtract
        """
        self.val["count"] -= other.val["count"]
        self.val["indexSum"] -= other.val["indexSum"]
        self.val["valueSum"] -= other.val["valueSum"]
        self.val["hashSum"] ^= other.val["hashSum"]

    def is_zero(self) -> bool:
        """
        Check if the symbol is in an empty (zero) state

        Returns:
            True if all fields are zero, False otherwise
        """
        return (
            self.val["count"] == 0 and
            self.val["indexSum"] == 0 and
            self.val["valueSum"] == 0 and
            self.val["hashSum"] == 0
        )

    def is_pure(self) -> bool:
        """
        Check if the symbol contains exactly one item (count == ±1) and is decodable

        Returns:
            True if the symbol is pure, False otherwise
        """
        if abs(self.val["count"]) != 1:
            return False
        expected_hash = self._hash_int(self.val["indexSum"])
        return expected_hash == self.val["hashSum"]

    @classmethod
    def from_kv(cls, index: int, value: int) -> 'IBLTSymbol':
        """
        Factory method to construct an IBLTSymbol from a single key-value pair

        Args:
            index: The integer key (index)
            value: The integer value

        Returns:
            An IBLTSymbol representing one inserted element
        """
        return cls({
            "count": 1,
            "indexSum": index,
            "valueSum": value,
            "hashSum": cls._hash_int(index),
        })


def test_iblt_symbol():
    sym1 = IBLTSymbol.from_kv(10, 100)
    sym2 = IBLTSymbol.from_kv(20, 200)

    print("Symbol1:", sym1)
    print("Symbol2:", sym2)

    sym1.add(sym2)
    print("\nAfter add:")
    print("sym1:", sym1)

    sym1.sub(sym2)
    print("\nAfter sub (should return to original sym1):")
    print("sym1:", sym1)

    print("\nIs sym1 pure?", sym1.is_pure())
    print("Is sym1 zero?", sym1.is_zero())


test_iblt_symbol()
