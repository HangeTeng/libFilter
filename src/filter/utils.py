# src/filter/utils.py

import hashlib
import math
import random
from typing import Final, Iterator, List, Union, Dict, Any

# --- Public Types & Constants ---
ALLOWED_HASH_TYPES: Final = frozenset(["sha256", "blake2b", "sha1", "blake2s"])
InputType = Union[int, str, bytes]

# --- Internal Helper Functions ---
def _inputtype_to_bytes(data: InputType) -> bytes:
    """Normalizes a standard input type to bytes for hashing."""
    if isinstance(data, bytes): return data
    if isinstance(data, str): return data.encode("utf-8")
    if isinstance(data, int):
        if data < 0: raise ValueError("Integer keys cannot be negative.")
        byte_length = (data.bit_length() + 7) // 8 or 1
        return data.to_bytes(byte_length, "little")
    raise TypeError(f"Unsupported input type: {type(data).__name__}")

def _xor_bytes(b1: bytes, b2: bytes) -> bytes:
    """Performs a fast, length-tolerant XOR operation on two byte strings."""
    if len(b1) < len(b2): b1, b2 = b2, b1
    result = bytearray(b1)
    for i, byte_val in enumerate(b2): result[i] ^= byte_val
    return bytes(result)

# --- Primary Utility Classes ---
class prng:
    """A seedable, reproducible pseudo-random number generator (PRNG)."""
    __slots__ = '_rng'
    def __init__(self, seed: int): self._rng = random.Random(seed)
    def random(self) -> float: return self._rng.random()

class Hasher:
    """A configurable, seeded hasher for generating deterministic hash values."""
    __slots__ = 'algo', 'seed', '_base_hasher'

    def __init__(self, seed: InputType, algo: str = "sha256"):
        algo = algo.lower()
        if algo not in ALLOWED_HASH_TYPES: raise ValueError(f"Unsupported hash: {algo}")
        self.algo = algo
        self.seed = seed
        self._base_hasher = hashlib.new(algo)
        self._base_hasher.update(_inputtype_to_bytes(seed))

    def digest_int(self, data: InputType, nbytes: int = 8) -> int:
        """Computes the hash of data and returns it as an integer."""
        h = self._base_hasher.copy()
        h.update(_inputtype_to_bytes(data))
        return int.from_bytes(h.digest()[:nbytes], "big")

    def prng(self, data: InputType) -> prng:
        """Creates a deterministic PRNG seeded with the hash of the input data."""
        return prng(self.digest_int(data))

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable configuration of the hasher."""
        return {'seed': self.seed, 'algo': self.algo}
    
    def __repr__(self) -> str:
        return f"Hasher(seed={self.seed!r}, algo='{self.algo}')"

class HashMapping:
    """Provides k-hash mapping for standard filters (BF, CBF, IBLT)."""
    __slots__ = 'hashers', 'table_size'

    def __init__(self, hashers: List[Hasher], table_size: int):
        if not table_size > 0: raise ValueError("table_size must be positive.")
        self.hashers = hashers
        self.table_size = table_size

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HashMapping":
        """Creates a HashMapping from a configuration dictionary."""
        hashers = [Hasher(**hasher_config) for hasher_config in config['hashers']]
        return cls(hashers, config['table_size'])

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable configuration of the hash mapping."""
        return {
            'hashers': [h.get_config() for h in self.hashers],
            'table_size': self.table_size
        }

    @classmethod
    def from_seeds(cls, seeds: List, table_size: int, algo: str = "sha256") -> "HashMapping":
        """A convenient factory to create a HashMapping from a list of seeds."""
        hashers = [Hasher(seed=s, algo=algo) for s in seeds]
        return cls(hashers, table_size)

    def indices(self, key: InputType) -> Iterator[int]:
        """Generates k hash indices for a given key within the table size."""
        if self.table_size <= 0: return
        for hasher in self.hashers:
            yield hasher.digest_int(key) % self.table_size

    def __repr__(self) -> str:
        return f"HashMapping(k={len(self.hashers)}, m={self.table_size})"

class IndexGenerator:
    """Generates an index sequence for RIBLT based on fountain code principles."""
    __slots__ = '_prng', 'curr', '_key', '_seed'

    def __init__(self, key: InputType, seed: InputType):
        self._key, self._seed = key, seed
        self._prng = Hasher(seed).prng(key)
        self.curr = 0

    def jump(self) -> None:
        """Calculates and jumps to the next index using the RIBLT formula."""
        r = self._prng.random()
        factor = ((1 << 32) / math.sqrt(r + 1e-9)) - 1.0
        increment = math.ceil((self.curr + 1.5) * factor)
        self.curr += max(1, int(increment))

    def __repr__(self) -> str:
        return f"IndexGenerator(key={self._key!r}, seed={self._seed!r})"