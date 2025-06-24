import hashlib
import math
import random
from typing import Final, Iterator, List, Union

# --- Public Types & Constants ---
ALLOWED_HASH_TYPES: Final = frozenset(["sha256", "blake2b", "sha1", "blake2s"])
"""A set of supported hash algorithms."""

InputType = Union[int, str, bytes]
"""The allowed primitive types for filter keys."""

# --- Internal Helper Functions ---
def _inputtype_to_bytes(data: InputType) -> bytes:
    """Normalizes a standard input type to bytes for hashing."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, int):
        if data < 0:
            raise ValueError("Integer keys cannot be negative.")
        # Handle the edge case of 0, which has a bit_length of 0.
        byte_length = (data.bit_length() + 7) // 8 or 1
        return data.to_bytes(byte_length, "little")
    raise TypeError(f"Unsupported input type: {type(data).__name__}")

def _xor_bytes(b1: bytes, b2: bytes) -> bytes:
    """Performs a fast, length-tolerant XOR operation on two byte strings."""
    if len(b1) < len(b2):
        b1, b2 = b2, b1  # Ensure b1 is the longer one
    
    # Use bytearray for mutable, efficient operations
    result = bytearray(b1)
    # The loop safely stops at the end of the shorter byte string
    for i, byte_val in enumerate(b2):
        result[i] ^= byte_val
    return bytes(result)

# --- Primary Utility Classes ---
class prng:
    """A seedable, reproducible pseudo-random number generator (PRNG)."""
    __slots__ = '_rng'

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def random(self) -> float:
        """Returns a random float in the range [0.0, 1.0)."""
        return self._rng.random()

class Hasher:
    """A configurable, seeded hasher for generating deterministic hash values."""
    __slots__ = 'algo', 'seed', '_base_hasher'

    def __init__(self, seed: InputType, algo: str = "sha256"):
        """Initializes the hasher with a seed and a specific hash algorithm."""
        algo = algo.lower()
        if algo not in ALLOWED_HASH_TYPES:
            raise ValueError(f"Unsupported hash algorithm: {algo}")
        
        self.algo = algo
        self.seed = seed
        # Pre-hash the seed to create a unique base state for this hasher instance.
        self._base_hasher = hashlib.new(algo)
        self._base_hasher.update(_inputtype_to_bytes(seed))

    def digest_int(self, data: InputType, nbytes: int = 8) -> int:
        """Computes the hash of the given data and returns it as an integer."""
        # Copy the base hasher to avoid re-hashing the seed.
        h = self._base_hasher.copy()
        h.update(_inputtype_to_bytes(data))
        return int.from_bytes(h.digest()[:nbytes], "big")

    def prng(self, data: InputType) -> prng:
        """Creates a deterministic PRNG seeded with the hash of the input data."""
        return prng(self.digest_int(data))
    
    def __repr__(self) -> str:
        return f"Hasher(seed={self.seed!r}, algo='{self.algo}')"

class HashMapping:
    """Provides k-hash mapping for standard filters (BF, CBF, IBLT)."""
    __slots__ = 'hashers', 'table_size'

    def __init__(self, hashers: List[Hasher], table_size: int):
        if not table_size > 0:
            raise ValueError("table_size must be positive.")
        self.hashers = hashers
        self.table_size = table_size

    @classmethod
    def from_seeds(cls, seeds: List[int], table_size: int, algo: str = "sha256") -> "HashMapping":
        """A convenient factory method to create a HashMapping from a list of seeds."""
        hashers = [Hasher(seed=s, algo=algo) for s in seeds]
        return cls(hashers, table_size)

    def indices(self, key: InputType) -> Iterator[int]:
        """Generates k hash indices for a given key within the table size."""
        if self.table_size <= 0:
            return
        for hasher in self.hashers:
            yield hasher.digest_int(key) % self.table_size

    def __repr__(self) -> str:
        return f"HashMapping(hashers={self.hashers!r}, table_size={self.table_size})"

class IndexGenerator:
    """Generates an index sequence for RIBLT based on fountain code principles."""
    __slots__ = '_prng', 'curr', '_key', '_seed' # Store key/seed for repr

    def __init__(self, key: InputType, seed: InputType):
        self._key = key
        self._seed = seed
        # Each key gets its own unique, deterministic random sequence.
        hasher = Hasher(seed)
        self._prng = hasher.prng(key)
        self.curr = 0

    def jump(self) -> None:
        """Calculates and jumps to the next index using the RIBLT 'magic formula'."""
        r = self._prng.random()
        # Add a small epsilon to avoid division by zero if r is exactly 0.
        factor = ((1 << 32) / math.sqrt(r + 1e-9)) - 1.0
        increment = math.ceil((self.curr + 1.5) * factor)
        # Ensure the index always advances by at least 1.
        self.curr += max(1, int(increment))

    def __repr__(self) -> str:
        return f"IndexGenerator(key={self._key!r}, seed={self._seed!r})"