# libfilter/core/utils.py

"""
Core utility components for the filter library.

This module provides fundamental building blocks used across different filter
implementations. It includes type definitions, data normalization helpers,
and essential classes for hashing and deterministic random number generation.
"""

import hashlib
import math
import random
from typing import Final, Iterator, List, Union, Dict, Any

# --- Public Types & Constants ---

ALLOWED_HASH_TYPES: Final = frozenset({"sha256", "blake2b", "sha1", "blake2s"})
"""A set of supported hashing algorithm names provided by hashlib."""

InputType = Union[int, str, bytes]
"""A type alias for standard data types that can be processed by filters."""


# --- Internal Helper Functions ---

def _normalize_input(data: InputType) -> bytes:
    """
    Converts a standard input type to bytes for consistent hashing.

    Args:
        data: The input data, which can be an int, str, or bytes.

    Returns:
        The byte representation of the input data.

    Raises:
        ValueError: If an integer key is negative.
        TypeError: If the input type is not supported.
    """
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, int):
        if data < 0:
            raise ValueError("Integer keys cannot be negative.")
        # Calculate the minimum number of bytes required to represent the integer.
        # The 'or 1' clause correctly handles the case where data is 0.
        byte_length = (data.bit_length() + 7) // 8 or 1
        return data.to_bytes(byte_length, "little")
    raise TypeError(f"Unsupported input type: {type(data).__name__}")


def _xor_bytes(b1: bytes, b2: bytes) -> bytes:
    """
    Performs a length-tolerant XOR operation on two byte strings.

    The operation extends to the length of the longer byte string. If one
    string is shorter, it is effectively padded with zeros during the XOR.

    Args:
        b1: The first byte string.
        b2: The second byte string.

    Returns:
        A new byte string containing the result of the XOR operation.
    """
    # Ensure b1 is the longer or equal-length string to simplify the loop.
    if len(b1) < len(b2):
        b1, b2 = b2, b1

    result = bytearray(b1)
    # The loop iterates up to the length of the shorter string (b2).
    for i, byte_val in enumerate(b2):
        result[i] ^= byte_val
    return bytes(result)


# --- Primary Utility Classes ---

class PRNG:
    """
    A simple, seedable pseudo-random number generator (PRNG).
    
    This class wraps Python's `random.Random` to create a deterministic and
    reproducible sequence of random numbers from a given integer seed.
    It is not intended for cryptographic use.
    """
    __slots__ = ('_rng',)

    def __init__(self, seed: int):
        """Initializes the PRNG with a specific seed."""
        self._rng = random.Random(seed)

    def random(self) -> float:
        """Returns the next random float in the range [0.0, 1.0)."""
        return self._rng.random()


class Hasher:
    """
    A configurable, seeded hasher for generating deterministic hash values.
    
    It uses a base seed and a specified algorithm from `hashlib` to produce
    consistent hashes. It pre-computes a seeded base hasher to optimize
    performance when hashing multiple items.
    """
    __slots__ = ('algo', 'seed', '_base_hasher')

    def __init__(self, seed: InputType, algo: str = "sha256"):
        """
        Initializes the Hasher.

        Args:
            seed: A seed that is mixed into every hash to ensure uniqueness.
            algo: The name of the hash algorithm to use (e.g., "sha256").

        Raises:
            ValueError: If the specified algorithm is not in ALLOWED_HASH_TYPES.
        """
        algo = algo.lower()
        if algo not in ALLOWED_HASH_TYPES:
            raise ValueError(f"Unsupported hash algorithm: {algo}")

        self.algo = algo
        self.seed = seed
        
        # Create a base hasher instance pre-updated with the seed.
        # This avoids re-seeding for every hash operation.
        self._base_hasher = hashlib.new(algo)
        self._base_hasher.update(_normalize_input(seed))

    def digest_int(self, data: InputType, nbytes: int = 8) -> int:
        """
        Computes the hash of data and returns its integer representation.

        Args:
            data: The data to hash.
            nbytes: The number of bytes from the digest to use for the integer.

        Returns:
            The integer representation of the first `nbytes` of the hash digest.
        """
        # Copy the base hasher to avoid affecting the pre-seeded state.
        h = self._base_hasher.copy()
        h.update(_normalize_input(data))
        return int.from_bytes(h.digest()[:nbytes], "big")

    def create_prng(self, data: InputType) -> PRNG:
        """
        Creates a deterministic PRNG seeded with the hash of the input data.

        Args:
            data: The data to use for seeding the PRNG.

        Returns:
            A new, deterministically seeded PRNG instance.
        """
        seed_value = self.digest_int(data)
        return PRNG(seed_value)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable configuration of the hasher."""
        return {'seed': self.seed, 'algo': self.algo}
    
    def __repr__(self) -> str:
        return f"Hasher(seed={self.seed!r}, algo='{self.algo}')"


class HashMapping:
    """
    Provides k-hash mapping for standard filters (e.g., BF, CBF, IBLT).

    This class uses a list of `k` independent `Hasher` instances to map a
    given key to `k` different indices within a table of a specified size.
    """
    __slots__ = ('hashers', 'table_size')

    def __init__(self, hashers: List[Hasher], table_size: int):
        """
        Initializes the HashMapping.

        Args:
            hashers: A list of k Hasher instances.
            table_size: The size (m) of the table to map indices into.

        Raises:
            ValueError: If table_size is not a positive integer.
        """
        if table_size <= 0:
            raise ValueError("table_size must be a positive integer.")
        self.hashers = hashers
        self.table_size = table_size

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HashMapping":
        """Creates a HashMapping instance from a configuration dictionary."""
        hashers = [Hasher(**hasher_config) for hasher_config in config['hashers']]
        return cls(hashers, config['table_size'])

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable configuration of the hash mapping."""
        return {
            'hashers': [h.get_config() for h in self.hashers],
            'table_size': self.table_size
        }

    @classmethod
    def from_seeds(
        cls, seeds: List[InputType], table_size: int, algo: str = "sha256"
    ) -> "HashMapping":
        """A convenience factory to create a HashMapping from a list of seeds."""
        hashers = [Hasher(seed=s, algo=algo) for s in seeds]
        return cls(hashers, table_size)

    def indices(self, key: InputType) -> Iterator[int]:
        """
        Generates k hash indices for a given key using its hashers.

        Args:
            key: The key to map.

        Yields:
            A sequence of k integer indices in the range [0, table_size-1].
        """
        for hasher in self.hashers:
            yield hasher.digest_int(key) % self.table_size

    def __repr__(self) -> str:
        return f"HashMapping(k={len(self.hashers)}, m={self.table_size})"


class IndexGenerator:
    """
    Generates a fountain-code-like index sequence for RIBLT.
    
    This generator produces a deterministic, monotonically increasing sequence
    of indices for a given key. The distribution of indices is inspired by
    the formula in the original RIBLT paper to ensure good decoding properties.
    """
    __slots__ = ('_prng', 'curr', '_key', '_seed')

    def __init__(self, key: InputType, seed: InputType):
        """
        Initializes the IndexGenerator.

        Args:
            key: The key for which to generate the index sequence.
            seed: A seed to make the sequence deterministic and unique.
        """
        self._key, self._seed = key, seed
        # The sequence is determined by a PRNG seeded from the key and a global seed.
        self._prng = Hasher(seed).create_prng(key)
        self.curr = 0

    def jump(self) -> None:
        """Calculates and advances to the next index in the sequence."""
        # Get a random float in (0, 1] to avoid division by zero.
        r = 1.0 - self._prng.random() # Maps to (0.0, 1.0]
        
        # The factor is now simply derived from 1/sqrt(r).
        # The scale of the increment can be tuned by an optional constant `C` if needed.
        # Here we assume C=1.
        factor = (1.0 / math.sqrt(r)) - 1.0
        # factor = (1.0 / r) - 1.0
        
        increment = math.ceil((float(self.curr) + 1.5) * factor)
        
        # The robust increment is still a good idea.
        self.curr += max(1, int(increment))

    def __repr__(self) -> str:
        return f"IndexGenerator(key={self._key!r}, seed={self._seed!r})"
    

# --- Data Serialization Helpers ---

class _DataType:
    """Internal enum-like class for type identifiers in serialization."""
    BYTES: Final[int] = 0
    STR: Final[int] = 1
    INT: Final[int] = 2

def serialize_typed_value(data: InputType) -> bytes:
    """
    Serializes an InputType (int, str, bytes) into a typed byte string.
    
    The format is: [1-byte type_id] + [value_bytes].
    
    Args:
        data: The input data to serialize.

    Returns:
        A byte string containing the type information and the value.
    """
    if isinstance(data, bytes):
        return _DataType.BYTES.to_bytes(1, 'little') + data
    if isinstance(data, str):
        return _DataType.STR.to_bytes(1, 'little') + data.encode('utf-8')
    if isinstance(data, int):
        if data < 0:
            raise ValueError("Integer keys cannot be negative.")
        byte_length = (data.bit_length() + 7) // 8 or 1
        return _DataType.INT.to_bytes(1, 'little') + data.to_bytes(byte_length, 'little')
    raise TypeError(f"Unsupported type for serialization: {type(data).__name__}")


def deserialize_typed_value(data: bytes) -> InputType:
    """
    Deserializes a typed byte string back into its original Python type.

    Args:
        data: The byte string created by `serialize_typed_value`.

    Returns:
        The deserialized data as an int, str, or bytes.
        
    Raises:
        ValueError: If the data is malformed or has an unknown type ID.
    """
    if not data:
        # A common case for empty XOR sums.
        return b''
    if len(data) < 1:
        raise ValueError("Invalid data: cannot read type identifier.")

    type_id, value_bytes = data[0], data[1:]
    
    if type_id == _DataType.BYTES:
        return value_bytes
    if type_id == _DataType.STR:
        return value_bytes.decode('utf-8')
    if type_id == _DataType.INT:
        return int.from_bytes(value_bytes, 'little')
        
    raise ValueError(f"Unknown type_id in serialized data: {type_id}")